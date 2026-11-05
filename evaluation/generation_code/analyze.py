"""Recalculate all generation metrics from immutable unedited output records."""
from __future__ import annotations
from collections import Counter, defaultdict
import itertools
import json
import math
import re
import statistics
from pathlib import Path

from generation import ROOT, EOS, METHODS, atomic_new, encoder

DEFINITIONS = {
    'analysis_token_sequence': 'Generated content_token_ids only: prompt excluded; a final EOT is excluded from repetition/diversity/whitespace metrics. Generated length and 128-token-cap metrics include EOT.',
    'repeated_ngram_rate': '(number of contiguous BPE n-gram occurrences minus number of distinct n-gram types) / number of occurrences. n=2,3,4; null/NA when content length<n. Decoder means exclude NA and give eligible counts.',
    'distinct_1': 'Unique generated BPE token types / content token count; null for empty content.',
    'distinct_2': 'Unique contiguous BPE bigram types / number of bigram occurrences; null for fewer than 2 content tokens.',
    'longest_repeated_token_subsequence': 'Length of the longest exact contiguous generated BPE token span occurring at least twice at non-overlapping positions. This is a contiguous substring statistic, not a noncontiguous subsequence metric. Zero means no token repeats.',
    'identical_phrases': 'Within-output repeated contiguous word 3-grams through 8-grams, after Unicode regex word extraction and casefolding. Counts include overlapping occurrences. Top 10 ranked by count descending, then longer phrase, then text; full aggregate phrase counts also saved.',
    'obvious_loop_proxy': 'Exact contiguous BPE motif with period 2–32 tokens repeated at least 3 consecutive times and spanning at least 12 tokens. Earliest qualifying onset is reported; longest span breaks onset ties. This is an explicit automatic proxy, not a semantic judgment and can include legitimate code/format repetition.',
    'empty_decoded_output': 'Visible completion after excluding terminal EOT has zero Unicode characters.',
    'whitespace_only_output': 'Visible completion has >=1 character and every character passes str.isspace(). Empty outputs are counted separately.',
    'spaces_only_output': 'Visible completion has >=1 character and consists exclusively of ASCII U+0020 spaces.',
    'newlines_only_output': 'Visible completion has >=1 character and consists exclusively of LF U+000A characters.',
    'unusual_whitespace_run': 'At least 16 consecutive Unicode whitespace characters in visible completion, fixed in the preregistered protocol.',
    'space_220': 'Token220 is the GPT-2 token for one ASCII space. Occurrences=count220; adjacent repeated occurrences=sum(ids[i]==ids[i-1]==220); longest run=count of consecutive220 tokens.',
    'whitespace_token_fraction': 'Fraction of content BPE IDs whose token bytes decode strictly as UTF-8 to a nonempty string that passes str.isspace(). Invalid/incomplete UTF-8 token bytes are not marked whitespace.',
    'lexical_diversity': 'Unique casefolded word forms / number of Unicode regex words matched by [^\\W_]+(?:[’\x27][^\\W_]+)*. Null for zero words; punctuation excluded. BPE distinct metrics are separate.',
    'early_eot': 'Stopping reason=endoftext and generated_length_including_eot<128. EOT at exactly token128 is an EOT stop but not early.',
    'maximum_length': 'generated_length_including_eot==128 regardless of stop reason.',
    'unique_generated_tokens': 'Count of distinct content BPE IDs in each output or union across all outputs in a decoder aggregate, explicitly labeled.',
    'duplicate_output_rate_across_seeds': 'For each sampled decoder and prompt, (3 minus number of unique content-token sequences across the 3 seeds); sum across prompts / 90 outputs. Also report identical pairs /90 within-prompt seed pairs. Greedy is NA because it has one output per prompt.',
    'raw_probabilities': 'Softmax of actual unmodified FP32 next-token logits at temperature1 before top-k/top-p. Entropy=-sum(p log p), in nats; margin=p(highest)-p(second-highest). Sampling probability separately uses the configured filtered/tempered distribution; greedy selection probability is1 by definition.',
}

def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None

def runs(values, predicate):
    maximum = current = 0
    for value in values:
        current = current + 1 if predicate(value) else 0
        maximum = max(maximum, current)
    return maximum

def longest_repeat(ids):
    n = len(ids)
    # Longest common prefixes for each pair of start positions; separation caps overlap.
    lcp = [[0] * (n + 1) for _ in range(n + 1)]
    best = (0, None, None)
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            if ids[i] == ids[j]:
                lcp[i][j] = 1 + lcp[i + 1][j + 1]
                size = min(lcp[i][j], j - i)
                if size > best[0]:
                    best = (size, i, j)
    return {'length_tokens': best[0], 'first_start_zero_based': best[1], 'second_start_zero_based': best[2]}

def find_loop(ids):
    candidates = []
    for start in range(len(ids)):
        for period in range(2, min(32, (len(ids) - start) // 3) + 1):
            motif = ids[start:start + period]
            repeats = 1
            while ids[start + repeats * period:start + (repeats + 1) * period] == motif:
                repeats += 1
            if repeats >= 3 and period * repeats >= 12:
                candidates.append({'start_zero_based': start, 'period_tokens': period,
                                   'repeats': repeats, 'span_tokens': period * repeats,
                                   'motif_token_ids': motif})
        if candidates:
            return max(candidates, key=lambda c: (c['span_tokens'], -c['period_tokens']))
    return None

def words(text):
    return re.findall(r"[^\W_]+(?:[’'][^\W_]+)*", text.casefold())

def analyze_record(record, whitespace_ids):
    ids = record['content_token_ids']
    text = record['completion']
    n = len(ids)
    metrics = {'output_id': record['output_id'], 'prompt_id': record['prompt_id'],
               'method': record['method'], 'seed': record['seed'], 'category': record['category'],
               'generated_length_including_eot': len(record['generated_token_ids']), 'content_length': n,
               'early_eot': record['stopping_reason'] == 'endoftext' and len(record['generated_token_ids']) < 128,
               'eot_stop': record['stopping_reason'] == 'endoftext',
               'reached_128_tokens': len(record['generated_token_ids']) == 128,
               'empty_output': text == '', 'whitespace_only': bool(text) and text.isspace(),
               'spaces_only': bool(text) and set(text) == {' '},
               'newlines_only': bool(text) and set(text) == {'\n'},
               'longest_whitespace_run_chars': runs(text, str.isspace),
               'space220_occurrences': ids.count(220),
               'space220_adjacent_repeated_occurrences': sum(a == b == 220 for a, b in zip(ids, ids[1:])),
               'space220_longest_run': runs(ids, lambda x: x == 220),
               'whitespace_token_count': sum(i in whitespace_ids for i in ids),
               'whitespace_token_fraction': sum(i in whitespace_ids for i in ids) / n if n else None,
               'unique_generated_tokens': len(set(ids)), 'distinct_1': len(set(ids)) / n if n else None,
               'longest_repeated_token_span': longest_repeat(ids), 'loop_proxy': find_loop(ids)}
    metrics['unusual_whitespace_run'] = metrics['longest_whitespace_run_chars'] >= 16
    for width in [2, 3, 4]:
        grams = [tuple(ids[i:i + width]) for i in range(max(0, n - width + 1))]
        metrics[f'repeated_{width}gram_rate'] = (len(grams) - len(set(grams))) / len(grams) if grams else None
        if width == 2:
            metrics['distinct_2'] = len(set(grams)) / len(grams) if grams else None
    lexical = words(text)
    metrics['word_count'] = len(lexical)
    metrics['unique_word_count'] = len(set(lexical))
    metrics['lexical_diversity'] = len(set(lexical)) / len(lexical) if lexical else None
    phrase_counts = Counter()
    for width in range(3, 9):
        counts = Counter(tuple(lexical[i:i + width]) for i in range(max(0, len(lexical) - width + 1)))
        phrase_counts.update({' '.join(k): v for k, v in counts.items() if v > 1})
    metrics['identical_phrase_types'] = len(phrase_counts)
    metrics['top_repeated_phrases'] = [{'phrase': k, 'occurrences': v} for k, v in sorted(
        phrase_counts.items(), key=lambda kv: (-kv[1], -len(kv[0].split()), kv[0]))[:10]]
    return metrics, phrase_counts

def aggregate(records, metrics):
    by_id = {r['output_id']: r for r in records}
    summary = {}
    bools = ['empty_output', 'whitespace_only', 'spaces_only', 'newlines_only', 'unusual_whitespace_run',
             'early_eot', 'eot_stop', 'reached_128_tokens']
    average = ['distinct_1', 'distinct_2', 'repeated_2gram_rate', 'repeated_3gram_rate', 'repeated_4gram_rate',
               'whitespace_token_fraction', 'lexical_diversity', 'space220_occurrences', 'space220_longest_run',
               'unique_generated_tokens', 'generated_length_including_eot', 'content_length']
    for method in METHODS:
        ms = [m for m in metrics if m['method'] == method]
        rs = [by_id[m['output_id']] for m in ms]
        s = {'outputs': len(ms)}
        for name in bools:
            s[name + '_count'] = sum(m[name] for m in ms)
            s[name + '_percent'] = 100 * s[name + '_count'] / len(ms)
        for name in average:
            s[name + '_mean'] = mean([m[name] for m in ms])
            s[name + '_eligible_outputs'] = sum(m[name] is not None for m in ms)
        s['length_median'] = statistics.median(m['generated_length_including_eot'] for m in ms)
        s['loop_proxy_count'] = sum(m['loop_proxy'] is not None for m in ms)
        s['loop_proxy_percent'] = 100 * s['loop_proxy_count'] / len(ms)
        s['longest_repeated_span_mean'] = mean([m['longest_repeated_token_span']['length_tokens'] for m in ms])
        s['longest_repeated_span_max'] = max(m['longest_repeated_token_span']['length_tokens'] for m in ms)
        s['space220_total_occurrences'] = sum(m['space220_occurrences'] for m in ms)
        s['space220_total_adjacent_repeats'] = sum(m['space220_adjacent_repeated_occurrences'] for m in ms)
        s['space220_max_run'] = max(m['space220_longest_run'] for m in ms)
        s['unique_content_token_union'] = len(set(itertools.chain.from_iterable(r['content_token_ids'] for r in rs)))
        content_tokens = sum(m['content_length'] for m in ms)
        s['pooled_whitespace_token_fraction'] = (
            sum(m['whitespace_token_count'] for m in ms) / content_tokens if content_tokens else None)
        groups = defaultdict(list)
        for r in rs:
            groups[r['prompt_id']].append(tuple(r['content_token_ids']))
        if method == 'greedy':
            s['duplicate_output_rate_across_seeds'] = None
            s['identical_seed_pair_rate'] = None
        else:
            s['duplicate_output_rate_across_seeds'] = sum(len(x) - len(set(x)) for x in groups.values()) / len(rs)
            pairs = list(itertools.chain.from_iterable(itertools.combinations(x, 2) for x in groups.values()))
            s['identical_seed_pair_rate'] = sum(a == b for a, b in pairs) / len(pairs)
        summary[method] = s
    return summary

def diagnostics(records, metrics, enc):
    by_id = {r['output_id']: r for r in records}
    selected = []
    for method in METHODS:
        rows = [m for m in metrics if m['method'] == method]
        for selector, why in [(lambda m: m['loop_proxy']['span_tokens'] if m['loop_proxy'] else -1, 'largest detected loop span'),
                              (lambda m: m['space220_longest_run'], 'longest token220 run')]:
            m = max(rows, key=lambda m: (selector(m), m['output_id']))
            if m['output_id'] in [x['output_id'] for x in selected]:
                continue
            r = by_id[m['output_id']]
            onset = m['loop_proxy']['start_zero_based'] if m['loop_proxy'] else 0
            positions = sorted(set([0] + list(range(max(0, onset - 3), min(len(r['probability_trace']), onset + 7)))))
            trace = []
            for pos in positions:
                t = dict(r['probability_trace'][pos])
                t['top10_decoded_tokens'] = [enc.decode([i]) for i in t['top10_token_ids']]
                t['generated_token_decoded'] = enc.decode([t['token_id']])
                trace.append(t)
            selected.append({'output_id': r['output_id'], 'method': method, 'reason_selected': why,
                             'prompt': r['prompt'], 'completion_unedited': r['completion'],
                             'loop_proxy': m['loop_proxy'], 'space220_longest_run': m['space220_longest_run'],
                             'probability_trace_selected': trace})
    return {'selection_rule': 'Per decoder: maximum detected exact-loop span and maximum token220 run; deterministic lexicographic output-ID tie break; deduplicate selected IDs. Include initial distribution and onset-3 through onset+6 when available.',
            'probabilities_are_raw_temperature_1': True, 'selected': selected}

def pct(v):
    return 'NA' if v is None else f'{100*v:.2f}%'

def decimal(v, places=3):
    return 'NA' if v is None else f'{v:.{places}f}'

def main():
    path = ROOT / 'GENERATION_RESULTS.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == len({r['output_id'] for r in records}) == 300
    protocol = json.loads((ROOT / 'GENERATION_PROMPTS.json').read_text())
    assert all(r['protocol_sha256'] == protocol['protocol_sha256'] for r in records)
    assert Counter(r['method'] for r in records) == {'greedy': 30, 'top_k': 90, 'top_p': 90, 'combined': 90}
    enc = encoder()
    whitespace_ids = set()
    for token in range(enc.n_vocab):
        try:
            text = enc.decode_single_token_bytes(token).decode('utf-8')
            if text and text.isspace():
                whitespace_ids.add(token)
        except UnicodeDecodeError:
            pass
    metrics, phrase_totals = [], Counter()
    for record in records:
        m, phrases = analyze_record(record, whitespace_ids)
        metrics.append(m)
        phrase_totals.update(phrases)
    summary = aggregate(records, metrics)
    payload = {'schema_version': 1, 'outputs': 300, 'protocol_sha256': protocol['protocol_sha256'],
               'definitions': DEFINITIONS, 'whitespace_token_ids': sorted(whitespace_ids),
               'by_decoder': summary, 'per_output': metrics,
               'aggregate_repeated_phrase_occurrences': dict(phrase_totals.most_common())}
    atomic_new(ROOT / 'GENERATION_METRICS.json', json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    diag = diagnostics(records, metrics, enc)
    atomic_new(ROOT / 'NEXT_TOKEN_DIAGNOSTICS.json', json.dumps(diag, indent=2, ensure_ascii=False) + '\n')
    report = ['# Repetition and whitespace analysis', '', 'All 300 frozen-schedule outputs are analyzed: 30 greedy and 90 per sampled decoder. No outputs were discarded. All text metrics exclude the prompt and terminal EOT. Metrics are descriptive for this fixed 30-prompt suite, not population estimates.', '',
              '## Repetition and diversity', '', '| Decoder | N | Repeated 2-gram | Repeated 3-gram | Repeated 4-gram | Distinct-1 | Distinct-2 | Exact-loop proxy | Longest repeat mean/max |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, s in summary.items():
        report.append(f'| {method} | {s["outputs"]} | {pct(s["repeated_2gram_rate_mean"])} | {pct(s["repeated_3gram_rate_mean"])} | {pct(s["repeated_4gram_rate_mean"])} | {decimal(s["distinct_1_mean"])} | {decimal(s["distinct_2_mean"])} | {s["loop_proxy_count"]}/{s["outputs"]} ({s["loop_proxy_percent"]:.2f}%) | {s["longest_repeated_span_mean"]:.2f}/{s["longest_repeated_span_max"]} tokens |')
    report += ['', '## Whitespace', '', '| Decoder | Empty | Whitespace only | Spaces only | Newlines only | ≥16-char whitespace run | Token220 count / adjacent repeats / max run | Pooled whitespace-token fraction |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, s in summary.items():
        report.append(f'| {method} | {s["empty_output_count"]} | {s["whitespace_only_count"]} | {s["spaces_only_count"]} | {s["newlines_only_count"]} | {s["unusual_whitespace_run_count"]}/{s["outputs"]} | {s["space220_total_occurrences"]} / {s["space220_total_adjacent_repeats"]} / {s["space220_max_run"]} | {pct(s["pooled_whitespace_token_fraction"])} |')
    report += ['', '## Length and seed diversity', '', '| Decoder | Mean/median generated length | Early EOT | Reached128 | Unique BPE IDs across decoder | Mean lexical diversity | Duplicate outputs across seeds |', '|---|---:|---:|---:|---:|---:|---:|']
    for method, s in summary.items():
        report.append(f'| {method} | {s["generated_length_including_eot_mean"]:.2f}/{s["length_median"]} | {s["early_eot_count"]}/{s["outputs"]} | {s["reached_128_tokens_count"]}/{s["outputs"]} | {s["unique_content_token_union"]} | {decimal(s["lexical_diversity_mean"])} | {pct(s["duplicate_output_rate_across_seeds"])} |')
    report += ['', '## Definitions and limitations', '']
    report.extend(f'- **{name}:** {definition}' for name, definition in DEFINITIONS.items())
    report += ['', 'Per-output metrics, eligibility counts for short sequences, phrase frequencies, and exact loop positions are in `GENERATION_METRICS.json`. `NEXT_TOKEN_DIAGNOSTICS.json` contains the deterministic diagnostic subset with actual logits-derived raw probabilities. Complete position-by-position traces are retained in every raw generation record.', '', '## Representative measured probability diagnostics', '', '| Output | Position | Selected token ID | Selected raw probability | Space220 probability | Top-two margin | Entropy (nats) |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in diag['selected']:
        t = r['probability_trace_selected'][0]
        report.append(f'| {r["output_id"]} | {t["generated_position_zero_based"]} | {t["token_id"]} | {t["raw_probability"]:.6f} | {t["space_220_probability"]:.6f} | {t["top_two_probability_margin"]:.6f} | {t["entropy_nats"]:.4f} |')
    report += ['', 'A greedy selection is an argmax decision; it does not imply a nearly certain model probability. The table and saved traces report the actual distribution. Loop flags can capture code indentation or repeated list structure, so semantic review must distinguish these from prose degeneration.', '']
    atomic_new(ROOT / 'REPETITION_AND_WHITESPACE_ANALYSIS.md', '\n'.join(report))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    ROOT=args.output_root.resolve()
    main()
