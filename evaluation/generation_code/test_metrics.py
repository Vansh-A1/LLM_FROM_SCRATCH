import unittest
from unittest.mock import patch
import torch
import generation as g
import analyze as a

class MetricTests(unittest.TestCase):
    def test_immediate_eot_empty_decoder_aggregates_preserve_na(self):
        records, metrics = [], []
        for method, settings in g.METHODS.items():
            for seed in settings['seeds']:
                record = {'output_id': f'x_{method}_{seed}', 'prompt_id': 'x',
                          'method': method, 'seed': seed, 'category': 'sentence_continuation',
                          'content_token_ids': [], 'generated_token_ids': [50256],
                          'completion': '', 'stopping_reason': 'endoftext'}
                records.append(record)
                metrics.append(a.analyze_record(record, {220})[0])
        result = a.aggregate(records, metrics)
        for method, summary in result.items():
            self.assertIsNone(summary['pooled_whitespace_token_fraction'])
            self.assertIsNone(summary['whitespace_token_fraction_mean'])
            self.assertIsNone(summary['distinct_1_mean'])
            self.assertIsNone(summary['distinct_2_mean'])
            self.assertIsNone(summary['repeated_4gram_rate_mean'])
            self.assertIsNone(summary['lexical_diversity_mean'])
            self.assertEqual(summary['distinct_1_eligible_outputs'], 0)
            self.assertEqual(summary['empty_output_count'], summary['outputs'])
            self.assertEqual(summary['generated_length_including_eot_mean'], 1)
        self.assertEqual(a.pct(None), 'NA')
        self.assertEqual(a.decimal(None), 'NA')

    def test_short_ngram_is_na(self):
        record = {'output_id': 'x', 'prompt_id': 'x', 'method': 'greedy', 'seed': None,
                  'category': 'sentence_continuation', 'content_token_ids': [220],
                  'generated_token_ids': [220, 50256], 'completion': ' ', 'stopping_reason': 'endoftext'}
        m, _ = a.analyze_record(record, {220})
        self.assertIsNone(m['repeated_2gram_rate'])
        self.assertIsNone(m['distinct_2'])
        self.assertEqual(m['whitespace_token_fraction'], 1)
        self.assertTrue(m['spaces_only'])
        self.assertFalse(m['empty_output'])
        self.assertEqual(m['content_length'], 1)
        self.assertEqual(m['generated_length_including_eot'], 2)

    def test_nonoverlapping_repeat(self):
        self.assertEqual(a.longest_repeat([1, 2, 3, 1, 2, 3])['length_tokens'], 3)
        self.assertEqual(a.longest_repeat([1, 1, 1, 1, 1])['length_tokens'], 2)
        self.assertEqual(a.longest_repeat([1, 2, 3])['length_tokens'], 0)

    def test_exact_loop_threshold(self):
        self.assertIsNone(a.find_loop([1, 2] * 3))
        loop = a.find_loop([9, 1, 2, 3, 4] + [1, 2, 3, 4] * 2)
        self.assertEqual(loop['start_zero_based'], 1)
        self.assertEqual(loop['period_tokens'], 4)
        self.assertEqual(loop['repeats'], 3)

    def test_eot_terminates_after_one_token(self):
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        logits = torch.full((300,), -100.)
        logits[299] = 1.
        with patch.object(g, 'next_logits', return_value=logits):
            ids, reason, traces = g.generate(model, [1], 'greedy', None, eos=299)
        self.assertEqual(ids, [299])
        self.assertEqual(reason, 'endoftext')
        self.assertEqual(len(traces), 1)

    def test_maximum_length_and_actual_raw_probability(self):
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        logits = torch.zeros(300)
        logits[220] = 0.01
        with patch.object(g, 'next_logits', return_value=logits):
            ids, reason, traces = g.generate(model, [1], 'greedy', None, max_new_tokens=4, eos=299)
        self.assertEqual(ids, [220] * 4)
        self.assertEqual(reason, 'max_new_tokens')
        self.assertLess(traces[0]['raw_probability'], 0.004)
        self.assertEqual(traces[0]['sampling_probability'], 1.0)

if __name__ == '__main__':
    unittest.main()
