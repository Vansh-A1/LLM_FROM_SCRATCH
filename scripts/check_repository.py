#!/usr/bin/env python3
"""Fast, offline publication checks; no third-party dependencies or model loading."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

def main():
    listing = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
    files = sorted({ROOT / p for p in listing.decode().split('\0') if p})
    errors = []
    python_count = link_count = 0
    for path in files:
        if not path.is_file():
            errors.append(f'Missing tracked file: {path.relative_to(ROOT)}')
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f'Artifact exceeds 5 MiB publication limit: {rel}')
        if path.suffix in {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.arrow', '.parquet'}:
            errors.append(f'Weights or dataset binary in repository: {rel}')
        if path.suffix == '.py':
            try: ast.parse(path.read_text(), filename=rel)
            except SyntaxError as exc: errors.append(f'{rel}: {exc}')
            python_count += 1
        if path.suffix == '.md':
            for match in re.finditer(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)', path.read_text()):
                target = match.group(1).strip('<>')
                if re.match(r'^[a-z]+:', target) or target.startswith('#'): continue
                clean = unquote(target.split('#', 1)[0])
                if not (path.parent / clean).exists(): errors.append(f'Broken link in {rel}: {target}')
                link_count += 1
    provenance = json.loads((ROOT / 'results/source_provenance.json').read_text())
    for name, expected in provenance['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            errors.append(f'Historical source hash changed: {name}')
    evaluation = json.loads((ROOT / 'results/evaluation_source_provenance.json').read_text())
    for item in evaluation['files']:
        if hashlib.sha256((ROOT / item['published_file']).read_bytes()).hexdigest() != item['published_sha256']:
            errors.append(f"Evaluation source hash changed: {item['published_file']}")
    training = json.loads((ROOT / 'results/training_metrics_provenance.json').read_text())
    for name, info in training['public_files'].items():
        if hashlib.sha256((ROOT / 'results' / name).read_bytes()).hexdigest() != info['sha256']:
            errors.append(f'Metric CSV hash changed: {name}')
    with (ROOT / 'results/training_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    if [int(r['step']) for r in rows] != list(range(1, 17235)):
        errors.append('Training CSV must contain each update 1–17,234 exactly once, in order.')
    for row in rows:
        if not all(math.isfinite(float(v)) for v in row.values()):
            errors.append(f'Nonfinite training metric at update {row["step"]}')
            break
    config = json.loads((ROOT / 'configs/gpt2-small-epoch1.json').read_text())
    if config['stop_at'] * config['train_config']['total_batch_size'] != config['training_tokens']:
        errors.append('Recorded update/token budget does not agree.')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {python_count} Python files, {link_count} local Markdown links, source/CSV hashes, 17,234 updates, and artifact policy.')

if __name__ == '__main__':
    main()
