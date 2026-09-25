"""Recompute the order-maintenance timing table from the original paired rows.

No timing experiment is run. Wall and CPU fields are seconds for a block of
`reps` complete estimator evaluations. A source_row is the zero-based position
in its condition's original JSON, hence the within-condition execution order.
The saved design used five paired rounds, four array evaluations per block,
and 32 baseline evaluations per block. Baselines and intermediate N are retained
in the CSV even though the main table compares the two arrays at N=512,4096.
"""
import csv
import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_rows():
    with (ROOT/'data/ordering_timing_rows.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ('condition_order', 'source_row', 'n', 'round', 'reps', 'calls', 'transitions'):
            row[key] = int(row[key])
        for key in ('wall', 'cpu'):
            row[key] = float(row[key])
    keys = [(r['condition'], r['n'], r['method'], r['round']) for r in rows]
    assert len(keys) == len(set(keys)) == 720
    for row in rows:
        assert row['reps'] == (4 if row['method'] in ('BINARY', 'MERGE') else 32)
        assert row['calls'] == 3*row['n']*row['reps']
        assert row['wall'] > 0 and row['cpu'] >= 0
    return rows


def summarize(rows):
    conditions = sorted({(r['condition_order'], r['condition']) for r in rows})
    assert len(conditions) == 16
    rng = np.random.default_rng(2091221)
    result = []
    for n in (512, 4096):
        ratios, cpu_ratios, boot_logs = [], [], []
        for _, condition in conditions:
            selected = {method: sorted(
                (r for r in rows if r['condition'] == condition and r['n'] == n and r['method'] == method),
                key=lambda r: r['round']) for method in ('BINARY', 'MERGE')}
            for values in selected.values():
                assert [r['round'] for r in values] == list(range(5))
            wall = {m: np.array([r['wall']/r['reps'] for r in v]) for m, v in selected.items()}
            cpu = {m: np.array([r['cpu']/r['reps'] for r in v]) for m, v in selected.items()}
            ratios.append(wall['MERGE'].mean()/wall['BINARY'].mean())
            cpu_ratios.append(cpu['MERGE'].mean()/cpu['BINARY'].mean())
            # Resample rounds jointly for the two methods, separately within
            # each fixed condition. Conditions are not resampled as new models.
            indices = rng.integers(0, 5, size=(2000, 5))
            boot_logs.append(np.log(wall['MERGE'][indices].mean(axis=1)/
                                    wall['BINARY'][indices].mean(axis=1)))
        interval = np.quantile(np.exp(np.mean(boot_logs, axis=0)), [.025, .975])
        result.append(dict(n=n, wall_ratio=float(np.exp(np.mean(np.log(ratios)))),
                           cpu_ratio=float(np.exp(np.mean(np.log(cpu_ratios)))),
                           wall_ratio_lo=float(interval[0]), wall_ratio_hi=float(interval[1]),
                           faster_conditions=int(sum(r < 1 for r in ratios))))
    return result


def verify():
    results = summarize(load_rows())
    saved = json.loads((ROOT/'data/saved_results.json').read_text(encoding='utf-8'))['ordering_summary']
    for got, expected in zip(results, saved):
        assert got['n'] == expected['n']
        for key in ('wall_ratio', 'cpu_ratio', 'wall_ratio_lo', 'wall_ratio_hi'):
            assert np.isclose(got[key], expected[key], rtol=1e-13, atol=1e-15), (key, got, expected)
        assert got['faster_conditions'] == 16
    return results


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
