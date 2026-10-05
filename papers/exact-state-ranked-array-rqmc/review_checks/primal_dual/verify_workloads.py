"""Verify integer observations and exact rational demand; preserve old summaries."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import numpy as np
import workload as wl


def left_sum(values):
    """Explicit legacy Python 3.11 accumulation, independent of built-in sum."""
    total = 0
    for value in values:
        total += value
    return total


def fraction_pair(value):
    return [value.numerator, value.denominator]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-exact', action='store_true',
                        help='Create the separate derived fraction record once; never overwrite it.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent/'workloads'
    original_bytes = (root/'summary.json').read_bytes()
    record = json.loads(original_bytes)
    tested = widths = float_changes = 0
    rational = dict(source_summary_sha256=hashlib.sha256(original_bytes).hexdigest(), workloads=[])
    for saved in record['workloads']:
        rebuilt = wl.extract(saved['spec'], saved['seed'])
        with np.load(root/f"{saved['index']}.npz") as f:
            for key, array in rebuilt.items():
                assert np.array_equal(array, f[key]), (saved['index'], 'workload extraction', key)
            expected_shifts = np.random.default_rng(saved['shift_seed']).integers(
                0, 1 << saved['spec']['w'], saved['shifts'], dtype=np.int64)
            assert np.array_equal(expected_shifts, f['shifts'])
            pred = wl.prediction(f['C'], f['U'], f['V'], f['starts'], f['ks'], f['thresholds'])
            J, xors = wl.sample_shifts(f['C'], f['U'], f['V'], f['starts'], f['ks'],
                                      f['thresholds'], f['shifts'])
            assert np.array_equal(J, f['J']) and np.array_equal(xors, f['row_xors'])
            assert len(pred) == len(saved['widths'])
            rows = []
            for a, b in zip(pred, saved['widths']):
                for key in ('k', 'queries', 'dependencies', 'distinct_prefixes', 'probabilities'):
                    assert a[key] == b[key], (saved['index'], a['k'], key)
                p = a['probabilities']
                mean = left_sum(p)
                legacy = dict(exact_mean=mean,
                              exact_variance=max(0., left_sum((2*s+1)*v for s,v in enumerate(p))-mean*mean),
                              exact_row_xors=left_sum(x*y for x,y in zip(a['source_row_weights'], p)),
                              psi=left_sum(min(1.,a['queries']/(1 << s)) for s in range(a['dependencies'])))
                for key, value in legacy.items():
                    assert value == b[key], (saved['index'], a['k'], 'legacy accumulation', key)
                    float_changes += a[key] != b[key]
                assert float(J[:, a['k']].mean()) == b['sample_mean']
                rows.append({key:a[key] for key in ('k','queries','dependencies','distinct_prefixes',
                                                   'source_row_weights','exact_fractions')})
                widths += 1
            totals = {name:fraction_pair(sum((Fraction(*p['exact_fractions'][name]) for p in pred), Fraction(0)))
                      for name in ('mean', 'row_xors', 'psi')}
            rational['workloads'].append(dict(index=saved['index'], widths=rows, totals=totals))
            tested += len(J)
    target = root/'expectations-exact.json'
    if args.write_exact:
        with target.open('x', encoding='utf-8') as out:
            out.write(json.dumps(rational, indent=2)+'\n')
    assert rational == json.loads(target.read_text()), 'exact rational expectations changed'
    assert (root/'summary.json').read_bytes() == original_bytes
    report = dict(status='PASS', fixed_workloads=len(record['workloads']), widths=widths,
                  shift_realizations=tested, incoming_workloads_reconstructed=True,
                  integer_observations_exact=True, rational_expectations_exact=True,
                  legacy_float_accumulation_exact=True, legacy_summary_unchanged=True,
                  correctly_rounded_display_fields_different_from_legacy=int(float_changes),
                  latency_measurements_repeated=False)
    (root/'verified.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
