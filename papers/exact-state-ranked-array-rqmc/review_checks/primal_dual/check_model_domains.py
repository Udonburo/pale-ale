"""Boundary regressions and published-cell compatibility of the current wrapper.

The confirmation source and its timings are read-only inputs. No new timing
observations are recorded; elapsed values returned by execute are discarded.
"""
from pathlib import Path
import difflib
import hashlib
import importlib.util
import json
import numpy as np
import executors as ex

HERE = Path(__file__).resolve().parent


def main():
    source, data = HERE/'confirmation/source', HERE/'confirmation/data'
    meta = json.loads((data/'metadata.json').read_text())
    for name, expected in meta['source'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected, name
        if name != 'executors.py':
            assert (HERE/name).read_bytes() == (source/name).read_bytes(), name
    # Only the model wrapper differs; both modules use identical retained kernels.
    loader = importlib.util.spec_from_file_location('measured_executors', source/'executors.py')
    measured = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(measured)
    changed = ''.join(difflib.unified_diff(
        (source/'executors.py').read_text().splitlines(keepends=True),
        (HERE/'executors.py').read_text().splitlines(keepends=True),
        fromfile='confirmation/source/executors.py', tofile='executors.py'))
    assert changed == (HERE/'model-wrapper.diff.txt').read_text()

    for K in range(3, 18):
        spec = dict(model='repair', K=K, m=3, w=8, horizon=3)
        _, _, weights = ex.problem(spec)
        assert weights[:, 1].tolist() == [int(2*x >= K+1) for x in range(K+1)]
    case = dict(model='repair', K=4, m=8, w=8, horizon=10)
    expected = None
    for method in ex.METHODS:
        _, _, result = ex.execute(case, 103000000, method)
        assert result[0][1].tolist() == [225, 30, 1, 0, 0]
        assert result[1][1, 1] == 0
        assert np.array_equal(result[1][:, 1], result[0][:, 3:].sum(axis=1))
        if expected is None:
            expected = result[:2]
        assert all(np.array_equal(a, b) for a, b in zip(expected, result[:2]))

    rejected = 0
    for w in (1, 2, 3):
        spec = dict(model='tandem', m=1, w=w, horizon=2)
        entries = [lambda: ex.tandem_tables(w, 2), lambda: ex.problem(spec)]
        entries += [lambda method=method: ex.execute(spec, 103000000, method)
                    for method in ex.METHODS]
        for entry in entries:
            try:
                entry()
            except ValueError as error:
                assert str(error) == 'tandem benchmark requires word width >= 4'
                rejected += 1
            else:
                raise AssertionError(('unsupported tandem width accepted', w))
    for w in (4, 8, 30, 52):
        cuts, _, _ = ex.tandem_tables(w, 2)
        expected_cuts = [(7*(1 << w))//16+1, (12*(1 << w))//16+1]
        assert np.all(cuts == expected_cuts)

    trajectories = steps = 0
    for cell, spec in enumerate(meta['cells']):
        current, original = ex.problem(spec), measured.problem(spec)
        assert all(np.array_equal(a, b) for a, b in zip(current, original)), spec
        seed = meta['seed_base']+100*cell
        with np.load(data/'trajectories'/f'{cell:02d}-00.npz') as saved:
            for method in ex.METHODS:
                _, _, result = ex.execute(spec, seed, method)
                ex.validate(result, spec)
                assert np.array_equal(result[0], saved['hist']), (cell, method, 'hist')
                assert np.array_equal(result[1], saved['readouts']), (cell, method, 'readouts')
                trajectories += 1
                steps += spec['horizon']
        print('current-wrapper compatibility cell '+str(cell)+': PASS', flush=True)
    report = dict(status='PASS', repair_capacities_checked=15,
                  even_capacity_fixture_methods=len(ex.METHODS), rejected_entries=rejected,
                  published_model_tables_unchanged=len(meta['cells']),
                  current_wrapper_method_trajectories=trajectories,
                  current_wrapper_method_steps=steps, paired_seeds_per_cell=1,
                  measured_source_hashes_verified=len(meta['source']),
                  timing_observations_added=0)
    (HERE/'model-domain-checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
