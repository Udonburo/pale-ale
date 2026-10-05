"""Prospectively specified paired timings, exact outputs and resumable records."""
from pathlib import Path
import argparse
import hashlib
import json
import random
import statistics
import time
import numpy as np
import executors as ex


def source_id(root):
    return {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts}


def summarize(out):
    meta = json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in (out/'timings.jsonl').read_text(encoding='utf-8').splitlines()]
    grouped = {}
    for row in rows:
        grouped.setdefault((row['cell'], row['rep'], row['method']), []).append(row['seconds'])
    cells = []
    for cell, spec in enumerate(meta['cells']):
        times = {method: [] for method in ex.METHODS}
        for method in ex.METHODS:
            for rep in range(meta['replications']):
                v = grouped.get((cell, rep, method), [])
                if len(v) == meta['rounds']:
                    times[method].append(statistics.median(v))
        if any(len(v) != meta['replications'] for v in times.values()):
            continue
        ratios = {method: [t/u for t, u in zip(times[method], times['reuse'])]
                  for method in ex.METHODS if method != 'reuse'}
        cells.append(dict(cell=cell, spec=spec, seed_median_seconds=times,
                          median_seconds={k: statistics.median(v) for k, v in times.items()},
                          paired_over_reuse=ratios,
                          median_paired_over_reuse={k: statistics.median(v) for k, v in ratios.items()}))
    result = dict(complete=len(cells) == len(meta['cells']), timing_rows=len(rows), cells=cells)
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def run(out, development):
    root = Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    (out/'trajectories').mkdir(exist_ok=True)
    selected = ([dict(model='repair', K=63, w=52, m=m, horizon=100) for m in (12, 20)] +
                [dict(model='tandem', w=52, m=m, horizon=200) for m in (12, 20)]) if development else ex.cells()
    reps, rounds, seedbase = (2, 3, 102000000) if development else (8, 5, 103000000)
    meta_path = out/'metadata.json'
    metadata = dict(design='development' if development else 'confirmation',
                    cells=selected, replications=reps, rounds=rounds, seed_base=seedbase,
                    methods=ex.METHODS, source=source_id(root), environment=ex.original_model.environment(),
                    started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding='utf-8'))
        for k in ('cells', 'replications', 'rounds', 'seed_base', 'source'):
            assert old[k] == metadata[k], ('resume mismatch', k)
    else:
        meta_path.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    records = out/'timings.jsonl'
    done = set()
    if records.exists():
        for line in records.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = row['cell'], row['rep'], row['round'], row['method']
            assert key not in done
            done.add(key)
    ex.warmup()
    print('warmup complete; starting '+metadata['design'], flush=True)
    comparisons, histogram_steps = 0, 0
    with records.open('a', encoding='utf-8') as stream:
        for cell, spec in enumerate(selected):
            for rep in range(reps):
                seed = seedbase+100*cell+rep
                saved = out/'trajectories'/f'{cell:02d}-{rep:02d}.npz'
                if saved.exists():
                    with np.load(saved) as data:
                        hist, readouts = data['hist'], data['readouts']
                else:
                    _, _, canonical = ex.execute(spec, seed, 'reuse', True)
                    _, _, rebuilt = ex.execute(spec, seed, 'rebuild', True)
                    ex.validate(canonical, spec)
                    for i in (0, 1, 3, 4, 5):
                        assert np.array_equal(canonical[i], rebuilt[i]), (cell, rep, 'audit', i)
                    hist, readouts = canonical[:2]
                    np.savez_compressed(saved, hist=hist, readouts=readouts, work=canonical[2],
                                        rebuild_work=rebuilt[2], widths=canonical[3],
                                        max_tests=canonical[4], occupied=canonical[5])
                for round_no in range(rounds):
                    order = list(ex.METHODS)
                    random.Random(seed*31+round_no).shuffle(order)
                    for method in order:
                        key = cell, rep, round_no, method
                        if key in done:
                            continue
                        seconds, setup, result = ex.execute(spec, seed, method, False)
                        assert np.array_equal(hist, result[0]) and np.array_equal(readouts, result[1]), key
                        comparisons += 1
                        histogram_steps += spec['horizon']
                        row = dict(cell=cell, rep=rep, seed=seed, round=round_no, method=method,
                                   seconds=seconds, setup_seconds=setup,
                                   checked_steps=spec['horizon'], cumulative_readouts=readouts.sum(axis=0).tolist())
                        stream.write(json.dumps(row)+'\n')
                        stream.flush()
                        done.add(key)
            result = summarize(out)
            last = result['cells'][-1]
            print(json.dumps(dict(cell=cell, spec=spec, ratios=last['median_paired_over_reuse'])), flush=True)
    report = dict(comparisons_this_run=comparisons, histogram_steps_this_run=histogram_steps,
                  status='PASS', source=metadata['source'])
    (out/'run_check.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', timing_rows=len(done), comparisons=comparisons)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.out)))
    else:
        run(args.out.resolve(), args.development)
