"""Paired, setup-inclusive confirmation; preserves every timing round."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import numpy as np
import study_run as run
import model

HERE=Path(__file__).resolve().parent


def snapshot(output):
    root=output/'source'
    child=root/'reuse_study'
    child.mkdir(parents=True)
    for name in ('kernel.py','counter_reference.py','run.py','model.py'):
        shutil.copy2(HERE.parent/name,root/name)
    for p in HERE.glob('*.py'):
        shutil.copy2(p,child/p.name)
    shutil.copy2(HERE/'PLAN.md',child/'PLAN.md')
    hashes={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}
    (output/'source_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')


def benchmark(output,development=False):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    data=output/'paths';data.mkdir()
    snapshot(output)
    nseeds,rounds,seed_base=(2,3,9610000) if development else (8,5,9700000)
    specs=run.cells(development)
    header=dict(started_utc=datetime.now(timezone.utc).isoformat(),
        mode='development' if development else 'confirmation',environment=model.environment(),
        cells=[dict(group=g,spec=s,methods=ms) for g,s,ms in specs],
        paired_seeds=nseeds,timing_rounds=rounds,seed_base=seed_base,
        output_contract='dense histogram at every step and two integer readouts',
        source='same eager inverse-LMS Philox tape for every method',
        timing='all setup and complete trajectory; JIT, counters, validation and serialization excluded',
        instrumentation='work counters in a separate untimed replay')
    (output/'header.json').write_text(json.dumps(header,indent=2)+'\n',encoding='utf-8')
    run.warmup()
    tasks=[(i,rep) for i in range(len(specs)) for rep in range(nseeds)]
    random.Random(seed_base+619).shuffle(tasks)
    with (output/'timings.jsonl').open('x',encoding='utf-8') as f:
        for task,(cell,rep) in enumerate(tasks):
            group,spec,methods=specs[cell]
            seed=seed_base+1000*cell+rep
            observations={name:[] for name in methods}
            expected=None
            for repeat in range(rounds):
                order=list(methods);random.Random(seed+7919*repeat).shuffle(order)
                for position,name in enumerate(order):
                    seconds,setup,result=run.execute(spec,seed,name)
                    run.validate(result,spec)
                    if expected is None:expected=result[:2]
                    elif not all(np.array_equal(a,b) for a,b in zip(expected,result[:2])):
                        np.savez_compressed(output/'failure.npz',expected_hist=expected[0],
                            actual_hist=result[0],expected_readout=expected[1],actual_readout=result[1])
                        raise AssertionError((cell,seed,name,repeat))
                    observations[name].append(dict(round=repeat,position=position,
                                                    seconds=seconds,setup_seconds=setup))
            saved=dict(histogram=expected[0],readouts=expected[1])
            summaries={}
            for name in methods:
                summaries[name]=dict(seconds=statistics.median(x['seconds'] for x in observations[name]),
                    setup_seconds=statistics.median(x['setup_seconds'] for x in observations[name]),
                    rounds=observations[name])
                if name not in ('reuse','rebuild'):continue
                _,_,audited=run.execute(spec,seed,name,True)
                assert all(np.array_equal(a,b) for a,b in zip(expected,audited[:2]))
                saved[name+'_work']=audited[2]
                saved[name+'_widths']=audited[3]
                saved[name+'_max_tests']=audited[4]
                saved['occupied']=audited[5]
                assert np.array_equal(audited[3][:,2,:],audited[4])
                summaries[name]['work_totals']=audited[2].sum(axis=0).tolist()
            assert np.array_equal(saved['reuse_widths'],saved['rebuild_widths'])
            file=f'cell{cell:02d}_seed{seed}.npz'
            np.savez_compressed(data/file,**saved)
            record=dict(cell=cell,group=group,spec=spec,replicate=rep,seed=seed,
                        path='paths/'+file,rows=summaries,all_equal=True)
            f.write(json.dumps(record)+'\n');f.flush()
            print(json.dumps(dict(case=task+1,total=len(tasks),group=group,spec=spec,
                  seconds={n:round(v['seconds'],6) for n,v in summaries.items()})),flush=True)
    (output/'complete.json').write_text(json.dumps(dict(cases=len(tasks),all_equal=True,
        timing_observations=sum(len(specs[i][2])*rounds for i,_ in tasks)),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--development',action='store_true');a=p.parse_args()
    benchmark(a.output,a.development)
