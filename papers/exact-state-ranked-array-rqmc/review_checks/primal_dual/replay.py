"""Replay every paired case using the source that actually ran confirmation."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'confirmation/source'
sys.path.insert(0,str(SOURCE))
import executors as ex


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true',help='One small repair and tandem pair; full replay is the default.')
    args=parser.parse_args()
    data=HERE/'confirmation/data'
    meta=json.loads((data/'metadata.json').read_text())
    for name,expected in meta['source'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==expected, name
    matrices=json.loads((data/'matrices.json').read_text())
    for key,row in matrices.items():
        assert ex.original.base_rows(row['m'],row['w']).tolist()==row['C_rows'],key
    ex.warmup()
    trajectories=steps=0
    for cell,spec in enumerate(meta['cells']):
        if args.quick and cell not in (0,11):continue
        for rep in range(1 if args.quick else meta['replications']):
            seed=meta['seed_base']+100*cell+rep
            with np.load(data/'trajectories'/f'{cell:02d}-{rep:02d}.npz') as f:
                for method in ex.METHODS:
                    audit=method in ('reuse','rebuild')
                    _,_,result=ex.execute(spec,seed,method,audit)
                    ex.validate(result,spec)
                    assert np.array_equal(result[0],f['hist']) and np.array_equal(result[1],f['readouts'])
                    if audit:
                        assert np.array_equal(result[2],f['work' if method=='reuse' else 'rebuild_work'])
                        for idx,name in ((3,'widths'),(4,'max_tests'),(5,'occupied')):
                            assert np.array_equal(result[idx],f[name])
                    trajectories+=1;steps+=spec['horizon']
        print('replayed cell '+str(cell),flush=True)
    report=dict(status='PASS',quick=args.quick,method_trajectories=trajectories,
                method_steps=steps,source_identity_verified=True,matrices_verified=len(matrices))
    (HERE/('replay_quick.json' if args.quick else 'replay.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
