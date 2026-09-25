"""Small engineering timing comparison; not a rerun of the calibrated study."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
import numpy as np
import scipy
import numba
import projection
import ordered
import reference


def evaluate(c, method, n, seed):
    # Four children reproduce the source driver's stream partition; two are unused.
    streams = np.random.SeedSequence(seed).spawn(4)
    rng, noise = (np.random.default_rng(s) for s in streams[:2])
    if method == "LIBRARY":
        return reference.array_run(c, n, rng, noise)
    if method == "DIRECT":
        return projection.array_kernel(c, n, rng, noise)
    return ordered.array_kernel(c, n, rng, noise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[512,4096])
    parser.add_argument("--repeats", type=int, default=3)
    args=parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error("repeats must be in 1..100")
    for n in args.sizes:
        projection.log2_size(n)
        if n > 65536:
            parser.error("This deliberately small driver caps N at 65,536.")
    config=json.loads((Path(__file__).resolve().parent / "data/cases.json").read_text(encoding="utf-8"))
    c=reference.Condition(**next(x for x in config["conditions"] if x["name"]=="case_08_distant"))
    methods=("LIBRARY","DIRECT","MAINTAINED")
    for m in methods:
        evaluate(replace(c,horizon=4), m, 8, [90412,0])
    scheduler=np.random.default_rng(904120)
    measurements=[]
    for n in args.sizes:
        direct=evaluate(c,"DIRECT",n,[90412,n])
        maintained=evaluate(c,"MAINTAINED",n,[90412,n])
        assert direct[0].tobytes()==maintained[0].tobytes()
        assert direct[1].tobytes()==maintained[1].tobytes()
        rows={m:[] for m in methods}
        for rep in range(args.repeats):
            for index in scheduler.permutation(len(methods)):
                m=methods[index]
                start=perf_counter()
                evaluate(c,m,n,[90412,n,rep])
                rows[m].append(perf_counter()-start)
        measurements.extend({"method":m,"n":n,"k":1,"repeats":args.repeats,
                             "median_warm_ms":1000*median(v)} for m,v in rows.items())
    print(json.dumps({"scope":"Engineering smoke timing only; no MSE calibration or new scientific confirmation.",
                      "same_seed_direct_vs_maintained":"PASS",
                      "library_vs_direct_same_seed":"Not claimed",
                      "python":platform.python_version(),"numpy":np.__version__,
                      "scipy":scipy.__version__,"numba":numba.__version__,
                      "rows":measurements},indent=2))


if __name__=="__main__":
    main()
