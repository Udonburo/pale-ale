"""Reproduce current evidence and PDFs without repeating performance timings."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parent


def run(label,args,cwd=ROOT):
    result=subprocess.run([sys.executable,*map(str,args)],cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace')
    log=ROOT/'tmp/reproduction'/f'{label}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        print((result.stdout+result.stderr)[-10000:])
        raise RuntimeError(label+' failed; see '+str(log))
    print(label+': PASS',flush=True)


def legacy_error_example():
    name='retained-dense-inputs.zip'
    archive=next((p for p in (ROOT/'archive'/name,ROOT/'output'/name) if p.exists()),None)
    if archive is None:raise FileNotFoundError(name)
    destination=(ROOT/'tmp/reproduction/legacy_dense').resolve()
    destination.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive) as z:
        for item in z.infolist():
            target=(destination/item.filename).resolve()
            if not target.is_relative_to(destination):raise ValueError('unsafe archive path')
        z.extractall(destination)
    legacy=destination/'array-rqmc-paper1'
    script=legacy/'analyze.py'
    command=("import matplotlib; matplotlib.rcParams['svg.hashsalt']='rqmc-retained-error-20261003'; "
             "import runpy; runpy.run_path("+repr(str(script))+",run_name='__main__')")
    run('legacy_error_analysis',['-c',command],legacy)
    for name in ('tables/errors.md','figures/errors.svg','figures/errors.png'):
        shutil.copy2(legacy/name,ROOT/name)
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--skip-pdf',action='store_true')
    parser.add_argument('--require-identical',action='store_true',help='Require rebuilt PDFs to match saved bytes on this environment.')
    args=parser.parse_args()
    names=('implicit-array-rqmc','retained-studies')
    before={n:hashlib.sha256((ROOT/'output/pdf'/f'{n}.pdf').read_bytes()).hexdigest() for n in names if (ROOT/'output/pdf'/f'{n}.pdf').exists()}
    study=ROOT/'review_checks/primal_dual'
    run('scalar_checks',[study/'confirmation/source/check.py'])
    run('primal_counter',[study/'check_primal_counter.py'])
    run('model_domains',[study/'check_model_domains.py'])
    run('point_sorting_bridge',[study/'check_point_sorting_bridge.py'])
    run('worked_example',[ROOT/'review_checks/check_worked_example.py'])
    run('saved_source_replay',[study/'replay.py',*(['--quick'] if args.quick else [])])
    run('conditional_workload',[study/'verify_workloads.py'])
    run('current_analysis',[study/'analyze.py'])
    run('mechanism_figures',[ROOT/'draw_mechanisms.py'])
    legacy_hash=legacy_error_example()
    matches={}
    if not args.skip_pdf:
        run('main_pdf',[ROOT/'build_pdf.py'])
        run('supplement_pdf',[ROOT/'build_pdf.py','--supplement'])
        matches={n:hashlib.sha256((ROOT/'output/pdf'/f'{n}.pdf').read_bytes()).hexdigest()==before.get(n) for n in names}
        if args.require_identical and not all(matches.values()):
            raise AssertionError(('rebuilt PDF bytes differ',matches))
    report=dict(status='PASS',quick=args.quick,pdf_rebuilt=not args.skip_pdf,
                pdf_bytes_match_saved=matches,legacy_archive_sha256=legacy_hash,
                performance_timings_repeated=False,mechanism_figures_checked=True)
    (ROOT/'output/reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
