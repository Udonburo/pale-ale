"""Regenerate current tables, figures, matrices and accounting from raw records."""
from pathlib import Path
import argparse
import json
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import executors as ex
from confirmation.source import benchmark

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]


def label(spec):
    return ('R'+str(spec['K']) if spec['model']=='repair' else 'T')


def exhaustive_geometry():
    tested=0
    for m in range(1,5):
        N=1 << m
        for mask in range(1 << (N-1)):
            edges=[0]+[j for j in range(1,N) if (mask >> (j-1)) & 1]+[N]
            actual=[0]*(m+1)
            for a,b in zip(edges,edges[1:]):
                while a<b:
                    k=(b-a).bit_length()-1
                    if a:k=min(k,(a & -a).bit_length()-1)
                    actual[k]+=1;a+=1 << k
            I=[0]+[len({a//(1 << k) for a in edges[1:-1] if a % (1 << k)}) for k in range(1,m+1)]
            expected=[2*I[k+1]-I[k] for k in range(m)]+[1-I[m]]
            assert actual==expected and sum(actual)==1+sum(I)
            assert max(actual)<=2*(len(edges)-1)
            tested+=1
    return tested


def timing_ratios(cell, numerator, denominator):
    times=cell['seed_median_seconds']
    return [a/b for a,b in zip(times[numerator],times[denominator])]


def write_timing_table(result):
    rows=[]
    for cell in result['cells']:
        s=cell['spec']
        fields=[label(s),f"{s['w']}, {s['m']}",f"{1000*cell['median_seconds']['basis_direct']:.3f}"]
        pairs=(('rank_stream','basis_direct'),('reuse','basis_direct'),
               ('basis_prefix','basis_direct'),('primal_reuse','basis_direct'),('rebuild','reuse'))
        fields += [f'{statistics.median(timing_ratios(cell,a,b)):.3f}' for a,b in pairs]
        rows.append('| '+' | '.join(fields)+' |')
    table=['**Table 2. Six-method confirmation with direct basis as the execution reference.** R63/R255 denote repair capacities (100 steps); T denotes tandem (200 steps). Direct milliseconds are medians of eight seed medians of five technical timings. Ratios are formed within seed before taking their median; values above one favor the denominator. The final column compares matched dual reconstruction with coefficient retention.', '',
           '| Model | $w,m$ | Direct ms | Stream / direct | Dual reuse / direct | Prefix / direct | Primal / direct | Rebuild / dual reuse |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |',*rows]
    (PAPER/'tables/primal_dual_times.md').write_text('\n'.join(table)+'\n',encoding='utf-8')


def timing_findings(result):
    return dict(
        reuse_over_direct=[statistics.median(timing_ratios(c,'reuse','basis_direct')) for c in result['cells']],
        stream_over_direct=[statistics.median(timing_ratios(c,'rank_stream','basis_direct')) for c in result['cells']],
        direct_median_seconds=[c['median_seconds']['basis_direct'] for c in result['cells']],
        rebuild_over_reuse=[statistics.median(timing_ratios(c,'rebuild','reuse')) for c in result['cells']],
        paired_basis_direct_over_primal=[statistics.median(timing_ratios(c,'basis_direct','primal_reuse')) for c in result['cells']])


def write_timing_figure(result,svg_metadata):
    pairs=(('basis_prefix','basis_direct'),('primal_reuse','basis_direct'),
           ('rebuild','reuse'),('reuse','basis_direct'))
    titles=('A  Interval\ndecomposition','B  Primal\nretention',
            'C  Dual\nretention','D  Executor\ncomparison')
    labels=('Prefix basis /\nDirect basis','Primal reuse /\nDirect basis',
            'Rebuild /\nDual reuse','Dual reuse /\nDirect basis')
    ticks=((1,1.5,2),(.8,1,1.2,1.4),(.8,1,1.2,1.4,1.6),(1,2,3))
    colors=('#32698f','#667987','#27756d','#ba6235')
    fig,axes=plt.subplots(1,4,figsize=(7.15,4.4),sharey=True,layout='constrained')
    panels=[]
    for ax,(numerator,denominator),title,xlabel,xticks,color in zip(axes,pairs,titles,labels,ticks,colors):
        rows=[]
        for i,cell in enumerate(result['cells']):
            values=timing_ratios(cell,numerator,denominator)
            assert len(values)==8 and all(x>0 for x in values)
            median=statistics.median(values)
            rows.append(dict(cell=cell['cell'],spec=cell['spec'],seed_indices=list(range(8)),
                             paired_ratios=values,median=median))
            ax.scatter(values,i+np.linspace(-.16,.16,8),s=12,color=color,alpha=.7,edgecolors='none')
            ax.plot(median,i,marker='|',color='black',ms=9,mew=1.1)
        all_values=[x for row in rows for x in row['paired_ratios']]
        lo=min(1,min(all_values),min(xticks));hi=max(1,max(all_values),max(xticks))
        padding=.04*(hi-lo)
        limits=(lo-padding,hi+padding)
        assert min(all_values)>limits[0] and max(all_values)<limits[1]
        ax.axvline(1,color='#777777',lw=.8,ls='--')
        ax.axhline(10.5,color='#bbbbbb',lw=.5)
        ax.set_title(title,fontsize=8)
        ax.set_xticks(xticks);ax.set_xlim(*limits)
        ax.tick_params(axis='x',labelsize=7.3)
        ax.grid(axis='x',color='#eeeeee');ax.set_xlabel(xlabel,fontsize=7.8)
        panels.append(dict(numerator=numerator,denominator=denominator,
                           axis_limits=limits,seed_ratio_range=[min(all_values),max(all_values)],cells=rows))
    axes[0].set_yticks(range(17),[f"{label(c['spec'])}  {c['spec']['w']}/{c['spec']['m']}" for c in result['cells']])
    axes[0].invert_yaxis()
    fig.savefig(PAPER/'figures/primal_dual_pairs.svg',metadata={**svg_metadata,'Date':'2026-10-04T00:00:00Z'})
    fig.savefig(PAPER/'figures/primal_dual_pairs.png',dpi=600)
    fig.savefig(PAPER/'figures/primal_dual_pairs.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    (HERE/'figure1-paired-ratios.json').write_text(json.dumps(dict(status='PASS',technical_timings=5,seeds_per_cell=8,
           cells=17,panels=panels,aggregation='median of within-seed ratios of five-timing medians'),indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing-only',action='store_true',help='Reaggregate Table 2 and timing findings from saved observations only.')
    args=parser.parse_args()
    data=HERE/'confirmation/data'
    result=benchmark.summarize(data)
    assert result['complete'] and result['timing_rows']==4080
    write_timing_table(result)
    if args.timing_only:
        review=json.loads((HERE/'results.json').read_text(encoding='utf-8'))
        review.update(timing_findings(result))
        (HERE/'results.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(status='PASS',timing_rows=result['timing_rows'],cells=len(result['cells']),performance_timings_repeated=False)))
        return
    # Validate each recorded per-step accounting identity, without re-timing.
    audited=0;case_metrics={};matrices={}
    for cell in result['cells']:
        spec=cell['spec'];m,w=spec['m'],spec['w']
        C=ex.original.base_rows(m,w)
        deps,heads,_,_=ex.base.prepare(C,m)
        weight=[];position=[]
        for k in range(m+1):
            p=np.flatnonzero(deps[k])
            weight.append(np.concatenate(([0],np.cumsum([int(deps[k,j]).bit_count()+int(heads[k,j]).bit_count() for j in p]))))
            position.append(np.concatenate(([0],p+1)))
        key=f'm{m}-w{w}'
        if key not in matrices:
            engine=ex.original.qmc.Sobol(d=2,scramble=False,bits=max(m,w))
            matrices[key]=dict(m=m,w=w,bit_depth=max(m,w),C_rows=C.tolist(),
                               direction_columns=engine._sv[:,:m].astype(np.int64).tolist(),
                               convention='Rows of C are packed MSB-first rank bits; output rows run MSB to LSB. Direction integers use SciPy bit-depth convention. C=C_y C_s^{-1}.')
        case_metrics[cell['cell']]=[]
        for rep in range(8):
            with np.load(data/'trajectories'/f"{cell['cell']:02d}-{rep:02d}.npz") as f:
                work=f['work'];rebuild=f['rebuild_work'];J=f['max_tests'];widths=f['widths'];hist=f['hist'];occupied=f['occupied']
                assert np.array_equal(J,widths[:,2,:])
                assert np.array_equal(work[:,2],J.sum(axis=1))
                assert np.array_equal(work[:,[0,1,2,8,9,10]],rebuild[:,[0,1,2,8,9,10]])
                for tick in range(spec['horizon']):
                    Xi=sum(weight[k][J[tick,k]] for k in range(m+1))
                    E=sum(position[k][J[tick,k]] for k in range(m+1))
                    assert Xi==work[tick,3] and E==rebuild[tick,6]
                    previous=np.array([1 << m],dtype=np.int64) if tick==0 else hist[tick-1][hist[tick-1]>0]
                    assert len(previous)==occupied[tick]
                    endpoints=np.cumsum(previous)[:-1]
                    I=np.zeros(m+2,dtype=np.int64)
                    for k in range(1,m+1):
                        I[k]=len({int(a)//(1 << k) for a in endpoints if int(a) % (1 << k)})
                    H=np.array([2*I[k+1]-I[k] for k in range(m)]+[1-I[m]],dtype=np.int64)
                    assert work[tick,8]==int(H.sum())
                    assert np.array_equal(widths[tick,0],2*H)
                    audited+=1
                v=dict(S=float(occupied.mean()),H=float(work[:,8].mean()),D=float(work[:,1].mean()),
                       J=float(work[:,2].mean()),Xi=float(work[:,3].mean()),E=float(rebuild[:,6].mean()),
                       G=float(rebuild[:,7].mean()),reuse_xors=float((work[:,3]+4*work[:,2]).mean()),
                       rebuild_xors=float((rebuild[:,5]+4*rebuild[:,7]).mean()))
                case_metrics[cell['cell']].append(v)
    selected=(5,7,10,14,16)
    metrics=[('Occupied states $S_t$','S'),('Rank blocks $H_t$','H'),('Dependent tests $D_t$','D'),
             ('New relations $J_t$','J'),('Dual reuse row XORs $\Xi_t$','Xi'),('Rebuilt rows $E_t$','E'),
             ('Gaussian reductions $G_t$','G'),('Dual reuse construction XORs','reuse_xors'),('Rebuild construction XORs','rebuild_xors')]
    lines=['**Table 3. Shared work in the confirmation, $w=52$.** Entries are marginal medians over eight seeds of per-step means. R63/R255 denote repair, and T denotes tandem. Construction XORs use the source-level identities of Section 4.3. Displayed marginal medians need not satisfy the per-trajectory identities arithmetically; counts are not latency measurements.', '',
           '| Quantity | R63, $m=12$ | R63, $m=20$ | R255, $m=20$ | T, $m=12$ | T, $m=20$ |',
           '| --- | --- | --- | --- | --- | --- |']
    for name,key in metrics:
        lines.append('| '+' | '.join([name]+[f"{statistics.median(x[key] for x in case_metrics[c]):.2f}" for c in selected])+' |')
    (PAPER/'tables/primal_dual_work.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (data/'matrices.json').write_text(json.dumps(matrices,indent=2)+'\n',encoding='utf-8')
    accounting=dict(status='PASS',geometry_partitions=exhaustive_geometry(),paired_trajectories=136,paired_steps=audited,case_metrics=case_metrics,
                    identities=['shared J=max depth','identical dual query profiles','Xi weights','rebuild row positions','rank boundary geometry','Q_k=2 H_k'])
    (HERE/'accounting.json').write_text(json.dumps(accounting,indent=2)+'\n',encoding='utf-8')
    workloads=json.loads((HERE/'workloads/summary.json').read_text())
    # Observations remain in the original summary; theoretical curves and table
    # entries are rounded once from the separately verified rational values.
    exact=json.loads((HERE/'workloads/expectations-exact.json').read_text())
    from fractions import Fraction
    assert len(workloads['workloads']) == len(exact['workloads'])
    for observed, rational in zip(workloads['workloads'], exact['workloads']):
        assert observed['index'] == rational['index']
        observed['exact_total_mean']=float(Fraction(*rational['totals']['mean']))
        observed['psi_total']=float(Fraction(*rational['totals']['psi']))
        assert len(observed['widths']) == len(rational['widths'])
        for row, value in zip(observed['widths'], rational['widths']):
            assert row['k'] == value['k']
            row['exact_mean']=float(Fraction(*value['exact_fractions']['mean']))
            row['psi']=float(Fraction(*value['exact_fractions']['psi']))
    lines=['**Table 4. Fixed incoming workloads and resampled shifts.** All use $w=52$. Repair snapshots are before step 51; tandem snapshots are before step 101, with steps numbered from one. State intervals, $C,U,V$ and thresholds are held fixed. Each mean uses 4,096 independent shift replicas; parentheses give the empirical standard error of the total demand. The last column sums the width-wise $\Psi$ bounds.', '',
           '| Workload | $S$ | Queries | Exact $\mathbb E\sum_k J_k$ | Sampled mean (SE) | $\sum_k\Psi$ |',
           '| --- | --- | --- | --- | --- | --- |']
    for x in workloads['workloads']:
        s=x['spec']
        lines.append('| '+ ' | '.join([f"{label(s)}, $m={s['m']}$",str(x['occupied']),str(x['queries']),
                                       f"{x['exact_total_mean']:.3f}",f"{x['sampled_total_mean']:.3f} ({x['sampled_total_se']:.3f})",f"{x['psi_total']:.3f}"])+' |')
    (PAPER/'tables/workload_prediction.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    phases=json.loads((HERE/'workloads/phases.json').read_text())
    lines=['**Table 5. Staged workload diagnostics, microseconds.** R63 uses $m=20$ and tandem uses $m=20$; both have $w=52$. Construction includes the relevant cache initialization. Query consumes prepared constraints or bases; aggregation includes compaction and readouts. The demanded constraint set is identified in an untimed replay. Staged execution materializes queries and flows and includes Python dispatch, so its marginal phase medians are not a decomposition of the separately timed fused step.', '',
           '| Workload / method | Source | Construction | Query | Aggregation | Staged total | Fused total |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    display={'reuse':'Dual reuse','rebuild':'Rebuild','basis_direct':'Direct basis','primal_reuse':'Primal reuse'}
    for i in (1,4):
        for x in phases['summaries']:
            if x['workload']!=i:continue
            fields=[('R63 / ' if i==1 else 'T / ')+display[x['method']]]
            fields += [f"{1e6*x[k]:.1f}" for k in ('source','construction','query','aggregation','staged_total','fused_total')]
            lines.append('| '+' | '.join(fields)+' |')
    (PAPER/'tables/workload_phases.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=['**Table C1. Fixed-workload repetitions, microseconds per step.** R63 and T are the preselected repair and tandem snapshots with $m=20,w=52$. Preparation is measured separately in nine batches of 200 calls; the remaining columns are medians of nine setup-inclusive sequences. The workload and tape are identical at every repetition, with step caches cleared. Python dispatch and host variation remain present.', '',
           '| Workload / method | Preparation once | $T=1$ | $T=4$ | $T=64$ | $T=256$ |',
           '| --- | --- | --- | --- | --- | --- |']
    for i in (1,4):
        for method in display:
            prep=next(x['median_seconds'] for x in phases['preparations'] if x['workload']==i and x['method']==method)
            times=[statistics.median(x['seconds_per_step'] for x in phases['amortization'] if x['workload']==i and x['method']==method and x['repeats']==T) for T in (1,4,64,256)]
            lines.append('| '+' | '.join([('R63 / ' if i==1 else 'T / ')+display[method],f'{prep*1e6:.1f}']+[f'{x*1e6:.1f}' for x in times])+' |')
    (PAPER/'tables/fixed_amortization.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'rqmc-primal-dual-20261003'})
    svg_metadata={'Date':'2026-10-03T00:00:00Z','Creator':'Array-RQMC review companion'}
    write_timing_figure(result,svg_metadata)
    fig,axes=plt.subplots(1,3,figsize=(7.15,2.45),sharey=True,layout='constrained')
    for panel,(ax,i) in enumerate(zip(axes,(0,1,4))):
        row=workloads['workloads'][i];points=[p for p in row['widths'] if p['queries']]
        k=[p['k'] for p in points]
        ax.plot(k,[p['psi'] for p in points],color='#999999',ls='--',label='Upper bound')
        ax.plot(k,[p['exact_mean'] for p in points],color='#32698f',lw=1.3,label='Exact expectation')
        ax.errorbar(k,[p['sample_mean'] for p in points],yerr=[1.96*p['sample_sd']/64 for p in points],fmt='o',ms=3,color='#ba6235',capsize=1,label='Shift mean +/- 1.96 SE')
        ax.set_title(f"{'ABC'[panel]}  {label(row['spec'])}, m={row['spec']['m']}")
        ax.set_xlabel('Free rank bits k');ax.grid(axis='y',color='#eeeeee')
    axes[0].set_ylabel('Shared constraint demand')
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=7.5,loc='outside lower center',ncol=3)
    fig.savefig(PAPER/'figures/workload_demand.svg',metadata=svg_metadata)
    fig.savefig(PAPER/'figures/workload_demand.png',dpi=600)
    fig.savefig(PAPER/'figures/workload_demand.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    sharp=[]
    for q in range(10):
        Q=1 << q
        thresholds=[(i << (10-q))+1 for i in range(Q)]
        demand=[max(10 if y==t else 11-(y^t).bit_length() for t in thresholds) for y in range(1024)]
        exact=sum(demand)/1024
        psi=sum(min(1.,Q/(1 << s)) for s in range(10))
        assert exact==psi
        sharp.append(dict(Q=Q,enumerated_mean=exact,psi=psi,coarse=min(10,q+2)))
    (HERE/'sharpness.json').write_text(json.dumps(dict(status='PASS',shifts_per_batch=1024,batches=sharp),indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(5.8,2.1),layout='constrained')
    ax.plot([x['Q'] for x in sharp],[x['psi'] for x in sharp],color='#32698f',label='Finite-sum bound')
    ax.scatter([x['Q'] for x in sharp],[x['enumerated_mean'] for x in sharp],color='#ba6235',s=22,label='All-shift mean',zorder=3)
    ax.plot([x['Q'] for x in sharp],[x['coarse'] for x in sharp],color='#999999',ls='--',label='Coarse logarithmic bound')
    ax.set_xscale('log',base=2);ax.set_xlabel('Queries Q');ax.set_ylabel('Shared demand');ax.legend(fontsize=8)
    ax.grid(color='#eeeeee')
    fig.savefig(PAPER/'figures/reuse_shared_work_bound.svg',metadata=svg_metadata)
    fig.savefig(PAPER/'figures/reuse_shared_work_bound.png',dpi=600)
    fig.savefig(PAPER/'figures/reuse_shared_work_bound.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    review=dict(confirmation_complete=True,paired_steps=audited,**timing_findings(result))
    (HERE/'results.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',paired_steps=audited,
                         rebuild_over_reuse=(min(review['rebuild_over_reuse']),max(review['rebuild_over_reuse'])),
                         reuse_over_direct=(min(review['reuse_over_direct']),max(review['reuse_over_direct'])))))


if __name__=='__main__':main()
