"""Draw the manuscript's existing examples; perform no timing experiments."""
from pathlib import Path
from datetime import datetime, timezone
import json, runpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT=Path(__file__).resolve().parent
STAMP=datetime(2026,10,4,tzinfo=timezone.utc)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'svg.hashsalt':'rqmc-round3-mechanisms-20261004',
                     'pdf.fonttype':42})
BLUE='#28678d'; TEAL='#28776b'; INK='#253441'; GREY='#63717c'

def save(fig,stem):
    fig.savefig(ROOT/'figures'/f'{stem}.svg',metadata={'Date':'2026-10-04','Creator':'RQMC manuscript'})
    fig.savefig(ROOT/'figures'/f'{stem}.pdf',metadata={'Creator':'RQMC manuscript','CreationDate':STAMP,'ModDate':STAMP})
    fig.savefig(ROOT/'figures'/f'{stem}.png',dpi=600)
    plt.close(fig)

def box(ax,x,y,w,h,text,color=BLUE,size=11):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.012,rounding_size=0.018',
                              ec=color,fc='#f4f7f8',lw=1))
    ax.text(x,y,text,ha='center',va='center',color=INK,fontsize=size,linespacing=1.5)

def arrow(ax,a,b,color=GREY):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.2,'color':color,'shrinkA':1,'shrinkB':1})

def four_rank():
    funcs=runpy.run_path(str(ROOT/'review_checks/check_worked_example.py'))
    mv,bits,integer,solve=(funcs[n] for n in ('mv','bits','integer','solve_lower'))
    C=[[1,0,1],[0,1,1],[0,1,0],[1,0,0]]
    U=[[1,0,0],[1,1,0],[0,1,1]]
    V=[[1,0,0,0],[1,1,0,0],[0,1,1,0],[1,0,1,1]]
    c=[0,1,0,1]
    outputs=[]
    for r in range(4,8):
        rhs=[a^b for a,b in zip(mv(C,mv(U,bits(r,3))),c)]
        outputs.append(integer(solve(V,rhs)))
    span={0,0b1001,0b0100,0b1001^0b0100}
    dual={y for y in range(16) if bits(y,4)[2]==0 and
          (bits(y,4)[0]^bits(y,4)[2]^bits(y,4)[3])==0}
    assert outputs==[13,4,0,9] and span==dual==set(outputs)
    below={tau:sum(y<tau for y in outputs) for tau in (0,9,10,16)}
    flows=[below[9],below[10]-below[9],below[16]-below[10]]
    assert below=={0:0,9:2,10:3,16:4} and flows==[2,1,1]
    fig=plt.figure(figsize=(7.1,2.55))
    ax=fig.add_axes((.02,.03,.96,.94));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.145,.64,.245,.43,'One aligned block\nRanks 4, 5, 6, 7\nOutputs in rank order\n(13, 4, 0, 9)')
    box(ax,.475,.82,.28,.25,'Primal image\nspan(1001, 0100)',size=11)
    box(ax,.475,.48,.28,.27,'Dual constraints\n$y_3=0$\n$y_1\\oplus y_3\\oplus y_4=0$',color=TEAL,size=11)
    box(ax,.83,.65,.275,.50,'Strict counts\n$F(9)=2,\\ F(10)=3$\nInterval flows\n[0, 9): 2\n[9, 10): 1\n[10, 16): 1',size=10.8)
    arrow(ax,(.278,.73),(.32,.82));arrow(ax,(.278,.55),(.32,.48))
    arrow(ax,(.63,.82),(.677,.74));arrow(ax,(.63,.48),(.677,.54))
    box(ax,.655,.125,.65,.13,'Sum all block flows  →  next histogram  →  rerank',color=GREY,size=10.7)
    arrow(ax,(.83,.385),(.83,.21))
    ax.text(.14,.20,'Other blocks',ha='center',va='center',color=GREY,fontsize=11)
    arrow(ax,(.245,.20),(.32,.125))
    save(fig,'four_rank_mechanism')
    return {'ranks':[4,5,6,7],'rank_ordered_outputs':outputs,'primal_generators':[9,4],
            'primal_span':sorted(span),'dual_output_set':sorted(dual),'strict_counts':below,
            'interval_flows':flows,'aggregation_scope':'this block plus all other blocks before reranking'}

def blocks(a,b):
    answer=[]
    while a<b:
        k=(b-a).bit_length()-1
        if a:k=min(k,(a&-a).bit_length()-1)
        end=a+(1<<k);answer.append([a,end]);a=end
    return answer

def structure():
    geometry={str(boundary):[blocks(0,boundary),blocks(boundary,16)] for boundary in (8,7)}
    assert geometry=={'8':[[[0,8]],[[8,16]]],
                      '7':[[[0,4],[4,6],[6,7]],[[7,8],[8,16]]]}
    batches=([1]*16,[64*i+1 for i in range(16)])
    occupied=[];means=[]
    for thresholds in batches:
        sets=[{tau>>(10-length) if length else 0 for tau in thresholds} for length in range(10)]
        counts=[len(s) for s in sets]
        mean=sum(count/(1<<length) for length,count in enumerate(counts))
        visits=[max(10 if y==tau else 11-(y^tau).bit_length() for tau in thresholds) for y in range(1024)]
        assert sum(visits)/1024==mean
        occupied.append(counts);means.append(mean)
    assert occupied==[[1]*10,[1,2,4,8,16,16,16,16,16,16]]
    assert means==[1.998046875,5.96875]
    fig=plt.figure(figsize=(7.1,2.9))
    a=fig.add_axes((.05,.23,.35,.61));a.set(xlim=(-.2,16.2),ylim=(-.16,1.0));a.axis('off')
    a.text(0,1.08,'A  Rank-boundary placement',weight='bold',fontsize=11)
    a.text(0,.95,'$N=16,\\ S=2$',fontsize=11)
    for y,boundary in ((.64,8),(.25,7)):
        a.text(0,y+.16,f'Boundary {boundary}: '+('$H=2$' if boundary==8 else '$H=5$'),fontsize=10.7)
        for state,parts in enumerate(geometry[str(boundary)]):
            for left,right in parts:
                a.add_patch(Rectangle((left,y-.09),right-left,.18,fc=BLUE if state==0 else TEAL,ec='white',lw=1.3))
        a.plot([boundary,boundary],[y-.13,y+.12],color=INK,lw=1)
    for rank in (0,4,6,7,8,16):
        a.text(rank,-.02,str(rank),ha='center',va='top',fontsize=10.5)
    a.text(8,-.13,'rank',ha='center',va='top',fontsize=10.5,color=GREY)
    b=fig.add_axes((.47,.25,.48,.58));b.set(xlim=(-.10,1.10),ylim=(.08,1.17));b.axis('off')
    b.text(-.10,1.29,'B  Occupied query prefixes',weight='bold',fontsize=11)
    b.text(-.06,1.13,'depth',ha='center',fontsize=10.5,color=GREY)
    b.text(.14,1.13,'$\\tau=1$',ha='center',fontsize=10.5)
    b.text(.64,1.13,'$64i+1$',ha='center',fontsize=10.5)
    for length in range(5):
        y=1-.18*length
        b.text(-.06,y,str(length),ha='center',va='center',fontsize=10.5,color=GREY)
        b.plot(.14,y,'o',ms=4.5,color=BLUE)
        b.text(.25,y,'1',ha='center',va='center',fontsize=10.5,color=BLUE)
        if length: b.plot([.14,.14],[y+.18,y],color=BLUE,lw=1)
        positions=[.35+.58*(i+.5)/(1<<length) for i in range(1<<length)]
        for i,x in enumerate(positions):
            b.plot(x,y,'o',ms=3.5,color=TEAL)
            if length:
                parent=.35+.58*((i//2)+.5)/(1<<(length-1))
                b.plot([parent,x],[y+.18,y],color=TEAL,lw=.8)
        b.text(1.02,y,str(1<<length),ha='center',va='center',fontsize=10.5,color=TEAL)
    fig.text(.47,.15,'Depths 5–9: 1 versus 16 prefixes',fontsize=10.5,color=GREY)
    fig.text(.47,.055,'Expected $J$: 1.998046875 versus 5.96875',fontsize=10.5,color=INK)
    save(fig,'structural_work_examples')
    return {'geometry':{'N':16,'S':2,'partitions':geometry,'block_counts':[2,5]},
            'prefix_example':{'k':0,'w':10,'r0':0,'U':'I','V':'I','Q':16,
                              'thresholds':batches,'occupied_prefix_counts_depth_0_to_9':occupied,
                              'common_uniform_syndrome':True,'expected_demands':means,
                              'exhausted_shifts':1024}}

def main():
    report={'status':'PASS','new_observations':False,'four_rank':four_rank(),'structural_examples':structure()}
    (ROOT/'review/diagram-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
