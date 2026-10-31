"""Generate report assets from completed evidence. No invented performance values."""
import argparse
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent


def load():
    return [json.loads((ROOT/path).read_text(encoding='utf-8')) for path in
            ['results/registry.json','results/runtime.json','results/counterexample/summary.json',
             'results/validation.json','results/learned_selector.json']]


def charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    r,rt,c,v,ml=load()
    pp=[p for p in r['profiles'] if p['workload']=='uniform_large_holdout' and p['policy']=='resident']
    pp.sort(key=lambda p:['index_add','pyg','sparse','cuda_csr'].index(p['backend']))
    colors=['#8d99ae','#5d8ba6','#c09356','#23876a']
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.0),layout='constrained')
    labels=['Index-add','PyG scatter','Sparse CSR','Custom CUDA']
    axes[0].bar(labels,[p['step_ms'] for p in pp],color=colors,
                yerr=np.array([[p['step_ms']-p['ci95_ms'][0] for p in pp],
                               [p['ci95_ms'][1]-p['step_ms'] for p in pp]]),capsize=3)
    axes[0].set_ylabel('Full training step (ms)');axes[0].set_title('Median of 5 independent-run medians',fontsize=10)
    axes[1].bar(labels,[p['peak_allocated_bytes']/2**20 for p in pp],color=colors)
    axes[1].set_ylabel('Peak tensor allocated (MiB)');axes[1].set_title('Worst observed run peak',fontsize=10)
    for axis in axes:
        axis.spines[['top','right']].set_visible(False);axis.tick_params(axis='x',labelsize=9)
        axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    folder=ROOT/'report';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'large_workload.png',dpi=180);plt.close(fig)
    data=ROOT/'web/data';data.mkdir(parents=True,exist_ok=True)
    for src,dst in [('results/registry.json','registry.json'),('results/runtime.json','runtime.json'),
                    ('results/validation.json','validation.json'),('results/counterexample/summary.json','counterexample.json'),
                    ('results/benchmark/manifest.json','benchmark_manifest.json'),('results/learned_selector.json','learned_selector.json')]:
        shutil.copy2(ROOT/src,data/dst)
    write_markdown(r,rt,c,v,ml)


def write_markdown(r,rt,c,v,ml):
    large=next(x for x in r['case_comparisons'] if x['workload']=='uniform_large_holdout')
    byid={p['id']:p for p in r['profiles']}
    custom=byid[large['custom']];library=byid[large['best_library']]
    lines=['# Graph Memory Lab — measured results','',
           'Drona Koppula Putta · 24 October 2026 · NVIDIA Munich JR2027497 work sample','',
           '## What works','',
           'C++/CUDA dual-CSR aggregation and transpose backward; PyTorch/PyG/sparse references; '+
           'resident/checkpoint/dense-activation offload; validated profile registry; specialist council '+
           'and real step-boundary executor; typed read-only experiment assistant; traceable reports.','',
           f'{v["count"]} kernel/training checks passed. The benchmark has 60 configurations and '+
           '300 measured runs, each with 3 warm-up and 7 recorded steps. No numerical-gate rejection occurred.','',
           '## Main measured comparison','',
           f'On the held-out uniform workload (16,384 nodes, 393,216 directed edges, width 96, depth 6), '+
           f'custom CSR resident training took {custom["step_ms"]:.2f} ms versus {library["step_ms"]:.2f} ms '+
           f'for the fastest tested resident library path ({library["backend"]}): {large["speedup_vs_best_library"]:.2f}x latency ratio. '+
           f'Peak tensor allocation was {custom["peak_allocated_bytes"]/2**20:.1f} versus '+
           f'{library["peak_allocated_bytes"]/2**20:.1f} MiB ({100*large["allocated_reduction_vs_best_library"]:.1f}% lower).','',
           'These are full repeated-model training steps, not an isolated-kernel speedup. '+
           'Compilation is excluded; graph conversion, cold step and raw GPU component timings are recorded separately.','',
           '| Workload | Best tested resident library | Library ms | Custom ms | Library/custom |',
           '|---|---|---:|---:|---:|']
    for row in r['case_comparisons']:
        lines.append(f'| {row["workload"]} | {row["best_library"].split("/")[1]} | '+
                     f'{row["library_step_ms"]:.2f} | {row["custom_step_ms"]:.2f} | {row["speedup_vs_best_library"]:.2f}x |')
    lines += ['', '## The interaction finding','',
      'The offline study evaluates allocated-memory caps between observed configuration peaks. '+
      'A is selected by resident latency; B is selected on the original index-add path under the cap; '+
      'their composition is compared with retuning B after A and exhaustive joint selection. '+
      'All methods draw from the same fully measured registry; this is not zero-cost online tuning.','',
      f'Fresh counterexample study status: **{c["status"]}**. '+
      'See results/counterexample/summary.json for the selected pair, all attempts and bounded size reductions.','']
    if 'confirmation' in c:
        cf=c['confirmation'];cert=c['exploratory_certificate']
        lines += [f'Exploratory case: {cert["workload"]}, cap {cert["budget_bytes"]/2**20:.1f} MiB. '+
          f'Fresh paired run ratio: {cf.get("median_ratio",float("nan")):.2f}x; 95% bootstrap interval '+
          f'{cf.get("ratio_ci95",[0,0])[0]:.2f}–{cf.get("ratio_ci95",[0,0])[1]:.2f}. '+
          f'Both fresh configurations fit the original cap: {cf.get("fits_original_budget")}.','']
    ties=sum(x['sequential_equals_joint'] for x in r['budget_comparisons'])
    lines += [f'**Negative result:** sequential retuning matched exhaustive joint choice in {ties}/{len(r["budget_comparisons"])} '+
              'evaluated budget cases. The evidence supports retuning after a kernel change, not requiring a larger search.','',
              '## Automatic controller execution','',
              '| Controller | Trained / 10 | Pauses | Switches | Median decision ms | Observed cap violations |',
              '|---|---:|---:|---:|---:|---:|']
    for t in rt['trials']:
        lines.append(f'| {t["strategy"]} | {t["trained_steps"]} | {t["pauses"]} | {t["switches"]} | '+
                     f'{t["decision_ms_median"]:.3f} | {t["budget_violations"]} |')
    lines += ['',rt['scope'],'',
      'All executed steps matched a shadow index-add computation for outputs, input gradients and optimizer-updated parameters. '+
      'The impossible-budget point pauses instead of running. The greedy and threshold baselines match the council’s completed-step count. '+
      'The council adds audit detail and decision overhead; this replay does not establish a council performance advantage. '+
      'Total replay timings are descriptive only: strategy order and warm state differ.','',
      '## Learned-policy experiment','',
      f'Local ridge regression used {ml["training_profiles"]} training profiles, grouped cross-validation, and three held-out regimes. '+
      f'Decision: **{ml["decision"]}**. It tied the fixed training-selected rule on held-out choice regret. '+
      'The training data has only two families with identical shape, so this is weak evidence for cost generalization. '+
      'The model is retained as an evaluated experiment, not used to control the GPU. No LLM/API was used.','',
      '## Method, boundaries and reproducibility','',
      '- Five independent sequential worker processes; random configuration order within each. No discarded timing samples.',
      '- Fixed graph seed 123, model seed 101; exact graph and source hashes in the manifest.',
      '- Peak allocated/reserved memory separately; sampled host RSS, logical offload payload, pinned allocator stats and transfer bytes retained.',
      '- Bootstrap intervals use independent-run medians, not the 35 correlated step observations.',
      '- CPU-offload keeps sparse/integer graph structure resident. It uses pinned current-stream transfers; no custom prefetch scheduler is claimed.',
      '- Custom kernel supports float32/float64, first-order gradients, fixed immutable graph structure, one GPU. No FP16/BF16 or second-order guarantee.',
      '- Native CSR graph storage and conversion are counted; sparse-library duplicate entries are coalesced with summed weights.',
      '- Setup amortization uses the full recorded graph/data/model setup, not the partial conversion_ms component. Compiler/process/optimizer startup is not fully covered. Sparse setup includes temporary CSR work that could be optimized away.',
      '- Public reference: NetworkX Zachary karate graph, unweighted/bidirectional; performance-scale workloads are synthetic mechanism tests.',
      '- No comparison against PhysicsNeMo, Transformer Engine, torch.compile/Triton, or all optimized GNN implementations. Triton and PhysicsNeMo were absent from the checked environment.',
      '- One RTX 5060 Laptop, ~8GB, driver 621.01, PyTorch 2.14.1+cu130. No data-center, game-FPS, arbitrary-app, virtual-GPU or novel-hardware result.',
      '- Actual training was captured with PyTorch/Kineto CPU+CUDA profiling. Initial Nsight launches failed; a later prebuilt trace contains custom kernels but its target was terminated after a shutdown wait. Nsight Compute counters remain permission-blocked. See results/profiler/README.md.',
      '- Evidence hash chains detect edits against a retained root; they are not signed authentication or tamper-proof storage.','',
      'Reproduction commands: see README.md and Run_Project.ps1. Raw data: results/benchmark/repeat_*.jsonl. '+
      'Report UI: python server.py, then http://127.0.0.1:8765. The read-only assistant does not run GPU jobs.','',
      '## Public technical context','',
      '- NVIDIA PhysicsNeMo: https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html',
      '- Stadler’s public GNN talk: https://events.ecmwf.int/event/460/contributions/5223/attachments/3262/5436/HPC-WS_Stadler.pdf',
      '- PyTorch checkpointing: https://pytorch.org/blog/activation-checkpointing-techniques/',
      '- MOLT prior art: https://arxiv.org/abs/2610.05748',
      '- NetworkX public graph: https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html',
      '- NetworkX BSD license: https://github.com/networkx/networkx/blob/main/LICENSE.txt','',
      'This is independent work adjacent to public engineering topics. No exact internship project, hiring-manager ownership, or proprietary NVIDIA need is established.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def pdf():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    r,rt,c,v,ml=load();byid={p['id']:p for p in r['profiles']}
    large=next(x for x in r['case_comparisons'] if x['workload']=='uniform_large_holdout')
    custom=byid[large['custom']];library=byid[large['best_library']]
    folder=ROOT/'report';folder.mkdir(exist_ok=True)
    path=folder/'Drona_Graph_Memory_Lab_Technical_Overview.pdf'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleLab',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=colors.HexColor('#132c3e'),spaceAfter=9))
    styles.add(ParagraphStyle(name='SubLab',fontName='Helvetica',fontSize=9,leading=13,textColor=colors.HexColor('#536878'),spaceAfter=12))
    styles.add(ParagraphStyle(name='BodyLab',fontName='Helvetica',fontSize=9.5,leading=13.5,spaceAfter=8))
    styles.add(ParagraphStyle(name='HeadLab',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=colors.HexColor('#14684f'),spaceBefore=9,spaceAfter=6))
    styles.add(ParagraphStyle(name='SmallLab',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=6,textColor=colors.HexColor('#425565')))
    def p(text,style='BodyLab'):return Paragraph(text,styles[style])
    def table(rows,widths):
        data=[[p(str(x),'SmallLab') for x in row] for row in rows]
        t=Table(data,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5f0eb')),('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                              ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4),
                              ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d5dee4'))]))
        return t
    story=[p('Graph Memory Lab','TitleLab'),
       p('Drona Koppula Putta | M.Sc. HPC &amp; Quantum Computing, TH Deggendorf<br/>24 October 2026 | Independent work sample for NVIDIA Munich AI Devtech, JR2027497','SubLab'),
       p('A faster graph kernel can change which activation-memory policy is worth using. This study implements the kernel, validates training semantics, and measures the combined decision.'),
       p('Implementation','HeadLab'),
       p('C++/CUDA dual-CSR gather-sum and transpose backward, compared with PyTorch index-add, PyG scatter and a sparse CSR library path. A repeated graph processor runs with resident activations, checkpointing or dense floating activation offload. Fixed sparse/integer graph structure remains on the GPU.'),
       p('Measured result','HeadLab'),
       p(f'On the held-out uniform graph (16,384 nodes; 393,216 directed edges; width 96; depth 6), custom resident training took <b>{custom["step_ms"]:.2f} ms</b> versus <b>{library["step_ms"]:.2f} ms</b> for the fastest tested resident library path: <b>{large["speedup_vs_best_library"]:.2f}x</b> latency ratio. Peak tensor allocation was {custom["peak_allocated_bytes"]/2**20:.1f} versus {library["peak_allocated_bytes"]/2**20:.1f} MiB.'),
       Image(str(folder/'large_workload.png'),width=174*mm,height=50.2*mm),
       p('Five independent-run medians per configuration; latency error bars are percentile-bootstrap intervals. Full training steps include forward, backward and SGD. Compilation is excluded; graph conversion and cold-step costs are recorded separately.','SmallLab'),
       p('Why combine kernel and memory analysis?','HeadLab')]
    if 'confirmation' in c:
        cf=c['confirmation'];cert=c['exploratory_certificate']
        story += [p(f'The selected {cert["workload"]} case used a {cert["budget_bytes"]/2**20:.1f} MiB tensor-allocation cap. Independently combining the chosen kernel with checkpointing was slower than retuning to resident activations. Fresh confirmation status: <b>{c["status"]}</b>; paired ratio {cf.get("median_ratio",0):.2f}x (95% bootstrap interval {cf.get("ratio_ci95",[0,0])[0]:.2f}-{cf.get("ratio_ci95",[0,0])[1]:.2f}).')]
    else:story += [p('No robust budget counterexample was confirmed. The experiment preserves this negative outcome.')]
    story += [p('<b>Important negative result:</b> sequential retuning matched exhaustive joint selection in all evaluated budget cases. The evidence favors retuning after a kernel change; it does not establish a need for a more complex search.'),PageBreak(),
       p('Control with evidence','TitleLab'),p('Numerical vetoes, bounded execution, and limits that remain visible','SubLab'),
       p('The automated specialist council','HeadLab'),
       p('Memory, performance and correctness specialists share one observation. The coordinator selects only compatible measured configurations and changes our own training workload at clean step boundaries. A hash-chained event ledger records the decision and execution. An impossible budget pauses training.'),
       table([['Controller','Trained / 10','Pauses','Median decision'],*[[t['strategy'],str(t['trained_steps']),str(t['pauses']),f'{t["decision_ms_median"]:.3f} ms'] for t in rt['trials']]],[69*mm,31*mm,24*mm,50*mm]),
       Spacer(1,8),p('All executed steps matched a shadow reference; no budget violation was observed. This is real GPU training under a logical extra-working-set budget replay plus a bounded same-process reservation, not an unrelated-game experiment. One replay per strategy makes timings descriptive only.','SmallLab'),
       p('<b>Keep the simpler rule when it works:</b> greedy and threshold baselines matched the council\'s completion count with less decision overhead. A local ridge selector tied the fixed training-selected choice on all three held-out regimes, so it was not promoted to runtime control.'),
       p('Validation and protocol','HeadLab'),
       p(f'{v["count"]} kernel/training checks cover duplicate edges, isolated nodes, finite-difference gradients, graph invariances, multi-step updates and dropout RNG parity. The benchmark contains 60 configurations and 300 measured runs: five sequential independent workers, randomized order, three warm-up and seven measured steps. All configuration gates passed; no timing samples were discarded.'),
       p('A bounded evidence assistant compares recorded configurations, explains measurements and proposes typed experiments. It rejects unknown graphs, stale evidence, unsupported precision and edits to validation status. These agents are deterministic programs; no LLM/API or model council consumes GPU memory.'),
       p('Scope and reproducibility','HeadLab'),
       p('One RTX 5060 Laptop (~8GB), driver 621.01, PyTorch 2.14.1+cu130. Generated graph families plus the small public NetworkX karate graph. Float32/float64 and first-order gradients only. Raw runs, setup/conversion costs, allocated/reserved memory, host payload/RSS, source hashes, controller traces and negative findings are retained.'),
       p('No benchmark against PhysicsNeMo, Transformer Engine or torch.compile/Triton; no claim of best-possible performance, new hardware, data-center scaling, production savings or universal formal verification. Architecture simulation is a documented extension, not an implemented result. Local report access is tested separately from public deployment.','SmallLab'),
       p('Public technical context','HeadLab'),
       p('PhysicsNeMo already combines graph optimizations with checkpoint/offload; MOLT and earlier systems study GPU memory sharing. The contribution here is a reproducible kernel/memory interaction study and validation/control workflow, not first-of-kind offloading.','SmallLab'),
       p('<link href="https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html">NVIDIA PhysicsNeMo MeshGraphNet</link> | <link href="https://arxiv.org/abs/2610.05748">MOLT (2026)</link> | <link href="https://pytorch.org/blog/activation-checkpointing-techniques/">PyTorch checkpointing</link><br/>Portfolio: <link href="https://kpdronaportfolio.netlify.app/">kpdronaportfolio.netlify.app</link> | koppulaputtadrona126@gmail.com','SmallLab')]
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#657887'))
        canvas.drawString(18*mm,12*mm,'Drona Koppula Putta | Independent engineering evidence')
        canvas.drawRightString(192*mm,12*mm,f'{doc.page} / 2')
    doc=SimpleDocTemplate(str(path),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,
                          topMargin=16*mm,bottomMargin=19*mm,title='Graph Memory Lab — Technical Overview',author='Drona Koppula Putta')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('charts','pdf'));args=parser.parse_args()
    charts() if args.mode=='charts' else pdf()
