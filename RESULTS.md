# Graph Memory Lab — measured results

Drona Koppula Putta · 24 October 2026 · NVIDIA Munich JR2027497 work sample

## What works

C++/CUDA dual-CSR aggregation and transpose backward; PyTorch/PyG/sparse references; resident/checkpoint/dense-activation offload; validated profile registry; specialist council and real step-boundary executor; typed read-only experiment assistant; traceable reports.

38 kernel/training checks passed. The benchmark has 60 configurations and 300 measured runs, each with 3 warm-up and 7 recorded steps. No numerical-gate rejection occurred.

## Main measured comparison

On the held-out uniform workload (16,384 nodes, 393,216 directed edges, width 96, depth 6), custom CSR resident training took 7.70 ms versus 19.18 ms for the fastest tested resident library path (sparse): 2.49x latency ratio. Peak tensor allocation was 190.7 versus 198.7 MiB (4.1% lower).

These are full repeated-model training steps, not an isolated-kernel speedup. Compilation is excluded; graph conversion, cold step and raw GPU component timings are recorded separately.

| Workload | Best tested resident library | Library ms | Custom ms | Library/custom |
|---|---|---:|---:|---:|
| karate_public | index_add | 8.39 | 7.12 | 1.18x |
| ring_holdout | pyg | 9.87 | 7.12 | 1.39x |
| skew_train | pyg | 9.05 | 6.33 | 1.43x |
| uniform_large_holdout | sparse | 19.18 | 7.70 | 2.49x |
| uniform_train | index_add | 8.65 | 6.38 | 1.36x |

## The interaction finding

The offline study evaluates allocated-memory caps between observed configuration peaks. A is selected by resident latency; B is selected on the original index-add path under the cap; their composition is compared with retuning B after A and exhaustive joint selection. All methods draw from the same fully measured registry; this is not zero-cost online tuning.

Fresh counterexample study status: **CONFIRMED**. See results/counterexample/summary.json for the selected pair, all attempts and bounded size reductions.

Exploratory case: ring_holdout, cap 245.8 MiB. Fresh paired run ratio: 2.14x; 95% bootstrap interval 1.89–2.24. Both fresh configurations fit the original cap: True.

**Negative result:** sequential retuning matched exhaustive joint choice in 15/15 evaluated budget cases. The evidence supports retuning after a kernel change, not requiring a larger search.

## Automatic controller execution

| Controller | Trained / 10 | Pauses | Switches | Median decision ms | Observed cap violations |
|---|---:|---:|---:|---:|---:|
| fixed | 7 | 3 | 0 | 0.018 | 0 |
| threshold | 9 | 1 | 2 | 0.024 | 0 |
| greedy_validated | 9 | 1 | 2 | 0.021 | 0 |
| council | 9 | 1 | 2 | 0.207 | 0 |

Actual GPU training, logical extra-working-set budget replay, real same-process bounded reservation; not an unrelated application or game.

All executed steps matched a shadow index-add computation for outputs, input gradients and optimizer-updated parameters. The impossible-budget point pauses instead of running. The greedy and threshold baselines match the council’s completed-step count. The council adds audit detail and decision overhead; this replay does not establish a council performance advantage. Total replay timings are descriptive only: strategy order and warm state differ.

## Learned-policy experiment

Local ridge regression used 24 training profiles, grouped cross-validation, and three held-out regimes. Decision: **RETAIN_SIMPLE_RULE**. It tied the fixed training-selected rule on held-out choice regret. The training data has only two families with identical shape, so this is weak evidence for cost generalization. The model is retained as an evaluated experiment, not used to control the GPU. No LLM/API was used.

## Method, boundaries and reproducibility

- Five independent sequential worker processes; random configuration order within each. No discarded timing samples.
- Fixed graph seed 123, model seed 101; exact graph and source hashes in the manifest.
- Peak allocated/reserved memory separately; sampled host RSS, logical offload payload, pinned allocator stats and transfer bytes retained.
- Bootstrap intervals use independent-run medians, not the 35 correlated step observations.
- CPU-offload keeps sparse/integer graph structure resident. It uses pinned current-stream transfers; no custom prefetch scheduler is claimed.
- Custom kernel supports float32/float64, first-order gradients, fixed immutable graph structure, one GPU. No FP16/BF16 or second-order guarantee.
- Native CSR graph storage and conversion are counted; sparse-library duplicate entries are coalesced with summed weights.
- Setup amortization uses the full recorded graph/data/model setup, not the partial conversion_ms component. Compiler/process/optimizer startup is not fully covered. Sparse setup includes temporary CSR work that could be optimized away.
- Public reference: NetworkX Zachary karate graph, unweighted/bidirectional; performance-scale workloads are synthetic mechanism tests.
- No comparison against PhysicsNeMo, Transformer Engine, torch.compile/Triton, or all optimized GNN implementations. Triton and PhysicsNeMo were absent from the checked environment.
- One RTX 5060 Laptop, ~8GB, driver 621.01, PyTorch 2.14.1+cu130. No data-center, game-FPS, arbitrary-app, virtual-GPU or novel-hardware result.
- Actual training was captured with PyTorch/Kineto CPU+CUDA profiling. Initial Nsight launches failed; a later prebuilt trace contains custom kernels but its target was terminated after a shutdown wait. Nsight Compute counters remain permission-blocked. See results/profiler/README.md.
- Evidence hash chains detect edits against a retained root; they are not signed authentication or tamper-proof storage.

Reproduction commands: see README.md and Run_Project.ps1. Raw data: results/benchmark/repeat_*.jsonl. Report UI: python server.py, then http://127.0.0.1:8765. The read-only assistant does not run GPU jobs.

## Public technical context

- NVIDIA PhysicsNeMo: https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html
- Stadler’s public GNN talk: https://events.ecmwf.int/event/460/contributions/5223/attachments/3262/5436/HPC-WS_Stadler.pdf
- PyTorch checkpointing: https://pytorch.org/blog/activation-checkpointing-techniques/
- MOLT prior art: https://arxiv.org/abs/2610.05748
- NetworkX public graph: https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html
- NetworkX BSD license: https://github.com/networkx/networkx/blob/main/LICENSE.txt

This is independent work adjacent to public engineering topics. No exact internship project, hiring-manager ownership, or proprietary NVIDIA need is established.
