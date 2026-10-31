# Graph Memory Lab: execution plan, 24 October 2026

Target: NVIDIA Munich JR2027497. Complete a bounded, reproducible single-GPU
engineering study, not an unvalidated universal memory manager.

1. **Kernel and semantics**: directed sum aggregation; PyTorch index_add, PyG
   scatter, native sparse-library reference, custom dual-CSR CUDA forward/backward.
   Preserve duplicate edges and isolated nodes; include format-conversion cost.
2. **Correctness**: double-precision CPU reference, gradcheck, graph invariances,
   parameter/input gradients, multiple optimizer steps and checkpoint RNG parity.
   Deliberately wrong variants must fail validation. No invalid result ranked.
3. **Measurement**: resident/checkpoint/CPU-offload training; repeated isolated
   randomized workers; synchronized full-step timing; allocated/reserved GPU memory,
   measured saved-tensor bytes, host RSS, cold/setup/conversion and raw samples.
   Calibrate workload size against real available memory; no intentional device OOM.
4. **Decision and execution**: measured profile registry, memory/performance/
   correctness specialists, controlled step-boundary transitions, pause/fail-closed,
   comparison with fixed/threshold/sequential/joint policies. No arbitrary shell tools.
5. **Experiments**: regular/skewed/held-out graph families plus public NetworkX
   Zachary karate graph; investigate interaction and rank reversals, minimize a
   found counterexample only if supported by repeat measurements. If no robust
   reversal is found, report that outcome. No invented or forced positive finding.
6. **Delivery**: local report viewer with typed experiment assistant, fresh-session
   browser checks, raw data and provenance hashes, plots, two-page PDF brief,
   truthful per-person outreach, reproducible source ZIP under Downloads/ChatGPT.

Initial declared scope: float32/float64, one device, one optimizer and repeated
message-passing model. No full-scale scientific-accuracy claim or external games.
GPU architecture simulation and language-model APIs are optional extensions, not
required for a working core. The automated specialists and experiment assistant
are deterministic programs unless a separate model integration is actually built.

Acceptance: meaningful tests pass; real GPU training and dynamic execution run;
all report numbers trace to raw files; negative results remain visible; package
can be inspected and replayed; no false claims of speedup, novelty or deployment.
Public hosting, paid resources, mail sends and application submission remain
separate external actions. Prepare a reviewable deployment bundle first.
