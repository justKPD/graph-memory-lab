# A+B scope audit — 24 October 2026

This audits the earlier experiment contract and latest chat request. Core implementation is distinct from proposed extensions.

| Requirement | Status | Evidence / boundary |
|---|---|---|
| A: CUDA training operator | IMPLEMENTED | csrc/aggregate.cu; gml/kernel.py — dual CSR gather and transpose backward |
| A: semantic and gradient contract | TESTED | 38 main checks plus 4 additional CUDA checks; duplicate/isolated/finite difference/RNG/multi-step/mutants |
| A: supported framework baselines | MEASURED | index_add, PyG scatter, coalesced sparse CSR; no Python-loop headline baseline |
| B: activation policies | MEASURED | resident, checkpoint and dense floating pinned-CPU offload; all four backends |
| A+B factorial comparison | MEASURED | 60 configurations, 300 main runs, five independent run units |
| Independent composition / sequential retuning / joint frontier | MEASURED | Independent composition has a fresh confirmed losing case; sequential equals joint in 15/15 offline budget cases |
| Fresh confirmation and bounded reduction | MEASURED | Five fresh paired workers; two smaller sizes; not globally minimal |
| Held-out graph regimes and public reference | MEASURED | Unseen size/family regimes; NetworkX karate unweighted/bidirectional |
| Allocated/reserved/host memory and transfers | MEASURED | Raw samples include host RSS, pinned payload and allocator stats; H2D/D2H calibration |
| Setup and conversion costs | PARTIAL | Recorded setup, first-step and conversion; amortization estimate. No end-to-end cold-process reuse sweep or full tuning-time comparison |
| Actual training traces | CAPTURED | Normally completed Kineto trace; qualified Nsight retry; hardware counters blocked |
| Specialist council and simple controller comparison | EXECUTED | Actual training transitions; fixed/threshold/greedy/council; 34 completed reference-checked steps |
| Typed plans / provenance / stale evidence | TESTED | Bounded proposals, raw hashes, measured-source snapshot, retained-root ledger checks |
| Learned selector | EVALUATED_NOT_PROMOTED | Local ridge ties fixed rule; no learned-control improvement claimed |
| Command chat / bounded local execution | ADDED | Scope, suggestions, compare/explain/propose, recorded replay; local single job with status/cancel/timeout |
| Exact Munich technical people / outreach | RESEARCHED | Stadler/Sohani public Munich evidence; individualized drafts; exact req ownership unknown |
| Two-page brief / raw evidence / UI | DELIVERED | Inspected PDF, report viewer, reproducible source ZIP |
| Transfer overlap predictor and prefetch | DEFERRED_EXTENSION | Serialized offload baseline implemented; no custom overlap, no calibrated window-based predictor |
| Variable checkpoint segments / large scientific model / TE / PhysicsNeMo | DEFERRED_EXTENSION | Current repeated processor checkpoints each block; frameworks not benchmarked; no superiority claim |
| Virtual/open GPU architecture; C vision path | DEFERRED_EXTENSION | Vortex/Accel-Sim researched, not installed. Vision C was parked while user selected A+B |
| LLM chat / arbitrary game allocations | NOT_IMPLEMENTED | Chat is a deterministic command router. No paid API or remote game/GPU allocation control |
