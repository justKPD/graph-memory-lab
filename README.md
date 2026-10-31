# Graph Memory Lab

Independent C++/CUDA and PyTorch engineering work sample for NVIDIA Munich
AI Developer Technology Intern, JR2027497. Drona Koppula Putta, 24 October 2026.

**Current state:** implemented and tested on an RTX 5060 Laptop GPU. This is a
bounded single-device research MVP. Public demo: https://graph-memory-lab.vercel.app . Public access has been verified; the source is independent. No NVIDIA affiliation or private project is implied.

## What it does

- Native dual-CSR sum aggregation with transpose backward; duplicate-edge semantics.
- Equivalent index-add, PyG scatter and sparse-library references.
- Resident activations, checkpointing with RNG preservation, and selective dense
  activation offload. Fixed sparse/integer graph resources remain resident.
- Independent-run benchmark registry, numerical gates, allocator/host measurements,
  source snapshots, fresh counterexample confirmation and bounded size reduction.
- Memory/performance/correctness specialists, real step-boundary execution,
  pause/fail-closed behavior, a hash-chained evidence ledger, and strong simple baselines.
- Read-only evidence tools plus command chat and a separate bounded local GPU replay controller.
- A local learned cost model evaluated on held-out cases and rejected for control
  when it did not improve the simple rule. No LLM or external API is used.

## Measured evidence

The main study has **60 configurations × 5 independent workers = 300 measured
runs**, each with 3 warm-up and 7 measured steps. All numerical gates passed.
On the held-out uniform graph, custom resident training was
7.70 ms versus 19.18 ms for the
fastest tested resident library path (2.49x ratio).
This is a local repeated-model result, not a general speedup claim.

Fresh counterexample status: CONFIRMED; ratio
2.14x. Sequential retuning matched exhaustive joint
selection in every evaluated budget case. The council and simpler adaptive
controllers completed the same number of replay steps; the council added decision
overhead and audit detail. Negative findings remain in RESULTS.md.

## Open the working report

From this source workspace:

```powershell
.\.venv\Scripts\python.exe server.py
```

Then open http://127.0.0.1:8765 . The server binds loopback only. It does not expose
your source folder or mutate the existing study evidence. The separate local chat controller
can launch one fixed, bounded replay with a fresh output directory, timeout, status and
cancellation. It requires an exact loopback Origin. The web/ folder is also
a static deployment bundle; its assistant works from the saved snapshot without
a backend. The public chat cannot launch GPU jobs. A static host needs HTTP, not file://.
See SCOPE_AUDIT.md for the full A+B implementation audit and deferred research extensions.

The delivered Run_Project.ps1 can locate the installed workspace environment when
launched from Downloads. The .venv itself is not duplicated into the source ZIP.

## Validate and reproduce

```powershell
. .\enter_dev_environment.ps1
.\.venv\Scripts\python.exe -m unittest test_council test_evidence_agents test_assistant -v
.\.venv\Scripts\python.exe -m gml.validate
.\.venv\Scripts\python.exe verify_additional_cuda.py
.\.venv\Scripts\python.exe verify_integrity.py
```

Use a **fresh output directory** for a new performance study:

```powershell
.\.venv\Scripts\python.exe -m gml.benchmark suite --repeats 5 --output results/reproduction
.\.venv\Scripts\python.exe -m gml.analysis --input results/reproduction --output results/reproduction_registry.json
```

Replaying the controller requires a fresh ledger filename:

```powershell
.\.venv\Scripts\python.exe -m gml.runtime --output results/runtime_reproduction.json
```

The CLI refuses to overwrite existing benchmark runs or a controller ledger.
Run_Project.ps1 offers individual stages; All is intended for a fresh source-only
checkout, not the delivered evidence directory. The GitHub CPU workflow is
included but has not run remotely. GPU validation requires a supported local GPU.

## Environment

Python 3.13.15; torch 2.14.1+cu130; PyG 2.8.1; CUDA Toolkit 13.0/nvcc 13.0.88;
MSVC 19.44; Ninja; RTX 5060 Laptop sm_120; driver 621.01. SETUP_VERIFIED.md records
the native setup and smoke checks. requirements-lock.txt pins Python packages.

For a new environment, install the native prerequisites separately, create .venv,
install torch==2.14.1+cu130 from https://download.pytorch.org/whl/cu130, then use
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130 .
The supplied development-shell script is configured for the observed Windows host;
adjust paths for another machine. Compilation/SDK setup is not supplied by pip.

PyTorch profiler captured real project kernels after Nsight Systems project-launch capture
failed. A prebuilt-extension Nsight trace also contains the custom kernels, but its target
was terminated after a shutdown wait. Logs are retained. Nsight Compute hardware counters
remain blocked by ERR_NVGPUCTRPERM; no permission setting was changed.

## Files worth reading

- RESULTS.md: methods, all main comparisons, negative findings and limits.
- report/Drona_Graph_Memory_Lab_Technical_Overview.pdf: two-page technical brief.
- gml/: kernel wrapper, graph/model, measurements, analysis, specialists, executor,
  evidence tools, learned experiment and counterexample study.
- csrc/: actual CUDA aggregation and C++ binding, not an emulation.
- results/benchmark/: raw independent runs and the exact measured source snapshot.
- results/runtime.json and runtime.ledger.jsonl: decisions, execution and validation.
- results/counterexample/: exploratory choice, fresh confirmation and all reductions.
- results/profiler/: real graph-training trace and kernel summary.
- web/: report viewer and static publishing bundle.
- OUTREACH.md: recipient-specific drafts based on completed evidence; not sent.
- INTERVIEW_DEFENSE.md and ENGINEERING_NOTES.md: decisions, tradeoffs and failures.

## What is deliberately not claimed

No universal memory manager, game-FPS improvement, control of unrelated app
allocations, production NVIDIA saving, new GPU architecture, hardware simulator
result, formal proof, FP16/BF16/second-order guarantee, or comparison against all
optimized GNN frameworks. PhysicsNeMo already combines graph optimizations with
checkpoint/offload. See the prior-art references in RESULTS.md and
Hardware_Control_and_Simulation.md.

Performance-scale graphs are generated mechanism tests; the public karate graph
is a tiny unweighted/bidirectional reference, not a scientific-accuracy benchmark.
Allocated, reserved, device-wide and pinned-host memory are distinct metrics.
The controller uses a logical extra-working-set budget and a bounded reservation
in its own process; it does not identify or manipulate a game.
