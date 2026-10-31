# Real hardware, synthetic pressure, and architecture simulation

The application-cooperation boundary is about data ownership and semantics, not
whether NVIDIA publishes every transistor or microarchitectural detail. CUDA lets
us execute kernels, manage our allocations, synchronize streams and collect
supported measurements. Moving somebody else's live data without its runtime's
cooperation can invalidate pointers or break execution. Offloading saved training
activations requires integration with autograd and the workload.

## Recommended project layers

1. Real RTX 5060 hardware: measure graph kernels, training correctness, memory,
   transfers and full-step costs. The agent controller manages our own workload.
2. Controlled synthetic pressure: a bounded competing workload or replayed
   resource traces tests adaptation. Label which observations are replayed versus
   actually sampled. A synthetic graph on a real GPU is still hardware execution;
   a simulated device is not.
3. Optional architecture experiment: use Accel-Sim/GPGPU-Sim for a narrowly defined
   hypothetical scheduling/cache/memory policy. Expose supported simulator controls
   through typed actions; compare with default and simple rule-based policies.

Accel-Sim is a cycle-level model that can replay captured or pre-collected traces.
Its repository currently documents Hopper support and experimental Blackwell work;
do not assume it precisely models this laptop. Model coverage, calibration and
simulation runtime must be assessed before choosing a hardware experiment.
https://github.com/accel-sim/accel-sim-framework

An architecture simulator is not a drop-in virtual graphics adapter running an
arbitrary Windows game. Trace replay also does not automatically capture changed
application behavior after a policy changes. Simulation findings must be labeled,
and control mechanisms implemented only in a simulator may require driver,
runtime or hardware changes on a real GPU. Replacing code/layout later is not
guaranteed to port the result to NVIDIA hardware.

## Where the agent idea adds testable value

Separate timescales: GPU kernels and runtime handle fast execution; lightweight
specialists review step-level telemetry; a slower optional planner proposes the
next bounded experiment. Avoid a language model making per-kernel decisions.
The council should observe memory pressure, validation status and measured costs,
then propose a supported policy at a safe boundary. No access to undocumented
registers, firmware modifications, or driver patching is needed for the core.

Primary hypothesis: joint kernel/memory decisions plus specialist vetoes reduce
invalid/infeasible choices under workload changes at acceptable overhead. Compare
against a single threshold controller with identical observations and constraints.
An agent council may lose; more agents do not establish innovation or lower memory.

Possible later hardware research question: does a simulated bandwidth-aware
scheduling policy improve tail latency/fairness for a chosen pair of workloads?
That is a different experiment from activation offloading and needs its own
baselines and validation. Keep it optional until the job-relevant real-GPU core works.

## An open hardware option matching your idea more literally

[Vortex](https://github.com/vortexgpgpu/vortex) is a full-stack open-source RISC-V
GPGPU with C++ simulation, RTL simulation, and FPGA backends. Its source exposes
hardware, runtime, drivers, and configurable microarchitecture. This is a credible
platform for editing a GPU design, rather than guessing undocumented NVIDIA details.
It is a different architecture/software stack: do not assume our CUDA/PyTorch
implementation will run unchanged or that a Vortex modification ports to NVIDIA.
Its documented OS/toolchain setup is Linux-oriented. WSL is not installed on this
machine in the current check; no Linux/simulator installation has been performed.

| Platform | Useful control | What a result establishes |
|---|---|---|
| Real RTX 5060 + CUDA | Our kernels, allocations, streams, workload policies; supported telemetry | Actual behavior of the measured device/software setup |
| Accel-Sim/GPGPU-Sim | Implemented architecture/performance-model policies and trace experiments | A result within the chosen model and workload assumptions |
| Vortex SimX/RTL | Open GPU architecture and software-stack changes | Behavior of that open design/model, not an NVIDIA hardware result |

For JR2027497, prioritize CUDA/PyTorch implementation and actual performance
analysis. Keep a small architecture study as an extension, with a specific testable
mechanism. If your main interest becomes implementing a GPU scheduler or hardware
interface itself, Vortex is a stronger literal match, but it substantially changes
the project scope. Reuse controller concepts and contracts, not an assumed binary
or plug-compatible implementation across these platforms.

## Additional prior art for automatic sharing

Dynamic GPU-resource coordination is already a researched field. Salus studies
iteration scheduling and memory sharing; Gandiva uses workload introspection for
resource fit; AntMan co-designs scheduling with framework-level dynamic scaling.
These illustrate why cooperative runtime integration matters, and why "agents
automatically share GPU memory" alone is not a new contribution.

A particularly relevant recent preprint is MOLT (5 October 2026), which describes
activation-level memory reclamation between LLM serving and background tuning,
with recomputation and synchronization precautions. This is different from our
graph-kernel/memory-policy interaction study, but prevents claiming the first
automatic activation-reclamation mechanism. Its reported results are author
claims, not reproduced by this project. Our current council does not implement
its mid-step reclamation mechanism.

The narrower experiment remains: determine whether kernel-dependent memory costs,
validation vetoes, and workload changes justify a specialist policy controller over
simple thresholds and sequential retuning. Include a negative result if they do not.

- https://arxiv.org/abs/1902.04610
- https://www.usenix.org/conference/osdi18/presentation/xiao
- https://www.usenix.org/conference/osdi20/presentation/xiao
- https://arxiv.org/abs/2610.05748

Sources checked 23 October 2026:
- https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html
- https://docs.nvidia.com/deploy/nvml-api/latest/index.html
- https://docs.nvidia.com/deeplearning/transformer-engine/features/other_optimizations/cpu_offloading/cpu_offloading.html
- https://github.com/accel-sim/accel-sim-framework
- https://github.com/vortexgpgpu/vortex

This document is a design assessment, not a claim of implemented simulator support,
novel hardware control, measured speedup, or relevance to a confirmed private project.
