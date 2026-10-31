# Graph Memory Lab â€” development environment verified

23 October 2026. This is setup verification, not an optimization benchmark.

## Installed and reused

- Isolated Python 3.13.15 environment: C:\Users\Drona\Documents\ChatGPT\applications\graph-memory-lab\.venv
- PyTorch **2.14.1+cu130**, CUDA runtime **13.0**, native **sm_120** support.
- NVIDIA CUDA Toolkit **13.0**, nvcc **13.0.88**, compiler runtime/IR support,
  CUDA headers/libraries, profiling support and Compute Sanitizer.
- Existing Visual Studio Build Tools 2022 / MSVC **19.44.35229**, x64 SDK and CMake reused.
- Ninja **1.13.2**, PyTorch Geometric **2.8.1**, NVML Python bindings, psutil,
  NumPy/SciPy, pandas, matplotlib and pytest. Full versions: requirements-lock.txt.
- Nsight Systems **2025.3.2** and Nsight Compute **2025.3.1**.
- Existing NVIDIA driver **621.01** retained. No driver replacement, firmware,
  power-limit change, or clock modification was performed.

The initially selected CUDA components omitted compiler-support headers. The first
native compile exposed missing crt/host_config.h. The signed NVIDIA network
installer supplied CRT/NVVM support; native and PyTorch-extension compilation then
passed. Do not treat the original install-success message as the verification.

## Checks that actually passed

| Check | Evidence |
|---|---|
| Native CUDA compilation and GPU execution | cuda_smoke.cu; 32 computed outputs checked |
| PyTorch CUDA execution and activation-policy parity | runtime_verified.json; resident/checkpoint/offload outputs and gradients |
| Graph training versus CPU reference | graph_verified.json; output/input and parameter gradients/SGD update, all three GPU policies |
| PyTorch C++/CUDA extension build, import and execution | extension_verified.json and extension_build.log |
| PyTorch Geometric GCNConv CUDA forward/backward | pyg_verified.json |
| NVML read-only telemetry | telemetry_setup.json; 5 real samples |
| Nsight Systems CUDA trace | setup_trace_retry.nsys-rep and nsys_kernel_summary.csv; add_one kernel captured |
| Controller behavior | 13 synthetic CPU unit tests passed |
| Package consistency | pip check returned no broken requirements |

The graph check is tiny and deterministic. The native/extension kernels are setup
smoke tests, not the planned custom graph aggregation implementation. No memory
savings, throughput improvement, multi-step convergence, or council advantage is
established by these checks. The telemetry sample is not a controller-overhead study.
PyG emitted an upstream torch.jit.script deprecation warning; its exercised CUDA
forward/backward path passed. Broader model/API compatibility remains to be tested.

## One profiling limitation

Nsight Compute launched but hardware-counter collection returned
**ERR_NVGPUCTRPERM**. Its installation is working; counter access needs an NVIDIA
administrator setting. That setting was not changed. Nsight Systems CUDA tracing
was verified separately. The user cancelled an unfamiliar nsys prompt, then
explicitly requested a retry; the retry completed and recorded the kernel.

Counter-permission reference: https://developer.nvidia.com/ERR_NVGPUCTRPERM

## Run the checks

Use Run_Lab_Checks.ps1 in this folder. It uses the installed workspace environment
and does not launch profiling permission prompts. It does rebuild/load the small
extension as needed. Development shell: dot-source enter_dev_environment.ps1 from
the workspace project. It loads MSVC/CUDA for that shell and sets MAX_JOBS=1.
The observed PyTorch extension toolchain selected C++20; do not force older
compiler settings based on an earlier tutorial.

Pinned Python dependencies were saved after successful installation. Recreating
the environment requires the CUDA wheel index for torch's +cu130 build:
https://download.pytorch.org/whl/cu130
CUDA/MSVC/SDK/Nsight are separate native installations, not supplied by pip freeze.
PyTorch install reference: https://pytorch.org/get-started/locally/
CUDA reference: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-installation-guide-microsoft-windows/index.html

## Hardware-control direction

Read Hardware_Control_and_Simulation.md. CUDA control of our workload is available
on the actual GPU. Another application's data still requires runtime cooperation.
Accel-Sim offers an architecture performance model; Vortex exposes an open GPU
hardware/software stack. Neither implies drop-in portability to NVIDIA hardware.
No simulator, WSL/Linux environment, or virtual graphics adapter has been installed.
These are optional research extensions, not required for the native CUDA core.

## Completed project stage — 24 October 2026

The comparable graph-training study, dual-CSR CUDA forward/backward, selective
activation offload, specialist controller and read-only evidence assistant are now
implemented. See README.md, RESULTS.md and PROJECT_STATUS.json for final evidence.
The main study has 60 configurations and 300 independent-worker measurements;
38 numerical/training checks and four additional CUDA checks passed.

Project-level Nsight Systems launch capture failed (exit 1/no kernel data);
retries and logs are preserved under results/profiler. A separate actual-workload
PyTorch/Kineto CPU+CUDA trace is the project profiling artifact. The setup-only
Nsight smoke trace remains valid and separate. A later prebuilt-extension Nsight
trace contains the custom kernels, but its target was terminated after a shutdown
wait; the normally completed PyTorch trace is the primary project artifact. Hardware-counter access remains
blocked by ERR_NVGPUCTRPERM, with no permission setting changed.
