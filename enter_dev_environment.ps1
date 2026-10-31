# Load MSVC and CUDA only for this PowerShell process. No global PATH edits.
$labVsRoot = & 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe' -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $labVsRoot) { throw 'MSVC C++ build tools were not found.' }
Import-Module (Join-Path $labVsRoot 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $labVsRoot -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
if (-not (Test-Path -LiteralPath "$env:CUDA_HOME\bin\nvcc.exe")) { throw 'CUDA 13.0 compiler is not installed.' }
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$PSScriptRoot\.venv\Scripts;$env:CUDA_HOME\bin;$env:PATH"
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:MAX_JOBS = '1'
$env:DISTUTILS_USE_SDK = '1'
Write-Output 'MSVC x64 / CUDA 13.0 loaded for this shell; project Python: .venv\Scripts\python.exe'
