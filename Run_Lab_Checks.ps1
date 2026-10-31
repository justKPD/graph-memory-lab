$ErrorActionPreference = 'Stop'
$labProjectRoot = $PSScriptRoot
if (-not (Test-Path -LiteralPath "$labProjectRoot\.venv\Scripts\python.exe")) {
    $labProjectRoot = 'C:\Users\Drona\Documents\ChatGPT\applications\graph-memory-lab'
}
Set-Location -LiteralPath $labProjectRoot
. '.\enter_dev_environment.ps1'
foreach ($labScript in @('verify_runtime.py','verify_graph.py','verify_extension.py','telemetry.py')) {
    & '.\.venv\Scripts\python.exe' $labScript
    if ($LASTEXITCODE -ne 0) { throw "Check failed: $labScript" }
}
& '.\.venv\Scripts\python.exe' -m unittest -v
if ($LASTEXITCODE -ne 0) { throw 'Controller tests failed' }
