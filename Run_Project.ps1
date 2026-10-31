param([ValidateSet('Validate','Benchmark','Analyze','Controller','Counterexample','Report','Serve','All')][string]$Stage='Validate')
$ErrorActionPreference='Stop'
$labRoot=$PSScriptRoot
if (-not (Test-Path -LiteralPath "$labRoot\.venv\Scripts\python.exe")) {
    $labRoot='C:\Users\Drona\Documents\ChatGPT\applications\graph-memory-lab'
}
Set-Location -LiteralPath $labRoot
. '.\enter_dev_environment.ps1'
function Invoke-LabPython([string[]]$Arguments) {
    & '.\.venv\Scripts\python.exe' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Lab command failed: $Arguments" }
}
if ($Stage -in @('Validate','All')) {
    Invoke-LabPython @('-m','unittest','test_council','test_evidence_agents','test_assistant','-v')
    Invoke-LabPython @('-m','gml.validate')
}
if ($Stage -in @('Benchmark','All')) { Invoke-LabPython @('-m','gml.benchmark','suite','--repeats','5','--output','results/benchmark') }
if ($Stage -in @('Analyze','All')) { Invoke-LabPython @('-m','gml.analysis'); Invoke-LabPython @('-m','gml.learned') }
if ($Stage -in @('Controller','All')) {
    if (Test-Path -LiteralPath '.\results\runtime.ledger.jsonl') { throw 'Preserve existing ledger. Use a new --output path with python -m gml.runtime to rerun.' }
    Invoke-LabPython @('-m','gml.runtime');Invoke-LabPython @('-m','gml.calibrate')
}
if ($Stage -in @('Counterexample','All')) { Invoke-LabPython @('-m','gml.counterexample','run') }
if ($Stage -in @('Report','All')) { Invoke-LabPython @('build_report.py','charts');Invoke-LabPython @('build_report.py','pdf') }
if ($Stage -eq 'Serve') { Invoke-LabPython @('server.py') }
