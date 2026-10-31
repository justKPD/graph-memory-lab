# Deploys only the static web bundle after signing into the intended existing account.
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
& npx --yes vercel@65.2.0 whoami
if ($LASTEXITCODE -ne 0) { throw 'Sign into Vercel first: npx vercel login' }
& npx --yes vercel@65.2.0 deploy .\web --prod --yes --project graph-memory-lab --create-if-missing
if ($LASTEXITCODE -ne 0) { throw 'Deployment failed; inspect the CLI output.' }
