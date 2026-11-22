# DocuMind AI - run lint + offline tests (no API key needed).
# Pass -Live to also run the live Gemini tests (requires GEMINI_API_KEY).
[CmdletBinding()]
param([switch]$Live)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

& (Join-Path $PSScriptRoot "setup.ps1") -Dev
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "`n[DocuMind] Ruff lint" -ForegroundColor Cyan
& $VenvPython -m ruff check .
$lint = $LASTEXITCODE
& $VenvPython -m ruff format --check .
$fmt = $LASTEXITCODE

Write-Host "`n[DocuMind] Pytest" -ForegroundColor Cyan
if ($Live) { & $VenvPython -m pytest -m "live or not live" } else { & $VenvPython -m pytest }
$tests = $LASTEXITCODE

if ($lint -ne 0 -or $fmt -ne 0 -or $tests -ne 0) {
    Write-Host "`n[DocuMind] FAILED (ruff=$lint format=$fmt pytest=$tests)" -ForegroundColor Red
    exit 1
}
Write-Host "`n[DocuMind] All checks passed." -ForegroundColor Green
