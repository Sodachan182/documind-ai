# DocuMind AI - set up (if needed) and launch the Streamlit app.
[CmdletBinding()]
param([int]$Port = 8501, [switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

& (Join-Path $PSScriptRoot "setup.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Url = "http://localhost:$Port"
Write-Host ""
Write-Host "[DocuMind] Starting at $Url  (press Ctrl+C to stop)" -ForegroundColor Cyan
$headless = if ($NoBrowser) { "true" } else { "false" }
& $VenvPython -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port --server.headless $headless
if ($LASTEXITCODE -ne 0) {
    Write-Host "[DocuMind] Streamlit exited with an error. Is port $Port already in use? Try: scripts\start.ps1 -Port 8502" -ForegroundColor Red
    exit $LASTEXITCODE
}
