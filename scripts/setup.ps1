# DocuMind AI - one-time / idempotent environment setup.
# Creates .venv with Python 3.12, installs dependencies only when requirements change,
# and creates .env from .env.example if missing. No administrator rights required.
[CmdletBinding()]
param([switch]$Dev)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

function Find-Python {
    # Prefer the Windows launcher with an explicit 3.12, then any python on PATH that is >= 3.12.
    $candidates = @(@("py", "-3.12"), @("python"), @("python3"))
    foreach ($c in $candidates) {
        $exe = $c[0]
        $extra = @()
        if ($c.Length -gt 1) { $extra = @($c[1..($c.Length - 1)]) }
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $ver = & $exe @extra -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        } catch { continue }
        if ($LASTEXITCODE -eq 0 -and $ver -and ([version]"$ver" -ge [version]"3.12")) {
            return , (@($exe) + $extra)
        }
    }
    return $null
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    $py = Find-Python
    if (-not $py) {
        Write-Host "[DocuMind] Python 3.12+ was not found." -ForegroundColor Red
        Write-Host "           Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')."
        exit 1
    }
    Write-Host "[DocuMind] Creating virtual environment with: $($py -join ' ')"
    $pyArgs = @()
    if ($py.Length -gt 1) { $pyArgs = $py[1..($py.Length - 1)] }
    & $py[0] @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "[DocuMind] Failed to create .venv" -ForegroundColor Red; exit 1 }
}

$ReqFile = if ($Dev) { "requirements-dev.txt" } else { "requirements.txt" }
$Stamp = Join-Path $Root ".venv\.documind-$($ReqFile).sha256"
$Hash = (Get-FileHash -LiteralPath (Join-Path $Root $ReqFile) -Algorithm SHA256).Hash
if (-not (Test-Path -LiteralPath $Stamp) -or (Get-Content -LiteralPath $Stamp -Raw).Trim() -ne $Hash) {
    Write-Host "[DocuMind] Installing dependencies from $ReqFile (first run can take a few minutes)..."
    & $VenvPython -m pip install --disable-pip-version-check -q --upgrade pip
    & $VenvPython -m pip install --disable-pip-version-check -q -r $ReqFile
    if ($LASTEXITCODE -ne 0) { Write-Host "[DocuMind] Dependency installation failed." -ForegroundColor Red; exit 1 }
    Set-Content -LiteralPath $Stamp -Value $Hash -Encoding ascii
} else {
    Write-Host "[DocuMind] Dependencies are up to date."
}

$EnvFile = Join-Path $Root ".env"
if (-not (Test-Path -LiteralPath $EnvFile)) {
    Copy-Item -LiteralPath (Join-Path $Root ".env.example") -Destination $EnvFile
    Write-Host "[DocuMind] Created .env from .env.example. Add your GEMINI_API_KEY to enable answers." -ForegroundColor Yellow
}

& $VenvPython -c "import sys; sys.path.insert(0, 'src'); from documind.config import load_settings; s = load_settings(); print('[DocuMind] Configuration OK. Gemini key:', 'set' if s.has_api_key else 'NOT set (offline retrieval-only mode)')"
if ($LASTEXITCODE -ne 0) { Write-Host "[DocuMind] Configuration check failed - review your .env values." -ForegroundColor Red; exit 1 }
exit 0
