<#
.SYNOPSIS
    Build the Quantiva Desktop Backend with PyInstaller (--onedir, CPU-only).

.DESCRIPTION
    1. Creates / reuses a build-only virtual environment under backend/.venv-build
    2. Installs CPU-only PyTorch (saves > 2 GB vs full CUDA bundle)
    3. Installs project dependencies + PyInstaller
    4. Runs PyInstaller with backend/quantiva-backend.spec
    5. Smoke-tests the built executable on an ephemeral port
    6. Reports pass / fail

.PARAMETER SkipVenv
    Reuse an existing .venv-build without reinstalling packages.

.PARAMETER SkipSmokeTest
    Skip the post-build startup/health/shutdown verification.

.PARAMETER Console
    Build with console=True (shows terminal window; useful for debugging).

.PARAMETER TestPort
    Port for the smoke test health check (default: 0 = auto-assigned).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_backend.ps1
    powershell -ExecutionPolicy Bypass -File scripts\build_backend.ps1 -Console -SkipSmokeTest
#>
[CmdletBinding()]
param(
    [switch]$SkipVenv,
    [switch]$SkipSmokeTest,
    [switch]$Console,
    [int]$TestPort = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Paths ─────────────────────────────────────────────────────────────────────

$RepoRoot   = (Resolve-Path "$PSScriptRoot\..").Path
$BackendDir = Join-Path $RepoRoot "backend"
$VenvDir    = Join-Path $BackendDir ".venv-build"
$SpecFile   = Join-Path $BackendDir "quantiva-backend.spec"
$DistDir    = Join-Path $BackendDir "dist" "quantiva-backend"
$ExePath    = Join-Path $DistDir "quantiva-backend.exe"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║          Quantiva Desktop Backend — PyInstaller Build           ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 1. Validate prerequisites ────────────────────────────────────────────────

Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

$PythonExe = Get-Command python -ErrorAction SilentlyContinue |
             Select-Object -ExpandProperty Source
if (-not $PythonExe) {
    Write-Error "Python is not on PATH. Install Python 3.11+ and try again."
    exit 1
}

$PyVersion = & python --version 2>&1
Write-Host "       Python: $PyVersion ($PythonExe)"

if (-not (Test-Path $SpecFile)) {
    Write-Error "Spec file not found: $SpecFile"
    exit 1
}
Write-Host "       Spec:   $SpecFile"
Write-Host ""

# ── 2. Create / activate build venv ──────────────────────────────────────────

Write-Host "[2/5] Preparing build virtual environment..." -ForegroundColor Yellow

if ($SkipVenv -and (Test-Path (Join-Path $VenvDir "Scripts" "python.exe"))) {
    Write-Host "       Reusing existing venv: $VenvDir"
} else {
    if (Test-Path $VenvDir) {
        Write-Host "       Removing stale venv..."
        Remove-Item -Recurse -Force $VenvDir
    }
    Write-Host "       Creating venv..."
    & python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create venv."; exit 1 }
}

$VenvPython = Join-Path $VenvDir "Scripts" "python.exe"
$VenvPip    = Join-Path $VenvDir "Scripts" "pip.exe"

# Upgrade pip + install build tooling
Write-Host "       Upgrading pip..."
& $VenvPython -m pip install --quiet --upgrade pip setuptools wheel 2>&1 | Out-Null

# ── 3. Install CPU-only PyTorch + project deps ──────────────────────────────

Write-Host "[3/5] Installing dependencies (CPU-only PyTorch)..." -ForegroundColor Yellow

# Install CPU-only PyTorch from the official CPU index.
# This saves > 2 GB compared to the full CUDA-bundled wheel.
Write-Host "       Installing CPU-only PyTorch..."
& $VenvPip install --quiet `
    torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cpu 2>&1 |
    Select-String -Pattern "ERROR|error" -SimpleMatch | ForEach-Object { Write-Warning $_ }

# Install project requirements.
$RequirementsFile = Join-Path $BackendDir "requirements.txt"
if (Test-Path $RequirementsFile) {
    Write-Host "       Installing requirements.txt..."
    & $VenvPip install --quiet -r $RequirementsFile 2>&1 |
        Select-String -Pattern "ERROR|error" -SimpleMatch | ForEach-Object { Write-Warning $_ }
} else {
    Write-Warning "       No requirements.txt found — skipping."
}

# Install PyInstaller itself.
Write-Host "       Installing PyInstaller..."
& $VenvPip install --quiet pyinstaller 2>&1 | Out-Null

Write-Host "       All dependencies installed."
Write-Host ""

# ── 4. Run PyInstaller ───────────────────────────────────────────────────────

Write-Host "[4/5] Running PyInstaller..." -ForegroundColor Yellow

# Clean previous build artifacts for reproducibility.
$BuildDir = Join-Path $BackendDir "build"
if (Test-Path (Join-Path $BuildDir "quantiva-backend")) {
    Remove-Item -Recurse -Force (Join-Path $BuildDir "quantiva-backend")
}
if (Test-Path $DistDir) {
    Remove-Item -Recurse -Force $DistDir
}

# Set environment for the spec file.
$env:QUANTIVA_BUILD_CONSOLE = if ($Console) { "1" } else { "" }

# Run PyInstaller from the backend directory so relative paths in the spec resolve.
Push-Location $BackendDir
try {
    $PyInstallerExe = Join-Path $VenvDir "Scripts" "pyinstaller.exe"
    & $PyInstallerExe --noconfirm --clean $SpecFile 2>&1 | ForEach-Object {
        $line = $_
        if ($line -match "ERROR|CRITICAL|error:") {
            Write-Host "  $line" -ForegroundColor Red
        } elseif ($line -match "WARNING|warn") {
            Write-Host "  $line" -ForegroundColor DarkYellow
        } else {
            Write-Host "  $line" -ForegroundColor DarkGray
        }
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "PyInstaller failed with exit code $LASTEXITCODE."
        exit 1
    }
} finally {
    Pop-Location
}

# Verify output exists.
if (-not (Test-Path $ExePath)) {
    Write-Error "Build succeeded but executable not found at: $ExePath"
    exit 1
}

$ExeSize = (Get-Item $ExePath).Length / 1MB
$DirSize = (Get-ChildItem $DistDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ""
Write-Host "       Build complete!" -ForegroundColor Green
Write-Host "       Executable:  $ExePath"
Write-Host "       EXE size:    $([math]::Round($ExeSize, 1)) MB"
Write-Host "       Bundle size: $([math]::Round($DirSize, 1)) MB"
Write-Host ""

# ── 5. Smoke test ────────────────────────────────────────────────────────────

if ($SkipSmokeTest) {
    Write-Host "[5/5] Smoke test skipped (-SkipSmokeTest)." -ForegroundColor DarkYellow
    Write-Host ""
    Write-Host "BUILD PASSED" -ForegroundColor Green
    exit 0
}

Write-Host "[5/5] Running smoke test..." -ForegroundColor Yellow

# Pick an ephemeral port if TestPort is 0.
if ($TestPort -eq 0) {
    $Listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $Listener.Start()
    $TestPort = $Listener.LocalEndpoint.Port
    $Listener.Stop()
}

$SmokeToken = "smoke-test-token-$(Get-Random)"
$HealthUrl  = "http://127.0.0.1:$TestPort/api/v1/system/health"

Write-Host "       Port:  $TestPort"
Write-Host "       URL:   $HealthUrl"

# Start the backend process.
$BackendProcess = Start-Process -FilePath $ExePath `
    -ArgumentList "--port", $TestPort, "--token", $SmokeToken `
    -PassThru `
    -NoNewWindow `
    -RedirectStandardOutput (Join-Path $BackendDir "build" "smoke-stdout.log") `
    -RedirectStandardError  (Join-Path $BackendDir "build" "smoke-stderr.log")

Write-Host "       PID:   $($BackendProcess.Id)"
Write-Host "       Waiting for health endpoint..."

# Poll the health endpoint.
$MaxWait    = 90        # seconds (transformers model load can take a while)
$PollMs     = 2000      # milliseconds between polls
$Elapsed    = 0
$Healthy    = $false

while ($Elapsed -lt $MaxWait) {
    Start-Sleep -Milliseconds $PollMs
    $Elapsed += ($PollMs / 1000)

    # Check if process died.
    if ($BackendProcess.HasExited) {
        $ExitCode = $BackendProcess.ExitCode
        Write-Host ""
        Write-Error "Backend process exited prematurely with code $ExitCode."
        $StdErr = Get-Content (Join-Path $BackendDir "build" "smoke-stderr.log") -ErrorAction SilentlyContinue
        if ($StdErr) {
            Write-Host "--- stderr ---" -ForegroundColor Red
            $StdErr | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        }
        exit 1
    }

    try {
        $Response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        if ($Response.StatusCode -eq 200) {
            $Healthy = $true
            break
        }
    } catch {
        # Connection refused or timeout — keep polling.
        Write-Host "       ... waiting ($Elapsed/$MaxWait s)" -ForegroundColor DarkGray
    }
}

# Clean shutdown.
Write-Host "       Shutting down backend (PID $($BackendProcess.Id))..."
try {
    # Send Ctrl+C via taskkill /PID — graceful on Windows.
    # Uvicorn handles SIGINT → clean shutdown.
    Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
} catch {
    # Already exited — fine.
}

# Wait for process to fully exit (max 10 s).
$BackendProcess | Wait-Process -Timeout 10 -ErrorAction SilentlyContinue

if ($Healthy) {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║   SMOKE TEST PASSED — 200 OK on /api/v1/system/health          ║" -ForegroundColor Green
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "       Startup time: ~$Elapsed s"
    Write-Host "       Executable:   $ExePath"
    Write-Host ""
    Write-Host "BUILD PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║   SMOKE TEST FAILED — no 200 within $MaxWait seconds              ║" -ForegroundColor Red
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Red
    Write-Host ""
    $StdOut = Get-Content (Join-Path $BackendDir "build" "smoke-stdout.log") -ErrorAction SilentlyContinue -Tail 30
    $StdErr = Get-Content (Join-Path $BackendDir "build" "smoke-stderr.log") -ErrorAction SilentlyContinue -Tail 30
    if ($StdOut) {
        Write-Host "--- stdout (last 30 lines) ---" -ForegroundColor Yellow
        $StdOut | ForEach-Object { Write-Host "  $_" }
    }
    if ($StdErr) {
        Write-Host "--- stderr (last 30 lines) ---" -ForegroundColor Red
        $StdErr | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    }
    Write-Host ""
    Write-Host "BUILD FAILED" -ForegroundColor Red
    exit 1
}
