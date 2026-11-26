$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$DistDir = Join-Path $Root "dist"
$VenvDir = Join-Path $Root ".venv"
$PyInstallDir = Join-Path $Root "python_runtime"
$PyInstallerExe = Join-Path $Root "python-3.12.7-amd64.exe"

function Find-SystemPython {
    # 1. Try py launcher (prefer 3.12, then any 3.x)
    foreach ($ver in @("-3.12", "-3.11", "-3.13", "-3")) {
        try {
            $p = & py $ver -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p -and (Test-Path $p.Trim())) {
                return $p.Trim()
            }
        } catch {}
    }
    # 2. Try standard per-user / system install folders
    $roots = @(
        "$env:LocalAppData\Programs\Python\Python312",
        "$env:LocalAppData\Programs\Python\Python311",
        "$env:LocalAppData\Programs\Python\Python313",
        "$env:ProgramFiles\Python312",
        "$env:ProgramFiles\Python311",
        "C:\Python312"
    )
    foreach ($r in $roots) {
        $cand = Join-Path $r "python.exe"
        if (Test-Path $cand) { return $cand }
    }
    # 3. Try local python_runtime
    $localCand = Join-Path $PyInstallDir "python.exe"
    if (Test-Path $localCand) { return $localCand }
    return $null
}

Write-Host "=== 1. Locating Python 3 ===" -ForegroundColor Cyan
$SysPy = Find-SystemPython

if (-not $SysPy) {
    Write-Host "Python not found on PC. Installing local Python 3.12 runtime..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" -OutFile $PyInstallerExe
    Start-Process -FilePath $PyInstallerExe -ArgumentList "/quiet", "InstallAllUsers=0", "TargetDir=`"$PyInstallDir`"", "PrependPath=0", "Include_test=0", "Include_doc=0" -Wait
    Remove-Item $PyInstallerExe -Force -ErrorAction SilentlyContinue
    $SysPy = Find-SystemPython
}

if (-not $SysPy) {
    throw "Could not find or install Python 3. Please ensure Python 3.11+ is installed."
}
Write-Host "Using Python: $SysPy" -ForegroundColor Green

Write-Host "=== 2. Setting up build environment & installing dependencies ===" -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    & $SysPy -m venv $VenvDir
}
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install -r (Join-Path $Root "requirements.txt") pyinstaller

Write-Host "=== 3. Building standalone SnipVR.IQ.exe ===" -ForegroundColor Cyan
& $VenvPy -m PyInstaller --noconfirm --onefile --windowed --name "SnipVR.IQ" `
    --add-data "default_model.npz;." `
    --collect-binaries imageio_ffmpeg `
    (Join-Path $Root "app.py")

Write-Host ""
Write-Host "Done! Standalone app is ready at: $DistDir\SnipVR.IQ.exe" -ForegroundColor Green
