# HyperBone SAM2 Setup (PowerShell)
# Run from HyperBone workspace root: .\scripts\setup_sam2.ps1

$ErrorActionPreference = "Stop"

Write-Host "=" * 60
Write-Host "HyperBone SAM2 Setup"
Write-Host "=" * 60
Write-Host ""

# 1. Check PyTorch
Write-Host "[1/4] Checking PyTorch..."
$torchCheck = python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch not installed." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install PyTorch first (choose your CUDA version):"
    Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    Write-Host "  # or for CUDA 11.8:"
    Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    Write-Host ""
    Write-Host "After installing PyTorch, re-run this script."
    exit 1
}
$lines = $torchCheck -split "`n"
Write-Host "  PyTorch version: $($lines[0])"
Write-Host "  CUDA available:  $($lines[1])"
Write-Host ""

# 2. Install SAM2
Write-Host "[2/4] Installing SAM2..."
$sam2Check = python -c "from sam2.build_sam import build_sam2; print('ok')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  SAM2 already installed." -ForegroundColor Green
} else {
    Write-Host "  Installing SAM2 from pip..."
    pip install sam2
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  pip install failed. Trying from source..." -ForegroundColor Yellow
        if (Test-Path "sam2_repo") {
            Remove-Item -Recurse -Force "sam2_repo"
        }
        git clone https://github.com/facebookresearch/sam2.git sam2_repo
        Push-Location sam2_repo
        pip install -e .
        Pop-Location
    }
    # Verify
    python -c "from sam2.build_sam import build_sam2; print('SAM2 import OK')"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: SAM2 installation failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "  SAM2 installed successfully." -ForegroundColor Green
}
Write-Host ""

# 3. Download checkpoint
Write-Host "[3/4] Checking checkpoint..."
if (-not (Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints" | Out-Null
}

$ckptPath = "checkpoints/sam2.1_hiera_tiny.pt"
if (Test-Path $ckptPath) {
    $size = (Get-Item $ckptPath).Length / 1MB
    Write-Host "  Checkpoint exists: $ckptPath ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
} else {
    Write-Host "  Downloading sam2.1_hiera_tiny.pt..."
    $url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    Invoke-WebRequest -Uri $url -OutFile $ckptPath
    if ($LASTEXITCODE -ne 0 -and -not (Test-Path $ckptPath)) {
        # Fallback to curl
        curl -L $url -o $ckptPath
    }
    if (Test-Path $ckptPath) {
        $size = (Get-Item $ckptPath).Length / 1MB
        Write-Host "  Downloaded: $ckptPath ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Download failed." -ForegroundColor Red
        Write-Host "  Manual download: $url"
        exit 1
    }
}
Write-Host ""

# 4. Verify config
Write-Host "[4/4] Locating model config..."
$cfgPath = python -c "import sam2, os; p = os.path.join(os.path.dirname(sam2.__file__), '..', 'sam2', 'configs', 'sam2.1', 'sam2.1_hiera_t.yaml'); print(p if os.path.exists(p) else '')" 2>&1
if (-not $cfgPath -or $cfgPath -eq "") {
    # Try alternate paths
    $cfgPath = python -c "import sam2, os; base = os.path.dirname(sam2.__file__); candidates = [os.path.join(base, 'configs', 'sam2.1', 'sam2.1_hiera_t.yaml'), os.path.join(base, '..', 'configs', 'sam2.1', 'sam2.1_hiera_t.yaml')]; found = [c for c in candidates if os.path.exists(c)]; print(found[0] if found else 'NOT_FOUND')" 2>&1
}
Write-Host "  Config path: $cfgPath"
if ($cfgPath -eq "NOT_FOUND" -or $cfgPath -eq "") {
    Write-Host "  WARNING: Config file not found. SAM2 may resolve configs internally." -ForegroundColor Yellow
    Write-Host "  Try using config name: sam2.1_hiera_t"
}
Write-Host ""

# Done
Write-Host "=" * 60
Write-Host "Setup complete. Run diagnostic:" -ForegroundColor Green
Write-Host "  python scripts/check_sam2.py"
Write-Host ""
Write-Host "Then test:"
Write-Host "  python scripts/test_sam2_frame.py --image outputs/test_run/frames/frame_000024.png --out outputs/sam2_frame_test --checkpoint checkpoints/sam2.1_hiera_tiny.pt --device cuda"
