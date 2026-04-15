<#
.SYNOPSIS
    CrispEmbed benchmark: compare against HuggingFace and fastembed-rs on Windows.

.DESCRIPTION
    Benchmarks CrispEmbed (CPU/CUDA/Vulkan) against:
    - HuggingFace sentence-transformers (PyTorch)
    - fastembed (Python ONNX)
    - fastembed-rs (Rust ONNX, if available)

.PARAMETER Model
    GGUF model path (default: auto-detect in current dir)

.PARAMETER HfModel
    HuggingFace model ID for comparison

.PARAMETER Port
    Server port (default: 8090)

.PARAMETER Threads
    Number of CPU threads (default: auto)

.PARAMETER NRuns
    Number of benchmark iterations (default: 100)

.PARAMETER SkipHF
    Skip HuggingFace comparison

.PARAMETER SkipFastembed
    Skip fastembed comparison

.EXAMPLE
    .\benchmark.ps1 -Model .\models\all-MiniLM-L6-v2.gguf -HfModel "sentence-transformers/all-MiniLM-L6-v2"
    .\benchmark.ps1 -Model .\models\arctic-embed-xs.gguf -HfModel "snowflake/snowflake-arctic-embed-xs" -NRuns 200
#>

param(
    [string]$Model = "",
    [string]$HfModel = "",
    [int]$Port = 8090,
    [int]$Threads = 0,
    [int]$NRuns = 100,
    [switch]$SkipHF,
    [switch]$SkipFastembed
)

$ErrorActionPreference = "Stop"

# Auto-detect model and binary
$BinaryPaths = @(
    "build-cuda\crispembed.exe",
    "build-cuda\Release\crispembed.exe",
    "build-cuda\bin\crispembed.exe",
    "build-vulkan\crispembed.exe",
    "build-vulkan\Release\crispembed.exe",
    "build\crispembed.exe",
    "build\Release\crispembed.exe",
    "build\bin\crispembed.exe"
)
$ServerPaths = @(
    "build-cuda\crispembed-server.exe",
    "build-cuda\Release\crispembed-server.exe",
    "build-cuda\bin\crispembed-server.exe",
    "build-vulkan\crispembed-server.exe",
    "build-vulkan\Release\crispembed-server.exe",
    "build\crispembed-server.exe",
    "build\Release\crispembed-server.exe",
    "build\bin\crispembed-server.exe"
)

$Binary = $null
$Server = $null
foreach ($p in $BinaryPaths) { if (Test-Path $p) { $Binary = $p; break } }
foreach ($p in $ServerPaths) { if (Test-Path $p) { $Server = $p; break } }

# Fallback: recursive search
if (-not $Binary) {
    $found = Get-ChildItem -Path "." -Filter "crispembed.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $Binary = $found.FullName }
}
if (-not $Server) {
    $found = Get-ChildItem -Path "." -Filter "crispembed-server.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $Server = $found.FullName }
}

if (-not $Binary) {
    Write-Host "[ERROR] No crispembed.exe found. Build first:" -ForegroundColor Red
    Write-Host "  build-windows.bat   (CPU)" -ForegroundColor Yellow
    Write-Host "  build-cuda.bat      (NVIDIA GPU)" -ForegroundColor Yellow
    Write-Host "  build-vulkan.bat    (Vulkan GPU)" -ForegroundColor Yellow
    exit 1
}

# Auto-detect model: search current dir, models/, and common cache dirs
if (-not $Model) {
    $SearchDirs = @(".", "models", "$env:USERPROFILE\.cache\crispembed")
    foreach ($dir in $SearchDirs) {
        if (Test-Path $dir) {
            $found = Get-ChildItem -Path $dir -Filter "*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) { $Model = $found.FullName; break }
        }
    }
    if (-not $Model) {
        Write-Host "[ERROR] No .gguf model found. Specify with -Model <path-to-file.gguf>" -ForegroundColor Red
        Write-Host "  Download models from: https://huggingface.co/cstr" -ForegroundColor Yellow
        exit 1
    }
}

# Validate model path exists or treat as model name
if (-not (Test-Path $Model)) {
    # Search for matching .gguf file
    $found = Get-ChildItem -Path "." -Filter "*$Model*.gguf" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        $Model = $found.FullName
    } else {
        # Not a file path — treat as model name (CrispEmbed will auto-download)
        Write-Host "[INFO] '$Model' is not a local file. CrispEmbed will auto-download from HuggingFace." -ForegroundColor Yellow
        # Model name is passed directly to crispembed -m (model_mgr handles it)
    }
}

# Map common GGUF names to HF models
$HfMap = @{
    "all-MiniLM-L6-v2" = "sentence-transformers/all-MiniLM-L6-v2"
    "gte-small" = "thenlper/gte-small"
    "arctic-embed-xs" = "snowflake/snowflake-arctic-embed-xs"
    "multilingual-e5-small" = "intfloat/multilingual-e5-small"
    "octen-0.6b" = "Octen/Octen-Embedding-0.6B"
}

if (-not $HfModel) {
    $basename = [System.IO.Path]::GetFileNameWithoutExtension($Model) -replace "-q[0-9].*", ""
    foreach ($k in $HfMap.Keys) {
        if ($basename -match [regex]::Escape($k)) { $HfModel = $HfMap[$k]; break }
    }
}

$TestText = "The quick brown fox jumps over the lazy dog near the river bank"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CrispEmbed Benchmark" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Binary:  $Binary"
Write-Host "Server:  $Server"
Write-Host "Model:   $Model"
Write-Host "HF:      $HfModel"
Write-Host "Runs:    $NRuns"
Write-Host ""

# --- CrispEmbed CLI benchmark ---
Write-Host "--- CrispEmbed CLI ---" -ForegroundColor Yellow

# Warmup (first call may download the model, so run twice)
Write-Host "  Loading model (may download on first run)..."
$null = & $Binary -m $Model "warmup" 2>$null
$warmup = & $Binary -m $Model $TestText 2>$null
if (-not $warmup) {
    Write-Host "  [ERROR] CLI produced no output. Check model path." -ForegroundColor Red
} else {
    $dim = ($warmup.Trim() -split '\s+').Count
    Write-Host "  Dimension: $dim"

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    for ($i = 0; $i -lt $NRuns; $i++) {
        $null = & $Binary -m $Model $TestText 2>$null
    }
    $sw.Stop()
    $cliMs = $sw.ElapsedMilliseconds / $NRuns
    $cliTps = $NRuns / ($sw.ElapsedMilliseconds / 1000.0)
    Write-Host ("  CLI:    {0:F1}ms/text  {1:F0} texts/s (includes model load)" -f $cliMs, $cliTps)
}

# --- CrispEmbed Server benchmark ---
if ($Server) {
    Write-Host ""
    Write-Host "--- CrispEmbed Server ---" -ForegroundColor Yellow

    $threadArg = if ($Threads -gt 0) { @("-t", "$Threads") } else { @() }
    $serverArgs = @("-m", $Model, "--port", "$Port") + $threadArg
    $serverProc = Start-Process -FilePath $Server -ArgumentList $serverArgs -PassThru -WindowStyle Hidden

    # Wait for server to be ready (CUDA init can take 5-10s)
    Write-Host "  Waiting for server to load model..."
    $ready = $false
    for ($wait = 0; $wait -lt 30; $wait++) {
        Start-Sleep -Seconds 1
        try {
            $null = Invoke-RestMethod -Uri "http://localhost:${Port}/health" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
            $ready = $true; break
        } catch { }
    }
    if (-not $ready) {
        Write-Host "  [WARN] Server may not be ready after 30s, trying anyway..." -ForegroundColor Yellow
    }

    try {
        # Warmup
        for ($i = 0; $i -lt 5; $i++) {
            try {
                $body = @{ texts = @($TestText) } | ConvertTo-Json
                $null = Invoke-RestMethod -Uri "http://localhost:${Port}/embed" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 60
            } catch { Start-Sleep -Seconds 1 }
        }

        # Benchmark
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        for ($i = 0; $i -lt $NRuns; $i++) {
            $body = @{ texts = @($TestText) } | ConvertTo-Json
            $null = Invoke-RestMethod -Uri "http://localhost:${Port}/embed" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 30
        }
        $sw.Stop()
        $srvMs = $sw.ElapsedMilliseconds / $NRuns
        $srvTps = $NRuns / ($sw.ElapsedMilliseconds / 1000.0)
        Write-Host ("  Server: {0:F1}ms/text  {1:F0} texts/s" -f $srvMs, $srvTps)

        # Batch benchmark
        $batchTexts = @($TestText) * 10
        $body = @{ texts = $batchTexts } | ConvertTo-Json
        $batchRuns = [math]::Max(10, $NRuns / 10)
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        for ($i = 0; $i -lt $batchRuns; $i++) {
            $null = Invoke-RestMethod -Uri "http://localhost:${Port}/embed" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 60
        }
        $sw.Stop()
        $batchMs = $sw.ElapsedMilliseconds / $batchRuns
        $batchTps = ($batchRuns * 10) / ($sw.ElapsedMilliseconds / 1000.0)
        Write-Host ("  Batch:  {0:F1}ms/10texts  {1:F0} texts/s" -f $batchMs, $batchTps)
    }
    finally {
        Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
    }
}

# --- HuggingFace benchmark ---
if (-not $SkipHF -and $HfModel) {
    Write-Host ""
    Write-Host "--- HuggingFace sentence-transformers ---" -ForegroundColor Yellow

    $pyScript = @"
import time
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('$HfModel', trust_remote_code=True)
    texts = ['$TestText']
    model.encode(texts * 3, normalize_embeddings=True)
    N = $NRuns
    t0 = time.perf_counter()
    for _ in range(N):
        model.encode(texts, normalize_embeddings=True)
    elapsed = time.perf_counter() - t0
    print(f'HF_SINGLE:{elapsed/N*1000:.1f}:{N/elapsed:.0f}')
    batch = texts * 10
    t0 = time.perf_counter()
    for _ in range(max(10, N//10)):
        model.encode(batch, normalize_embeddings=True, batch_size=32)
    runs = max(10, N//10)
    elapsed = time.perf_counter() - t0
    print(f'HF_BATCH:{elapsed/runs*1000:.1f}:{runs*10/elapsed:.0f}')
except Exception as e:
    print(f'HF_ERROR:{e}')
"@
    $result = python -c $pyScript 2>$null
    foreach ($line in $result) {
        if ($line -match "^HF_SINGLE:(.+):(.+)$") {
            Write-Host ("  Single: {0}ms/text  {1} texts/s" -f $Matches[1], $Matches[2])
        }
        if ($line -match "^HF_BATCH:(.+):(.+)$") {
            Write-Host ("  Batch:  {0}ms/10texts  {1} texts/s" -f $Matches[1], $Matches[2])
        }
        if ($line -match "^HF_ERROR:(.+)$") {
            Write-Host "  ERROR: $($Matches[1])" -ForegroundColor Red
        }
    }
}

# --- fastembed (Python ONNX) benchmark ---
if (-not $SkipFastembed) {
    Write-Host ""
    Write-Host "--- fastembed (ONNX Runtime) ---" -ForegroundColor Yellow

    $feModel = $HfModel
    # fastembed uses different model IDs for some models
    if ($feModel -eq "thenlper/gte-small") { $feModel = "" }  # not supported

    if ($feModel) {
        $pyScript = @"
import time
try:
    from fastembed import TextEmbedding
    model = TextEmbedding('$feModel')
    texts = ['$TestText']
    list(model.embed(texts * 3))
    N = $NRuns
    t0 = time.perf_counter()
    for _ in range(N):
        list(model.embed(texts))
    elapsed = time.perf_counter() - t0
    print(f'FE_SINGLE:{elapsed/N*1000:.1f}:{N/elapsed:.0f}')
    batch = texts * 10
    t0 = time.perf_counter()
    runs = max(10, N//10)
    for _ in range(runs):
        list(model.embed(batch))
    elapsed = time.perf_counter() - t0
    print(f'FE_BATCH:{elapsed/runs*1000:.1f}:{runs*10/elapsed:.0f}')
except Exception as e:
    print(f'FE_ERROR:{e}')
"@
        $result = python -c $pyScript 2>$null
        foreach ($line in $result) {
            if ($line -match "^FE_SINGLE:(.+):(.+)$") {
                Write-Host ("  Single: {0}ms/text  {1} texts/s" -f $Matches[1], $Matches[2])
            }
            if ($line -match "^FE_BATCH:(.+):(.+)$") {
                Write-Host ("  Batch:  {0}ms/10texts  {1} texts/s" -f $Matches[1], $Matches[2])
            }
            if ($line -match "^FE_ERROR:(.+)$") {
                Write-Host "  ERROR: $($Matches[1])" -ForegroundColor Red
            }
        }
    } else {
        Write-Host "  Skipped (model not supported by fastembed)"
    }
}

# --- fastembed-rs benchmark ---
if (-not $SkipFastembed) {
    Write-Host ""
    Write-Host "--- fastembed-rs (if available) ---" -ForegroundColor Yellow

    $feRsBinary = $null
    $feRsPaths = @(
        "..\fastembed-rs\target\release\fastembed-bench.exe",
        "..\fastembed-rs\target\release\examples\bench.exe",
        "fastembed-bench.exe"
    )
    foreach ($p in $feRsPaths) { if (Test-Path $p) { $feRsBinary = $p; break } }

    if ($feRsBinary) {
        Write-Host "  Binary: $feRsBinary"
        # Run fastembed-rs benchmark if available
        & $feRsBinary --model $HfModel --text $TestText --n-runs $NRuns 2>$null
    } else {
        Write-Host "  Not found. Build fastembed-rs with: cd ..\fastembed-rs && cargo build --release"
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Benchmark Complete" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
