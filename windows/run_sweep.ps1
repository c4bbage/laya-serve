# Precision x concurrency sweep for one GPU: parity (vs Python bf16 reference) + goserve/bench.
# First run of each variant builds a TensorRT engine for this GPU (several minutes, cached in trtcache_<name>\).
# Usage:  powershell -ExecutionPolicy Bypass -File run_sweep.ps1 [-Gpu 0] [-Only fp16,fp8] [-SkipParity] [-Fixtures basic.jsonl]
param(
    [string]$Gpu = "0",
    [string[]]$Only = @(),
    [switch]$SkipParity,
    [int[]]$Conc = @(1, 64, 128, 192, 256, 384),
    [int]$BatchMax = 48,
    [string]$Fixtures = "basic.jsonl"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# name, onnx, TRT precision flag. fp8 graphs carry explicit Q/DQ; the flag sets the precision of the rest.
$variants = @(
    @{ name = "fp16"; onnx = "laya32s.onnx"; prec = "fp16" },
    @{ name = "bf16"; onnx = "laya32s.onnx"; prec = "bf16" },
    @{ name = "fp32"; onnx = "laya32s.onnx"; prec = "fp32" },
    @{ name = "fp8";  onnx = "laya8.onnx";   prec = "fp16" }
)
if ($Only.Count -gt 0) { $variants = $variants | Where-Object { $Only -contains $_.name } }

$env:ORT_LIB = Join-Path $PSScriptRoot "onnxruntime.dll"
$env:GOSERVE_PROVIDER = "trt"
$env:GOSERVE_GPUS = $Gpu
$env:BATCH_MAX = "$BatchMax"
$env:GOSERVE_MEM_LIMIT = "$(4GB)"
$port = 8329
$out = Join-Path $PSScriptRoot ("results_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
function Log($s) { Write-Host $s; Add-Content -Path $out -Value $s -Encoding utf8 }

Log ("# " + (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader -i $Gpu))
foreach ($v in $variants) {
    if (-not (Test-Path $v.onnx)) { Log "$($v.name): skip, $($v.onnx) not found"; continue }
    $env:ONNX_PATH = $v.onnx
    $env:TRT_PRECISION = $v.prec
    $env:TRT_CACHE = Join-Path $PSScriptRoot ("trtcache_" + $v.name)
    New-Item -ItemType Directory -Force $env:TRT_CACHE | Out-Null

    if (-not $SkipParity) {
        Write-Host "[$($v.name)] parity (builds the engine on first run) ..."
        $p = & .\paritycheck.exe -mode full -provider trt -gpu $Gpu -onnx $v.onnx -fixtures $Fixtures 2>&1 | Out-String
        $line = ($p -split "`n" | Where-Object { $_ -match "^full:|^engine:" }) -join " "
        if (-not $line) { $line = "FAILED (see parity_$($v.name).log)"; $p | Set-Content "parity_$($v.name).log" }
        Log "$($v.name) parity: $line"
    }

    $env:LISTEN = ":$port"
    $srv = Start-Process -FilePath .\goserve.exe -NoNewWindow -PassThru `
        -RedirectStandardOutput "goserve_$($v.name).out" -RedirectStandardError "goserve_$($v.name).log"
    $ready = $false
    for ($i = 0; $i -lt 300; $i++) {   # up to 15 min for a first engine build
        if ($srv.HasExited) { break }
        try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$port/healthz" | Out-Null; $ready = $true; break } catch { Start-Sleep 3 }
    }
    if (-not $ready) { Log "$($v.name): goserve did not start (see goserve_$($v.name).log)"; if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }; continue }

    foreach ($c in $Conc) {
        $d = if ($c -eq 1) { "10s" } else { "20s" }
        $r = & .\bench.exe -u "http://127.0.0.1:$port/predict" -c $c -d $d -warm 3s -f $Fixtures -slo 600ms 2>&1 | Out-String
        $rps = [regex]::Match($r, "rps=[0-9.]+").Value
        $lat = ([regex]::Matches($r, "p(50|95|99)=[0-9.]+ms") | ForEach-Object { $_.Value }) -join " "
        $slo = [regex]::Match($r, "[0-9.]+% within -> (PASS|FAIL)").Value
        Log ("{0,-6} c={1,-4} {2,-10} {3}  SLO600 {4}" -f $v.name, $c, $rps, $lat, $slo)
    }
    Stop-Process -Id $srv.Id -Force
    Start-Sleep 3
}
Log "results: $out"
