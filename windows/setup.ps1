# Download the runtime DLLs next to the exes:
#   ONNX Runtime 1.30.0 (CUDA 12 build, with CUDA + TensorRT EPs)
#   TensorRT 10.13.3 (cu12)  - supports RTX 5090 (sm_120)
#   CUDA 12.9 runtime + cuBLAS, cuDNN 9
# ~3.2 GB download. Needs an NVIDIA driver that supports CUDA 12.9 (R575+).
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$dl = Join-Path $root "_downloads"
New-Item -ItemType Directory -Force $dl | Out-Null

$pkgs = @(
    @{ name = "onnxruntime"; url = "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-win-x64-gpu_cuda12-1.30.0.zip" },
    @{ name = "tensorrt";    url = "https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-10.13.3.9-py2.py3-none-win_amd64.whl" },
    @{ name = "cudart";      url = "https://files.pythonhosted.org/packages/59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl" },
    @{ name = "cublas";      url = "https://files.pythonhosted.org/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl" },
    @{ name = "cudnn";       url = "https://files.pythonhosted.org/packages/c5/ee/baebebf270df5a57830e40879b4016de47ca43961095cf18b7749452150f/nvidia_cudnn_cu12-9.26.0.51-py3-none-win_amd64.whl" }
)

foreach ($p in $pkgs) {
    $zip = Join-Path $dl ($p.name + ".zip")   # wheels are zip archives
    if (-not (Test-Path $zip)) {
        Write-Host "downloading $($p.name) ..."
        # curl.exe ships with Windows 10+; far faster than Invoke-WebRequest for GB files
        & curl.exe -L --fail --retry 3 -o $zip $p.url
        if ($LASTEXITCODE -ne 0) { throw "download failed: $($p.url)" }
    }
    $out = Join-Path $dl $p.name
    if (-not (Test-Path $out)) {
        Write-Host "extracting $($p.name) ..."
        New-Item -ItemType Directory -Force $out | Out-Null
        & tar.exe -xf $zip -C $out
        if ($LASTEXITCODE -ne 0) { throw "extract failed: $zip" }
    }
    Get-ChildItem -Path $out -Recurse -Filter *.dll | ForEach-Object {
        Copy-Item $_.FullName -Destination $root -Force
    }
}

$need = "onnxruntime.dll", "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_tensorrt.dll", "nvinfer_10.dll", "nvonnxparser_10.dll",
        "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"
$missing = $need | Where-Object { -not (Test-Path (Join-Path $root $_)) }
if ($missing) { throw "missing DLLs: $($missing -join ', ')" }
Write-Host "OK: runtime DLLs in $root  (you can delete _downloads\ now)"
& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
