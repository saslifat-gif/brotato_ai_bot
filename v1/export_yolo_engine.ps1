param(
    [string]$Model = "runs/classify/train/weights/best.pt",
    [int]$ImgSize = 256,
    [string]$Device = "0",
    [switch]$Half = $true,
    [int]$Batch = 1,
    [double]$Workspace = 4
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Model)) {
    throw "Model not found: $Model"
}

$halfArg = ""
if ($Half) {
    $halfArg = "half=True"
}

$cmd = @(
    "yolo export",
    "model=$Model",
    "format=engine",
    "imgsz=$ImgSize",
    "device=$Device",
    "batch=$Batch",
    "workspace=$Workspace",
    "dynamic=False",
    $halfArg
) -join " "

Write-Host "[export] $cmd"
Invoke-Expression $cmd

$enginePath = [System.IO.Path]::ChangeExtension($Model, ".engine")
if (Test-Path $enginePath) {
    Write-Host "[done] TensorRT engine export finished: $enginePath"
    exit 0
}

Write-Error "TensorRT engine export failed. Engine file not found: $enginePath"
exit 1
