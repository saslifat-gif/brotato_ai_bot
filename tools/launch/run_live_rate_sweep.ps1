$ErrorActionPreference = "Stop"
Set-Location C:\ml\brotato
New-Item -ItemType Directory -Force -Path C:\ml\brotato\reports | Out-Null

$python = "C:\Users\lifat\miniconda3\envs\bota_ai\python.exe"
$env:PYTHONPATH = "C:\ml\brotato\src;C:\ml\brotato"
$env:BROTATO_V4_OUTPUT_DIR = "C:\ml\brotato\models\version_3\ranged_smg_v2"
$env:BROTATO_V4_UI_BUILD_PROFILE = "ranged_smg"
$env:BROTATO_V4_UI_DATASET = "C:\ml\brotato\models\version_3\ui_decisions_ranged_smg_v2.jsonl"
$env:BROTATO_V4_SAFETY_SHIELD = "1"
$env:BROTATO_V4_FULL_RESTART = "1"
$env:BROTATO_TORCH_THREADS = "8"

$rates = @(10, 15, 20, 24, 30, 60)
if ($args.Count -gt 0) {
    $rates = $args | ForEach-Object { [double]$_ }
}

foreach ($rate in $rates) {
    $stamp = "{0}hz" -f $rate
    Write-Host ("===== live rate {0} Hz =====" -f $rate)
    $env:RATE_HZ = "$rate"
    $env:EPISODES = "3"
    $raw = "C:\ml\brotato\models\version_3\raw_records\live_rate_$stamp.jsonl"
    $recorder = Start-Process -FilePath $python -ArgumentList @(
        "-u", "-m", "brotato_ai.data.recorder",
        "--host", "127.0.0.1", "--port", "4243",
        "--output", $raw, "--max-gib", "10"
    ) -WorkingDirectory "C:\ml\brotato" -WindowStyle Hidden -PassThru
    try {
        & C:\ml\brotato\tools\launch\run_frozen_rate.bat
        $code = $LASTEXITCODE
    } finally {
        if ($recorder -and -not $recorder.HasExited) {
            Stop-Process -Id $recorder.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host ("===== live rate {0} Hz exit={1} =====" -f $rate, $code)
    if ($code -ne 0) {
        throw "frozen runner failed for $rate Hz with exit $code"
    }
}
Write-Host "===== live rate sweep complete ====="
