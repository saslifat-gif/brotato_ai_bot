@echo off
setlocal EnableDelayedExpansion
title Close Brotato V4 Trainer

set "MODEL_DIR=C:\ml\brotato\models\version_3\ranged_smg_v2"
set "STOP_FILE=%MODEL_DIR%\v4_stop.request"

for /f %%P in ('powershell -NoProfile -Command "(Get-NetTCPConnection -LocalPort 4242 -State Listen -ErrorAction SilentlyContinue).OwningProcess"') do if not defined TRAINER_PID set "TRAINER_PID=%%P"
if not defined TRAINER_PID (
    echo [v4-close] No trainer is listening; bridge port 4242 is already free.
    exit /b 0
)

echo [v4-close] Requesting a normal checkpoint-preserving stop for PID !TRAINER_PID!...
powershell -NoProfile -Command "Set-Content -Path '%STOP_FILE%' -Value 'stop' -Encoding ascii"

for /l %%N in (1,1,90) do (
    powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 4242 -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
    if not errorlevel 1 goto stopped
    timeout /t 1 /nobreak >nul
)

echo [v4-close] WARNING: graceful stop timed out after 90 seconds.
echo [v4-close] Leaving PID !TRAINER_PID! running so its checkpoint is not corrupted.
exit /b 2

:stopped
echo [v4-close] Trainer stopped normally, final checkpoint saved, and port 4242 is free.
exit /b 0
