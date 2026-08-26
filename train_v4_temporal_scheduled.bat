@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\lifat\miniconda3\envs\ml\python.exe"
set "MODEL_DIR=C:\ml\brotato\models\version_3"

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 4242 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if not errorlevel 1 goto trainer_start
echo.
echo [v4-scheduled] A trainer is already listening on bridge port 4242.
echo [v4-scheduled] Close the existing trainer before starting another one.
pause
exit /b 2

:trainer_start
set "RESUME_MODEL=%MODEL_DIR%\v4_temporal_bootstrap.zip"
set "LATEST_MODEL="

for /f "delims=" %%F in ('dir /b /a-d /o-d "%MODEL_DIR%\v4_temporal_checkpoints\v4_temporal_ppo_*_steps.zip" 2^>nul') do if not defined LATEST_MODEL set "LATEST_MODEL=%%F"
if defined LATEST_MODEL set "RESUME_MODEL=%MODEL_DIR%\v4_temporal_checkpoints\%LATEST_MODEL%"

echo [v4-scheduled] resume=%RESUME_MODEL%
echo [v4-scheduled] launch_token=%V4_LAUNCH_TOKEN% >> "%MODEL_DIR%\v4_temporal_train.log"
"%PYTHON%" -u -m v4.train_temporal_hierarchical --resume "%RESUME_MODEL%" --raw-cache-only --state-hz 24 --torch-threads 1 --device cuda --timesteps 1000000 >> "%MODEL_DIR%\v4_temporal_train.log" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
echo [v4-scheduled] trainer exited with code %EXIT_CODE%. See %MODEL_DIR%\v4_temporal_train.log
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
