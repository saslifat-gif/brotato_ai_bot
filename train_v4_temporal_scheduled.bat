@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\lifat\miniconda3\envs\bota_ai\python.exe"
set "MODEL_DIR=C:\ml\brotato\models\version_3"
set "RESUME_MODEL=%MODEL_DIR%\v4_temporal_bootstrap.zip"
set "LATEST_MODEL="

for /f "delims=" %%F in ('dir /b /a-d /o-d "%MODEL_DIR%\v4_temporal_checkpoints\v4_temporal_ppo_*_steps.zip" 2^>nul') do if not defined LATEST_MODEL set "LATEST_MODEL=%%F"
if defined LATEST_MODEL set "RESUME_MODEL=%MODEL_DIR%\v4_temporal_checkpoints\%LATEST_MODEL%"

echo [v4-scheduled] resume=%RESUME_MODEL%
"%PYTHON%" -u -m v4.train_temporal_hierarchical --resume "%RESUME_MODEL%" --state-hz 24 --torch-threads 1 --timesteps 1000000 >> "%MODEL_DIR%\v4_temporal_train.log" 2>&1
