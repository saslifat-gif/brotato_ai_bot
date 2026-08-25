@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\lifat\miniconda3\envs\bota_ai\python.exe"
set "MODEL_DIR=C:\ml\brotato\models\version_3"
set "RESUME_MODEL=%MODEL_DIR%\v4_temporal_checkpoints\v4_temporal_ppo_240000_steps.zip"
set "BROTATO_V3_LATE_WAVE_FOCUS=1"
set "BROTATO_V3_TIMESTEPS=1000000"

echo [v4-late-wave] resume=%RESUME_MODEL%
echo [v4-late-wave] focus=waves 18-20 threat avoidance and survival
"%PYTHON%" -u -m v4.train_temporal_hierarchical --resume "%RESUME_MODEL%" --state-hz 20 --timesteps %BROTATO_V3_TIMESTEPS%
pause
