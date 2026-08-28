@echo off
setlocal
cd /d C:\ml\brotato
if not exist C:\ml\brotato\reports mkdir C:\ml\brotato\reports

set "PYTHON=C:\Users\lifat\miniconda3\envs\bota_ai\python.exe"
set "PYTHONPATH=C:\ml\brotato\src;C:\ml\brotato;%PYTHONPATH%"
set "BROTATO_V3_OUTPUT_DIR=C:\ml\brotato\models\version_3\ranged_smg_v2"
set "BROTATO_V3_UI_BUILD_PROFILE=ranged_smg"
set "BROTATO_V3_UI_DATASET=C:\ml\brotato\models\version_3\ui_decisions_ranged_smg_v2.jsonl"
set "BROTATO_V3_SAFETY_SHIELD=1"
set "BROTATO_V4_FULL_RESTART=1"
if "%BROTATO_TORCH_THREADS%"=="" set "BROTATO_TORCH_THREADS=8"
set "OMP_NUM_THREADS=%BROTATO_TORCH_THREADS%"
set "MKL_NUM_THREADS=%BROTATO_TORCH_THREADS%"
set "OPENBLAS_NUM_THREADS=%BROTATO_TORCH_THREADS%"

if "%RATE_HZ%"=="" set "RATE_HZ=24"
if "%EPISODES%"=="" set "EPISODES=3"
if "%MODEL%"=="" set "MODEL=C:\ml\brotato\models\version_3\ranged_smg_v2\v4_temporal_best\best_training_agent.zip"

set "STAMP=%RATE_HZ%hz"
set "OUT=C:\ml\brotato\reports\live_rate_%STAMP%.out.log"
set "ERR=C:\ml\brotato\reports\live_rate_%STAMP%.err.log"
set "RESULTS=C:\ml\brotato\reports\live_rate_%STAMP%.json"
set "DECISIONS=C:\ml\brotato\reports\live_rate_%STAMP%_decisions.jsonl"

echo [frozen-rate] model=%MODEL%
echo [frozen-rate] state_hz=%RATE_HZ% episodes=%EPISODES%
echo [frozen-rate] results=%RESULTS%
echo [frozen-rate] torch_threads=%BROTATO_TORCH_THREADS% device=cpu
"%PYTHON%" -u -m v4.run_frozen --model "%MODEL%" --state-hz %RATE_HZ% --episodes %EPISODES% --results "%RESULTS%" --combat-dataset "%DECISIONS%" --device cpu --torch-threads %BROTATO_TORCH_THREADS% >> "%OUT%" 2>> "%ERR%"
echo [frozen-rate] exit=%ERRORLEVEL%
exit /b %ERRORLEVEL%
