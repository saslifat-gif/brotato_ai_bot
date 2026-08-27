@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\lifat\miniconda3\envs\ml\python.exe"
set "MODEL_ROOT=C:\ml\brotato\models\version_3"
set "MODEL_DIR=%MODEL_ROOT%\ranged_smg_v1"
set "BROTATO_V3_OUTPUT_DIR=%MODEL_DIR%"
set "BROTATO_V3_UI_BUILD_PROFILE=ranged_smg"
set "BROTATO_V3_UI_MODEL="
set "BROTATO_V3_UI_DATASET=%MODEL_ROOT%\ui_decisions_ranged_smg_v2.jsonl"
set "SOURCE_MODEL=%MODEL_ROOT%\bullet_hell_finetune_best\best_training_agent.zip"
set "SEMANTIC_DATASET=%MODEL_ROOT%\human_semantic_combat_v2.jsonl"
set "RAW_DATASET=%MODEL_ROOT%\raw_records"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 4242 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if not errorlevel 1 goto trainer_start
echo.
echo [v4-scheduled] A trainer is already listening on bridge port 4242.
echo [v4-scheduled] Close the existing trainer before starting another one.
pause
exit /b 2

:trainer_start
set "RESUME_MODEL="
set "RESUME_OPTION="
set "LATEST_MODEL="

for /f "delims=" %%F in ('dir /b /a-d /o-d "%MODEL_DIR%\v4_temporal_checkpoints\v4_temporal_ppo_*_steps.zip" 2^>nul') do if not defined LATEST_MODEL set "LATEST_MODEL=%%F"
if defined LATEST_MODEL (
    set "RESUME_MODEL=%MODEL_DIR%\v4_temporal_checkpoints\%LATEST_MODEL%"
) else if exist "%MODEL_DIR%\v4_temporal_bootstrap.zip" (
    set "RESUME_MODEL=%MODEL_DIR%\v4_temporal_bootstrap.zip"
)
if defined RESUME_MODEL set "RESUME_OPTION=--resume %RESUME_MODEL%"

if defined RESUME_MODEL (echo [v4-scheduled] resume=%RESUME_MODEL%) else echo [v4-scheduled] resume=fresh-ranged-lineage
echo [v4-scheduled] launch_token=%V4_LAUNCH_TOKEN% >> "%MODEL_DIR%\v4_temporal_train.log"
"%PYTHON%" -u -m v4.train_temporal_hierarchical --source-model "%SOURCE_MODEL%" --dataset "%SEMANTIC_DATASET%" --raw-dataset "%RAW_DATASET%" %RESUME_OPTION% --raw-cache-only --state-hz 24 --torch-threads 1 --device cuda --timesteps 1000000 >> "%MODEL_DIR%\v4_temporal_train.log" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
echo [v4-scheduled] trainer exited with code %EXIT_CODE%. See %MODEL_DIR%\v4_temporal_train.log
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
