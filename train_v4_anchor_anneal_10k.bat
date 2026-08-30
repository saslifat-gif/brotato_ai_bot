@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=C:\Users\lifat\miniconda3\envs\bota_ai\python.exe"
set "PYTHONPATH=C:\ml\brotato\src;C:\ml\brotato;%PYTHONPATH%"
set "MODEL_ROOT=C:\ml\brotato\models\version_3"
set "PARENT_DIR=%MODEL_ROOT%\ranged_smg_v2"
set "RUN_DIR=%PARENT_DIR%\anchor_anneal_20_to_0_10k"
set "BROTATO_V4_OUTPUT_DIR=%RUN_DIR%"
set "BROTATO_V4_UI_BUILD_PROFILE=ranged_smg"
set "BROTATO_V4_UI_MODEL="
set "BROTATO_V4_UI_DATASET=%MODEL_ROOT%\ui_decisions_ranged_smg_v2.jsonl"
set "SOURCE_MODEL=%MODEL_ROOT%\\bullet_hell_finetune_best\\best_training_agent.zip"
set "SEMANTIC_DATASET=%MODEL_ROOT%\human_semantic_combat_v2.jsonl"
set "RAW_DATASET=%MODEL_ROOT%\raw_records"

if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 4242 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if not errorlevel 1 goto start
echo [anchor-anneal] Trainer bridge port 4242 is already occupied.
echo [anchor-anneal] Stop the existing trainer before starting this fresh run.
pause
exit /b 2

:start
echo [anchor-anneal] Fresh v4 model output=%RUN_DIR%
echo [anchor-anneal] Human anchor: 0.20 -> 0.00 over 10000 timesteps
"%PYTHON%" -u -m brotato_ai.training.ppo --source-model "%SOURCE_MODEL%" --dataset "%SEMANTIC_DATASET%" --raw-dataset "%RAW_DATASET%" --raw-cache-only --state-hz 24 --torch-threads 1 --device cuda --timesteps 10000 --bc-coefficient 0.20 --bc-coefficient-final 0.0 --bc-anneal-steps 10000 --run-name V4AnchorAnneal20To0_10k
set "EXIT_CODE=%ERRORLEVEL%"
echo [anchor-anneal] Trainer exited with code %EXIT_CODE%.
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
