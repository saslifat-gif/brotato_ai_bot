@echo off
chcp 65001 >nul
title Brotato AI v4 - Temporal Hierarchical PPO
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set "MODEL_ROOT=%~dp0models\version_3"
set "MODEL_DIR=%MODEL_ROOT%\ranged_smg_v2"
set "BROTATO_V3_OUTPUT_DIR=%MODEL_DIR%"
set "BROTATO_V3_UI_BUILD_PROFILE=ranged_smg"
set "BROTATO_V3_UI_MODEL="
set "BROTATO_V3_UI_DATASET=%MODEL_ROOT%\ui_decisions_ranged_smg_v2.jsonl"
python -u -m v4.train_temporal_hierarchical --source-model "%MODEL_ROOT%\bullet_hell_finetune_best\best_training_agent.zip" --dataset "%MODEL_ROOT%\human_semantic_combat_v2.jsonl" --raw-dataset "%MODEL_ROOT%\raw_records" --state-hz 24 --torch-threads 1
pause
