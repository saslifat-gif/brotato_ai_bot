@echo off
chcp 65001 >nul
title Brotato AI V4 - Frozen Safe Collector
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set "BROTATO_V4_SAFETY_SHIELD=1"
set "BROTATO_V4_UI_BUILD_PROFILE=ranged_smg"
set "BROTATO_V4_UI_MODEL="
set "BROTATO_V4_UI_DATASET=%~dp0models\version_3\ui_decisions_ranged_smg_v1.jsonl"
python -u -m v4.run_frozen --model "%~dp0models\version_3\combat_peak_100883_agent.zip" --combat-dataset "%~dp0models\version_3\combat_decisions_v1.jsonl"
pause
