@echo off
chcp 65001 >nul
title Brotato AI v3 - Frozen Safe Collector
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set "BROTATO_V3_SAFETY_SHIELD=1"
set "BROTATO_V3_UI_MODEL="
set "BROTATO_V3_UI_DATASET=%~dp0models\version_3\ui_decisions_stick_melee_v2.jsonl"
python -u -m v3.run_frozen --model "%~dp0models\version_3\combat_peak_100883_agent.zip" --combat-dataset "%~dp0models\version_3\combat_decisions_v1.jsonl"
pause
