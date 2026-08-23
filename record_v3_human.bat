@echo off
chcp 65001 >nul
title Brotato AI v3 - Human Demonstration Recorder
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set "BROTATO_V3_UI_MODEL="
set "BROTATO_V3_UI_DATASET=%~dp0models\version_3\ui_decisions_stick_melee_v2.jsonl"
python -u -m v3.record_human --output "%~dp0models\version_3\human_combat_v1.jsonl" --sample-hz 8 --idle-hz 2
pause
