@echo off
chcp 65001 >nul
title Brotato AI v2 - Frame Curation
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
python -m v2.curate_recording --session latest --stride 10
pause
