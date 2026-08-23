@echo off
chcp 65001 >nul
title Brotato AI v2 - Gameplay Recorder
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set BROTATO_CAPTURE_BACKEND=windows-capture
python -m v2.record_gameplay --fps 10
pause
