@echo off
chcp 65001 >nul
title Brotato AI v2 - Recurrent Vision Agent
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set BROTATO_CAPTURE_BACKEND=windows-capture
set BROTATO_V2_DEVICE=cpu
set BROTATO_V2_IMGSZ=416
python -m v2.train_agent
pause
