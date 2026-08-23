@echo off
chcp 65001 >nul
title Brotato AI v2 - Detector Validation
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set BROTATO_CAPTURE_BACKEND=mss
set BROTATO_V2_DEVICE=cpu
python -m v2.validate_detector --task combat
pause
