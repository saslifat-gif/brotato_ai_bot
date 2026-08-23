@echo off
chcp 65001 >nul
title Brotato AI v2 - Recurrent Vision Agent
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set BROTATO_CAPTURE_BACKEND=obs-camera
if not defined BROTATO_OBS_CAMERA_INDEX set BROTATO_OBS_CAMERA_INDEX=0
set BROTATO_V2_DEVICE=cpu
set BROTATO_V2_IMGSZ=416
python -m v2.train_agent
pause
