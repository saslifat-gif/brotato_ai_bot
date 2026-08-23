@echo off
chcp 65001 >nul
title Brotato AI v3 - API Recurrent Agent
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set BROTATO_V3_DEVICE=auto
set BROTATO_V3_TIMESTEPS=1000000
python -m v3.train
pause
