@echo off
chcp 65001 >nul
title Brotato AI v2 - Import OBS Recording
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
python -m v2.import_obs_video %*
pause
