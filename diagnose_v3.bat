@echo off
chcp 65001 >nul
title Brotato AI v3 - API Bridge Diagnosis
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
python -m v3.diagnose_bridge --states 10
pause
