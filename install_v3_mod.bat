@echo off
chcp 65001 >nul
title Brotato AI v3 - Install Training Bridge
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
python -m v3.install_mod
pause
