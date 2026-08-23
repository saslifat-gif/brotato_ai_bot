@echo off
chcp 65001 >nul
title Brotato AI - Runtime Preflight
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped

set BROTATO_CAPTURE_BACKEND=mss
set BROTATO_INPUT_MODE=physical_foreground
set BROTATO_CONTROL_PANEL=false

echo [diagnose] window and capture check
python v1\diagnose_runtime.py
if errorlevel 1 pause
