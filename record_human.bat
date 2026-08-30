@echo off
setlocal
title Brotato AI V4 - Human Demonstration Recorder
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set "PYTHONPATH=%~dp0src;%~dp0;%PYTHONPATH%"
set "BROTATO_V4_POLICY_MODE=HANDCRAFTED"
set "BROTATO_V4_AUTOMATE_MENUS=0"
if "%~1"=="" set "OUTPUT=%~dp0models\version_3\human_demos\manual_run.sqlite"
if not "%~1"=="" set "OUTPUT=%~1"
echo [v4-human-demo] output=%OUTPUT%
echo [v4-human-demo] play manually; press F9 to bookmark meaningful states
python -u -m v4.record_human_demo --output "%OUTPUT%" --run-label manual --require-capture %2 %3 %4 %5 %6 %7 %8 %9
pause
