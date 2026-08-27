@echo off
setlocal
cd /d "%~dp0\..\.."
set "PYTHON=C:\Users\lifat\miniconda3\envs\bota_ai\python.exe"
set "PYTHONPATH=C:\ml\brotato\src;C:\ml\brotato;%PYTHONPATH%"

if "%~1"=="" (
    echo Usage: tools\inspect\backtest_v4.bat RECORDING.jsonl [MAX_RECORDS]
    exit /b 2
)

set "MAX_RECORDS=%~2"
if "%MAX_RECORDS%"=="" set "MAX_RECORDS=0"
set "REPORT_DIR=C:\ml\brotato\models\version_3\evaluation"
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

"%PYTHON%" -m brotato_ai.evaluation.backtest "%~1" --max-records %MAX_RECORDS% --json "%REPORT_DIR%\v4_controller_backtest.json" --markdown "%REPORT_DIR%\v4_controller_backtest.md"
exit /b %ERRORLEVEL%

