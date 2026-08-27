@echo off
setlocal
set "PYTHON=C:\Users\lifat\miniconda3\envs\bota_ai\python.exe"
set "PYTHONPATH=C:\ml\brotato\src;C:\ml\brotato;%PYTHONPATH%"
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%T"
set "OUT=C:\ml\brotato\models\version_3\raw_records\raw_%STAMP%.jsonl"
echo [raw-recorder] Connecting to bridge recorder port 4243...
"%PYTHON%" -m brotato_ai.data.recorder --host 127.0.0.1 --port 4243 --output "%OUT%" --max-gib 10
echo.
echo [raw-recorder] Finished. Output: %OUT%
pause
