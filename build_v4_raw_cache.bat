@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\lifat\miniconda3\envs\ml\python.exe"
set "LOG=C:\ml\brotato\models\version_3\v4_raw_cache_build.log"

echo [raw-cache] Background refresh started.>> "%LOG%"
"%PYTHON%" -u -m v4.build_raw_anchor_cache --raw-dataset "C:\ml\brotato\models\version_3\raw_records" --max-records 50000 --stride 3 >> "%LOG%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
echo [raw-cache] Refresh exited with code %EXIT_CODE%.>> "%LOG%"
exit /b %EXIT_CODE%
