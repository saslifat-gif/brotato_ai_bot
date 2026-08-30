@echo off
setlocal
cd /d "%~dp0"
set "MODEL_ROOT=C:\ml\brotato\models\version_3"
set "MODEL_DIR=%MODEL_ROOT%\ranged_smg_v2"
set "BROTATO_V4_LATE_WAVE_FOCUS=1"
set "BROTATO_V4_OUTPUT_DIR=%MODEL_DIR%"
set "BROTATO_V4_UI_BUILD_PROFILE=ranged_smg"
set "BROTATO_V4_UI_MODEL="
set "BROTATO_V4_UI_DATASET=%MODEL_ROOT%\ui_decisions_ranged_smg_v2.jsonl"
set "PYTHONPATH=C:\ml\brotato\src;C:\ml\brotato;%PYTHONPATH%"
if exist "%MODEL_DIR%\v4_stop.request" del /q "%MODEL_DIR%\v4_stop.request"

echo [train] Checking trainer port...
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 4242 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if not errorlevel 1 goto start_trainer
echo [train] A trainer is already listening on bridge port 4242.
echo [train] Close the existing trainer before starting another run.
pause
exit /b 2

:start_trainer
set "TRAIN_TOKEN=%RANDOM%_%RANDOM%"
set "V4_LAUNCH_TOKEN=%TRAIN_TOKEN%"
echo [train] Starting trainer; raw-cache refresh will run in the background...
for /f %%P in ('powershell -NoProfile -Command "$p=Start-Process -FilePath $env:ComSpec -ArgumentList '/d','/c','call C:\ml\brotato\train_v4_temporal_scheduled.bat' -WorkingDirectory 'C:\ml\brotato' -WindowStyle Normal -PassThru; $p.Id"') do set "TRAINER_PID=%%P"
echo [train] trainer_pid=%TRAINER_PID%

:wait_for_trainer
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command "$lines=Get-Content '%MODEL_DIR%\v4_temporal_train.log'; $start=-1; for($i=0; $i -lt $lines.Count; $i++){ if($lines[$i] -like '*launch_token=%TRAIN_TOKEN%*'){ $start=$i } }; if($start -ge 0){ $tail=$lines | Select-Object -Skip ($start+1); if($tail -match 'raw_anchor_records='){ exit 0 } }; exit 1"
if not errorlevel 1 goto start_recorder
tasklist /FI "PID eq %TRAINER_PID%" | find "%TRAINER_PID%" >nul
if errorlevel 1 goto trainer_failed
goto wait_for_trainer

:start_recorder
echo [train] Starting 60 Hz raw recorder in a separate window...
for /f %%P in ('powershell -NoProfile -Command "$p=Start-Process -FilePath $env:ComSpec -ArgumentList '/d','/c','call C:\ml\brotato\record_v4_raw.bat' -WindowStyle Minimized -PassThru; $p.Id"') do set "RECORDER_PID=%%P"
echo [train] recorder_pid=%RECORDER_PID%
echo [train] Control bridge is ready; training is running immediately.

echo [train] Starting background raw-cache refresh...
for /f %%P in ('powershell -NoProfile -Command "$p=Start-Process -FilePath $env:ComSpec -ArgumentList '/d','/c','call C:\ml\brotato\build_v4_raw_cache.bat' -WorkingDirectory 'C:\ml\brotato' -WindowStyle Minimized -PassThru; $p.Id"') do set "CACHE_PID=%%P"
echo [train] cache_refresh_pid=%CACHE_PID%

:wait_for_training
timeout /t 5 /nobreak >nul
tasklist /FI "PID eq %TRAINER_PID%" | find "%TRAINER_PID%" >nul
if not errorlevel 1 goto wait_for_training

if defined RECORDER_PID (
    echo [train] Stopping recorder PID %RECORDER_PID%...
    taskkill /PID %RECORDER_PID% /T /F >nul 2>&1
)
echo [train] Combined run finished.
exit /b 0

:trainer_failed
echo [train] Trainer exited before startup completed. See %MODEL_DIR%\v4_temporal_train.log.
pause
exit /b 3
