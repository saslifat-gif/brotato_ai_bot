@echo off
setlocal EnableDelayedExpansion
title Close Brotato V4 Trainer

for /l %%N in (1,1,3) do (
    set "TRAINER_PID="
    for /f %%P in ('powershell -NoProfile -Command "(Get-NetTCPConnection -LocalPort 4242 -State Listen -ErrorAction SilentlyContinue).OwningProcess"') do if not defined TRAINER_PID set "TRAINER_PID=%%P"
    if defined TRAINER_PID (
        echo [v4-close] Stopping trainer PID !TRAINER_PID! and its child processes...
        taskkill /PID !TRAINER_PID! /T /F >nul 2>&1
        timeout /t 2 /nobreak >nul
    ) else (
        goto port_check
    )
)

:port_check
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 4242 -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo [v4-close] Trainer stopped and bridge port 4242 is free.
) else (
    echo [v4-close] WARNING: port 4242 is still occupied.
)
pause
