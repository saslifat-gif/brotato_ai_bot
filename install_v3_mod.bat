@echo off
chcp 65001 >nul
title Brotato AI v3 - Install Training Bridge
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set "GAME_DIR=C:\Program Files (x86)\Steam\steamapps\common\Brotato"
if not exist "%GAME_DIR%\Brotato.exe" set "GAME_DIR=C:\Program Files\Steam\steamapps\common\Brotato"
if not exist "%GAME_DIR%\Brotato.exe" (
  echo [error] Brotato.exe was not found at the standard Steam paths.
  echo [error] Set GAME_DIR in this file to the folder containing Brotato.exe.
  pause
  exit /b 1
)
echo [v3-install] game_root=%GAME_DIR%
python -m v3.install_mod --game-dir "%GAME_DIR%"
pause
