@echo off
chcp 65001 >nul
title Brotato AI v3 - Human Base PPO Fine-tuning
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
python -u -m v3.train_combat_finetune
pause
