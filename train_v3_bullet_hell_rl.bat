@echo off
chcp 65001 >nul
title Brotato AI v3 - Bullet Hell PPO Fine-tuning
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
python -u -m v3.train_bullet_hell_finetune --state-hz 12
pause
