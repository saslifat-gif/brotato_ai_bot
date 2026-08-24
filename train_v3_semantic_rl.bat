@echo off
chcp 65001 >nul
title Brotato AI v3 - Semantic PPO Fine-tuning
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
python -u -m v3.train_semantic_finetune --state-hz 12
pause
