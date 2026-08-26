@echo off
chcp 65001 >nul
title Brotato AI v4 - Temporal Hierarchical PPO
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
set "BROTATO_V3_UI_BUILD_PROFILE=ranged_smg"
set "BROTATO_V3_UI_MODEL="
python -u -m v4.train_temporal_hierarchical --state-hz 24 --torch-threads 1
pause
