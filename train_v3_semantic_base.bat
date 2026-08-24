@echo off
chcp 65001 >nul
title Brotato AI v3 - Semantic Combat Base
cd /d "%~dp0"
call conda activate bota_ai 2>nul || echo [warn] conda activate skipped
python -u -m v3.train_semantic_combat_bc --dataset "%~dp0models\version_3\human_semantic_combat_v2.jsonl" --base-model "%~dp0models\version_3\human_combat_base_candidate.pt" --output "%~dp0models\version_3\semantic_combat_base_candidate.pt"
pause
