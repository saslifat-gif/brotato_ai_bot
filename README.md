# Brotato AI (v1)

Windows-only reinforcement learning project for Brotato.

这是一个面向 Brotato 的 Windows 平台强化学习项目。

## Features | 功能

- PPO training loop (`v1/train.py`)
- Custom game environment (`v1/env/brotato_env.py`)
- Reward / shop / runtime modules
- Utility scripts for HP labeling and YOLO classification prep

- PPO 训练主流程（`v1/train.py`）
- 自定义游戏环境（`v1/env/brotato_env.py`）
- 奖励、商店策略、运行时控制模块
- HP 标注与 YOLO 分类数据准备脚本

## Quick Start | 快速开始

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: configure environment variables via `.env.example`.
4. Start training:

```bash
python v1/train.py
```

Or run:

```bat
train_mask.bat
```

1. 创建并激活 Python 虚拟环境。
2. 安装依赖：`pip install -r requirements.txt`
3. 可选：参考 `.env.example` 配置环境变量。
4. 启动训练：`python v1/train.py`（或运行 `train_mask.bat`）

## Test | 测试

Default unit tests:

```bash
pytest
```

默认会运行 `test/v1/unit` 下的单元测试。

## Key Environment Variables | 关键环境变量

- `ROBOFLOW_API_KEY`: optional, enables Roboflow detector path.
- `BROTATO_OUTPUT_DIR`: model/checkpoint output directory.
- `BROTATO_WINDOW_TITLE`, `BROTATO_EXE_NAME`: target game window.

完整配置见 `v1/config/runtime_config.py`。

## GitHub Publish Checklist | 发布检查清单

1. Rotate leaked credentials before publishing.
2. Ensure large artifacts are ignored (`models/`, `point/`, `.venv/`).
3. Run tests locally (`pytest`).
4. Commit only source/config/docs.
5. Push to your GitHub repository.

1. 发布前先轮换密钥（尤其是 Roboflow API Key）。
2. 确认大文件产物已忽略（`models/`、`point/`、`.venv/`）。
3. 本地跑测试（`pytest`）。
4. 只提交源码、配置和文档。
5. 推送到你的 GitHub 仓库。

## Notes | 说明

- This project is tightly coupled to Windows APIs (`ctypes.windll`, WinRT OCR, windows-capture).
- Publish model artifacts with GitHub Releases or external storage if needed.

- 项目高度依赖 Windows API（`ctypes.windll`、WinRT OCR、windows-capture）。
- 模型产物建议通过 GitHub Releases 或外部存储发布。
