# Brotato AI Bot (v1)

Windows-only reinforcement learning project for **Brotato**.

这是一个运行在 **Windows** 平台上的 Brotato 强化学习项目，核心使用 PPO 训练策略网络，并包含游戏环境、奖励系统、商店策略与运行时控制模块。

## 功能概览

- PPO 训练主流程：`v1/train.py`
- 自定义游戏环境：`v1/env/brotato_env.py`
- 奖励引擎：`v1/reward/reward_engine.py`
- 商店策略与 OCR：`v1/shop/`
- 实用工具脚本：HP 标注、YOLO 分类数据准备等

## 环境要求

- 操作系统：Windows 10/11
- Python：建议 3.11
- 游戏窗口：Brotato（可通过环境变量配置窗口标题/进程名）

## 快速开始

1. 创建并激活虚拟环境

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 配置环境变量（可选）

- 复制 `.env.example` 并按需设置。
- 若使用 Roboflow 检测器，需提供 `ROBOFLOW_API_KEY`。

4. 启动训练

```bash
python v1/train.py
```

或使用批处理脚本：

```bat
train_mask.bat
```

## 测试

默认单元测试目录为 `test/v1/unit`：

```bash
pytest
```

## 常用环境变量

- `ROBOFLOW_API_KEY`：可选，启用 Roboflow 检测路径
- `BROTATO_OUTPUT_DIR`：模型与 checkpoint 输出目录
- `BROTATO_WINDOW_TITLE`：游戏窗口标题
- `BROTATO_EXE_NAME`：游戏进程名（如 `Brotato.exe`）

完整配置项请查看：`v1/config/runtime_config.py`

## 训练控制热键

- `F7`：开始/暂停自动化
- `F8`：请求停止训练并保存
- `F6`：显示/隐藏调试窗口

## 项目结构

```text
.
├─v1/                 # 核心训练与环境代码
│  ├─config/          # 运行时配置
│  ├─env/             # 游戏环境与检测适配
│  ├─reward/          # 奖励计算
│  ├─runtime/         # 输入、停止控制、调试窗口
│  └─shop/            # 商店策略与 OCR
├─test/               # 测试代码
├─raw_models/         # 原始模型/资源/实验脚本
├─train_mask.bat      # Windows 启动脚本
└─requirements.txt    # 依赖列表
```

## 发布到 GitHub 前检查

1. 轮换并移除所有泄露过的密钥（尤其是 API Key）
2. 确认大文件产物不入库（`models/`、`point/`、`.venv/`）
3. 本地运行 `pytest` 确保通过
4. 仅提交源码、配置、文档

## 注意事项

- 项目高度依赖 Windows API（`ctypes.windll`、WinRT OCR、windows-capture）。
- 训练产出的模型文件建议通过 GitHub Releases 或外部对象存储发布。

## License

See `LICENSE`.
