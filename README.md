# H3D

个性化虚拟化身工作台：上传 1–N 张人像/局部特写 → **真实**衣下素体推理（SMPL-X）→ Web 姿态/表情/身材编辑 → 导出标准 GLB + Pose JSON。

**默认管线是 `real`。** 未设置任何环境变量时 `H3D_PIPELINE` 默认为 `real`。仅调试时可显式 `export H3D_PIPELINE=mock`。

## 仓库结构

```text
apps/
  api/          FastAPI + 可插拔推理管线（默认 real）
  web/          React + Three.js 工作台
packages/
  contracts/    骨骼 / Morph / Pose JSON 合约
scripts/
  fetch_models.py   离线/CI 拉权重（与启动自动拉取同一模块）
data/models/    权重目录（gitignore；启动可自动拉取）
```

## 硬件与依赖

| 项目 | 要求 |
|------|------|
| OS | **macOS**（Apple Silicon / Intel）或 Linux |
| Python | 3.11+（推荐 3.11–3.12） |
| RAM | ≥8 GB（推荐 16 GB） |
| 加速 | 默认 `H3D_DEVICE=auto`：Mac 优先 **MPS**，否则 CPU；Linux 有 CUDA 则用 CUDA |
| 磁盘 | ≈1 GB 权重（SMPL-X ≈520 MB + HF 缓存） |

### 许可

- **SMPL-X**：Max Planck Institute 研究许可。使用前请在 [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) 注册并遵守条款。
- **SegFormer clothes**（SCHP 开源等价）：`mattmdjaga/segformer_b2_clothes`
- **MediaPipe Pose**：Apache-2.0

---

## macOS 启动（推荐）

适用于 Apple Silicon（M1/M2/M3/…）与 Intel Mac。

### 1. 系统依赖（Homebrew）

```bash
# 可选但推荐：确保有 Python 3.12 与常见编解码库
brew install python@3.12
# OpenCV / MediaPipe 一般用 pip 轮子，无需 libEGL（那是 Linux 包名）
```

若 `python3` 已是 3.11+，可跳过 brew Python。

### 2. API 虚拟环境与 PyTorch

```bash
cd apps/api
python3 -m venv .venv
source .venv/bin/activate

# macOS：直接装官方 wheel（含 MPS 支持），不要用 Linux CUDA 的 index-url
pip install -U pip
pip install torch torchvision
pip install -r requirements.txt
```

> 不要在 Mac 上使用 `https://download.pytorch.org/whl/cpu` 的 Linux 指令当唯一方式；macOS 用默认 PyPI/`pip install torch torchvision` 即可获得带 MPS 的构建。

### 3. 启动（自动拉权重 + 默认 real）

```bash
# 无需 export H3D_PIPELINE — 默认就是 real
# 可选：export H3D_DEVICE=mps   # 或 cpu / auto（默认）
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

启动日志应类似：

```text
Startup: default pipeline is REAL (H3D_PIPELINE env='real' → backend='real')…
Device auto-selected: mps (platform=darwin, …)   # 或 cpu
[check] Verifying real-pipeline model weights…
[ok] … / [fetch] …
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool
# backend=real, defaultPipeline=real, device.resolved=mps|cpu
```

### 4. Web

```bash
cd apps/web
npm install
npm run dev
# http://127.0.0.1:5173
```

### Mac 上 MediaPipe / OpenCV / Torch 说明

| 组件 | Mac 说明 |
|------|----------|
| **PyTorch** | `pip install torch torchvision`；`H3D_DEVICE=auto` 优先 MPS |
| **MediaPipe** | pip 轮子，**不需要** Linux 的 `libegl1`；若 Tasks 初始化失败，升级 `mediapipe` 或设 `H3D_DEVICE=cpu` 排查 |
| **OpenCV** | `opencv-python-headless`（requirements 已含），无需系统 OpenCV |
| **MPS 不稳时** | `export H3D_DEVICE=cpu` 强制 CPU（部分算子在 MPS 上仍可能回退/报错） |

启动仍会自动检查并下载权重（`H3D_AUTO_FETCH_MODELS=1` 默认）。离线：`python scripts/fetch_models.py`。

---

## Linux 快速启动

```bash
cd apps/api
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # 或 cu124
pip install -r requirements.txt
# MediaPipe 在部分 Linux 需要：sudo apt install libegl1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## 环境变量

| Env | Default | 含义 |
|-----|---------|------|
| `H3D_PIPELINE` | **`real`** | 仅 `mock` 启用假人管线 |
| `H3D_DEVICE` | **`auto`** | `auto` / `mps` / `cpu` / `cuda` |
| `H3D_AUTO_FETCH_MODELS` | `1` | 启动自动拉缺失权重；`0` 关闭 |
| `H3D_DENSEPOSE` | `smplx_iuv` | 可选 `detectron2`（需自装） |

自动拉取失败时 API **不会**静默回退 mock，启动会报错（除非显式 `H3D_PIPELINE=mock`）。

## 推理阶段（默认 real）

| 阶段 | 实现 | 说明 |
|------|------|------|
| 图像分流 | MediaPipe Pose + SegFormer clothes | SCHP 开源等价；DensePose 默认 `smplx_iuv` |
| 衣下体型 | SMPL-X 可微拟合 + 衣物松弛补偿 | SHAPY/BCNet **开源等价** |
| 局部融合 | PnP + Poisson `seamlessClone` + Sobel 法线 | 真实几何/图像处理 |
| 资产组装 | 官方 SMPL-X → GLB | **10475 verts**、55+4 骨、morph、`_PART_ID` |

## 资产合约

- 拓扑：SMPL-X **10475** 顶点（real）
- 59 joints：55 SMPL-X + 4 helper（Breast/Glute）
- Morph：身材 + 表情通道；`_PART_ID` 0–7
- Pose JSON：`packages/contracts/pose.schema.json`

## 已知缺口

- 官方 SHAPY / Detectron2 DensePose 权重未捆绑；默认用开源等价物，接口可替换
- UV 为 SMPL-X 自带 vt 按顶点折叠（非独立全展开烘焙）
- 表情 morph 为基于分区的几何偏移，非 FLAME 表情基
- MPS 上个别 torch 算子可能需回退 CPU（设 `H3D_DEVICE=cpu`）
