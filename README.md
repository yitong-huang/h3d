# H3D

个性化虚拟化身工作台：上传 1–N 张人像/局部特写 → **真实**衣下素体推理（SMPL-X）→ Web 姿态/表情/身材编辑 → 导出标准 GLB + Pose JSON。

默认后端为 **real**（`H3D_PIPELINE=real`）。仅调试时可 `H3D_PIPELINE=mock` 回退程序化假人。

## 仓库结构

```text
apps/
  api/          FastAPI + 可插拔推理管线（默认 real）
  web/          React + Three.js 工作台
packages/
  contracts/    骨骼 / Morph / Pose JSON 合约
scripts/
  fetch_models.py   下载 SMPL-X / MediaPipe / 预热 SegFormer
data/models/    权重目录（gitignore，需本机拉取）
```

## 硬件与依赖

| 项目 | 要求 |
|------|------|
| OS | Linux / macOS（需 `libEGL` 供 MediaPipe） |
| Python | 3.11+ |
| RAM | ≥8 GB（推荐 16 GB；CPU 拟合） |
| GPU | 可选。设 `H3D_DEVICE=cuda` 可加速；无 GPU 时默认 CPU |
| 磁盘 | ≈1 GB 权重（SMPL-X ≈520 MB + HF 缓存） |

### 许可

- **SMPL-X**：Max Planck Institute 研究许可。使用前请在 [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) 注册并遵守条款。`fetch_models.py` 仅拉取开发用镜像权重，**不**代替官方授权。
- **SegFormer clothes**（SCHP 开源等价）：Hugging Face `mattmdjaga/segformer_b2_clothes`
- **MediaPipe Pose**：Apache-2.0

## 快速启动

### 1. 拉取权重

```bash
cd apps/api
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # 或 cu124
pip install -r requirements.txt
python ../../scripts/fetch_models.py
```

### 2. API

```bash
cd apps/api && source .venv/bin/activate
# 默认 real；调试假人：export H3D_PIPELINE=mock
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

`GET /api/health` → `"backend":"real"`

### 3. Web

```bash
cd apps/web && npm install && npm run dev
# http://127.0.0.1:5173
```

## 推理阶段（默认 real）

| 阶段 | 实现 | 说明 |
|------|------|------|
| 图像分流 | MediaPipe Pose + SegFormer clothes | SCHP 开源等价；DensePose 默认 `smplx_iuv` 投影 |
| 衣下体型 | SMPL-X 可微拟合 + 衣物松弛补偿 | SHAPY/BCNet **开源等价**（非官方 SHAPY 权重） |
| 局部融合 | PnP (`solvePnPRansac`) + `seamlessClone` 泊松 + Sobel 法线 | 真实几何/图像处理 |
| 资产组装 | 官方 SMPL-X 前向 → GLB | **10475 verts**、55+4 骨、morph、`_PART_ID` |

关闭 mock（默认已关闭）：不要设置 `H3D_PIPELINE=mock`。  
强制 mock：`export H3D_PIPELINE=mock`。

可选：`H3D_DEVICE=cuda`、`H3D_DENSEPOSE=detectron2`（需自行安装 Detectron2 DensePose）。

## 资产合约

- 拓扑：SMPL-X **10475** 顶点（real）
- 59 joints：55 SMPL-X + 4 helper（Breast/Glute）
- Morph：身材 + 表情通道；`_PART_ID` 0–7
- Pose JSON：`packages/contracts/pose.schema.json`

## 已知缺口

- 官方 SHAPY / Detectron2 DensePose 权重未捆绑；默认用开源等价物，接口可替换
- UV 为 SMPL-X 自带 vt 按顶点折叠（非独立全展开烘焙）；换装级非重叠 UV 可后续替换
- 表情 morph 为基于分区的几何偏移，非 FLAME 表情基
