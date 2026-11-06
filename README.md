# H3D

个性化虚拟化身工作台：上传 1–N 张人像/局部特写 → 衣下素体生成（MVP 为 mock GPU 管线）→ Web 姿态/表情/身材编辑 → 导出标准 GLB + Pose JSON。

## 仓库结构

```text
apps/
  api/          FastAPI + 可插拔推理管线（默认 inline mock）
  web/          React + Three.js 工作台
packages/
  contracts/    骨骼 / Morph / Pose JSON 合约
data/           本地上传与任务产物（gitignore）
```

## 快速启动

### 1. API

```bash
cd apps/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

健康检查：`GET http://127.0.0.1:8000/api/health`  
演示 GLB：`POST http://127.0.0.1:8000/api/demo/generate-glb`

### 2. Web

```bash
cd apps/web
npm install
npm run dev
```

打开 `http://127.0.0.1:5173`（Vite 已将 `/api` 代理到 `:8000`）。

### 3. 端到端演示

1. 点击「加载演示模型」，或上传 1–10 张 JPG/PNG/WEBP  
2. 等待任务完成（mock 管线约 1 秒）  
3. 在右侧工作台调整姿态 / 手势 / 表情 / 身材 / 分区遮罩  
4. 导出 `h3d-avatar.glb` 与 `h3d-pose.json`

## 管线阶段（可替换）

| 阶段 | 接口 | MVP 实现 |
|------|------|----------|
| 图像分流 | `ImageParseRouter` | 启发式 mock（DensePose/SCHP 桩） |
| 衣下体型 | `UnderClothingEstimator` | 图像哈希 → β ∈ ℝ¹⁰（SHAPY 桩） |
| 局部融合 | `LocalDetailMerger` | 记录区域与虚拟相机（PnP/泊松桩） |
| 资产组装 | `AssetAssembler` | 程序化 skinned GLB（59 骨 + morph + `_PART_ID`） |

Redis/Celery：见 `app/pipeline/celery_app.py`；开发默认 `USE_INLINE_PIPELINE=True`（线程内跑任务）。

## 资产合约（摘要）

- 59 joints：55 基础骨 + 4 辅助骨（`Spine3_Breast_*`, `Pelvis_Glute_*`）  
- Morph：`breast_scale` / `waist_scale` / `glute_scale` / `thigh_scale` + 表情通道  
- 顶点属性 `_PART_ID`：0–7 身体分区遮罩  
- Pose JSON：`packages/contracts/pose.schema.json`

> MVP 网格为 SMPL-X **拓扑契约兼容的简化人体**（约 1.2k 顶点），非完整 10,475 顶点官方拓扑；换真实 SMPL-X 权重时保持同名骨骼 / morph / `_PART_ID` 即可。
