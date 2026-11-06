# H3D API

FastAPI gateway + **real** (default) / mock inference pipeline.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python ../../scripts/fetch_models.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Env:

| Variable | Default | Meaning |
|----------|---------|---------|
| `H3D_PIPELINE` | `real` | `real` or `mock` |
| `H3D_DEVICE` | `cpu` | `cpu` / `cuda` |
| `H3D_SMPLX_DIR` | `data/models` | Parent of `smplx/` weights |
| `H3D_DENSEPOSE` | `smplx_iuv` | Dense correspondence backend |
| `H3D_INLINE_PIPELINE` | `1` | Inline thread worker vs Celery |

Routes: `POST /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/avatar.glb`, `POST /api/demo/generate-glb`, `GET /api/health`.
