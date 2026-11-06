# H3D API

FastAPI gateway. **Default pipeline is `real`** (`H3D_PIPELINE` defaults to `real` when unset).

```bash
python3 -m venv .venv && source .venv/bin/activate

# macOS (MPS-capable wheels):
pip install torch torchvision
# Linux CPU example:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Startup (when `H3D_PIPELINE=real`): auto-downloads missing weights via `app.models_fetch` unless `H3D_AUTO_FETCH_MODELS=0`. Same logic as `scripts/fetch_models.py`.

Env:

| Variable | Default | Meaning |
|----------|---------|---------|
| `H3D_PIPELINE` | **`real`** | `real` or `mock` |
| `H3D_DEVICE` | **`auto`** | `auto` / `mps` / `cpu` / `cuda` |
| `H3D_AUTO_FETCH_MODELS` | `1` | Auto-download missing weights on startup |
| `H3D_SMPLX_DIR` | `data/models` | Parent of `smplx/` weights |
| `H3D_DENSEPOSE` | `smplx_iuv` | Dense correspondence backend |
| `H3D_INLINE_PIPELINE` | `1` | Inline thread worker vs Celery |

`GET /api/health` reports `backend`, `defaultPipeline: "real"`, `device`, and `models`.
