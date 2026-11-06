# H3D API

FastAPI gateway + mockable inference pipeline.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Key routes: `POST /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/avatar.glb`, `POST /api/demo/generate-glb`.
