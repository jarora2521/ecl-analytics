# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import os
import pandas as pd
import uvicorn
from fastapi import Body

from services import ecl_service

# Simple in-memory saved reports store (keeps for process life)
SAVED_REPORTS = []

app = FastAPI(title="ECL Analytics API")

# === CORS (allow local frontend to call API) ===
# You can set FRONTEND_ORIGIN env var to your production frontend (e.g. https://my-site.netlify.app)
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]
frontend_origin = os.getenv("FRONTEND_ORIGIN")
if frontend_origin:
    origins.append(frontend_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- serve frontend if present ----------------
def _find_frontend_index():
    here = Path(__file__).resolve().parent      # backend/
    repo_root = here.parent.resolve()           # repo root
    candidates = [
        repo_root / "frontend" / "index.html",
        repo_root / "frontend" / "dist" / "index.html",
        repo_root / "frontend" / "build" / "index.html",
        repo_root / "build" / "index.html",
        repo_root / "static" / "index.html",
        repo_root / "index.html",
        here / "frontend" / "index.html",
    ]
    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            pass
        if p.exists():
            return (str(p), str(p.parent))
    return (None, None)

INDEX_HTML, FRONTEND_DIR = _find_frontend_index()
if INDEX_HTML and FRONTEND_DIR:
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(INDEX_HTML)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        fav = Path(FRONTEND_DIR) / "favicon.ico"
        if fav.exists():
            return FileResponse(str(fav))
        raise HTTPException(status_code=404, detail="favicon not found")

    print(f"[frontend] Serving frontend index from: {INDEX_HTML}")
else:
    @app.get("/", include_in_schema=False)
    def root_info():
        return {"message": "API running. Visit /docs or /redoc for API docs."}
    print("[frontend] No frontend found — API-only service")

# ---------------- Data loading on startup (no import-time loading) ----------------
def _get_df_or_500():
    df = getattr(app.state, "df", None)
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    return df

@app.on_event("startup")
def startup_event():
    try:
        df = ecl_service.load_data()       # loader will honor DATA_PATH env var if set
        df = ecl_service.clean_data(df)
        app.state.df = df
        print("Startup: dataset loaded, rows:", len(df))
    except Exception as exc:
        print("Startup: failed to load dataset:", exc)
        # fail fast so Render shows the error (change if you want to keep app up)
        raise

# small debug endpoint to inspect paths (helpful on Render)
@app.get("/debug/paths")
def debug_paths():
    cwd = os.getcwd()
    data_exists = os.path.exists(os.path.join(cwd, "data", "loan_data.csv"))
    return {
        "cwd": cwd,
        "data_exists_in_cwd_data": data_exists,
        "env_DATA_PATH": os.getenv("DATA_PATH"),
        "env_FRONTEND_ORIGIN": os.getenv("FRONTEND_ORIGIN")
    }

# ---------------- API logic ----------------
AVAILABLE_SEGMENTS = [
    "gender",
    "education",
    "home_ownership",
    "loan_purpose",
    "income_bucket",
    "credit_history_bucket"
]

USERS = {
    "analyst": {"password": "analyst123", "role": "analyst"},
    "cro": {"password": "cro123", "role": "cro"}
}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(req: LoginRequest):
    username = req.username
    password = req.password
    user = USERS.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"role": user["role"], "token": user["role"]}

def bucket_credit_score(df):
    df = df.copy()
    df["credit_score_bucket"] = pd.cut(
        df["credit_score"],
        bins=[0, 500, 600, 700, 800, 1000],
        labels=["<500", "500-599", "600-699", "700-799", "800+"],
        include_lowest=True
    )
    return df

@app.get("/segments")
def get_segments():
    return {"segments": AVAILABLE_SEGMENTS}

@app.get("/ecl/{segment_col}")
def ecl_by_segment(segment_col: str):
    if segment_col not in AVAILABLE_SEGMENTS and segment_col != "credit_score_bucket":
        raise HTTPException(status_code=400, detail=f"Unsupported segment. Use /segments to see supported columns.")
    df = _get_df_or_500().copy()
    if segment_col == "credit_score_bucket":
        df = bucket_credit_score(df)
    summary = ecl_service.compute_segment_ecl(df, segment_col)
    return {"segment": segment_col, "summary": summary.to_dict(orient="records")}

@app.get("/curve/{segment_col}/{segment_value}")
def ecl_curve(segment_col: str, segment_value: str):
    if segment_col not in AVAILABLE_SEGMENTS and segment_col != "credit_score_bucket":
        raise HTTPException(status_code=400, detail="Unsupported segment. Use /segments to see supported columns.")
    df = _get_df_or_500().copy()
    if segment_col == "credit_score_bucket":
        df = bucket_credit_score(df)
    if df[segment_col].dtype == "object" or pd.api.types.is_categorical_dtype(df[segment_col]):
        mask = df[segment_col].astype(str).str.upper() == str(segment_value).upper()
    else:
        mask = df[segment_col] == segment_value
    df_sub = df[mask]
    if df_sub.empty:
        raise HTTPException(status_code=404, detail="No rows found for that segment value.")
    curve = ecl_service.compute_segment_ecl(df_sub, "credit_history_bucket")
    return {"segment_col": segment_col, "segment_value": segment_value, "curve": curve.to_dict(orient="records")}

@app.post("/reports")
def save_report(payload: dict = Body(...)):
    import time, uuid
    rec = {
        "id": str(uuid.uuid4()),
        "ts": int(time.time()),
        "name": payload.get("name", f"report_{len(SAVED_REPORTS)+1}"),
        "payload": payload
    }
    SAVED_REPORTS.append(rec)
    return {"status": "ok", "report": rec}

@app.get("/reports")
def list_reports():
    return {"reports": SAVED_REPORTS}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
