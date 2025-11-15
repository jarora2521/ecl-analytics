# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from services import ecl_service
from pydantic import BaseModel
import uvicorn
from fastapi import Body

# Simple in-memory saved reports store (keeps for process life)
SAVED_REPORTS = []


app = FastAPI(title="ECL Analytics API")

# === CORS (allow local frontend to call API) ===
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "https://<YOUR_VERCEL_FRONTEND_DOMAIN>" 
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and clean once on startup
DF = ecl_service.load_data()
DF = ecl_service.clean_data(DF)

# Helper: possible segment columns we support
AVAILABLE_SEGMENTS = [
    "gender",
    "education",
    "home_ownership",
    "loan_purpose",
    "income_bucket",
    "credit_history_bucket"
]

# Simple in-memory "users" for demo login (hardcoded for speed)
# In real world, you'd use hashed passwords and a DB
USERS = {
    "analyst": {"password": "analyst123", "role": "analyst"},
    "cro": {"password": "cro123", "role": "cro"}
}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(req: LoginRequest):
    """
    Simple demo login.
    POST JSON: {"username": "analyst", "password": "analyst123"}
    Returns: {"role": "analyst", "token": "analyst"} (token is just the role string for demo)
    """
    username = req.username
    password = req.password
    user = USERS.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"role": user["role"], "token": user["role"]}

# Create a credit_score bucket if user asks (not in AVAILABLE_SEGMENTS by default)
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
    """Return list of supported segment columns"""
    return {"segments": AVAILABLE_SEGMENTS}

@app.get("/ecl/{segment_col}")
def ecl_by_segment(segment_col: str):
    """
    Returns ECL summary grouped by the requested segment_col.
    Example: /ecl/loan_purpose
    """
    if segment_col not in AVAILABLE_SEGMENTS and segment_col != "credit_score_bucket":
        raise HTTPException(status_code=400, detail=f"Unsupported segment. Use /segments to see supported columns.")
    df = DF.copy()
    if segment_col == "credit_score_bucket":
        df = bucket_credit_score(df)
    summary = ecl_service.compute_segment_ecl(df, segment_col)
    # Convert to JSON-friendly
    return {"segment": segment_col, "summary": summary.to_dict(orient="records")}

@app.get("/curve/{segment_col}/{segment_value}")
def ecl_curve(segment_col: str, segment_value: str):
    """
    Returns ECL across credit_history_bucket for a specific segment value.
    Example: /curve/loan_purpose/PERSONAL
    """
    if segment_col not in AVAILABLE_SEGMENTS and segment_col != "credit_score_bucket":
        raise HTTPException(status_code=400, detail="Unsupported segment. Use /segments to see supported columns.")
    df = DF.copy()
    if segment_col == "credit_score_bucket":
        df = bucket_credit_score(df)
    # Filter rows that match the segment value
    if df[segment_col].dtype == "object" or pd.api.types.is_categorical_dtype(df[segment_col]):
        mask = df[segment_col].astype(str).str.upper() == str(segment_value).upper()
    else:
        mask = df[segment_col] == segment_value

    df_sub = df[mask]
    if df_sub.empty:
        raise HTTPException(status_code=404, detail="No rows found for that segment value.")

    # Compute ECL per credit_history_bucket (our pseudo-time)
    curve = ecl_service.compute_segment_ecl(df_sub, "credit_history_bucket")
    # Return buckets in order (fill missing buckets with zeros)
    return {"segment_col": segment_col, "segment_value": segment_value, "curve": curve.to_dict(orient="records")}

@app.post("/reports")
def save_report(payload: dict = Body(...)):
    """
    Save a small JSON report (in-memory). Payload should include:
    { "name": "My report name", "segment": "gender", "segment_value": "female", "summary": [...] }
    Returns the saved report with server timestamp/id.
    """
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
    """
    List all saved reports (most recent last).
    """
    return {"reports": SAVED_REPORTS}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
