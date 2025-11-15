import pandas as pd
import os
import numpy as np
import math

DATA_PATH = os.path.join("..", "data", "loan_data.csv")

def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.read_csv(os.path.join("data", "loan_data.csv"))

    return df


def clean_data(df):
    # Rename for consistency
    df = df.rename(columns={
        "person_gender": "gender",
        "person_education": "education",
        "person_home_ownership": "home_ownership",
        "loan_intent": "loan_purpose",
        "loan_amnt": "loan_amount",
        "cb_person_cred_hist_length": "credit_history_yrs"
    })

    # Create buckets for credit history to simulate "time" for ECL curve
    df["credit_history_bucket"] = pd.cut(
        df["credit_history_yrs"],
        bins=[0, 1, 2, 3, 4, 5, 10],
        labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5+"],
        include_lowest=True
    )

    # Create income bucket
    df["income_bucket"] = pd.cut(
        df["person_income"],
        bins=[0, 20000, 50000, 100000, 200000],
        labels=["low", "medium", "high", "very_high"],
        include_lowest=True
    )

    return df


def compute_segment_ecl(df, segment_col):
    """
    Returns a dataframe with PD, exposure, LGD (fixed 0.6), ECL per segment.
    Sanitizes NaN/inf values so the result is JSON-safe.
    """
    LGD = 0.6  # fixed for assignment simplicity

    # compute raw summary
    summary = df.groupby(segment_col).apply(
        lambda x: pd.Series({
            "total_loans": len(x),
            "defaults": int((x["loan_status"] == 0).sum()),
            "PD": float((x["loan_status"] == 0).mean()) if len(x) > 0 else 0.0,
            "exposure": float(x["loan_amount"].sum())
        })
    )

    # attach LGD and compute ECL
    summary["LGD"] = LGD
    # compute ECL, guard against invalid math
    summary["ECL"] = summary["PD"] * summary["LGD"] * summary["exposure"]

    # Replace infinities and NaNs with zeros (JSON-safe)
    summary = summary.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Ensure types are Python native (no numpy types)
    def to_python_val(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            # if not finite, return 0.0
            return float(v) if np.isfinite(v) else 0.0
        if pd.isna(v):
            return 0
        return v

    summary = summary.applymap(to_python_val)

    return summary.reset_index()

def compute_curve_for_segment(df, segment_col, segment_value):
    """
    Safely compute ECL per credit_history_bucket for a given segment value.
    Returns a list of dicts with sanitized numeric values.
    """
    if segment_col not in df.columns:
        return []

    df = df.copy()
    df[segment_col] = df[segment_col].astype(str)
    seg_val_str = str(segment_value)

    mask = df[segment_col].str.strip().str.upper() == seg_val_str.strip().upper()
    df_sub = df[mask]
    if df_sub.empty:
        return []

    if "credit_history_bucket" not in df_sub.columns and "credit_history_yrs" in df_sub.columns:
        df_sub["credit_history_bucket"] = pd.cut(
            df_sub["credit_history_yrs"],
            bins=[0, 1, 2, 3, 4, 5, 10],
            labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5+"],
            include_lowest=True
        )

    bucket_order = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]
    rows = []
    for b in bucket_order:
        df_b = df_sub[df_sub["credit_history_bucket"] == b]
        if df_b.empty:
            rows.append({
                "credit_history_bucket": b,
                "total_loans": 0,
                "defaults": 0,
                "PD": 0.0,
                "exposure": 0.0,
                "LGD": 0.6,
                "ECL": 0.0
            })
        else:
            # compute per-bucket summary using the sanitized compute_segment_ecl
            summary = compute_segment_ecl(df_b, "credit_history_bucket")
            # summary will have one row for this bucket; find it
            rec = summary[summary["credit_history_bucket"] == b]
            if not rec.empty:
                r = rec.iloc[0].to_dict()
                rows.append(r)
            else:
                # fallback computation with sanitization
                total_loans = int(len(df_b))
                defaults = int((df_b["loan_status"] == 0).sum())
                PD = float(defaults/total_loans) if total_loans>0 else 0.0
                exposure = float(df_b["loan_amount"].sum())
                LGD = 0.6
                ECL = PD * LGD * exposure
                # sanitize values
                if not np.isfinite(PD): PD = 0.0
                if not np.isfinite(exposure): exposure = 0.0
                if not np.isfinite(ECL): ECL = 0.0
                rows.append({
                    "credit_history_bucket": b,
                    "total_loans": total_loans,
                    "defaults": defaults,
                    "PD": PD,
                    "exposure": exposure,
                    "LGD": LGD,
                    "ECL": ECL
                })

    # final safety: replace any NaN/inf in rows
    safe_rows = []
    for r in rows:
        safe = {}
        for k, v in r.items():
            if isinstance(v, (int, np.integer)):
                safe[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                # convert non-finite to 0.0
                safe[k] = float(v) if np.isfinite(v) else 0.0
            elif pd.isna(v):
                safe[k] = 0
            else:
                safe[k] = v
        safe_rows.append(safe)

    return safe_rows


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)

    print("\n=== Cleaned Columns ===")
    print(df.columns)

    print("\n=== Example: ECL by loan_purpose ===")
    print(compute_segment_ecl(df, "loan_purpose"))
