from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import joblib
import pandas as pd
import numpy as np
import shap

from pathlib import Path

app = FastAPI(title="Credit Decision Support System")

# ============================================================
# Load Models
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgb_pipeline.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

QUICK_MODEL_PATH = BASE_DIR / "models" / "xgb_quick.pkl"
QUICK_LABEL_PATH = BASE_DIR / "models" / "label_encoder_quick.pkl"

quick_model = joblib.load(QUICK_MODEL_PATH)
quick_label_encoder = joblib.load(QUICK_LABEL_PATH)

print("✅ Models Loaded Successfully")

# ============================================================
# SHAP Explainers
# ============================================================

quick_preprocessor = quick_model.named_steps["preprocessor"]
quick_xgb_model = quick_model.named_steps["model"]
quick_explainer = shap.TreeExplainer(quick_xgb_model)

bulk_preprocessor = model.named_steps["preprocessor"]
bulk_xgb_model = model.named_steps["model"]
bulk_explainer = shap.TreeExplainer(bulk_xgb_model)

print("✅ SHAP Explainers Ready")

# ============================================================
# Human-readable display names
# ============================================================

DISPLAY_NAMES = {
    "Credit_Score": "Credit Score",
    "NETMONTHLYINCOME": "Monthly Income",
    "AGE": "Age",
    "Time_With_Curr_Empr": "Years with Current Employer",
    "EDUCATION_12TH": "Education (12th)",
    "EDUCATION_GRADUATE": "Education (Graduate)",
    "EDUCATION_OTHERS": "Education (Other)",
    "EDUCATION_POST-GRADUATE": "Education (Post-Graduate)",
    "EDUCATION_PROFESSIONAL": "Education (Professional)",
    "EDUCATION_SSC": "Education (SSC)",
    "EDUCATION_UNDER GRADUATE": "Education (Under Graduate)",
    "tot_enq": "Total Loan Enquiries",
    "enq_L3m": "Enquiries in Last 3 Months",
    "enq_L6m": "Enquiries in Last 6 Months",
    "enq_L12m": "Enquiries in Last 12 Months",
    "CC_enq": "Credit Card Enquiries",
    "CC_enq_L6m": "Credit Card Enquiries (6 Months)",
    "CC_enq_L12m": "Credit Card Enquiries (12 Months)",
    "PL_enq": "Personal Loan Enquiries",
    "PL_enq_L6m": "Personal Loan Enquiries (6 Months)",
    "PL_enq_L12m": "Personal Loan Enquiries (12 Months)",
    "time_since_recent_enq": "Time Since Most Recent Enquiry",
    "pct_of_active_TLs_ever": "% of Accounts Currently Active",
    "pct_opened_TLs_L6m_of_L12m": "% of Accounts Opened Recently (6 of 12 Months)",
    "pct_currentBal_all_TL": "% Current Balance Across All Accounts",
    "CC_utilization": "Credit Card Utilization",
    "CC_Flag": "Has a Credit Card",
    "PL_utilization": "Personal Loan Utilization",
    "PL_Flag": "Has a Personal Loan",
    "HL_Flag": "Has a Home Loan",
    "GL_Flag": "Has a Gold Loan",
    "max_unsec_exposure_inPct": "Max Unsecured Loan Exposure (%)",
    "num_times_delinquent": "Number of Missed Payments (Ever)",
    "max_delinquency_level": "Worst Missed-Payment Severity",
    "max_recent_level_of_deliq": "Recent Missed-Payment Severity",
    "recent_level_of_deliq": "Most Recent Missed-Payment Level",
    "num_deliq_6mts": "Missed Payments (Last 6 Months)",
    "num_deliq_12mts": "Missed Payments (Last 12 Months)",
    "num_times_30p_dpd": "Payments 30+ Days Late (Count)",
    "num_times_60p_dpd": "Payments 60+ Days Late (Count)",
    "time_since_recent_payment": "Time Since Last Payment",
    "time_since_first_deliquency": "Time Since First Missed Payment",
    "time_since_recent_deliquency": "Time Since Most Recent Missed Payment",
    "Total_TL": "Total Accounts",
    "Tot_Closed_TL": "Total Closed Accounts",
    "Tot_Active_TL": "Total Active Accounts",
    "MARITALSTATUS_Married": "Marital Status (Married)",
    "MARITALSTATUS_Single": "Marital Status (Single)",
    "GENDER_M": "Gender (Male)",
    "GENDER_F": "Gender (Female)",
}

INFLUENCE_LABELS = [
    (50, "Primary driver"),
    (15, "Strong influence"),
    (5, "Moderate influence"),
    (0, "Minor influence"),
]


def get_influence_label(bar_pct):
    """Converts a relative bar percentage into a plain-language influence label."""
    for threshold, label in INFLUENCE_LABELS:
        if bar_pct >= threshold:
            return label
    return "Minor influence"

RISK_LABELS = {
    "P1": "Very Low Risk",
    "P2": "Low Risk",
    "P3": "Medium Risk",
    "P4": "High Risk"
}


def clean_feature_name(raw_name):
    name = raw_name.replace("remainder__", "").replace("cat__", "")
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def get_confidence_breakdown(input_df):
    proba = quick_model.predict_proba(input_df)[0]
    classes = quick_label_encoder.classes_

    breakdown = []
    for cls, p in zip(classes, proba):
        breakdown.append({
            "category": cls,
            "label": RISK_LABELS.get(cls, cls),
            "percent": round(float(p) * 100, 1)
        })
    return breakdown


def get_global_top_factors(df, predicted_class_indices, top_n=5):
    transformed = bulk_preprocessor.transform(df)
    feature_names = bulk_preprocessor.get_feature_names_out()

    shap_values = bulk_explainer(transformed)
    n_samples = shap_values.values.shape[0]

    per_row_class_values = np.array([
        shap_values.values[i, :, predicted_class_indices[i]]
        for i in range(n_samples)
    ])

    mean_abs_impact = np.abs(per_row_class_values).mean(axis=0)

    ranked = list(zip(feature_names, mean_abs_impact))
    ranked.sort(key=lambda x: -x[1])
    top = ranked[:top_n]

    max_impact = max(val for _, val in top) if top else 1

    readable = []
    for name, val in top:
        clean_name = clean_feature_name(name)
        bar_pct = round((val / max_impact) * 100, 1) if max_impact > 0 else 0
        readable.append({
            "feature": clean_name,
            "avg_impact": round(float(val), 3),
            "bar_pct": bar_pct,
            "influence_label": get_influence_label(bar_pct)
        })
    return readable


def get_product_risk_breakdown(df):
    """
    Cross-tabs last_prod_enq2 (last product enquired for) against the
    predicted risk category, showing what percentage of each product's
    applicants fall into each P1-P4 bucket. Sorted by highest combined
    P3+P4 ("elevated risk") rate first, so the riskiest product segments
    show up at the top.
    """

    if "last_prod_enq2" not in df.columns:
        return []

    grouped = df.groupby("last_prod_enq2")["Prediction"].value_counts(normalize=True).unstack(fill_value=0)

    for cls in ["P1", "P2", "P3", "P4"]:
        if cls not in grouped.columns:
            grouped[cls] = 0.0

    grouped = grouped[["P1", "P2", "P3", "P4"]] * 100

    counts = df.groupby("last_prod_enq2").size()

    grouped["elevated_risk_pct"] = grouped["P3"] + grouped["P4"]
    grouped = grouped.sort_values("elevated_risk_pct", ascending=False)

    breakdown = []

    for product, row in grouped.iterrows():
        breakdown.append({
            "product": str(product),
            "total": int(counts.get(product, 0)),
            "p1_pct": round(row["P1"], 1),
            "p2_pct": round(row["P2"], 1),
            "p3_pct": round(row["P3"], 1),
            "p4_pct": round(row["P4"], 1),
            "elevated_risk_pct": round(row["elevated_risk_pct"], 1)
        })

    return breakdown


ENQUIRY_DISPLAY_NAMES = {
    "enq_L3m": "Enquiries in last 3 months",
    "enq_L6m": "Enquiries in last 6 months",
    "enq_L12m": "Enquiries in last 12 months"
}


def phrase_count(val):
    """Converts a raw average into a more readable approximate phrase."""
    if val < 0.5:
        return "less than 1"
    return str(round(val))


def get_enquiry_pattern_insight(df):
    """
    Shows average recent loan-enquiry activity (enq_L3m, enq_L6m, enq_L12m)
    broken down by predicted risk category, to reveal whether "credit hungry"
    behavior (many recent enquiries) correlates with higher risk categories
    in this uploaded batch. Also generates a plain-language headline
    highlighting the P1-to-P4 trend using the 12-month figure.
    """

    enquiry_cols = [c for c in ["enq_L3m", "enq_L6m", "enq_L12m"] if c in df.columns]

    if not enquiry_cols:
        return [], [], ""

    grouped = df.groupby("Prediction")[enquiry_cols].mean()
    grouped = grouped.reindex(["P1", "P2", "P3", "P4"]).fillna(0)

    max_val = grouped.values.max() if grouped.values.size else 1

    insight = []

    for category in ["P1", "P2", "P3", "P4"]:
        row = grouped.loc[category]
        entry = {
            "category": category,
            "label": RISK_LABELS[category],
            "metrics": {}
        }
        for col in enquiry_cols:
            val = float(row[col])
            entry["metrics"][col] = {
                "label": ENQUIRY_DISPLAY_NAMES.get(col, col),
                "avg": round(val, 2),
                "approx": phrase_count(val),
                "bar_pct": round((val / max_val) * 100, 1) if max_val > 0 else 0
            }
        insight.append(entry)

    # Auto-generate a headline using the longest-window column available
    headline = ""
    primary_col = "enq_L12m" if "enq_L12m" in enquiry_cols else enquiry_cols[-1]

    p1_val = grouped.loc["P1", primary_col]
    p4_val = grouped.loc["P4", primary_col]

    if p1_val > 0:
        multiplier = round(p4_val / p1_val, 1)
        headline = (
            f"On average, High Risk (P4) customers made about {phrase_count(p4_val)} "
            f"loan enquiries per person in the last 12 months, compared to about "
            f"{phrase_count(p1_val)} for Very Low Risk (P1) customers — roughly "
            f"{multiplier}x more."
        )

    return insight, enquiry_cols, headline


# ============================================================
# Static Files / Templates
# ============================================================

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ============================================================
# Routes
# ============================================================

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


@app.get("/single")
async def single(request: Request):
    return templates.TemplateResponse(
        request=request, name="single.html", context={"request": request}
    )


@app.get("/bulk")
async def bulk(request: Request):
    return templates.TemplateResponse(
        request=request, name="bulk.html", context={"request": request}
    )


@app.post("/predict_single")
async def predict_single(
    request: Request,
    Credit_Score: int = Form(...),
    NETMONTHLYINCOME: float = Form(...),
    AGE: int = Form(...),
    Time_With_Curr_Empr: float = Form(...),
    EDUCATION: str = Form(...)
):

    input_df = pd.DataFrame({
        "Credit_Score": [Credit_Score],
        "NETMONTHLYINCOME": [NETMONTHLYINCOME],
        "AGE": [AGE],
        "Time_With_Curr_Empr": [Time_With_Curr_Empr],
        "EDUCATION": [EDUCATION]
    })

    prediction_encoded = quick_model.predict(input_df)
    prediction = quick_label_encoder.inverse_transform(prediction_encoded)[0]

    confidence_breakdown = get_confidence_breakdown(input_df)

    return templates.TemplateResponse(
        request=request,
        name="result_single.html",
        context={
            "request": request,
            "prediction": prediction,
            "risk": RISK_LABELS[prediction],
            "confidence_breakdown": confidence_breakdown
        }
    )


@app.post("/predict_csv")
async def predict_csv(request: Request, file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    predictions_encoded = model.predict(df)
    predictions = label_encoder.inverse_transform(predictions_encoded)

    df["Prediction"] = predictions

    total_customers = len(df)
    prediction_counts = df["Prediction"].value_counts().to_dict()
    preview = df.head(20)

    global_top_factors = get_global_top_factors(
        df.drop(columns=["Prediction"]),
        predictions_encoded,
        top_n=5
    )

    product_risk_breakdown = get_product_risk_breakdown(df)

    enquiry_insight, enquiry_cols, enquiry_headline = get_enquiry_pattern_insight(df)

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "request": request,
            "total_customers": total_customers,
            "prediction_counts": prediction_counts,
            "table": preview,
            "global_top_factors": global_top_factors,
            "product_risk_breakdown": product_risk_breakdown,
            "enquiry_insight": enquiry_insight,
            "enquiry_cols": enquiry_cols,
            "enquiry_headline": enquiry_headline
        }
    )