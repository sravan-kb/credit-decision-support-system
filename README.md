---
title: Credit Decision Support System
emoji: 📚
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
short_description: AI-powered credit risk classifier (P1–P4)
---

# Credit Decision Support System

**by Sravan Branwal**

AI-powered credit risk classification system that predicts a customer's
credit risk category — **P1 (Very Low Risk) to P4 (High Risk)** — using
internal banking data and external CIBIL bureau data.

🔗 **[Live Demo](https://huggingface.co/spaces/sravankb/credit-decision-support-system)**

*CDAC Thiruvananthapuram — PGCP Big Data Analytics, Mini Project (Feb 2026 batch)*

---

## Overview

This system helps financial institutions assess creditworthiness by
analyzing credit score, repayment history, loan enquiries, credit
utilization, income, and employment details. It's built as a **decision
support tool** — designed to help a loan officer make a faster, more
consistent, explainable decision, not to replace human judgment.

## Features

- 📄 **Bulk CSV Prediction** — the core feature. Upload a customer dataset and get a predicted risk category (P1–P4) for every row, along with SHAP-based explanations, product-wise risk breakdowns, and enquiry pattern insights — all computed automatically across the batch.
- ⬇️ **Downloadable Results** — after processing, the full predicted dataset (every row, not just a preview) is available as a one-click CSV download, ready to hand off or analyze further.
- 🔍 **SHAP Explainability** — predictions come with the top factors that actually drove them, not just a bare label.
- 👤 **Single Customer Prediction** — a lightweight 5-field form for quick, one-off lookups when a full dataset isn't available.

## How Bulk Prediction Works

1. **Upload** a CSV of customer records (see the dataset section below for expected columns)
2. The **Full Model** predicts a risk category (P1–P4) for every row
3. View instant summary stats: total customers, category breakdown, and a live chart
4. Explore automatically generated insights — which products carry the most risk, and how enquiry activity correlates with risk category
5. **Download** the complete results as a CSV, with the predicted category added as a new column

## Architecture

Two XGBoost models — the **Full Model** handles the primary bulk workflow with nearly the entire feature set; a lightweight **Quick Model** exists as a fallback for one-off lookups with limited data:

| | Full Model | Quick Model |
|---|---|---|
| **Features used** | ~60 (nearly the full dataset) | 5 (Credit Score, Income, Age, Employment, Education) |
| **Powers** | Bulk CSV upload | Single-customer form |
| **Accuracy** | 99.51% | 99.40% |

## Model Selection

Four algorithms were trained and compared on identical train/test splits:

| Model | Accuracy |
|---|---|
| Logistic Regression | 66.48% |
| Random Forest | 98.82% |
| Decision Tree | 99.33% |
| **XGBoost** ✅ | **99.51%** |

XGBoost was chosen over the closely-tied Decision Tree for better
generalization to unseen data and native SHAP compatibility.

## Key Finding

Credit Score drives roughly **91% of the model's decisions** — verified
through a feature ablation test: removing it drops accuracy from 99.5% to
75.3%. This reflects Credit Score's role as a bureau-computed summary of
much of the other behavioral data (delinquency, utilization, enquiries),
not a weakness in the model.

## Dataset

**Source:** [Leading Indian Bank & CIBIL Real-World Dataset](https://www.kaggle.com/datasets/saurabhbadole/leading-indian-bank-and-cibil-real-world-dataset) (Kaggle)

51,336 customer records, 60 predictive features, combining a bank's internal
data with external CIBIL bureau data. Target: `Approved_Flag` (P1–P4).

## Tech Stack

**Python** · **FastAPI** · **XGBoost** · **scikit-learn** · **SHAP** · **Docker** · **Hugging Face Spaces**

## Running Locally

```bash
git clone https://github.com/sravan-kb/credit-decision-support-system.git
cd credit-decision-support-system

uv sync
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

---

**Sravan Kumar Branwal**
CDAC Thiruvananthapuram — PGCP Big Data Analytics