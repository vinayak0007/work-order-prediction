# Work Order Prediction System

An end-to-end machine learning pipeline that predicts whether a device requires a **Work Order (WO)** based on its temperature time-series data.

This project demonstrates structured ML development with modular feature engineering, reproducible training, and JSON-based inference supporting both single and bulk predictions.

---

## 📌 Problem Statement

Given a device’s temperature time-series data, classify whether it requires:

- `NO_WORK_ORDER`
- `WO`

Each device contains a sequence of temperature readings. The goal is to engineer meaningful features from this time-series data and build a reliable classification system.

---

## Solution Overview

The system is structured as a production-style ML pipeline consisting of:

- Exploratory Data Analysis (EDA)
- Centralized Feature Engineering
- Reproducible Training Pipeline
- JSON-Based Inference (Single & Bulk Support)
- Clean Modular Project Architecture

---

## Exploratory Data Analysis (EDA)

EDA was conducted in `notebooks/task_notebook.ipynb` and includes:

- Dataset inspection and schema validation
- Class distribution analysis (≈ 60–40 split)
- Time-series length distribution
- Feature distribution comparison by class
- Volatility pattern analysis
- Shuffled-label sanity check to validate absence of leakage

### Key Insight

Devices labeled `WO` exhibit significantly higher:

- Standard deviation
- Temperature range
- Extreme maximum values

This volatility-driven behavior creates strong separability between classes.

---

## Feature Engineering

All feature logic is centralized in:

`src/features.py`

Extracted features include:

- Mean temperature  
- Standard deviation  
- Minimum and Maximum  
- Temperature range  
- Skewness  
- Kurtosis  
- Number of points  
- Linear trend slope  
- Trend R² score  

Centralizing feature engineering ensures:

- No training–inference drift  
- Clean separation of concerns  
- Reusability across pipelines  

---

## Model

Model Used: **Logistic Regression**

Configuration:

- `class_weight="balanced"`
- Standard scaling via `sklearn.Pipeline`

### Evaluation Metrics

- Precision  
- Recall  
- F1-score  
- ROC-AUC  

### Observed Performance

- ROC-AUC: **1.0**

The high performance is driven by strong volatility-based class separation within the dataset.

---

## Project Structure

```
work_order_project/
│
├── README.md
├── requirements.txt
├── bulk_input.json
├── single_input.json
│
├── data/
│   └── training_data.pkl
│
├── notebooks/
│   └── task_notebook.ipynb
│
└── src/
    ├── config.py
    ├── features.py
    ├── train.py
    └── inference.py
```

---

## Training Pipeline

To train the model:

```bash
cd src
python train.py
```

This will:

- Load dataset  
- Compute engineered features  
- Perform train/test split  
- Train the model  
- Evaluate performance  
- Save model to `models/model.pkl`  

---

## Inference Pipeline

The inference system accepts JSON input and supports:

- Single device prediction  
- Bulk device prediction  

---

### ▶ Single Prediction

Example input (`single_input.json`):

```json
{
  "series": [50, 51, 52, 49, 50]
}
```

Run:

```bash
cd src
python inference.py --json ../single_input.json
```

Example Output:

```json
{
  "prediction": "NO_WORK_ORDER",
  "probability": 0.011
}
```

---

### ▶ Bulk Prediction

Example input (`bulk_input.json`):

```json
{
  "series_list": [
    [50, 51, 52],
    [110, 75, 80, 95, 60, 85, 120]
  ]
}
```

Run:

```bash
cd src
python inference.py --json ../bulk_input.json
```

Example Output:

```json
[
  {"prediction": "NO_WORK_ORDER", "probability": 0.011},
  {"prediction": "WO", "probability": 1.0}
]
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Design Decisions

- Centralized feature engineering to prevent training–inference inconsistency  
- Used `Pipeline` to avoid data leakage from scaling  
- Performed shuffled-label sanity check to validate robustness  
- Modular separation between EDA and production code  
- JSON-based inference to simulate deployment readiness  

---

## Limitations

- Dataset is strongly separable due to volatility-driven features  
- Perfect ROC-AUC suggests near-deterministic separation  
- Model uses full-series aggregation (no temporal forecasting)

---

## Potential Improvements

- Cross-validation for stronger robustness assessment  
- Model comparison (Random Forest, Gradient Boosting)  
- API wrapper using FastAPI  
- Dockerization for deployment  
- Monitoring and drift detection  

---

## Author

Vinayak Pushkar  
