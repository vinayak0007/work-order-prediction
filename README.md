# Work Order Prediction System

## 📌 Problem Statement

The objective of this project is to build a machine learning system that predicts whether a device requires a **Work Order (WO)** based on its temperature time series data.

Each device contains a time series of temperature readings. The task is to classify the device into:

- `NO_WORK_ORDER`
- `WO`

---

## 🧠 Approach Overview

The solution is structured as a modular, production-style ML pipeline with:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training pipeline
- JSON-based inference pipeline (single & bulk support)

---

## 📊 Exploratory Data Analysis (EDA)

EDA was performed inside the `notebooks/` directory and includes:

- Dataset shape and schema inspection
- Class distribution analysis (60–40 split)
- Time series length distribution
- Feature distribution by class
- Volatility analysis
- Leakage validation via shuffled-label test

### 🔍 Key Finding

Devices labeled as `WO` exhibit significantly higher:

- Standard deviation
- Temperature range
- Extreme max values

This creates strong separability between the two classes.

---

## ⚙️ Feature Engineering

All feature engineering is centralized in:


src/features.py


Extracted features include:

- Mean temperature
- Standard deviation
- Min / Max
- Range
- Skewness
- Kurtosis
- Number of points
- Linear trend slope
- Trend R² score

This ensures consistency between training and inference.

---

## 🤖 Model

Model used:

- **Logistic Regression**
- `class_weight="balanced"`
- Standard scaling via `Pipeline`

Evaluation Metrics:

- Precision
- Recall
- F1-score
- ROC-AUC

Observed performance:

- ROC-AUC: 1.0
- Perfect separation due to strong volatility features

---

## 🏗 Project Structure


work_order_project/
│
├── bulk_input.json
├── single_input.json
├── data/
│ └── training_data.pkl
├── models/
│ └── model.pkl
├── notebooks/
│ └── eda_modeling.ipynb
├── requirements.txt
└── src/
├── config.py
├── features.py
├── train.py
└── inference.py


---

## 🚀 Training Pipeline

To train the model:

```bash
cd src
python train.py

This will:

Load data

Extract features

Train model

Evaluate performance

Save model to models/model.pkl

🔮 Inference Pipeline

Inference accepts JSON input and supports both:

Single upload

Bulk upload

▶ Single Prediction

Example JSON:

{
  "series": [50, 51, 52, 49, 50]
}

Run:

cd src
python inference.py --json ../single_input.json

Output:

{
  "prediction": "NO_WORK_ORDER",
  "probability": 0.011
}
▶ Bulk Prediction

Example JSON:

{
  "series_list": [
    [50, 51, 52],
    [110, 75, 80, 95, 60, 85, 120]
  ]
}

Run:

cd src
python inference.py --json ../bulk_input.json

Output:

[
  {"prediction": "NO_WORK_ORDER", "probability": 0.011},
  {"prediction": "WO", "probability": 1.0}
]
📦 Installation

Install dependencies:

pip install -r requirements.txt
🛡 Design Decisions

Centralized feature logic to prevent training–inference drift

Used pipeline to avoid data leakage from scaling

Performed shuffled-label sanity check

Modular structure separating EDA and production code

JSON-based inference for deployment readiness

⚠️ Limitations

Dataset is strongly separable due to volatility features

Perfect ROC-AUC suggests deterministic behavior

Does not currently implement temporal forecasting (uses full series)
