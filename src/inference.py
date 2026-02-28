import argparse
import json
import joblib
import pandas as pd

from config import MODEL_PATH
from features import compute_features


def load_model():
    return joblib.load(MODEL_PATH)


def predict_single(model, series):
    features = compute_features(series)
    features_df = pd.DataFrame([features])

    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1]

    label_map = {0: "NO_WORK_ORDER", 1: "WO"}

    return {
        "prediction": label_map[prediction],
        "probability": round(float(probability), 4)
    }


def predict_bulk(model, series_list):
    results = []
    for series in series_list:
        result = predict_single(model, series)
        results.append(result)
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Work Order Prediction Inference")

    parser.add_argument(
        "--json",
        type=str,
        help="Path to JSON file containing input data"
    )

    args = parser.parse_args()

    model = load_model()

    if not args.json:
        raise ValueError("Please provide --json input file")

    with open(args.json, "r") as f:
        input_data = json.load(f)

    # Single upload
    if "series" in input_data:
        result = predict_single(model, input_data["series"])
        print(json.dumps(result, indent=2))

    # Bulk upload
    elif "series_list" in input_data:
        result = predict_bulk(model, input_data["series_list"])
        print(json.dumps(result, indent=2))

    else:
        raise ValueError("JSON must contain 'series' or 'series_list'")
