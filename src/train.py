import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE, LABEL_MAPPING
from features import compute_features


def main():

    # Load dataset
    df = pd.read_pickle(DATA_PATH)

    # Feature extraction
    feature_df = df["timeseries_data"].apply(compute_features)
    feature_df = pd.DataFrame(feature_df.tolist())

    # Map labels to numeric
    feature_df["label"] = df["label"].map(LABEL_MAPPING)

    X = feature_df.drop(columns=["label"])
    y = feature_df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    print("ROC-AUC:", roc_auc_score(y_test, probs))

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
