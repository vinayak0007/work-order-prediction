import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression


def compute_features(timeseries):
    """
    Extract statistical features from raw temperature time series.
    Works for:
    - list of dicts (training data)
    - list of floats (CLI input)
    """

    # Handle both training format (list of dicts) and CLI format (list of floats)
    if isinstance(timeseries[0], dict):
        temps = np.array([point["temperature"] for point in timeseries])
    else:
        temps = np.array(timeseries)

    features = {}

    features["mean_temp"] = np.mean(temps)
    features["std_temp"] = np.std(temps)
    features["min_temp"] = np.min(temps)
    features["max_temp"] = np.max(temps)
    features["range_temp"] = np.max(temps) - np.min(temps)
    features["skew_temp"] = skew(temps)
    features["kurtosis_temp"] = kurtosis(temps)
    features["num_points"] = len(temps)

    # Linear trend
    X = np.arange(len(temps)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, temps)

    features["trend_slope"] = model.coef_[0]
    features["trend_r2"] = model.score(X, temps)

    return features
