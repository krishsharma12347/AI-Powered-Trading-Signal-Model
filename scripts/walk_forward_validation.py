import pandas as pd
import os

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# =========================
# FILE PATH
# =========================

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "labeled",
    "featured_btc_5m.csv"
)

# =========================
# LOAD DATA
# =========================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset Shape: {df.shape}")

# =========================
# TIMESTAMP CONVERSION
# =========================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# EXTRACT YEAR

df["year"] = df["timestamp"].dt.year

print("\nYears Found:")

print(df["year"].unique())

# =========================
# FEATURES
# =========================

FEATURE_COLUMNS = [

    "open",
    "high",
    "low",
    "close",
    "volume",

    "RSI_14",
    "EMA_20",
    "MACD",
    "MACD_signal",
    "MACD_hist",

    "BB_upper",
    "BB_middle",
    "BB_lower",

    "body_size",
    "upper_wick",
    "lower_wick",
    "candle_range",
    "body_range_ratio",

    "ema_distance",
    "price_momentum",

    "bb_width",
    "rolling_volatility",

    "rsi_momentum",
    "macd_momentum",

    "relative_volume"
]

TARGET_COLUMN = "target"

# =========================
# WALK-FORWARD SPLITS
# =========================

walk_forward_tests = [

    {
        "train_years": [2022],
        "test_year": 2023
    },

    {
        "train_years": [2022, 2023],
        "test_year": 2024
    },

    {
        "train_years": [2022, 2023, 2024],
        "test_year": 2025
    }
]

# =========================
# START TESTING
# =========================

results = []

for test_case in walk_forward_tests:

    train_years = test_case["train_years"]

    test_year = test_case["test_year"]

    print("\n=========================")
    print(f"TRAIN: {train_years}")
    print(f"TEST : {test_year}")
    print("=========================")

    # =========================
    # TRAIN DATA
    # =========================

    train_df = df[
        df["year"].isin(train_years)
    ]

    # =========================
    # TEST DATA
    # =========================

    test_df = df[
        df["year"] == test_year
    ]

    print(f"\nTrain Shape: {train_df.shape}")

    print(f"Test Shape : {test_df.shape}")

    # =========================
    # CREATE X & y
    # =========================

    X_train = train_df[FEATURE_COLUMNS]

    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]

    y_test = test_df[TARGET_COLUMN]

    # =========================
    # CREATE MODEL
    # =========================

    model = XGBClassifier(

        n_estimators=500,

        max_depth=8,

        learning_rate=0.03,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        random_state=42,

        eval_metric="logloss"
    )

    # =========================
    # TRAIN MODEL
    # =========================

    print("\nTraining model...")

    model.fit(X_train, y_train)

    # =========================
    # PREDICTIONS
    # =========================

    y_pred = model.predict(X_test)

    # =========================
    # ACCURACY
    # =========================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # STORE RESULT

    results.append({

        "train_years": str(train_years),

        "test_year": test_year,

        "accuracy": accuracy
    })

# =========================
# FINAL SUMMARY
# =========================

results_df = pd.DataFrame(results)

print("\n=========================")
print("WALK-FORWARD SUMMARY")
print("=========================")

print(results_df)