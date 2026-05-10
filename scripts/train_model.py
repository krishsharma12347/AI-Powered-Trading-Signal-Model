import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# FILE PATHS
# =========================

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "labeled",
    "featured_btc_5m.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_model.json"
)

# =========================
# LOAD DATA
# =========================

print("\nLoading featured dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset Shape: {df.shape}")

# =========================
# FEATURES
# =========================

FEATURE_COLUMNS = [

    # Original OHLCV
    "open",
    "high",
    "low",
    "close",
    "volume",

    # Indicators
    "RSI_14",
    "EMA_20",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "BB_upper",
    "BB_middle",
    "BB_lower",

    # Price Action Features
    "body_size",
    "upper_wick",
    "lower_wick",
    "candle_range",
    "body_range_ratio",

    # Trend Features
    "ema_distance",
    "price_momentum",

    # Volatility Features
    "bb_width",
    "rolling_volatility",

    # Momentum Features
    "rsi_momentum",
    "macd_momentum",

    # Volume Features
    "relative_volume"
]

# =========================
# TARGET
# =========================

TARGET_COLUMN = "target"

# =========================
# CREATE X & y
# =========================

X = df[FEATURE_COLUMNS]

y = df[TARGET_COLUMN]

# =========================
# TRAIN / TEST SPLIT
# =========================

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

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

print("\nTraining advanced XGBoost model...")

model.fit(X_train, y_train)

print("\nModel training completed!")

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print("ADVANCED MODEL RESULTS")
print("=========================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

# =========================
# FEATURE IMPORTANCE
# =========================

feature_importance = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n=========================")
print("FEATURE IMPORTANCE")
print("=========================")

print(feature_importance)

# =========================
# SAVE MODEL
# =========================

model.save_model(MODEL_PATH)

print("\n=========================")
print("MODEL SAVED SUCCESSFULLY")
print("=========================")

print(f"\nModel Path:\n{MODEL_PATH}")