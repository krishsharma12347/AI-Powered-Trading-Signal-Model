import pandas as pd
import numpy as np
import os

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
    "labeled_btc_5m.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "labeled",
    "featured_btc_5m.csv"
)

# =========================
# LOAD DATA
# =========================

print("\nLoading labeled dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original Shape: {df.shape}")

# =========================
# PRICE ACTION FEATURES
# =========================

print("\nCreating price action features...")

# Candle body size
df["body_size"] = abs(df["close"] - df["open"])

# Upper wick
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)

# Lower wick
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

# Candle range
df["candle_range"] = df["high"] - df["low"]

# Body to range ratio
df["body_range_ratio"] = (
    df["body_size"] / (df["candle_range"] + 1e-9)
)

# =========================
# TREND FEATURES
# =========================

print("Creating trend features...")

# EMA distance
df["ema_distance"] = df["close"] - df["EMA_20"]

# Price momentum
df["price_momentum"] = df["close"].pct_change(5)

# =========================
# VOLATILITY FEATURES
# =========================

print("Creating volatility features...")

# Bollinger width
df["bb_width"] = df["BB_upper"] - df["BB_lower"]

# Rolling volatility
df["rolling_volatility"] = (
    df["close"].rolling(window=10).std()
)

# =========================
# MOMENTUM FEATURES
# =========================

print("Creating momentum features...")

# RSI momentum
df["rsi_momentum"] = df["RSI_14"].diff()

# MACD momentum
df["macd_momentum"] = df["MACD_hist"].diff()

# =========================
# VOLUME FEATURES
# =========================

print("Creating volume features...")

# Volume moving average
df["volume_ma"] = df["volume"].rolling(window=20).mean()

# Relative volume
df["relative_volume"] = (
    df["volume"] / (df["volume_ma"] + 1e-9)
)

# =========================
# SHIFT FEATURES
# AVOID DATA LEAKAGE
# =========================

print("Applying anti-leakage feature shifting...")

feature_columns_to_shift = [

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

for col in feature_columns_to_shift:

    df[col] = df[col].shift(1)

# =========================
# REMOVE NaNs
# =========================

df.dropna(inplace=True)

# =========================
# RESET INDEX
# =========================

df.reset_index(drop=True, inplace=True)

# =========================
# FINAL INFO
# =========================

print("\n=========================")
print("FEATURE ENGINEERING COMPLETED")
print("=========================")

print(f"\nFinal Shape: {df.shape}")

print(f"\nTotal Features: {len(df.columns)}")

# =========================
# SAVE FEATURED DATA
# =========================

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved At:\n{OUTPUT_FILE}")