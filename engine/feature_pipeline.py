import pandas as pd
import numpy as np
import os

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# =========================
# FILE PATHS
# =========================

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "live_btc_data.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "live_featured_data.csv"
)

# =========================
# LOAD DATA
# =========================

print("\nLoading live BTC data...")

df = pd.read_csv(INPUT_FILE)

print(f"Original Shape: {df.shape}")

# =========================
# EMA 20
# =========================

df["EMA_20"] = (
    df["close"]
    .ewm(span=20, adjust=False)
    .mean()
)

# =========================
# RSI 14
# =========================

delta = df["close"].diff()

gain = delta.where(delta > 0, 0)

loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()

avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / (avg_loss + 1e-9)

df["RSI_14"] = 100 - (
    100 / (1 + rs)
)

# =========================
# MACD
# =========================

ema_12 = (
    df["close"]
    .ewm(span=12, adjust=False)
    .mean()
)

ema_26 = (
    df["close"]
    .ewm(span=26, adjust=False)
    .mean()
)

df["MACD"] = ema_12 - ema_26

df["MACD_signal"] = (
    df["MACD"]
    .ewm(span=9, adjust=False)
    .mean()
)

df["MACD_hist"] = (
    df["MACD"]
    - df["MACD_signal"]
)

# =========================
# BOLLINGER BANDS
# =========================

rolling_mean = (
    df["close"]
    .rolling(window=20)
    .mean()
)

rolling_std = (
    df["close"]
    .rolling(window=20)
    .std()
)

df["BB_middle"] = rolling_mean

df["BB_upper"] = (
    rolling_mean
    + (rolling_std * 2)
)

df["BB_lower"] = (
    rolling_mean
    - (rolling_std * 2)
)

# =========================
# PRICE ACTION FEATURES
# =========================

df["body_size"] = abs(
    df["close"] - df["open"]
)

df["upper_wick"] = (
    df["high"]
    - df[["open", "close"]]
    .max(axis=1)
)

df["lower_wick"] = (
    df[["open", "close"]]
    .min(axis=1)
    - df["low"]
)

df["candle_range"] = (
    df["high"] - df["low"]
)

df["body_range_ratio"] = (
    df["body_size"]
    / (df["candle_range"] + 1e-9)
)

# =========================
# TREND FEATURES
# =========================

df["ema_distance"] = (
    df["close"] - df["EMA_20"]
)

df["price_momentum"] = (
    df["close"].pct_change(5)
)

# =========================
# VOLATILITY FEATURES
# =========================

df["bb_width"] = (
    df["BB_upper"]
    - df["BB_lower"]
)

df["rolling_volatility"] = (
    df["close"]
    .rolling(window=10)
    .std()
)

# =========================
# MOMENTUM FEATURES
# =========================

df["rsi_momentum"] = (
    df["RSI_14"].diff()
)

df["macd_momentum"] = (
    df["MACD_hist"].diff()
)

# =========================
# VOLUME FEATURES
# =========================

df["volume_ma"] = (
    df["volume"]
    .rolling(window=20)
    .mean()
)

df["relative_volume"] = (
    df["volume"]
    / (df["volume_ma"] + 1e-9)
)

# =========================
# SHIFT FEATURES
# =========================

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
# SAVE FEATURED DATA
# =========================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================
# FINAL INFO
# =========================

print("\n=========================")
print("LIVE FEATURE PIPELINE DONE")
print("=========================")

print(f"\nFinal Shape: {df.shape}")

print(f"\nSaved At:\n{OUTPUT_FILE}")