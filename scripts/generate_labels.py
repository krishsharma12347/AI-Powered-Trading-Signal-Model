import pandas as pd
import os
import numpy as np

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
    "cleaned",
    "cleaned_btc_5m.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "labeled",
    "labeled_btc_5m.csv"
)

# =========================
# LOAD DATA
# =========================

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset Shape: {df.shape}")

# =========================
# PARAMETERS
# =========================

LOOKAHEAD = 20

TP_PERCENT = 0.5
SL_PERCENT = 0.25

# =========================
# TARGET LIST
# =========================

targets = []

# =========================
# GENERATE LABELS
# =========================

print("\nGenerating labels...")

for i in range(len(df) - LOOKAHEAD):

    current_close = df.iloc[i]["close"]

    future_high = df.iloc[i+1:i+LOOKAHEAD+1]["high"].max()
    future_low = df.iloc[i+1:i+LOOKAHEAD+1]["low"].min()

    # BUY CONDITIONS

    buy_tp_price = current_close * (1 + TP_PERCENT / 100)
    buy_sl_price = current_close * (1 - SL_PERCENT / 100)

    # SELL CONDITIONS

    sell_tp_price = current_close * (1 - TP_PERCENT / 100)
    sell_sl_price = current_close * (1 + SL_PERCENT / 100)

    target = None

    # BUY LABEL

    if future_high >= buy_tp_price and future_low > buy_sl_price:
        target = 1

    # SELL LABEL

    elif future_low <= sell_tp_price and future_high < sell_sl_price:
        target = 0

    targets.append(target)

# =========================
# TRIM DATAFRAME
# =========================

df = df.iloc[:len(targets)]

# =========================
# ADD TARGETS
# =========================

df["target"] = targets

# =========================
# REMOVE NONE LABELS
# =========================

df.dropna(inplace=True)

# =========================
# RESET INDEX
# =========================

df.reset_index(drop=True, inplace=True)

# =========================
# LABEL DISTRIBUTION
# =========================

print("\n=========================")
print("LABEL DISTRIBUTION")
print("=========================")

print(df["target"].value_counts())

# =========================
# SAVE LABELED DATA
# =========================

df.to_csv(OUTPUT_FILE, index=False)

# =========================
# FINAL INFO
# =========================

print("\n=========================")
print("LABEL GENERATION COMPLETED")
print("=========================")

print(f"\nFinal Shape: {df.shape}")

print(f"\nSaved At:\n{OUTPUT_FILE}")