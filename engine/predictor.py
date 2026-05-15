import pandas as pd
import os
import requests

from xgboost import XGBClassifier

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# =========================
# FILE PATHS
# =========================

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "live_featured_data.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_model.json"
)

LOG_FILE = os.path.join(
    BASE_DIR,
    "results",
    "signal_log.csv"
)

# =========================
# TELEGRAM SETTINGS
# =========================

BOT_TOKEN = "Your_bot_key"

CHAT_ID = ""

# =========================
# CONFIDENCE SETTINGS
# =========================

CONFIDENCE_THRESHOLD = 85

# =========================
# LOAD LIVE FEATURE DATA
# =========================

print("\nLoading live featured data...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset Shape: {df.shape}")

# =========================
# FEATURE COLUMNS
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

# =========================
# GET LATEST ROW
# =========================

latest_row = df.iloc[-1]

# =========================
# CREATE INPUT DATA
# =========================

X_live = pd.DataFrame(
    [latest_row[FEATURE_COLUMNS]]
)

# =========================
# LOAD MODEL
# =========================

print("\nLoading trained AI model...")

model = XGBClassifier()

model.load_model(MODEL_FILE)

# =========================
# MAKE PREDICTION
# =========================

prediction = model.predict(X_live)[0]

probabilities = model.predict_proba(X_live)[0]

confidence = max(probabilities) * 100

# =========================
# SIGNAL LOGIC
# =========================

signal = "BUY" if prediction == 1 else "SELL"

# =========================
# CONFIDENCE FILTER
# =========================

if confidence < CONFIDENCE_THRESHOLD:

    signal = "NO TRADE"

# =========================
# OUTPUT
# =========================

print("\n=========================")
print("LIVE AI SIGNAL")
print("=========================")

print(f"\nSignal: {signal}")

print(f"Confidence: {confidence:.2f}%")

print(f"\nBTC Price: {latest_row['close']}")

print(f"\nTimestamp: {latest_row['timestamp']}")

# =========================
# TRADE LEVELS
# =========================

entry_price = latest_row["close"]

if signal == "BUY":

    tp = entry_price * 1.005

    sl = entry_price * 0.9975

elif signal == "SELL":

    tp = entry_price * 0.995

    sl = entry_price * 1.0025

else:

    tp = None

    sl = None

# =========================
# PRINT LEVELS
# =========================

print("\n=========================")
print("TRADE LEVELS")
print("=========================")

if signal == "NO TRADE":

    print("\nNo high-confidence trade found.")

else:

    print(f"\nEntry : {entry_price:.2f}")

    print(f"TP    : {tp:.2f}")

    print(f"SL    : {sl:.2f}")

    # =========================
    # TELEGRAM MESSAGE
    # =========================

    telegram_message = f'''
🔥 BTC AI SIGNAL 🔥

Pair: BTCUSDT

Signal: {signal}

Confidence: {confidence:.2f}%

Entry: {entry_price:.2f}

TP: {tp:.2f}

SL: {sl:.2f}

Time: {latest_row['timestamp']}
'''

    # =========================
    # SEND TELEGRAM ALERT
    # =========================

    url = (
        f"https://api.telegram.org/bot"
        f"{BOT_TOKEN}/sendMessage"
    )

    data = {

        "chat_id": CHAT_ID,

        "text": telegram_message
    }

    response = requests.post(
        url,
        data=data
    )

    if response.status_code == 200:

        print("\nTelegram signal sent!")

    else:

        print("\nTelegram send failed.")

    # =========================
    # SAVE SIGNAL LOG
    # =========================

    signal_data = {

        "timestamp": latest_row["timestamp"],

        "signal": signal,

        "confidence": round(confidence, 2),

        "entry": round(entry_price, 2),

        "tp": round(tp, 2),

        "sl": round(sl, 2)
    }

    log_df = pd.DataFrame([signal_data])

    # CREATE HEADER ONLY IF FILE EMPTY

    write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0

    log_df.to_csv(

        LOG_FILE,

        mode="a",

        header=write_header,

        index=False
    )

    print("\nSignal logged successfully!")

# =========================
# FINAL STATUS
# =========================

print("\n=========================")
print("AI PREDICTION COMPLETED")
print("=========================")
