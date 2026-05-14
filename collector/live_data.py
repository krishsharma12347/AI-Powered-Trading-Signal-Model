import websocket
import json
import pandas as pd
import os
from datetime import datetime

# =========================
# FILE PATH
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

LIVE_DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "live_btc_data.csv"
)

# =========================
# BINANCE WEBSOCKET
# =========================

SOCKET = "wss://stream.binance.com:9443/ws/btcusdt@kline_5m"

# =========================
# DATA STORAGE
# =========================

columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

# CREATE FILE IF NOT EXISTS

if not os.path.exists(LIVE_DATA_FILE):

    pd.DataFrame(columns=columns).to_csv(
        LIVE_DATA_FILE,
        index=False
    )

# =========================
# ON MESSAGE
# =========================

def on_message(ws, message):

    data = json.loads(message)

    candle = data["k"]

    is_candle_closed = candle["x"]

    # ONLY SAVE CLOSED CANDLES

    if is_candle_closed:

        candle_data = {

            "timestamp": datetime.fromtimestamp(
                candle["t"] / 1000
            ),

            "open": float(candle["o"]),

            "high": float(candle["h"]),

            "low": float(candle["l"]),

            "close": float(candle["c"]),

            "volume": float(candle["v"])
        }

        print("\n=========================")
        print("NEW 5m BTC CANDLE")
        print("=========================")

        print(candle_data)

        # SAVE TO CSV

        df = pd.DataFrame([candle_data])

        df.to_csv(
            LIVE_DATA_FILE,
            mode="a",
            header=False,
            index=False
        )

        print("\nSaved to live_btc_data.csv")

# =========================
# ON ERROR
# =========================

def on_error(ws, error):

    print("\nWebSocket Error:")

    print(error)

# =========================
# ON CLOSE
# =========================

def on_close(ws, close_status_code, close_msg):

    print("\nWebSocket Closed")

# =========================
# ON OPEN
# =========================

def on_open(ws):

    print("\nConnected to Binance WebSocket!")

# =========================
# START WEBSOCKET
# =========================

ws = websocket.WebSocketApp(

    SOCKET,

    on_open=on_open,

    on_message=on_message,

    on_error=on_error,

    on_close=on_close
)

ws.run_forever()