import pandas as pd
import numpy as np
import os

from xgboost import XGBClassifier

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# FILE PATHS
# =========================

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "labeled",
    "featured_btc_5m.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_model.json"
)

TRADES_FILE = os.path.join(
    BASE_DIR,
    "results",
    "trade_log.csv"
)

# =========================
# LOAD DATA
# =========================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset Shape: {df.shape}")

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

# =========================
# LOAD MODEL
# =========================

print("\nLoading trained model...")

model = XGBClassifier()

model.load_model(MODEL_FILE)

# =========================
# TEST DATA
# =========================

split_index = int(len(df) * 0.8)

test_df = df.iloc[split_index:].copy()

X_test = test_df[FEATURE_COLUMNS]

# =========================
# PREDICTIONS
# =========================

print("\nGenerating predictions...")

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

test_df["prediction"] = predictions

test_df["confidence"] = probabilities.max(axis=1)

# =========================
# BACKTEST SETTINGS
# =========================

INITIAL_BALANCE = 1000

balance = INITIAL_BALANCE

FIXED_RISK_PER_TRADE = 10

TAKE_PROFIT_PERCENT = 0.5
STOP_LOSS_PERCENT = 0.25

CONFIDENCE_THRESHOLD = 0.95

TRADING_FEE_PERCENT = 0.04

SLIPPAGE_PERCENT = 0.02

COOLDOWN_CANDLES = 5

LOOKAHEAD = 20

# =========================
# TRACKING VARIABLES
# =========================

total_trades = 0
winning_trades = 0
losing_trades = 0

gross_profit = 0
gross_loss = 0

equity_curve = []

max_balance = balance
max_drawdown = 0

last_trade_index = -COOLDOWN_CANDLES

trade_logs = []

# =========================
# START BACKTEST
# =========================

print("\nRunning realistic professional backtest...")

for i in range(len(test_df) - LOOKAHEAD):

    # COOLDOWN FILTER

    if i - last_trade_index < COOLDOWN_CANDLES:
        continue

    row = test_df.iloc[i]

    prediction = row["prediction"]

    confidence = row["confidence"]

    # CONFIDENCE FILTER

    if confidence < CONFIDENCE_THRESHOLD:
        continue

    entry_price = row["close"]

    # SLIPPAGE

    slippage = entry_price * (SLIPPAGE_PERCENT / 100)

    future_data = test_df.iloc[i+1:i+LOOKAHEAD+1]

    trade_result = None

    pnl = 0

    # =========================
    # BUY TRADE
    # =========================

    if prediction == 1:

        entry_price += slippage

        tp_price = entry_price * (
            1 + TAKE_PROFIT_PERCENT / 100
        )

        sl_price = entry_price * (
            1 - STOP_LOSS_PERCENT / 100
        )

        for _, future_row in future_data.iterrows():

            high = future_row["high"]
            low = future_row["low"]

            # TAKE PROFIT

            if high >= tp_price:

                reward = FIXED_RISK_PER_TRADE * 2

                fee = reward * (
                    TRADING_FEE_PERCENT / 100
                )

                pnl = reward - fee

                balance += pnl

                winning_trades += 1

                gross_profit += pnl

                trade_result = "WIN"

                break

            # STOP LOSS

            elif low <= sl_price:

                risk = FIXED_RISK_PER_TRADE

                fee = risk * (
                    TRADING_FEE_PERCENT / 100
                )

                pnl = -(risk + fee)

                balance += pnl

                losing_trades += 1

                gross_loss += abs(pnl)

                trade_result = "LOSS"

                break

    # =========================
    # SELL TRADE
    # =========================

    elif prediction == 0:

        entry_price -= slippage

        tp_price = entry_price * (
            1 - TAKE_PROFIT_PERCENT / 100
        )

        sl_price = entry_price * (
            1 + STOP_LOSS_PERCENT / 100
        )

        for _, future_row in future_data.iterrows():

            high = future_row["high"]
            low = future_row["low"]

            # TAKE PROFIT

            if low <= tp_price:

                reward = FIXED_RISK_PER_TRADE * 2

                fee = reward * (
                    TRADING_FEE_PERCENT / 100
                )

                pnl = reward - fee

                balance += pnl

                winning_trades += 1

                gross_profit += pnl

                trade_result = "WIN"

                break

            # STOP LOSS

            elif high >= sl_price:

                risk = FIXED_RISK_PER_TRADE

                fee = risk * (
                    TRADING_FEE_PERCENT / 100
                )

                pnl = -(risk + fee)

                balance += pnl

                losing_trades += 1

                gross_loss += abs(pnl)

                trade_result = "LOSS"

                break

    # =========================
    # STORE TRADE
    # =========================

    if trade_result is not None:

        total_trades += 1

        last_trade_index = i

        equity_curve.append(balance)

        # DRAWDOWN

        if balance > max_balance:
            max_balance = balance

        drawdown = (
            (max_balance - balance)
            / max_balance
        ) * 100

        if drawdown > max_drawdown:
            max_drawdown = drawdown

        # TRADE LOG

        trade_logs.append({

            "trade_number": total_trades,
            "prediction": prediction,
            "confidence": confidence,
            "result": trade_result,
            "pnl": pnl,
            "balance": balance
        })

# =========================
# FINAL METRICS
# =========================

win_rate = (
    winning_trades / total_trades * 100
    if total_trades > 0 else 0
)

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0 else 0
)

# =========================
# SAVE TRADE LOG
# =========================

trade_log_df = pd.DataFrame(trade_logs)

trade_log_df.to_csv(TRADES_FILE, index=False)

# =========================
# FINAL RESULTS
# =========================

print("\n=========================")
print("REALISTIC BACKTEST RESULTS")
print("=========================")

print(f"\nInitial Balance: ${INITIAL_BALANCE:.2f}")

print(f"Final Balance: ${balance:.2f}")

print(f"\nNet Profit: ${balance - INITIAL_BALANCE:.2f}")

print(f"\nTotal Trades: {total_trades}")

print(f"Winning Trades: {winning_trades}")

print(f"Losing Trades: {losing_trades}")

print(f"\nWin Rate: {win_rate:.2f}%")

print(f"\nProfit Factor: {profit_factor:.2f}")

print(f"\nMax Drawdown: {max_drawdown:.2f}%")

print(f"\nTrade log saved at:\n{TRADES_FILE}")