import time
import subprocess

# =========================
# LOOP SETTINGS
# =========================

WAIT_TIME = 330  # 5 minutes

print("\n=========================")
print("BTC AI LIVE SYSTEM STARTED")
print("=========================")

# =========================
# MAIN LOOP
# =========================

while True:

    try:

        print("\n=========================")
        print("STEP 1 -> RUN FEATURE PIPELINE")
        print("=========================")

        subprocess.run(

            ["python", "engine/feature_pipeline.py"]

        )

        # =========================
        # RUN PREDICTOR
        # =========================

        print("\n=========================")
        print("STEP 2 -> RUN AI PREDICTOR")
        print("=========================")

        subprocess.run(

            ["python", "engine/predictor.py"]

        )

        # =========================
        # WAIT FOR NEXT CANDLE
        # =========================

        print("\n=========================")
        print("WAITING FOR NEXT 5m CANDLE...")
        print("=========================")

        time.sleep(WAIT_TIME)

    except Exception as e:

        print("\nSYSTEM ERROR:")

        print(e)

        time.sleep(10)