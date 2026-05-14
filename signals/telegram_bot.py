import requests

# =========================
# TELEGRAM SETTINGS
# =========================

BOT_TOKEN = ""

CHAT_ID = ""

# =========================
# TEST MESSAGE
# =========================

message = """

🔥 BTC AI SIGNAL BOT ONLINE 🔥

System Connected Successfully ✅

"""

# =========================
# TELEGRAM API URL
# =========================

url = (
    f"https://api.telegram.org/bot"
    f"{BOT_TOKEN}/sendMessage"
)

# =========================
# DATA
# =========================

data = {

    "chat_id": CHAT_ID,

    "text": message
}

# =========================
# SEND MESSAGE
# =========================

response = requests.post(
    url,
    data=data
)

# =========================
# RESULT
# =========================

if response.status_code == 200:

    print("\nTelegram message sent successfully!")

else:

    print("\nFailed to send Telegram message.")

    print(response.text)
