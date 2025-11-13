#Execute eval for MOOC

import requests

BOT_TOKEN = "7991086857:AAHqeqj_9BPHDg7U3apIo0huoOuP8CsrwLw"
CHAT_ID = "-1003254176613"
def notify(text):
    requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        params={"chat_id": CHAT_ID, "text": text},
        timeout=5
    )
# after your work:
notify("Start execution")

import os
import subprocess

#code = os.system("python -m Evaluation.eval --dataset MOOC --explainer all")

#if code == 0:
#    notify("✅ Eval: MOOC finished!")
#else:
#    notify("❌ Eval: MOOC failed!")

code = os.system("python -m Evaluation.eval --dataset Reddit --explainer tgnn")

if code == 0:
   notify("✅ Eval: Reddit finished!")
else:
   notify("❌ Eval: Reddit failed!")

code = os.system("python -m Evaluation.eval --dataset Wikipedia --explainer all")

if code == 0:
    notify("✅ Eval: Wikipedia finished!")
else:
    notify("❌ Eval: Wikipedia failed!")

