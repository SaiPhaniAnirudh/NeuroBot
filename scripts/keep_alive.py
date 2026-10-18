"""
Render 24/7 Keep-Alive Pinger for NeuroBot.
Pings the Render /api/health endpoint every 10 minutes to prevent the free tier from spinning down.
"""

import time
import datetime
import urllib.request

URL = "https://neurobot-nrnn.onrender.com/api/health"
PING_INTERVAL_SECONDS = 600  # 10 minutes (Render sleeps at 15 mins)


def ping():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        req = urllib.request.Request(
            URL,
            headers={"User-Agent": "NeuroBot-KeepAlive/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            status = response.getcode()
            print(f"[{now}] Ping OK -> HTTP {status} (NeuroBot is awake)")
    except Exception as e:
        print(f"[{now}] Ping Warning -> {e} (Waking up...)")


def main():
    print("=" * 60)
    print("NeuroBot Render Keep-Alive Service Started")
    print(f"Target: {URL}")
    print(f"Interval: Every {PING_INTERVAL_SECONDS // 60} minutes")
    print("=" * 60)
    
    while True:
        ping()
        time.sleep(PING_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
