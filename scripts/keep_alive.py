"""
Render 24/7 Keep-Alive Pinger for NeuroBot.
Pings the Render /api/health endpoint every 10 minutes to prevent the free tier from spinning down.
"""

import datetime
import sys
import time
import urllib.request
import urllib.error

HEALTH_URL = "https://neurobot-nrnn.onrender.com/api/health"
UI_URL = "https://neurobot-nrnn.onrender.com/"
PING_INTERVAL_SECONDS = 600  # 10 minutes (Render free tier sleeps after 15 mins)
REQUEST_TIMEOUT_SECONDS = 90  # Allow up to 90s for cold boot


def ping_endpoint(url: str, name: str) -> bool:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NeuroBot-KeepAlive/2.0"}
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.getcode()
            print(f"[{now}] Ping OK [{name}] -> HTTP {status} (awake & active)")
            return status == 200
    except urllib.error.HTTPError as e:
        print(f"[{now}] Ping Warning [{name}] -> HTTP {e.code} (service responding)")
        return False
    except Exception as e:
        print(f"[{now}] Ping Notice [{name}] -> {e} (initiating wake-up...)")
        return False


def ping_cycle():
    print(f"\n--- Keep-Alive Cycle: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    ping_endpoint(HEALTH_URL, "API Health")
    ping_endpoint(UI_URL, "UI Web")


def main():
    print("=" * 65)
    print("NeuroBot Render Keep-Alive Service")
    print(f"Target Health API: {HEALTH_URL}")
    print(f"Target Web UI:     {UI_URL}")
    print(f"Interval:          Every {PING_INTERVAL_SECONDS // 60} minutes")
    print("=" * 65)

    try:
        while True:
            ping_cycle()
            time.sleep(PING_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nKeep-Alive service stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
