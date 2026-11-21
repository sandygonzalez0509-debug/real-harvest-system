import os
import time
import json
import threading
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SPORTS = ["basketball_nba", "americanfootball_nfl", "basketball_ncaab"]
MARKETS = "h2h,spreads,totals"
REGIONS = "us"
ODDS_FORMAT = "american"
CHECK_INTERVAL = 900  # Escanea cada 15 minutos


def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Faltan credenciales de Telegram en las variables de entorno.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except Exception as e:
        print(f"Error enviando a Telegram: {e}")


def fetch_odds(sport: str):
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
        f"?apiKey={ODDS_API_KEY}&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}"
    )
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def scan_markets():
    if not ODDS_API_KEY:
        print("Falta ODDS_API_KEY en las variables de entorno.")
        return

    alerts = []
    for sport in SPORTS:
        try:
            events = fetch_odds(sport)
            for event in events:
                home = event.get("home_team", "")
                away = event.get("away_team", "")
                bookmakers = event.get("bookmakers", [])
                if not bookmakers:
                    continue

                bm = bookmakers[0]
                bm_name = bm.get("title", "Book")
                for market in bm.get("markets", []):
                    m_key = market.get("key")
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price")
                        name = outcome.get("name")
                        point = outcome.get("point", "")
                        if price is not None and price >= -110:
                            line_str = f" ({point:+})" if isinstance(point, (int, float)) else ""
                            alerts.append(
                                f"<b>{sport.upper()}</b> | {away} @ {home}\n"
                                f"• {m_key.upper()}: <b>{name}{line_str}</b> @ <b>{price:+}</b> ({bm_name})"
                            )
                            if len(alerts) >= 5:
                                break
                    if len(alerts) >= 5:
                        break
        except Exception as e:
            print(f"Error escaneando {sport}: {e}")

    if alerts:
        msg = "🎯 <b>RADAR REAL HARVEST V2 — CUOTAS ACTIVAS</b>\n\n" + "\n\n".join(alerts)
        send_telegram(msg)
    else:
        print("Escaneo completado: sin nuevas alertas en este ciclo.")


def run_scanner():
    send_telegram("🚀 <b>Real Harvest Radar V2</b> conectado y escaneando mercados en vivo.")
    while True:
        scan_markets()
        time.sleep(CHECK_INTERVAL)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Real Harvest Radar V2 activo")

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    threading.Thread(target=run_scanner, daemon=True).start()
    port = int(os.getenv("PORT", "10000"))
    HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()
