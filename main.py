"""
Real Harvest Radar V3.1 — Motor +EV / ROI para Hard Rock Bet (Florida)
Fusión óptima: De-vig Sharp/Consenso + Filtro de Frescura + Kelly Fraccionado 0.25x + Control de Créditos
"""
import os
import json
import time
import csv
import math
import threading
import statistics
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- CONFIGURACIÓN Y SECRETOS ---
ODDS_API_KEY       = os.getenv("ODDS_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
PORT               = int(os.getenv("PORT", "10000"))

SPORTS         = [s.strip() for s in os.getenv("SPORTS", "basketball_nba,americanfootball_nfl,basketball_ncaab").split(",") if s.strip()]
MARKETS        = os.getenv("MARKETS", "h2h,spreads,totals")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "900"))   # 15 min base
HOURS_AHEAD    = int(os.getenv("HOURS_AHEAD", "24"))       # Solo eventos próximas 24h
MAX_AGE_SEC    = int(os.getenv("MAX_AGE_SEC", "300"))      # Máx 5 min de antigüedad de cuota

TARGET_BOOKS   = [b.strip() for b in os.getenv("TARGET_BOOKS", "hardrockbet_fl,hardrockbet").split(",") if b.strip()]
SHARP_BOOKS    = [b.strip() for b in os.getenv("SHARP_BOOKS", "pinnacle").split(",") if b.strip()]
REF_BOOKS      = [b.strip() for b in os.getenv("REF_BOOKS", "draftkings,fanduel,betmgm,williamhill_us,betrivers,espnbet").split(",") if b.strip()]

MIN_ROI        = float(os.getenv("MIN_ROI", "0.035"))        # +3.5% ROI obligatorio
NEAR_MISS_ROI  = float(os.getenv("NEAR_MISS_ROI", "0.03"))   # Registro de auditoría desde +3%
SUSPECT_ROI    = float(os.getenv("SUSPECT_ROI", "0.45"))     # Alerta de posible cuota colgada
MIN_REF_BOOKS  = int(os.getenv("MIN_REF_BOOKS", "3"))
BANKROLL       = float(os.getenv("BANKROLL", "0"))
MAX_STAKE_PCT  = float(os.getenv("MAX_STAKE_PCT", "0.02"))   # Tope 2% del bankroll
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))  # Kelly óptimo prudente (1/4)
MIN_CREDITS    = int(os.getenv("MIN_CREDITS", "50"))
LOG_FILE       = os.getenv("LOG_FILE", "senales.csv")

STATE = {
    "status": "running",
    "version": "V3.1",
    "last_scan_utc": None,
    "credits_left": None,
    "credits_last": None,
    "alerts_sent": 0,
    "near_misses": 0,
    "stale_rejected": 0,
}
SENT_CACHE = {}  # Deduplicación: firma -> última cuota alertada


def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Faltan credenciales de Telegram.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()
    except Exception as e:
        print(f"Error Telegram ({type(e).__name__})")


def parse_iso(ts: str):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def is_fresh(ts_str: str, now_utc: datetime) -> tuple[bool, int]:
    dt = parse_iso(ts_str)
    if not dt:
        return False, 999999
    age = int((now_utc - dt).total_seconds())
    return (-30 <= age <= MAX_AGE_SEC), max(0, age)


def fetch_odds(sport: str):
    books = list(dict.fromkeys(TARGET_BOOKS + SHARP_BOOKS + REF_BOOKS))[:10]
    now = datetime.now(timezone.utc)
    params = {
        "apiKey": ODDS_API_KEY,
        "bookmakers": ",".join(books),
        "markets": MARKETS,
        "oddsFormat": "decimal",
        "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": (now + timedelta(hours=HOURS_AHEAD)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as r:
        left = r.headers.get("x-requests-remaining")
        last = r.headers.get("x-requests-last")
        if left is not None:
            STATE["credits_left"] = int(float(left))
        if last is not None:
            STATE["credits_last"] = int(float(last))
        return json.loads(r.read().decode("utf-8"))


def fair_probs(market: dict) -> dict:
    outs = market.get("outcomes", [])
    if len(outs) < 2:
        return {}
    prices = [o.get("price") for o in outs]
    if any(not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 1.0 for p in prices):
        return {}
    implied = [1.0 / p for p in prices]
    total = sum(implied)
    if total <= 0:
        return {}
    return {(o["name"], o.get("point")): (1.0 / o["price"]) / total for o in outs}


def calc_kelly_pct(p: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0 or not (0.0 < p < 1.0):
        return 0.0
    f_full = ((b * p) - (1.0 - p)) / b
    return max(0.0, min(f_full * KELLY_FRACTION, MAX_STAKE_PCT))


def log_row(row: dict):
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def scan_event(sport: str, ev: dict, now_utc: datetime):
    commence = parse_iso(ev.get("commence_time", ""))
    if commence and commence <= now_utc:
        return  # Solo prepartido hasta sincronizar reloj/marcador live

    books = {b["key"]: b for b in ev.get("bookmakers", [])}
    target_key = next((k for k in TARGET_BOOKS if k in books), None)
    if not target_key:
        return

    target_book = books[target_key]
    for tmarket in target_book.get("markets", []):
        mkey = tmarket["key"]
        t_ts = tmarket.get("last_update") or target_book.get("last_update")
        fresh_t, age_t = is_fresh(t_ts, now_utc)
        if not fresh_t:
            STATE["stale_rejected"] += 1
            continue

        sharp_probs, ref_probs = {}, {}
        for bk, b in books.items():
            if bk in TARGET_BOOKS:
                continue
            m = next((x for x in b.get("markets", []) if x["key"] == mkey), None)
            if not m:
                continue
            r_ts = m.get("last_update") or b.get("last_update")
            fresh_r, _ = is_fresh(r_ts, now_utc)
            if not fresh_r:
                continue
            probs = fair_probs(m)
            for key_out, prob in probs.items():
                (sharp_probs if bk in SHARP_BOOKS else ref_probs).setdefault(key_out, []).append(prob)

        for o in tmarket.get("outcomes", []):
            O = o.get("price")
            if not isinstance(O, (int, float)) or not math.isfinite(O) or O <= 1.0:
                continue

            key_out = (o["name"], o.get("point"))
            if key_out in sharp_probs:
                p_est = statistics.mean(sharp_probs[key_out])
                source, n_refs = "Pinnacle (no-vig)", len(sharp_probs[key_out])
            elif len(ref_probs.get(key_out, [])) >= MIN_REF_BOOKS:
                p_est = statistics.median(ref_probs[key_out])
                source, n_refs = "Consenso (no-vig)", len(ref_probs[key_out])
            else:
                continue

            p_req = (1.0 + MIN_ROI) / O
            if p_req > 1.0:
                continue

            roi = (p_est * O) - 1.0
            if roi < NEAR_MISS_ROI:
                continue

            stake_f = calc_kelly_pct(p_est, O)
            is_push_line = isinstance(o.get("point"), (int, float)) and float(o["point"]).is_integer()

            row = {
                "hora_utc": now_utc.isoformat(timespec="seconds"),
                "event_id": ev["id"],
                "deporte": sport,
                "juego": f"{ev['away_team']} @ {ev['home_team']}",
                "inicio": ev["commence_time"],
                "mercado": mkey,
                "seleccion": o["name"],
                "linea": o.get("point"),
                "casa": target_key,
                "cuota_dec": round(O, 4),
                "p_imp": round(1.0 / O, 4),
                "p_ref": round(p_est, 4),
                "p_req_min": round(p_req, 4),
                "roi": round(roi, 4),
                "fuente": source,
                "n_casas": n_refs,
                "edad_seg": age_t,
                "stake_pct": round(stake_f, 4),
                "alertada": roi >= MIN_ROI,
            }
            log_row(row)

            if roi < MIN_ROI:
                STATE["near_misses"] += 1
                continue

            sig = f"{ev['id']}|{mkey}|{o['name']}|{o.get('point')}|{target_key}"
            if SENT_CACHE.get(sig, 0) >= O:
                continue
            SENT_CACHE[sig] = O

            linea_str = f" {o['point']:+g}" if o.get("point") is not None else ""
            monto_str = f" (${BANKROLL * stake_f:,.2f})" if BANKROLL > 0 else ""
            notas = []
            if is_push_line:
                notas.append("ℹ️ Línea entera (posible push).")
            if roi >= SUSPECT_ROI:
                notas.append("⚠️ ROI extremo: confirma precio vivo en Hard Rock antes de ejecutar.")
            nota_txt = ("\n" + "\n".join(notas)) if notas else ""

            send_telegram(
                f"🌾 <b>REAL HARVEST V3.1 — CANDIDATO +EV</b>\n"
                f"<b>{row['juego']}</b> ({sport})\n"
                f"• Mercado: {mkey.upper()} → <b>{o['name']}{linea_str}</b>\n"
                f"• Ejecución: <b>{target_key}</b> @ <b>{O:.2f}</b> (hace {age_t}s)\n"
                f"• p_ref: <b>{p_est:.1%}</b> ({source}, {n_refs}) | p_req({MIN_ROI:+.1%}): <b>{p_req:.1%}</b>\n"
                f"• ROI estimado: <b>{roi:+.1%}</b>\n"
                f"• Stake máx (0.25x Kelly): <b>{stake_f:.2%}</b>{monto_str}{nota_txt}"
            )
            STATE["alerts_sent"] += 1


def scanner_loop():
    send_telegram(
        f"🚀 <b>Real Harvest Radar V3.1</b> activo\n"
        f"Objetivo: Hard Rock FL | Filtro: ROI ≥ <b>{MIN_ROI:.1%}</b> | Tope: <b>{MAX_STAKE_PCT:.0%}</b>"
    )
    while True:
        if not ODDS_API_KEY:
            print("Falta ODDS_API_KEY.")
            time.sleep(60)
            continue

        now_utc = datetime.now(timezone.utc)
        for sport in SPORTS:
            try:
                events = fetch_odds(sport)
                for ev in events:
                    scan_event(sport, ev, now_utc)
            except Exception as e:
                print(f"[{sport}] Error en ciclo: {type(e).__name__}")

        STATE["last_scan_utc"] = now_utc.isoformat(timespec="seconds")
        wait = CHECK_INTERVAL
        if STATE["credits_left"] is not None and STATE["credits_left"] < MIN_CREDITS:
            wait = CHECK_INTERVAL * 4

        print(
            f"Ciclo OK | Créditos: {STATE['credits_left']} (último: {STATE['credits_last']}) | "
            f"Alertas: {STATE['alerts_sent']} | Casi-señales: {STATE['near_misses']} | "
            f"Descartadas por viejas: {STATE['stale_rejected']}"
        )
        time.sleep(wait)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(STATE).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


if __name__ == "__main__":
    threading.Thread(target=scanner_loop, daemon=True).start()
    HTTPServer(("0.0.0.0", PORT), HealthHandler).serve_forever()
