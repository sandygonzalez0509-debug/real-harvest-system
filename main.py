"""
Real Harvest Radar V3.3 ÉLITE — Motor +EV / ROI para Hard Rock Bet (Florida)
Evolución V3.3 (Costo $0 extra en API):
  1. Modo Turbo en Horas Pico (Calendario Inteligente ET):
     - Escanea cada 8 min en horas pico de reportes de lesiones/alineaciones SOLO deportes con juegos <= 3.5h
     - Mantiene 30 min en horario normal y 90 min en madrugada (ahorro de créditos inteligente)
  2. Interpolación de Medios Puntos (Alt-Line Pricing):
     - Calcula probabilidad justa exacta con distribución Normal calibrada por deporte (NBA, NCAAB, NCAAW, NFL, NCAAF)
     - Captura desfases de 0.5 a 3.5 pts cuando el mercado sharp se mueve y Hard Rock Bet queda rezagado
     - Guardia automática en números clave de NFL/NCAAF (3 y 7) y exclusión de deportes de goles (NHL/Fútbol)
  3. Auditoría Obligatoria RAG (92 libros en Pinecone):
     - Cero señales sin aprobación explícita de la biblioteca (protección total del bankroll)
  4. Núcleo V3.2 ÉLITE intacto:
     - De-vig Power Method + Escuadra Multi-Sharp + Consenso normalizado (suma 1.0)
     - Reloj fresco por deporte + sincronización de snapshots + matemática exacta de Push
     - Pre-filtro gratuito (/sports y /events) + confirmación real Telegram + Health Watchdog 24/7
"""
import os
import json
import time
import csv
import math
import html
import gzip
import random
import signal
import threading
import statistics
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from openai import OpenAI
from pinecone import Pinecone

# --- LLAVES DESDE VARIABLES DE ENTORNO (Render -> Environment) ---
ODDS_API_KEY = (os.getenv("ODDS_API_KEY") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
PINECONE_API_KEY = (os.getenv("PINECONE_API_KEY") or "").strip()
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

PINECONE_INDEX_NAME = (os.getenv("PINECONE_INDEX") or "rh-vectors-1024-v1").strip()
PINECONE_NAMESPACE = (os.getenv("PINECONE_NAMESPACE") or "rh-te3small-1024-v1").strip()

# Inicialización segura del Cerebro RAG (92 libros)
oai_client = None
rh_index = None

try:
    if OPENAI_API_KEY:
        oai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=15.0)
except Exception as e:
    print(f"Error inicializando OpenAI: {type(e).__name__}: {e}")

try:
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        rh_index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Error inicializando Pinecone: {type(e).__name__}: {e}")

AUDIT_CACHE = {}


def auditar_con_biblioteca(alerta: dict) -> dict:
    """Filtro obligatorio contra los 92 libros. Sin aprobación explícita, bloquea la señal."""
    cache_key = (
        f"{alerta.get('deporte')}|{alerta.get('mercado')}|"
        f"{alerta.get('cuota_hrb')}|{alerta.get('roi_pct')}|{alerta.get('delta_pts')}"
    )
    if cache_key in AUDIT_CACHE:
        return AUDIT_CACHE[cache_key]

    if oai_client is None or rh_index is None:
        return {
            "aprobada": False,
            "resumen": "Bloqueada: motor RAG (OpenAI/Pinecone) no inicializado.",
            "principio_clave": "Filtro Obligatorio 92 Libros",
        }

    try:
        query_text = (
            f"Auditoría +EV {alerta['deporte']} mercado {alerta['mercado']} "
            f"cuota {alerta['cuota_hrb']} ROI {alerta['roi_pct']}% "
            f"alt-line delta {alerta['delta_pts']} puntos gestión Kelly varianza"
        )
        emb_resp = oai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text,
            dimensions=1024,
        )
        res = rh_index.query(
            vector=emb_resp.data[0].embedding,
            top_k=6,
            namespace=PINECONE_NAMESPACE,
            include_metadata=True,
        )
        chunks = []
        for match in getattr(res, "matches", []) or []:
            meta = getattr(match, "metadata", None) or {}
            txt = meta.get("text") or meta.get("chunk_text") or ""
            if txt:
                chunks.append(str(txt))

        if not chunks:
            return {
                "aprobada": False,
                "resumen": "Bloqueada: sin fragmentos de respaldo en los 92 libros.",
                "principio_clave": "Evidencia RAG Requerida",
            }

        contexto_libros = "\n---\n".join(chunks)
        prompt_sistema = (
            "Eres el Auditor Cuantitativo de Real Harvest. Evalúa la señal matemática "
            "usando únicamente los principios de los 92 libros adjuntos. Aprueba solo si "
            "la ventaja es sólida, está a favor y cumple gestión de riesgo; de lo contrario "
            "devuelve 'aprobada': false. Responde estrictamente en JSON con las claves: "
            "'aprobada' (boolean), 'resumen' (máximo 2 líneas técnicas y directas) y "
            "'principio_clave' (ley o concepto de la biblioteca que respalda el veredicto)."
        )
        chat_resp = oai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt_sistema},
                {
                    "role": "user",
                    "content": f"SEÑAL:\n{json.dumps(alerta, ensure_ascii=False)}\n\nBIBLIOTECA (92 LIBROS):\n{contexto_libros}",
                },
            ],
        )
        parsed = json.loads(chat_resp.choices[0].message.content)
        result = {
            "aprobada": bool(parsed.get("aprobada", False)),
            "resumen": str(parsed.get("resumen", "Dictamen de biblioteca generado.")),
            "principio_clave": str(parsed.get("principio_clave", "Valor Esperado (+EV)")),
        }
        AUDIT_CACHE[cache_key] = result
        return result
    except Exception as e:
        print(f"Fallo auditoría RAG: {type(e).__name__}: {e}")
        return {
            "aprobada": False,
            "resumen": f"Bloqueada por seguridad ({type(e).__name__}): sin validación de los 92 libros.",
            "principio_clave": "Protección de Bankroll",
        }


try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except Exception:
    ET_TZ = timezone(timedelta(hours=-5))

PORT = int(os.getenv("PORT", "10000"))
API_BASE = "https://api.the-odds-api.com/v4"

SPORTS = [
    s.strip()
    for s in os.getenv(
        "SPORTS",
        "basketball_nba,americanfootball_nfl,basketball_ncaab,icehockey_nhl,"
        "americanfootball_ncaaf,basketball_ncaaw,soccer_epl,soccer_spain_la_liga,"
        "soccer_italy_serie_a,soccer_uefa_champs_league",
    ).split(",")
    if s.strip()
]

MARKETS = os.getenv("MARKETS", "h2h,spreads,totals")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "1800"))
PEAK_INTERVAL = int(os.getenv("PEAK_INTERVAL", "480"))
QUIET_INTERVAL = int(os.getenv("QUIET_INTERVAL", "5400"))
TURBO_MODE = os.getenv("TURBO_MODE", "true").lower() == "true"
TURBO_HORIZON_H = float(os.getenv("TURBO_HORIZON_H", "3.5"))
PEAK_START_ET = float(os.getenv("PEAK_START_ET", "16.0"))
PEAK_END_ET = float(os.getenv("PEAK_END_ET", "21.5"))
WKND_PEAK_START = float(os.getenv("WKND_PEAK_START", "11.0"))
QUIET_START_ET = float(os.getenv("QUIET_START_ET", "0.5"))
QUIET_END_ET = float(os.getenv("QUIET_END_ET", "9.0"))

HOURS_AHEAD = int(os.getenv("HOURS_AHEAD", "24"))
MAX_AGE_SEC = int(os.getenv("MAX_AGE_SEC", "300"))
MAX_SKEW_SEC = int(os.getenv("MAX_SKEW_SEC", "120"))
SMART_SCHEDULING = os.getenv("SMART_SCHEDULING", "true").lower() == "true"

TARGET_BOOKS = [b.strip() for b in os.getenv("TARGET_BOOKS", "hardrockbet_fl,hardrockbet").split(",") if b.strip()]
SHARP_BOOKS = [b.strip() for b in os.getenv("SHARP_BOOKS", "pinnacle,lowvig,betonlineag,betfair_ex_eu").split(",") if b.strip()]
REF_BOOKS = [b.strip() for b in os.getenv("REF_BOOKS", "draftkings,fanduel,betmgm,williamhill_us").split(",") if b.strip()]

MIN_ROI = float(os.getenv("MIN_ROI", "0.035"))
HIGH_ROI = float(os.getenv("HIGH_ROI", "0.10"))
ELITE_ROI = float(os.getenv("ELITE_ROI", "0.18"))
NEAR_MISS_ROI = float(os.getenv("NEAR_MISS_ROI", "0.03"))
SUSPECT_ROI = float(os.getenv("SUSPECT_ROI", "0.45"))

DEVIG_METHOD = os.getenv("DEVIG_METHOD", "power").lower()
MIN_OVERROUND = float(os.getenv("MIN_OVERROUND", "1.0"))
MAX_OVERROUND = float(os.getenv("MAX_OVERROUND", "1.15"))
MIN_REF_BOOKS = int(os.getenv("MIN_REF_BOOKS", "3"))
MIN_DEC_ODDS = float(os.getenv("MIN_DEC_ODDS", "1.833"))
MAX_REF_SPREAD = float(os.getenv("MAX_REF_SPREAD", "0.035"))
MAX_SHARP_GAP = float(os.getenv("MAX_SHARP_GAP", "0.04"))
MAX_DEC_ODDS = float(os.getenv("MAX_DEC_ODDS", "0"))
LONGSHOT_ODDS = float(os.getenv("LONGSHOT_ODDS", "6.0"))
LONGSHOT_KELLY = float(os.getenv("LONGSHOT_KELLY", "0.5"))
CONSENSUS_KELLY = float(os.getenv("CONSENSUS_KELLY", "0.5"))

ALT_LINE_ENABLED = os.getenv("ALT_LINE_ENABLED", "true").lower() == "true"
ALT_LINE_KELLY = float(os.getenv("ALT_LINE_KELLY", "0.70"))

ALT_LINE_PARAMS = {
    "basketball_nba":          {"spreads": (12.5, 3.5), "totals": (18.5, 4.5)},
    "basketball_ncaab":        {"spreads": (11.8, 3.5), "totals": (16.0, 4.5)},
    "basketball_ncaaw":        {"spreads": (12.2, 3.5), "totals": (16.0, 4.5)},
    "americanfootball_nfl":    {"spreads": (13.8, 1.5), "totals": (13.8, 2.0)},
    "americanfootball_ncaaf":  {"spreads": (15.8, 1.5), "totals": (16.5, 2.5)},
}
FOOTBALL_KEY_NUMBERS = (3.0, 7.0)

PUSH_PROB_SPREAD = float(os.getenv("PUSH_PROB_SPREAD", "0.045"))
PUSH_PROB_TOTAL = float(os.getenv("PUSH_PROB_TOTAL", "0.040"))

BANKROLL = float(os.getenv("BANKROLL", "0"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "0.02"))
MAX_EVENT_PCT = float(os.getenv("MAX_EVENT_PCT", "0.03"))
MAX_TOTAL_PCT = float(os.getenv("MAX_TOTAL_PCT", "0.10"))

MIN_CREDITS = int(os.getenv("MIN_CREDITS", "50"))
DAILY_CREDIT_BUDGET = int(os.getenv("DAILY_CREDIT_BUDGET", "0"))
ALERT_TTL_SEC = int(os.getenv("ALERT_TTL_SEC", "3600"))
REALERT_ROI_DELTA = float(os.getenv("REALERT_ROI_DELTA", "0.02"))
WATCHDOG_SEC = int(os.getenv("WATCHDOG_SEC", "900"))
LOG_FILE = os.getenv("LOG_FILE", "senales_v33.csv")
MAX_LOG_BYTES = int(os.getenv("MAX_LOG_BYTES", "50000000"))

CSV_COLUMNS = [
    "hora_utc", "event_id", "deporte", "juego", "inicio", "mercado",
    "seleccion", "linea", "alt_delta", "casa", "cuota_dec", "p_imp",
    "p_ref", "q_push", "p_req_min", "roi", "nivel", "fuente", "n_casas",
    "edad_seg", "stake_pct", "alertada",
]

STATE = {
    "status": "running",
    "version": "V3.3-ELITE-TURBO",
    "schedule_mode": "INIT",
    "last_scan_utc": None,
    "credits_left": None,
    "credits_last": None,
    "daily_credits_spent": 0,
    "alerts_sent": 0,
    "rag_blocked": 0,
    "alt_line_alerts": 0,
    "near_misses": 0,
    "stale_rejected": 0,
    "skew_rejected": 0,
    "dispersion_rejected": 0,
    "incomplete_rejected": 0,
    "longshot_blocked": 0,
    "sports_skipped_free": 0,
    "max_roi_seen": 0.0,
    "last_error": None,
}

SENT_CACHE = {}
EVENT_ALERTS = {}
SPORT_NEXT = {}
EVENT_CACHE = {}
ACTIVE_CACHE = {"ts": 0.0, "keys": set()}
CREDIT_DAY = {"day": None, "spent": 0}
API_BLOCK_UNTIL = 0.0

STOP = threading.Event()
CSV_LOCK = threading.Lock()
HEARTBEAT_MONO = time.monotonic()
SCANNER_THREAD = None
SQRT2 = math.sqrt(2.0)

TIER_ICONS = {"base": "🌾", "alto": "⚡", "elite": "🔥"}


def esc(val) -> str:
    return html.escape(str(val), quote=False)


def classify_tier(roi: float) -> tuple[str, str]:
    if roi >= ELITE_ROI:
        return "🔥 NIVEL ÉLITE (+18%+)", "elite"
    if roi >= HIGH_ROI:
        return "⚡ NIVEL ALTO (+10% a +17.9%)", "alto"
    return "🌾 NIVEL BASE (+3.5% a +9.9% — Mínimo Radar)", "base"


def format_cycle_summary(cycle_hits: list, mode_label: str) -> str:
    best = max(cycle_hits, key=lambda x: x["roi"])
    n_base = sum(1 for x in cycle_hits if x["tier_code"] == "base")
    n_alto = sum(1 for x in cycle_hits if x["tier_code"] == "alto")
    n_elite = sum(1 for x in cycle_hits if x["tier_code"] == "elite")
    n_alt = sum(1 for x in cycle_hits if x.get("is_alt"))

    dist_parts = []
    if n_base > 0:
        dist_parts.append(f"🌾 Base (3.5%–9.9%): <b>{n_base}</b>")
    if n_alto > 0:
        dist_parts.append(f"⚡ Alto (10%–17.9%): <b>{n_alto}</b>")
    if n_elite > 0:
        dist_parts.append(f"🔥 Élite (18%+): <b>{n_elite}</b>")
    if n_alt > 0:
        dist_parts.append(f"📐 Alt-Line: <b>{n_alt}</b>")

    total = len(cycle_hits)
    cuota_word = "cuota aprobada" if total == 1 else "cuotas aprobadas"
    best_icon = TIER_ICONS.get(best["tier_code"], "🌾")
    dist_line = " | ".join(dist_parts)

    return (
        f"📊 <b>RESUMEN DE ESCANEO [{mode_label}] ({total} {cuota_word} por 92 libros)</b>\n"
        f"• Distribución: {dist_line}\n"
        f"• {best_icon} Máximo del ciclo: <b>{best['roi']:+.1%}</b> en <b>{esc(best['juego'])}</b> ({esc(best['sport'])}) → {esc(best['pick'])} @ {best['cuota_am']}"
    )


def parse_iso(ts: str):
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def is_fresh(ts_str: str, now_utc: datetime) -> tuple[bool, int, datetime | None]:
    dt = parse_iso(ts_str)
    if dt is None:
        return False, 999999, None
    age = int((now_utc - dt).total_seconds())
    return (-30 <= age <= MAX_AGE_SEC), max(0, age), dt


def get_schedule_profile(now_utc: datetime) -> tuple[str, int]:
    if not TURBO_MODE:
        return "NORMAL", CHECK_INTERVAL
    local_dt = now_utc.astimezone(ET_TZ)
    hour_f = local_dt.hour + (local_dt.minute / 60.0)
    is_weekend = local_dt.weekday() >= 5
    peak_start = WKND_PEAK_START if is_weekend else PEAK_START_ET

    if peak_start <= hour_f < PEAK_END_ET:
        return "TURBO", PEAK_INTERVAL
    if QUIET_START_ET <= hour_f < QUIET_END_ET:
        return "AHORRO", QUIET_INTERVAL
    return "NORMAL", CHECK_INTERVAL


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def norm_ppf(p: float) -> float:
    lo, hi = -8.0, 8.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def crosses_football_key_number(pt_ref: float, pt_target: float) -> bool:
    lo, hi = sorted((abs(pt_ref), abs(pt_target)))
    return any(lo <= key <= hi for key in FOOTBALL_KEY_NUMBERS)


def interpolate_alt_prob(
    sport: str,
    mkey: str,
    outcome_name: str,
    p_ref: float,
    pt_ref: float,
    pt_target: float,
) -> float | None:
    if not ALT_LINE_ENABLED or mkey not in ("spreads", "totals"):
        return None
    sport_cfg = ALT_LINE_PARAMS.get(sport)
    if not sport_cfg or mkey not in sport_cfg:
        return None
    sigma, max_diff = sport_cfg[mkey]
    diff = abs(pt_target - pt_ref)
    if diff <= 1e-9 or diff > max_diff + 1e-9:
        return None
    if not (0.02 <= p_ref <= 0.98):
        return None
    if mkey == "spreads" and sport.startswith("americanfootball_") and crosses_football_key_number(pt_ref, pt_target):
        return None

    z_ref = norm_ppf(p_ref)
    if mkey == "spreads":
        dz = (pt_target - pt_ref) / sigma
    elif outcome_name == "Over":
        dz = (pt_ref - pt_target) / sigma
    elif outcome_name == "Under":
        dz = (pt_target - pt_ref) / sigma
    else:
        return None

    p_alt = norm_cdf(z_ref + dz)
    if not (0.01 <= p_alt <= 0.99):
        return None
    return p_alt


def outcome_key(o: dict) -> tuple[str, float | None]:
    point = o.get("point")
    if point is not None:
        if isinstance(point, bool) or not isinstance(point, (int, float)) or not math.isfinite(point):
            raise ValueError("Punto inválido")
        point = round(float(point), 4)
    return o["name"], point


def valid_market(m: dict) -> bool:
    try:
        outs = m["outcomes"]
        if len(outs) < 2:
            return False
        keys = [outcome_key(o) for o in outs]
        if len(set(keys)) != len(keys):
            return False
        prices = [o["price"] for o in outs]
        if any(
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or not math.isfinite(p)
            or p <= 1.0
            for p in prices
        ):
            return False

        kind = m["key"]
        if kind == "h2h":
            return len(outs) in (2, 3) and all(pt is None for _, pt in keys)
        if len(outs) != 2:
            return False
        if kind == "totals":
            return (
                {name for name, _ in keys} == {"Over", "Under"}
                and all(pt is not None for _, pt in keys)
                and math.isclose(keys[0][1], keys[1][1], abs_tol=1e-9)
            )
        if kind == "spreads":
            return (
                keys[0][0] != keys[1][0]
                and all(pt is not None for _, pt in keys)
                and math.isclose(keys[0][1] + keys[1][1], 0.0, abs_tol=1e-9)
            )
        return False
    except (KeyError, TypeError, ValueError):
        return False


def fair_probs(market: dict) -> dict:
    if not valid_market(market):
        return {}
    outs = market["outcomes"]
    q = [1.0 / o["price"] for o in outs]
    overround = math.fsum(q)
    if not (MIN_OVERROUND <= overround <= MAX_OVERROUND):
        return {}

    if DEVIG_METHOD == "multiplicative":
        probs = [v / overround for v in q]
    else:
        lo, hi = 1.0, 2.0
        while math.fsum(v ** hi for v in q) > 1.0:
            hi *= 2.0
            if hi > 1024.0:
                return {}
        for _ in range(70):
            mid = (lo + hi) * 0.5
            if math.fsum(v ** mid for v in q) > 1.0:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) * 0.5
        raw = [v ** k for v in q]
        total = math.fsum(raw)
        probs = [v / total for v in raw]

    return {outcome_key(o): p for o, p in zip(outs, probs)}


def estimate_push_prob(mkey: str, point) -> float | None:
    if mkey not in ("spreads", "totals"):
        return 0.0
    if point is None or isinstance(point, bool) or not isinstance(point, (int, float)):
        return None
    if not math.isclose(point * 2.0, round(point * 2.0), abs_tol=1e-9):
        return None
    if math.isclose(abs(point) % 1.0, 0.5, abs_tol=1e-9):
        return 0.0
    return PUSH_PROB_SPREAD if mkey == "spreads" else PUSH_PROB_TOTAL


def calc_metrics_and_kelly(p_cond: float, decimal_odds: float, q_push: float = 0.0, factor: float = 1.0) -> tuple[float, float, float]:
    roi = (1.0 - q_push) * ((p_cond * decimal_odds) - 1.0)
    p_req = (1.0 + (MIN_ROI / (1.0 - q_push))) / decimal_odds
    b = decimal_odds - 1.0
    if b <= 0 or not (0.0 < p_cond < 1.0) or roi <= 0:
        return roi, p_req, 0.0
    f_full = ((p_cond * decimal_odds) - 1.0) / b
    stake = max(0.0, min(f_full * KELLY_FRACTION * factor, MAX_STAKE_PCT))
    return roi, p_req, stake


def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Faltan credenciales de Telegram.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = json.loads(r.read().decode("utf-8"))
            if resp.get("ok") is True:
                time.sleep(0.35)
                return True
            print(f"Telegram respondió ok=False: {resp}")
            return False
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = 5
                try:
                    body = json.loads(e.read().decode("utf-8"))
                    retry_after = int(body.get("parameters", {}).get("retry_after", 5))
                except Exception:
                    pass
                time.sleep(min(30, retry_after) + 0.5)
                continue
            print(f"Error Telegram HTTP {e.code}")
            return False
        except Exception as e:
            if attempt == 2:
                print(f"Error Telegram ({type(e).__name__})")
                return False
            time.sleep(1.5 * (attempt + 1))
    return False


def _roll_credit_day():
    today = datetime.now(timezone.utc).date().isoformat()
    if CREDIT_DAY["day"] != today:
        CREDIT_DAY["day"] = today
        CREDIT_DAY["spent"] = 0
        STATE["daily_credits_spent"] = 0


def api_get(path: str, params: dict | None = None, expected_cost: int = 0):
    global API_BLOCK_UNTIL
    _roll_credit_day()

    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY vacía")

    if time.monotonic() < API_BLOCK_UNTIL:
        rem = int(API_BLOCK_UNTIL - time.monotonic())
        raise RuntimeError(f"API en pausa temporal ({rem}s)")

    if expected_cost > 0:
        left = STATE.get("credits_left")
        if left is not None and (left - expected_cost) < MIN_CREDITS:
            raise RuntimeError(f"Reserva mínima de créditos alcanzada ({left} < {MIN_CREDITS})")
        if DAILY_CREDIT_BUDGET > 0 and (CREDIT_DAY["spent"] + expected_cost) > DAILY_CREDIT_BUDGET:
            raise RuntimeError("Presupuesto diario de créditos alcanzado")

    query = dict(params or {}, apiKey=ODDS_API_KEY)
    url = f"{API_BASE}{path}?" + urllib.parse.urlencode(query)

    for attempt in range(3):
        req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding", "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))
                left = r.headers.get("x-requests-remaining")
                last = r.headers.get("x-requests-last")
                if left is not None:
                    STATE["credits_left"] = int(float(left))
                if last is not None:
                    cost = int(float(last))
                    STATE["credits_last"] = cost
                    CREDIT_DAY["spent"] += cost
                    STATE["daily_credits_spent"] = CREDIT_DAY["spent"]
                return data
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                API_BLOCK_UNTIL = time.monotonic() + 300
                raise
            if e.code == 429:
                try:
                    delay = float(e.headers.get("Retry-After", "60"))
                except (TypeError, ValueError):
                    delay = 60.0
                API_BLOCK_UNTIL = time.monotonic() + min(300.0, max(10.0, delay))
                raise
            if e.code < 500 or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
        time.sleep((2 ** attempt) + random.random())


def get_active_sports() -> set:
    now_mono = time.monotonic()
    if ACTIVE_CACHE["keys"] and (now_mono - ACTIVE_CACHE["ts"]) < 21600:
        return ACTIVE_CACHE["keys"]
    try:
        catalog = api_get("/sports", {}, expected_cost=0)
        keys = {s["key"] for s in catalog if s.get("active")}
        if keys:
            ACTIVE_CACHE["keys"] = keys
            ACTIVE_CACHE["ts"] = now_mono
            return keys
    except Exception as e:
        print(f"Aviso catálogo /sports: {type(e).__name__}: {e}")
    return set(SPORTS)


def get_upcoming_starts(sport: str, now_utc: datetime, is_turbo: bool = False) -> list[datetime]:
    now_mono = time.monotonic()
    ttl = 210.0 if is_turbo else 900.0
    cached = EVENT_CACHE.get(sport)
    if cached and (now_mono - cached[0]) < ttl:
        agenda = cached[1]
    else:
        agenda = api_get(
            f"/sports/{sport}/events",
            {
                "commenceTimeFrom": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "commenceTimeTo": (now_utc + timedelta(hours=HOURS_AHEAD)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            expected_cost=0,
        )
        EVENT_CACHE[sport] = (now_mono, agenda)

    end_utc = now_utc + timedelta(hours=HOURS_AHEAD)
    starts = []
    for ev in agenda:
        dt = parse_iso(ev.get("commence_time", ""))
        if dt and now_utc < dt <= end_utc:
            starts.append(dt)
    return starts


def fetch_odds(sport: str, now_utc: datetime) -> list[dict]:
    books = list(dict.fromkeys(TARGET_BOOKS + SHARP_BOOKS + REF_BOOKS))[:10]
    markets_list = list(dict.fromkeys(m.strip() for m in MARKETS.split(",") if m.strip()))
    expected_cost = len(markets_list) * math.ceil(len(books) / 10)
    return api_get(
        f"/sports/{sport}/odds",
        {
            "bookmakers": ",".join(books),
            "markets": ",".join(markets_list),
            "oddsFormat": "decimal",
            "commenceTimeFrom": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commenceTimeTo": (now_utc + timedelta(hours=HOURS_AHEAD)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        expected_cost=expected_cost,
    )


def log_row(row: dict):
    aligned_row = {col: row.get(col, "") for col in CSV_COLUMNS}
    with CSV_LOCK:
        parent = os.path.dirname(os.path.abspath(LOG_FILE))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > MAX_LOG_BYTES:
            os.replace(LOG_FILE, f"{LOG_FILE}.{int(time.time())}.bak")

        write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            if write_header:
                w.writeheader()
            w.writerow(aligned_row)


def prune_memory(now_utc: datetime):
    for sig in [k for k, v in SENT_CACHE.items() if v["commence"] and v["commence"] <= now_utc]:
        SENT_CACHE.pop(sig, None)
    for eid in [k for k, v in EVENT_ALERTS.items() if v["commence"] and v["commence"] <= now_utc]:
        EVENT_ALERTS.pop(eid, None)


def should_alert(sig: str, odds: float, roi: float) -> bool:
    prev = SENT_CACHE.get(sig)
    if prev is None:
        return True
    if odds > prev["odds"] + 1e-6:
        return True
    if roi >= prev["roi"] + REALERT_ROI_DELTA:
        return True
    if (time.time() - prev["sent_ts"]) >= ALERT_TTL_SEC:
        return True
    return False


def _sharp_label(keys: list[str], method_tag: str) -> str:
    if keys == ["pinnacle"]:
        return f"Pinnacle ({method_tag})"
    return f"Multi-Sharp ({method_tag})"


def build_reference_map(sport: str, tmarket: dict, target_dt: datetime, books: dict, now_utc: datetime) -> dict:
    mkey = tmarket["key"]
    wanted = {outcome_key(o) for o in tmarket["outcomes"]}
    wanted_names = {name for name, _ in wanted}
    target_pt_by_name = {name: pt for name, pt in wanted}

    allowed_sharp = set(SHARP_BOOKS) - set(TARGET_BOOKS)
    allowed_ref = set(REF_BOOKS) - set(TARGET_BOOKS)
    allowed_all = allowed_sharp | allowed_ref

    exact_sharp, exact_ref = {}, {}
    alt_sharp, alt_ref = {}, {}

    for bk, b in books.items():
        if bk not in allowed_all:
            continue
        best_alt_candidate = None
        for m in b.get("markets", []):
            if m.get("key") != mkey or not valid_market(m):
                continue
            r_ts = m.get("last_update") or b.get("last_update")
            fresh_r, _, r_dt = is_fresh(r_ts, now_utc)
            if not fresh_r or r_dt is None:
                continue
            if abs((r_dt - target_dt).total_seconds()) > MAX_SKEW_SEC:
                STATE["skew_rejected"] += 1
                continue

            m_keys = {outcome_key(o) for o in m["outcomes"]}
            if m_keys == wanted:
                probs = fair_probs(m)
                if probs:
                    (exact_sharp if bk in allowed_sharp else exact_ref)[bk] = probs
                    best_alt_candidate = None
                    break
            elif (
                ALT_LINE_ENABLED
                and mkey in ("spreads", "totals")
                and {name for name, _ in m_keys} == wanted_names
            ):
                probs = fair_probs(m)
                if not probs:
                    continue
                projected = {}
                deltas = []
                valid_proj = True
                for (name, pt_ref), p_ref in probs.items():
                    pt_tgt = target_pt_by_name.get(name)
                    if pt_ref is None or pt_tgt is None:
                        valid_proj = False
                        break
                    p_alt = interpolate_alt_prob(sport, mkey, name, p_ref, pt_ref, pt_tgt)
                    if p_alt is None:
                        valid_proj = False
                        break
                    projected[(name, pt_tgt)] = p_alt
                    deltas.append(abs(pt_tgt - pt_ref))
                if valid_proj and len(projected) == len(wanted):
                    tot_p = math.fsum(projected.values())
                    if tot_p > 0:
                        projected = {k: v / tot_p for k, v in projected.items()}
                        avg_delta = statistics.mean(deltas)
                        if best_alt_candidate is None or avg_delta < best_alt_candidate[1]:
                            best_alt_candidate = (projected, avg_delta)

        if best_alt_candidate is not None:
            (alt_sharp if bk in allowed_sharp else alt_ref)[bk] = best_alt_candidate

    method_tag = "Power" if DEVIG_METHOD != "multiplicative" else "Mult"

    def _build_consensus(obs_dict: dict) -> tuple[dict, dict]:
        if len(obs_dict) < MIN_REF_BOOKS:
            return {}, {}
        raw_meds = {k: statistics.median(obs[k] for obs in obs_dict.values()) for k in wanted}
        tot = math.fsum(raw_meds.values())
        if tot <= 0:
            return {}, {}
        vec = {k: v / tot for k, v in raw_meds.items()}
        spreads = {}
        for k in wanted:
            vals = [obs[k] for obs in obs_dict.values()]
            spreads[k] = max(vals) - min(vals)
        return vec, spreads

    cons_vec, cons_spread = _build_consensus(exact_ref)
    alt_sharp_probs = {bk: item[0] for bk, item in alt_sharp.items()}
    alt_ref_probs = {bk: item[0] for bk, item in alt_ref.items()}
    alt_cons_vec, alt_cons_spread = _build_consensus(alt_ref_probs)

    result = {}
    for k in wanted:
        has_exact_cons = (k in cons_vec) and (cons_spread.get(k, 1.0) <= MAX_REF_SPREAD)
        if k in cons_vec and not has_exact_cons:
            STATE["dispersion_rejected"] += 1

        if exact_sharp:
            p_sharp = statistics.mean(obs[k] for obs in exact_sharp.values())
            if has_exact_cons and abs(p_sharp - cons_vec[k]) > MAX_SHARP_GAP:
                STATE["dispersion_rejected"] += 1
                continue
            label = _sharp_label(list(exact_sharp.keys()), method_tag)
            result[k] = (p_sharp, label, len(exact_sharp), 1.0, 0.0)
            continue

        if has_exact_cons:
            if alt_sharp_probs:
                p_alt_sh = statistics.mean(obs[k] for obs in alt_sharp_probs.values())
                if abs(p_alt_sh - cons_vec[k]) > MAX_SHARP_GAP:
                    STATE["dispersion_rejected"] += 1
                    continue
            result[k] = (cons_vec[k], f"Consenso ({method_tag})", len(exact_ref), CONSENSUS_KELLY, 0.0)
            continue

        has_alt_cons = (k in alt_cons_vec) and (alt_cons_spread.get(k, 1.0) <= MAX_REF_SPREAD)
        if alt_sharp_probs:
            p_alt_sh = statistics.mean(obs[k] for obs in alt_sharp_probs.values())
            if has_alt_cons and abs(p_alt_sh - alt_cons_vec[k]) > MAX_SHARP_GAP:
                STATE["dispersion_rejected"] += 1
                continue
            avg_d = statistics.mean(item[1] for item in alt_sharp.values())
            label = _sharp_label(list(alt_sharp.keys()), method_tag) + f" Alt Δ{avg_d:g}"
            result[k] = (p_alt_sh, label, len(alt_sharp_probs), ALT_LINE_KELLY, avg_d)
            continue

        if has_alt_cons:
            avg_d = statistics.mean(item[1] for item in alt_ref.values())
            label = f"Consenso ({method_tag}) Alt Δ{avg_d:g}"
            result[k] = (alt_cons_vec[k], label, len(alt_ref_probs), CONSENSUS_KELLY * ALT_LINE_KELLY, avg_d)

    return result


def scan_event(sport: str, ev: dict, now_utc: datetime, cycle_hits: list):
    commence = parse_iso(ev.get("commence_time", ""))
    if commence is None or commence <= now_utc + timedelta(seconds=60):
        return

    books = {b["key"]: b for b in ev.get("bookmakers", [])}
    target_key = next((k for k in TARGET_BOOKS if k in books), None)
    if not target_key:
        return

    target_book = books[target_key]
    for tmarket in target_book.get("markets", []):
        mkey = tmarket.get("key", "")
        if not valid_market(tmarket):
            STATE["incomplete_rejected"] += 1
            continue

        t_ts = tmarket.get("last_update") or target_book.get("last_update")
        fresh_t, age_t, target_dt = is_fresh(t_ts, now_utc)
        if not fresh_t or target_dt is None:
            STATE["stale_rejected"] += 1
            continue

        references = build_reference_map(sport, tmarket, target_dt, books, now_utc)
        if not references:
            continue

        for o in tmarket.get("outcomes", []):
            O = o.get("price")
            if isinstance(O, bool) or not isinstance(O, (int, float)) or not math.isfinite(O) or O <= 1.0:
                continue

            key_out = outcome_key(o)
            ref = references.get(key_out)
            if ref is None:
                continue
            p_est, source, n_refs, k_factor, alt_delta = ref

            q_push = estimate_push_prob(mkey, o.get("point"))
            if q_push is None:
                continue

            is_longshot = O >= LONGSHOT_ODDS
            effective_k_factor = k_factor * (LONGSHOT_KELLY if is_longshot else 1.0)
            p_imp = 1.0 / O
            roi, p_req, stake_f = calc_metrics_and_kelly(p_est, O, q_push, effective_k_factor)
            if p_req > 1.0 or roi < NEAR_MISS_ROI:
                continue

            ev_rec_existing = EVENT_ALERTS.get(ev["id"])
            used_event_pct = ev_rec_existing["stake_sum"] if ev_rec_existing else 0.0
            remaining_event_pct = max(0.0, MAX_EVENT_PCT - used_event_pct)
            effective_stake_f = min(stake_f, remaining_event_pct)

            tier_label, tier_code = classify_tier(roi)
            blocked_longshot = (MAX_DEC_ODDS > 0) and (O > MAX_DEC_ODDS)

            if roi < MIN_ROI:
                STATE["near_misses"] += 1
                log_row({
                    "hora_utc": now_utc.isoformat(timespec="seconds"),
                    "event_id": ev["id"],
                    "deporte": sport,
                    "juego": f"{ev['away_team']} @ {ev['home_team']}",
                    "inicio": ev["commence_time"],
                    "mercado": mkey,
                    "seleccion": o["name"],
                    "linea": o.get("point"),
                    "alt_delta": round(alt_delta, 2),
                    "casa": target_key,
                    "cuota_dec": round(O, 4),
                    "p_imp": round(p_imp, 4),
                    "p_ref": round(p_est, 4),
                    "q_push": round(q_push, 4),
                    "p_req_min": round(p_req, 4),
                    "roi": round(roi, 4),
                    "nivel": tier_code,
                    "fuente": source,
                    "n_casas": n_refs,
                    "edad_seg": age_t,
                    "stake_pct": round(effective_stake_f, 4),
                    "alertada": False,
                })
                continue

            if blocked_longshot:
                STATE["longshot_blocked"] += 1
                continue

            if o.get("point") is not None:
                linea_str = f" {o['point']:g}" if mkey == "totals" else f" {o['point']:+g}"
            else:
                linea_str = ""
            cuota_am = f"+{round((O - 1) * 100)}" if O >= 2.0 else f"{round(-100 / (O - 1))}"
            juego_str = f"{ev['away_team']} @ {ev['home_team']}"

            auditoria = auditar_con_biblioteca({
                "deporte": sport,
                "mercado": f"{mkey.upper()} -> {o['name']}{linea_str}",
                "cuota_hrb": cuota_am,
                "p_ref": round(p_est * 100, 1),
                "roi_pct": round(roi * 100, 1),
                "delta_pts": round(alt_delta, 2),
                "kelly_pct": round(effective_stake_f * 100, 2),
            })

            aprobada_rag = bool(auditoria.get("aprobada", False))

            log_row({
                "hora_utc": now_utc.isoformat(timespec="seconds"),
                "event_id": ev["id"],
                "deporte": sport,
                "juego": juego_str,
                "inicio": ev["commence_time"],
                "mercado": mkey,
                "seleccion": o["name"],
                "linea": o.get("point"),
                "alt_delta": round(alt_delta, 2),
                "casa": target_key,
                "cuota_dec": round(O, 4),
                "p_imp": round(p_imp, 4),
                "p_ref": round(p_est, 4),
                "q_push": round(q_push, 4),
                "p_req_min": round(p_req, 4),
                "roi": round(roi, 4),
                "nivel": tier_code,
                "fuente": source,
                "n_casas": n_refs,
                "edad_seg": age_t,
                "stake_pct": round(effective_stake_f, 4),
                "alertada": aprobada_rag,
            })

            if not aprobada_rag:
                STATE["rag_blocked"] += 1
                continue

            cycle_hits.append({
                "juego": juego_str,
                "sport": sport,
                "pick": f"{mkey.upper()} → {o['name']}{linea_str}",
                "cuota_am": cuota_am,
                "roi": roi,
                "tier_code": tier_code,
                "is_alt": alt_delta > 0,
            })

            if roi > STATE["max_roi_seen"]:
                STATE["max_roi_seen"] = round(roi, 4)

            sig = f"{ev['id']}|{mkey}|{o['name']}|{key_out[1]}|{target_key}"
            if not should_alert(sig, O, roi):
                continue

            ventaja_sobre_piso = roi - MIN_ROI
            monto_str = f" (${BANKROLL * effective_stake_f:,.2f})" if BANKROLL > 0 else ""
            notas = []
            if alt_delta > 0:
                notas.append(
                    f"📐 Alt-Line Pricing (Δ{alt_delta:g} pts): Hard Rock rezagado frente a la línea real "
                    f"(Kelly ajustado a {ALT_LINE_KELLY:.2f}x)."
                )
            if ev_rec_existing and sig not in ev_rec_existing["picks"]:
                notas.append(
                    f"🔗 Correlación intra-partido: stake ajustado al remanente del partido "
                    f"(tope {MAX_EVENT_PCT:.0%}; usado {used_event_pct:.2%})."
                )
            if source.startswith("Consenso"):
                notas.append("ℹ️ Sin Sharp directo: Kelly ajustado a 0.5x por menor certeza.")
            if is_longshot:
                notas.append("⚡ Cuota larga (alta varianza): Kelly ajustado a 0.5x.")
            if q_push > 0:
                notas.append(f"ℹ️ Línea entera: ROI real ajustado por push estimado ({q_push:.1%}).")
            if roi >= SUSPECT_ROI:
                notas.append("⚠️ ROI extremo: confirma precio vivo en Hard Rock antes de ejecutar.")

            notas.append(
                f"📚 <b>Biblioteca (92 libros):</b> {esc(auditoria.get('resumen', ''))} "
                f"(<i>Base: {esc(auditoria.get('principio_clave', ''))}</i>)"
            )

            nota_txt = ("\n" + "\n".join(notas)) if notas else ""
            dt_miami = commence.astimezone(ET_TZ)
            hora_miami = dt_miami.strftime("%d/%m %I:%M %p ET")

            msg = (
                f"🕒 Horario Miami: <b>{hora_miami}</b>\n"
                f"{tier_label}\n"
                f"<b>{esc(juego_str)}</b> ({esc(sport)})\n"
                f"• Mercado: {esc(mkey.upper())} → <b>{esc(o['name'])}{esc(linea_str)}</b>\n"
                f"• Ejecución: <b>{esc(target_key)}</b> @ <b>{cuota_am}</b> (dec {O:.2f} | hace {age_t}s)\n"
                f"• Probabilidades: p_imp HRB <b>{p_imp:.1%}</b> | p_ref <b>{p_est:.1%}</b> ({esc(source)}, {n_refs}) | p_req({MIN_ROI:+.1%}) <b>{p_req:.1%}</b>\n"
                f"• ROI real estimado: <b>{roi:+.1%}</b> (<b>{ventaja_sobre_piso:+.1%}</b> sobre piso mínimo)\n"
                f"• Stake guía (0.25x Kelly): <b>{effective_stake_f:.2%}</b>{monto_str}{nota_txt}"
            )

            if send_telegram(msg):
                SENT_CACHE[sig] = {"odds": O, "roi": roi, "sent_ts": time.time(), "commence": commence}
                ev_rec = EVENT_ALERTS.setdefault(ev["id"], {"picks": [], "stake_sum": 0.0, "commence": commence})
                if sig not in ev_rec["picks"]:
                    ev_rec["picks"].append(sig)
                    ev_rec["stake_sum"] += effective_stake_f
                STATE["alerts_sent"] += 1
                if alt_delta > 0:
                    STATE["alt_line_alerts"] += 1


def heartbeat():
    global HEARTBEAT_MONO
    HEARTBEAT_MONO = time.monotonic()


def run_cycle(quota_alerted: dict) -> int:
    cycle_start = datetime.now(timezone.utc)
    prune_memory(cycle_start)
    mode_label, base_wait = get_schedule_profile(cycle_start)
    STATE["schedule_mode"] = mode_label
    cycle_hits = []
    activos = get_active_sports()
    has_near_game = False

    for sport in list(SPORTS):
        if STOP.is_set():
            break
        heartbeat()

        if sport not in activos:
            STATE["sports_skipped_free"] += 1
            continue

        now_mono = time.monotonic()
        if SMART_SCHEDULING and now_mono < SPORT_NEXT.get(sport, 0.0):
            cached = EVENT_CACHE.get(sport)
            if cached:
                now_tmp = datetime.now(timezone.utc)
                for ev in cached[1]:
                    dt = parse_iso(ev.get("commence_time", ""))
                    if dt and now_tmp < dt <= now_tmp + timedelta(hours=TURBO_HORIZON_H):
                        has_near_game = True
                        break
            STATE["sports_skipped_free"] += 1
            continue

        try:
            starts = get_upcoming_starts(sport, datetime.now(timezone.utc), is_turbo=(mode_label == "TURBO"))
            if not starts:
                STATE["sports_skipped_free"] += 1
                skip_wait = 600 if mode_label == "TURBO" else max(900, CHECK_INTERVAL)
                SPORT_NEXT[sport] = time.monotonic() + skip_wait - 15
                continue

            now_check = datetime.now(timezone.utc)
            hrs_to_next = (min(starts) - now_check).total_seconds() / 3600.0
            if hrs_to_next <= TURBO_HORIZON_H:
                has_near_game = True

            events = fetch_odds(sport, now_check)
            now_after = datetime.now(timezone.utc)
            for ev in events:
                heartbeat()
                try:
                    scan_event(sport, ev, now_after, cycle_hits)
                except Exception as ev_err:
                    STATE["last_error"] = f"{sport}/ev: {type(ev_err).__name__}: {ev_err}"

            if SMART_SCHEDULING:
                if mode_label == "TURBO" and hrs_to_next <= TURBO_HORIZON_H:
                    next_gap = PEAK_INTERVAL
                elif mode_label == "AHORRO" and hrs_to_next > TURBO_HORIZON_H:
                    next_gap = QUIET_INTERVAL
                elif hrs_to_next <= 2.0:
                    next_gap = min(CHECK_INTERVAL, 900) if mode_label == "TURBO" else CHECK_INTERVAL
                elif hrs_to_next <= 6.0:
                    next_gap = max(CHECK_INTERVAL, 1800)
                else:
                    next_gap = max(CHECK_INTERVAL, 3600)
                SPORT_NEXT[sport] = time.monotonic() + max(60, next_gap - 15)

        except urllib.error.HTTPError as e:
            STATE["last_error"] = f"{sport}: HTTP {e.code}"
            if e.code in (401, 403, 429):
                if not quota_alerted.get("sent"):
                    send_telegram(f"⛔ <b>The Odds API respondió HTTP {e.code}</b> (créditos o límite). Pausa protectora activada.")
                    quota_alerted["sent"] = True
                return CHECK_INTERVAL * 4
            print(f"[{sport}] HTTP {e.code}")
        except Exception as e:
            STATE["last_error"] = f"{sport}: {type(e).__name__}: {e}"
            print(f"[{sport}] Error en ciclo: {type(e).__name__}: {e}")

    quota_alerted["sent"] = False

    if cycle_hits:
        send_telegram(format_cycle_summary(cycle_hits, mode_label))

    STATE["last_scan_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if mode_label == "TURBO" and not has_near_game:
        wait = CHECK_INTERVAL
    else:
        wait = base_wait

    if STATE["credits_left"] is not None and STATE["credits_left"] < MIN_CREDITS:
        wait = max(wait, CHECK_INTERVAL * 4)

    print(
        f"Ciclo OK [{mode_label}] | Próx: {wait // 60}m | Créditos: {STATE['credits_left']} (últ: {STATE['credits_last']}) | "
        f"Alertas: {STATE['alerts_sent']} (Alt: {STATE['alt_line_alerts']} | Vetadas RAG: {STATE['rag_blocked']}) | "
        f"Casi: {STATE['near_misses']} | Viejas: {STATE['stale_rejected']} | Desfase: {STATE['skew_rejected']} | "
        f"Dispersión: {STATE['dispersion_rejected']} | Saltados gratis: {STATE['sports_skipped_free']}"
    )
    return wait


def monitored_wait(seconds: int):
    deadline = time.monotonic() + seconds
    while not STOP.is_set() and time.monotonic() < deadline:
        heartbeat()
        STOP.wait(min(10.0, max(0.5, deadline - time.monotonic())))


def scanner_loop():
    mode_now, _ = get_schedule_profile(datetime.now(timezone.utc))
    send_telegram(
        f"🚀 <b>Real Harvest Radar V3.3 ÉLITE</b> activo\n"
        f"Objetivo: Hard Rock FL | De-vig: <b>{DEVIG_METHOD.upper()}</b> | Alt-Line: <b>{'ON' if ALT_LINE_ENABLED else 'OFF'}</b> | Filtro 92 Libros: <b>OBLIGATORIO</b>\n"
        f"Turbo Pico: <b>{PEAK_INTERVAL // 60} min</b> | Normal: <b>{CHECK_INTERVAL // 60} min</b> | Ahorro: <b>{QUIET_INTERVAL // 60} min</b> (Modo: <b>{mode_now}</b>)\n"
        f"Escalones: 🌾 <b>{MIN_ROI:.1%}</b> | ⚡ <b>{HIGH_ROI:.0%}</b> | 🔥 <b>{ELITE_ROI:.0%}+</b> | Tope guía: <b>{MAX_STAKE_PCT:.0%}</b> / partido <b>{MAX_EVENT_PCT:.0%}</b>"
    )
    quota_alerted = {"sent": False}
    while not STOP.is_set():
        heartbeat()
        if not ODDS_API_KEY:
            print("Falta ODDS_API_KEY.")
            monitored_wait(60)
            continue
        try:
            wait = run_cycle(quota_alerted)
            STATE["status"] = "running"
        except Exception as e:
            STATE["status"] = "degraded"
            STATE["last_error"] = f"ciclo: {type(e).__name__}: {e}"
            print(f"Error general del ciclo: {type(e).__name__}: {e}")
            wait = 60
        monitored_wait(wait)


class HealthHandler(BaseHTTPRequestHandler):
    def _health_status(self):
        alive = SCANNER_THREAD is not None and SCANNER_THREAD.is_alive()
        hb_age = time.monotonic() - HEARTBEAT_MONO
        healthy = alive and (hb_age < WATCHDOG_SEC)
        return healthy, alive, round(hb_age, 1)

    def do_GET(self):
        if self.path not in ("/", "/health", "/healthz"):
            self.send_error(404)
            return
        healthy, alive, hb_age = self._health_status()
        body = json.dumps({
            **STATE,
            "scanner_alive": alive,
            "heartbeat_age_sec": hb_age,
        }).encode("utf-8")
        self.send_response(200 if healthy else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        healthy, _, _ = self._health_status()
        self.send_response(200 if healthy else 503)
        self.end_headers()

    def log_message(self, *args):
        return


if __name__ == "__main__":
    SCANNER_THREAD = threading.Thread(target=scanner_loop, name="radar-scanner", daemon=True)
    SCANNER_THREAD.start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), HealthHandler)

    def handle_shutdown(*_):
        STOP.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        STOP.set()
        server.server_close()
        SCANNER_THREAD.join(timeout=20)
