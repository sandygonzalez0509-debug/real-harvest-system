import csv
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple

import requests

try:
    from zoneinfo import ZoneInfo
    TZ_ET = ZoneInfo("America/New_York")
except Exception:
    TZ_ET = None


# ==============================================================================
# 1. CONFIGURACIÓN MAESTRA (VARIABLES DE ENTORNO EN RENDER)
# ==============================================================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

BANKROLL = float(os.getenv("BANKROLL", "1000.0"))
BOOKMAKER_TARGET = "hardrockbet"

# Casas Sharp de referencia (ordenadas por peso de precisión)
SHARP_WEIGHTS = {
    "pinnacle": 3.0,
    "circa": 2.5,
    "bookmaker": 2.2,
    "lowvig": 1.8,
    "betonlineag": 1.5,
    "fanduel": 1.0,
    "draftkings": 1.0,
}

# Cuota mínima permitida: -110 americano equivale a 1.90909 decimal
CUOTA_MIN_DECIMAL = float(os.getenv("CUOTA_MIN_DECIMAL", "1.905"))

# Ventana prepartido para escanear submercados y props (en horas antes del inicio)
SUBMARKET_WINDOW_HOURS = float(os.getenv("SUBMARKET_WINDOW_HOURS", "3.5"))
ENABLE_SUBMARKETS = os.getenv("ENABLE_SUBMARKETS", "true").lower() == "true"

# Topes de exposición agregada (correlación y blindaje del bankroll)
MAX_EXPOSICION_PARTIDO = float(os.getenv("MAX_EXPOSICION_PARTIDO", "0.025"))   # 2.5% por partido
MAX_EXPOSICION_JUGADOR = float(os.getenv("MAX_EXPOSICION_JUGADOR", "0.0075"))  # 0.75% por jugador
MAX_EXPOSICION_CICLO = float(os.getenv("MAX_EXPOSICION_CICLO", "0.050"))       # 5.0% por ciclo

# Antigüedad máxima aceptada del feed de la API (segundos)
MAX_FEED_AGE_PRINCIPAL = int(os.getenv("MAX_FEED_AGE_PRINCIPAL", "120"))
MAX_FEED_AGE_SUBMERCADO = int(os.getenv("MAX_FEED_AGE_SUBMERCADO", "90"))

# Cooldown para evitar alertas repetidas (en segundos)
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "900"))
CSV_LOG_FILE = os.getenv("CSV_LOG_FILE", "real_harvest_v4_log.csv")

# ==============================================================================
# 2. CALIBRACIÓN CUANTITATIVA DE 3 CAPAS (CONSENSO DE MÁXIMA PRECISIÓN)
# ==============================================================================
CONFIG_CATEGORIAS = {
    "principal": {
        "nombre": "JUEGO PRINCIPAL",
        "icono": "🎯",
        "min_ev": 0.035,        # +3.5% EV mínimo
        "delta_p": 0.005,       # 0.5% margen de incertidumbre
        "kelly_mult": 0.25,     # 0.25x Kelly sobre EV conservador
        "cap_pct": 0.015,       # Tope 1.50% del bankroll por jugada
        "exec_window_sec": 30,  # Ejecutar en <= 30 segundos
        "max_feed_age": MAX_FEED_AGE_PRINCIPAL,
        "devig": "power",
    },
    "mitades": {
        "nombre": "MITADES / PERIODOS",
        "icono": "⏱️",
        "min_ev": 0.050,        # +5.0% EV mínimo
        "delta_p": 0.010,       # 1.0% margen de incertidumbre
        "kelly_mult": 0.15,     # 0.15x Kelly sobre EV conservador
        "cap_pct": 0.010,       # Tope 1.00% del bankroll por jugada
        "exec_window_sec": 20,  # Ejecutar en <= 20 segundos
        "max_feed_age": MAX_FEED_AGE_SUBMERCADO,
        "devig": "shin",
    },
    "props": {
        "nombre": "PROPS & SUBMERCADOS",
        "icono": "⚡",
        "min_ev": 0.065,        # +6.5% EV mínimo
        "delta_p": 0.015,       # 1.5% margen de incertidumbre
        "kelly_mult": 0.12,     # 0.12x Kelly sobre EV conservador
        "cap_pct": 0.0075,      # Tope 0.75% del bankroll por jugada
        "exec_window_sec": 15,  # Ejecutar en <= 15 segundos
        "max_feed_age": MAX_FEED_AGE_SUBMERCADO,
        "devig": "shin",
    },
}

# ==============================================================================
# 3. LOS 10 DEPORTES ACTIVOS, ETIQUETAS LIMPIAS Y SUBMERCADOS POR DEPORTE
# ==============================================================================
SPORT_LABELS = {
    "basketball_nba": "NBA",
    "basketball_ncaab": "NCAAM",
    "americanfootball_nfl": "NFL",
    "americanfootball_ncaaf": "NCAAF",
    "icehockey_nhl": "NHL",
    "soccer_epl": "EPL",
    "soccer_spain_la_liga": "La Liga",
    "soccer_uefa_champs_league": "UCL",
    "soccer_usa_mls": "MLS",
    "baseball_mlb": "MLB",
}

DEPORTES_ACTIVOS = list(SPORT_LABELS.keys())

# Submercados oficiales The Odds API v4 separados por categoría para evitar errores 422
SPORT_SUBMARKETS = {
    "basketball_nba": {
        "mitades": ["h2h_h1", "spreads_h1", "totals_h1", "h2h_h2", "spreads_h2", "totals_h2"],
        "props": ["player_points", "player_rebounds", "player_assists", "player_threes"],
    },
    "basketball_ncaab": {
        "mitades": ["h2h_h1", "spreads_h1", "totals_h1", "h2h_h2", "spreads_h2", "totals_h2"],
        "props": [],
    },
    "americanfootball_nfl": {
        "mitades": ["h2h_h1", "spreads_h1", "totals_h1", "h2h_h2", "spreads_h2", "totals_h2"],
        "props": ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions"],
    },
    "americanfootball_ncaaf": {
        "mitades": ["h2h_h1", "spreads_h1", "totals_h1", "h2h_h2", "spreads_h2", "totals_h2"],
        "props": [],
    },
    "icehockey_nhl": {
        "mitades": ["h2h_p1", "spreads_p1", "totals_p1"],
        "props": ["player_shots_on_goal", "player_points", "player_assists"],
    },
    "soccer_epl": {
        "mitades": ["h2h_h1", "totals_h1"],
        "props": ["btts", "alternate_totals_corners"],
    },
    "soccer_spain_la_liga": {
        "mitades": ["h2h_h1", "totals_h1"],
        "props": ["btts", "alternate_totals_corners"],
    },
    "soccer_uefa_champs_league": {
        "mitades": ["h2h_h1", "totals_h1"],
        "props": ["btts", "alternate_totals_corners"],
    },
    "soccer_usa_mls": {
        "mitades": ["h2h_h1", "totals_h1"],
        "props": ["btts"],
    },
    "baseball_mlb": {
        "mitades": ["h2h_1st_5_innings", "spreads_1st_5_innings", "totals_1st_5_innings"],
        "props": ["batter_hits", "batter_total_bases", "pitcher_strikeouts"],
    },
}

MARKET_LABELS = {
    "h2h": "MONEYLINE",
    "spreads": "SPREAD",
    "totals": "TOTAL",
    "h2h_h1": "1H MONEYLINE",
    "spreads_h1": "1H SPREAD",
    "totals_h1": "1H TOTAL",
    "h2h_h2": "2H MONEYLINE",
    "spreads_h2": "2H SPREAD",
    "totals_h2": "2H TOTAL",
    "h2h_p1": "1P MONEYLINE",
    "spreads_p1": "1P SPREAD",
    "totals_p1": "1P TOTAL",
    "h2h_1st_5_innings": "F5 MONEYLINE",
    "spreads_1st_5_innings": "F5 SPREAD",
    "totals_1st_5_innings": "F5 TOTAL",
    "player_points": "PROP PUNTOS",
    "player_rebounds": "PROP REBOTES",
    "player_assists": "PROP ASISTENCIAS",
    "player_threes": "PROP TRIPLES",
    "player_pass_yds": "PROP YARDAS PASE",
    "player_rush_yds": "PROP YARDAS CARRERA",
    "player_reception_yds": "PROP YARDAS RECEPCIÓN",
    "player_receptions": "PROP RECEPCIONES",
    "player_shots_on_goal": "PROP TIROS A PUERTA",
    "batter_hits": "PROP HITS BATEADOR",
    "batter_total_bases": "PROP BASES TOTALES",
    "pitcher_strikeouts": "PROP PONCHES",
    "btts": "AMBOS ANOTAN (BTTS)",
    "alternate_totals_corners": "CÓRNERS TOTALES",
}


# ==============================================================================
# 4. ZONA HORARIA ET (MIAMI) CON RESPALDO SEGURO PARA RENDER LINUX SLIM
# ==============================================================================
def _us_eastern_offset(dt_utc: datetime) -> timezone:
    year = dt_utc.year
    # 2do domingo de marzo (07:00 UTC = 02:00 EST)
    mar1 = datetime(year, 3, 1, tzinfo=timezone.utc)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7, hours=7)
    # 1er domingo de noviembre (06:00 UTC = 02:00 EDT)
    nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7, hours=6)
    if dst_start <= dt_utc < dst_end:
        return timezone(timedelta(hours=-4))
    return timezone(timedelta(hours=-5))


def to_et(dt_utc: datetime) -> datetime:
    if TZ_ET is not None:
        return dt_utc.astimezone(TZ_ET)
    return dt_utc.astimezone(_us_eastern_offset(dt_utc))


def now_et() -> datetime:
    return to_et(datetime.now(timezone.utc))


def parse_iso_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def formato_cuenta_regresiva(minutos: float) -> str:
    mins = max(0, int(round(minutos)))
    if mins < 60:
        return f"en {mins} min"
    horas, rem = divmod(mins, 60)
    return f"en {horas}h {rem:02d}m"


def decimal_a_americana(dec: float) -> str:
    if dec >= 2.0:
        return f"+{int(round((dec - 1.0) * 100))}"
    if dec <= 1.0:
        return "N/A"
    return f"{int(round(-100.0 / (dec - 1.0)))}"


# ==============================================================================
# 5. MOTOR MATEMÁTICO: POWER DE-VIG, SHIN DE-VIG EXACTO Y KELLY CONSERVADOR
# ==============================================================================
def devig_power(odds_list: List[float]) -> Optional[List[float]]:
    if any(o <= 1.0 for o in odds_list) or len(odds_list) < 2:
        return None
    qs = [1.0 / o for o in odds_list]
    if sum(qs) <= 1.0:
        total = sum(qs)
        return [q / total for q in qs]

    lo, hi = 0.1, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = sum(q ** mid for q in qs)
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    probs = [q ** k for q in qs]
    norm = sum(probs)
    return [p / norm for p in probs]


def devig_shin(odds_list: List[float]) -> Optional[List[float]]:
    if any(o <= 1.0 for o in odds_list) or len(odds_list) < 2:
        return None
    qs = [1.0 / o for o in odds_list]
    M = sum(qs)
    if M <= 1.0:
        return [q / M for q in qs]

    def shin_probs(z: float) -> List[float]:
        denom = 2.0 * (1.0 - z)
        out = []
        for q in qs:
            disc = z * z + 4.0 * (1.0 - z) * (q * q / M)
            out.append((math.sqrt(max(0.0, disc)) - z) / denom)
        return out

    lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sum(shin_probs(mid)) > 1.0:
            lo = mid
        else:
            hi = mid
    probs = shin_probs(0.5 * (lo + hi))
    norm = sum(probs)
    return [p / norm for p in probs]


def calcular_metricas(
    p_real: float, cuota_dec: float, categoria: str
) -> Optional[Dict]:
    if cuota_dec < CUOTA_MIN_DECIMAL or cuota_dec <= 1.0 or not (0.0 < p_real < 1.0):
        return None

    cfg = CONFIG_CATEGORIAS[categoria]
    ev_est = p_real * cuota_dec - 1.0
    if ev_est < cfg["min_ev"]:
        return None

    p_cons = max(0.001, p_real - cfg["delta_p"])
    ev_cons = p_cons * cuota_dec - 1.0
    if ev_cons <= 0.0:
        return None

    # Kelly fraccionado aplicado sobre el edge conservador
    kelly_full = max(0.0, ev_cons / (cuota_dec - 1.0))
    frac_stake = min(cfg["cap_pct"], cfg["kelly_mult"] * kelly_full)

    # Cuota mínima válida en Hard Rock Bet para mantener piso y EV conservador positivo
    cuota_min_ev = (1.0 + cfg["min_ev"]) / p_real
    cuota_min_cons = 1.0001 / p_cons
    cuota_min_valida = max(CUOTA_MIN_DECIMAL, cuota_min_ev, cuota_min_cons)

    return {
        "ev_est": ev_est,
        "ev_cons": ev_cons,
        "p_real": p_real,
        "p_cons": p_cons,
        "p_imp_hrb": 1.0 / cuota_dec,
        "frac_stake": frac_stake,
        "cuota_min_valida": cuota_min_valida,
    }


# ==============================================================================
# 6. UTILIDADES DE API, TELEGRAM, CSV Y MODOS HORARIOS ET
# ==============================================================================
def obtener_modo_horario() -> Tuple[str, int]:
    """Devuelve (modo, intervalo_segundos) según hora local ET (Miami)."""
    hora = now_et().hour
    if 17 <= hora <= 23:
        return "TURBO", 180     # Ventana pico NBA/NCAAM/NHL/NFL (cada 3 min)
    if 11 <= hora < 17:
        return "NORMAL", 420    # Fútbol Europa + apertura líneas US (cada 7 min)
    return "AHORRO", 900        # Madrugada (cada 15 min)


def enviar_telegram(texto: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM DESACTIVADO]", texto)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": texto,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=12,
        )
    except Exception as exc:
        print(f"[ERROR TELEGRAM] {exc}")


def registrar_csv(op: Dict) -> None:
    file_exists = os.path.isfile(CSV_LOG_FILE)
    campos = [
        "timestamp_et", "liga", "hora_partido_et", "partido", "categoria",
        "mercado", "seleccion", "cuota_hrb_dec", "cuota_min_dec",
        "p_real", "p_cons", "ev_est_pct", "ev_cons_pct",
        "stake_pct", "stake_usd", "edad_dato_seg", "sharps_usadas",
    ]
    try:
        with open(CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp_et": now_et().strftime("%Y-%m-%d %H:%M:%S"),
                "liga": op["liga"],
                "hora_partido_et": op["hora_et_str"],
                "partido": op["partido"],
                "categoria": op["categoria"],
                "mercado": op["mercado_label"],
                "seleccion": op["seleccion_label"],
                "cuota_hrb_dec": f"{op['cuota_dec']:.3f}",
                "cuota_min_dec": f"{op['cuota_min_valida']:.3f}",
                "p_real": f"{op['p_real']:.4f}",
                "p_cons": f"{op['p_cons']:.4f}",
                "ev_est_pct": f"{op['ev_est'] * 100:.2f}",
                "ev_cons_pct": f"{op['ev_cons'] * 100:.2f}",
                "stake_pct": f"{op['stake_pct_final'] * 100:.2f}",
                "stake_usd": f"{op['stake_usd']:.2f}",
                "edad_dato_seg": op["edad_dato_seg"],
                "sharps_usadas": op["num_sharps"],
            })
    except Exception as exc:
        print(f"[ERROR CSV] {exc}")


# ==============================================================================
# 7. EXTRACCIÓN Y EVALUACIÓN EXACTA DE CONTRATOS (MISMO CONTRATO Y LÍNEA)
# ==============================================================================
def _contract_group_key(outcome: Dict, market_key: str) -> Tuple:
    """Agrupa estrictamente los lados opuestos del MISMO contrato y línea."""
    point = outcome.get("point")
    desc = (outcome.get("description") or "").strip().lower()
    if market_key.startswith(("player_", "batter_", "pitcher_")):
        return (desc, point)
    if market_key in ("totals", "totals_h1", "totals_h2", "totals_p1",
                      "totals_1st_5_innings", "alternate_totals_corners"):
        return ("total", point)
    if market_key in ("spreads", "spreads_h1", "spreads_h2", "spreads_p1",
                      "spreads_1st_5_innings"):
        # En spreads, -5.5 y +5.5 pertenecen al mismo par simétrico (abs(point))
        return ("spread", abs(float(point)) if point is not None else None)
    return ("base", None)


def _outcome_id(outcome: Dict) -> Tuple:
    name = (outcome.get("name") or "").strip()
    desc = (outcome.get("description") or "").strip()
    point = outcome.get("point")
    return (desc, name, point)


def _format_seleccion(outcome_id: Tuple) -> str:
    desc, name, point = outcome_id
    partes = []
    if desc:
        partes.append(desc)
    partes.append(name)
    if point is not None:
        pt_str = f"{point:+g}" if name not in ("Over", "Under") else f"{point:g}"
        partes.append(pt_str)
    return " ".join(partes)


def evaluar_evento(evento: Dict, sport_key: str, categoria_por_mercado: Dict[str, str]) -> List[Dict]:
    oportunidades = []
    now_utc = datetime.now(timezone.utc)
    commence_utc = parse_iso_utc(evento["commence_time"])

    # Regla dura: solo prepartido (evita trampa de latencia en vivo)
    minutos_inicio = (commence_utc - now_utc).total_seconds() / 60.0
    if minutos_inicio <= 1.0:
        return []

    bookmakers = evento.get("bookmakers", [])
    hrb_book = next((b for b in bookmakers if b.get("key") == BOOKMAKER_TARGET), None)
    if not hrb_book:
        return []

    sharp_books = [b for b in bookmakers if b.get("key") in SHARP_WEIGHTS]
    if not sharp_books:
        return []

    liga = SPORT_LABELS.get(sport_key, sport_key.upper())
    partido = f"{evento.get('away_team', '')} @ {evento.get('home_team', '')}"
    commence_et = to_et(commence_utc)
    hora_et_str = commence_et.strftime("%I:%M %p ET")

    # Indexar mercados de Hard Rock Bet
    for hrb_mkt in hrb_book.get("markets", []):
        m_key = hrb_mkt.get("key", "")
        categoria = categoria_por_mercado.get(m_key)
        if not categoria:
            continue

        cfg = CONFIG_CATEGORIAS[categoria]
        hrb_update = parse_iso_utc(hrb_mkt.get("last_update") or hrb_book.get("last_update"))
        hrb_age = max(0, int((now_utc - hrb_update).total_seconds()))
        if hrb_age > cfg["max_feed_age"]:
            continue

        # Recolectar probabilidades de-viggeadas de cada sharp para cada contrato exacto
        sharp_probs_acc: Dict[Tuple, List[Tuple[float, float, int]]] = {}

        for sb in sharp_books:
            sb_key = sb["key"]
            peso = SHARP_WEIGHTS[sb_key]
            sb_mkt = next((m for m in sb.get("markets", []) if m.get("key") == m_key), None)
            if not sb_mkt:
                continue

            sb_update = parse_iso_utc(sb_mkt.get("last_update") or sb.get("last_update"))
            sb_age = max(0, int((now_utc - sb_update).total_seconds()))
            if sb_age > cfg["max_feed_age"]:
                continue

            # Agrupar resultados por contrato exacto (exige ambos lados del mismo mercado)
            grupos: Dict[Tuple, List[Dict]] = {}
            for out in sb_mkt.get("outcomes", []):
                g_key = _contract_group_key(out, m_key)
                grupos.setdefault(g_key, []).append(out)

            for g_outcomes in grupos.values():
                # Obligatorio tener al menos 2 lados del mismo contrato para de-vig exacto
                if len(g_outcomes) < 2:
                    continue
                odds_vals = [float(o["price"]) for o in g_outcomes]
                probs = devig_shin(odds_vals) if cfg["devig"] == "shin" else devig_power(odds_vals)
                if not probs:
                    continue
                for o_obj, p_val in zip(g_outcomes, probs):
                    oid = _outcome_id(o_obj)
                    sharp_probs_acc.setdefault(oid, []).append((p_val, peso, sb_age))

        # Evaluar cada cuota de Hard Rock Bet contra el consenso sharp exacto
        for out in hrb_mkt.get("outcomes", []):
            oid = _outcome_id(out)
            if oid not in sharp_probs_acc:
                continue

            cuota_dec = float(out["price"])
            registros = sharp_probs_acc[oid]
            total_peso = sum(w for _, w, _ in registros)
            p_real = sum(p * w for p, w, _ in registros) / total_peso
            max_age = max(hrb_age, max(age for _, _, age in registros))

            metricas = calcular_metricas(p_real, cuota_dec, categoria)
            if not metricas:
                continue

            jugador_key = oid[0].lower() if oid[0] else ""
            oportunidades.append({
                "event_id": evento["id"],
                "sport_key": sport_key,
                "liga": liga,
                "partido": partido,
                "commence_utc": commence_utc,
                "hora_et_str": hora_et_str,
                "minutos_inicio": minutos_inicio,
                "categoria": categoria,
                "mercado_key": m_key,
                "mercado_label": MARKET_LABELS.get(m_key, m_key.upper()),
                "seleccion_id": oid,
                "seleccion_label": _format_seleccion(oid),
                "jugador_key": jugador_key,
                "cuota_dec": cuota_dec,
                "cuota_am": decimal_a_americana(cuota_dec),
                "edad_dato_seg": max_age,
                "num_sharps": len(registros),
                **metricas,
            })

    return oportunidades


# ==============================================================================
# 8. GESTOR DE EXPOSICIÓN AGREGADA (PARTIDO / JUGADOR / CICLO)
# ==============================================================================
def aplicar_topes_exposicion(oportunidades: List[Dict]) -> List[Dict]:
    # Priorizar mayor EV conservador al asignar presupuesto, luego ordenar cronológicamente
    ordenadas_ev = sorted(oportunidades, key=lambda x: x["ev_cons"], reverse=True)

    exp_partido: Dict[str, float] = {}
    exp_jugador: Dict[str, float] = {}
    exp_total = 0.0
    aprobadas = []

    for op in ordenadas_ev:
        if exp_total >= MAX_EXPOSICION_CICLO:
            break

        ev_id = op["event_id"]
        jug = op["jugador_key"]

        rem_total = MAX_EXPOSICION_CICLO - exp_total
        rem_partido = MAX_EXPOSICION_PARTIDO - exp_partido.get(ev_id, 0.0)
        rem_jugador = (
            MAX_EXPOSICION_JUGADOR - exp_jugador.get(jug, 0.0)
            if jug else MAX_EXPOSICION_CICLO
        )

        stake_pct = min(op["frac_stake"], rem_total, rem_partido, rem_jugador)
        if stake_pct <= 0.0005:  # Menos de 0.05% del bankroll se omite
            continue

        stake_usd = round(BANKROLL * stake_pct, 2)
        if stake_usd < 1.0:
            continue

        exp_total += stake_pct
        exp_partido[ev_id] = exp_partido.get(ev_id, 0.0) + stake_pct
        if jug:
            exp_jugador[jug] = exp_jugador.get(jug, 0.0) + stake_pct

        op["stake_pct_final"] = stake_pct
        op["stake_usd"] = stake_usd
        aprobadas.append(op)

    # Orden final 100% CRONOLÓGICO (partidos más cercanos primero)
    aprobadas.sort(key=lambda x: (x["commence_utc"], -x["ev_est"]))
    return aprobadas


# ==============================================================================
# 9. CONSULTAS A THE ODDS API (CAPA 1: 10 DEPORTES | CAPA 2: SUBMERCADOS)
# ==============================================================================
def obtener_deportes_en_temporada() -> List[str]:
    """Consulta gratuita (0 créditos) para escanear solo ligas con cartelera activa."""
    url = "https://api.the-odds-api.com/v4/sports/"
    try:
        r = requests.get(url, params={"apiKey": ODDS_API_KEY}, timeout=12)
        if r.status_code == 200:
            activos_api = {s["key"] for s in r.json() if s.get("active")}
            return [s for s in DEPORTES_ACTIVOS if s in activos_api]
    except Exception as exc:
        print(f"[WARN SPORTS] {exc}")
    return DEPORTES_ACTIVOS


def escanear_ciclo() -> Tuple[List[Dict], Optional[str]]:
    oportunidades_raw: List[Dict] = []
    creditos_restantes: Optional[str] = None
    now_utc = datetime.now(timezone.utc)
    deportes = obtener_deportes_en_temporada()

    for sport_key in deportes:
        # --- CAPA 1: JUEGO PRINCIPAL (h2h, spreads, totals) ---
        url_main = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        params_main = {
            "apiKey": ODDS_API_KEY,
            "regions": "us,eu",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal",
        }
        try:
            resp = requests.get(url_main, params=params_main, timeout=15)
            creditos_restantes = resp.headers.get("x-requests-remaining", creditos_restantes)
            if resp.status_code != 200:
                continue
            eventos = resp.json()
        except Exception as exc:
            print(f"[ERROR MAIN {sport_key}] {exc}")
            continue

        cat_main = {"h2h": "principal", "spreads": "principal", "totals": "principal"}
        eventos_ventana_sub = []

        for ev in eventos:
            oportunidades_raw.extend(evaluar_evento(ev, sport_key, cat_main))
            commence_utc = parse_iso_utc(ev["commence_time"])
            horas_faltan = (commence_utc - now_utc).total_seconds() / 3600.0
            if 0.05 <= horas_faltan <= SUBMARKET_WINDOW_HOURS:
                eventos_ventana_sub.append(ev)

        # --- CAPA 2: MITADES Y PROPS (SOLO EVENTOS DENTRO DE VENTANA PREPARTIDO) ---
        if not ENABLE_SUBMARKETS:
            continue

        sub_cfg = SPORT_SUBMARKETS.get(sport_key, {})
        for ev in eventos_ventana_sub:
            ev_id = ev["id"]
            for sub_cat in ("mitades", "props"):
                mkts = sub_cfg.get(sub_cat, [])
                if not mkts:
                    continue
                url_ev = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{ev_id}/odds"
                params_ev = {
                    "apiKey": ODDS_API_KEY,
                    "regions": "us,eu",
                    "markets": ",".join(mkts),
                    "oddsFormat": "decimal",
                }
                try:
                    r_sub = requests.get(url_ev, params=params_ev, timeout=15)
                    creditos_restantes = r_sub.headers.get("x-requests-remaining", creditos_restantes)
                    if r_sub.status_code != 200:
                        continue
                    ev_sub = r_sub.json()
                    cat_map = {m: sub_cat for m in mkts}
                    oportunidades_raw.extend(evaluar_evento(ev_sub, sport_key, cat_map))
                except Exception as exc:
                    print(f"[ERROR SUB {sport_key}/{ev_id}] {exc}")

    aprobadas = aplicar_topes_exposicion(oportunidades_raw)
    return aprobadas, creditos_restantes


# ==============================================================================
# 10. FORMATEO ULTRARRÁPIDO PARA TELEGRAM (ENTRADA DIRECTA A HARD ROCK BET)
# ==============================================================================
def formatear_alerta(op: Dict) -> str:
    cfg = CONFIG_CATEGORIAS[op["categoria"]]
    countdown = formato_cuenta_regresiva(op["minutos_inicio"])
    cuota_min_am = decimal_a_americana(op["cuota_min_valida"])

    return (
        f"{cfg['icono']} <b>{op['liga']} | ⏰ {op['hora_et_str']} ({countdown})</b>\n"
        f"🏟 <b>{op['partido']}</b>\n"
        f"📌 <b>{op['mercado_label']} → {op['seleccion_label']}</b>\n"
        f"💵 <b>Hard Rock Bet: {op['cuota_am']} (dec {op['cuota_dec']:.2f})</b>\n"
        f"🛡 Cuota mín. válida: <b>{cuota_min_am} (dec {op['cuota_min_valida']:.2f})</b>\n"
        f"💰 <b>Stake guía: ${op['stake_usd']:.2f} ({op['stake_pct_final']*100:.2f}% BR)</b>\n"
        f"📈 EV Est: <b>+{op['ev_est']*100:.1f}%</b> | EV Cons: <b>+{op['ev_cons']*100:.1f}%</b>\n"
        f"📊 p_real: {op['p_real']*100:.1f}% ({op['num_sharps']} sharps, {cfg['devig'].upper()}) | p_HRB: {op['p_imp_hrb']*100:.1f}%\n"
        f"⚡ Dato: hace {op['edad_dato_seg']}s | <b>Ejecutar en ≤ {cfg['exec_window_sec']}s</b>"
    )


def formatear_resumen(aprobadas: List[Dict], modo: str, creditos: Optional[str]) -> str:
    hora_actual = now_et().strftime("%I:%M %p ET")
    lineas = [
        f"📊 <b>RESUMEN CRONOLÓGICO [{modo}] — {hora_actual}</b>",
        f"🔥 Oportunidades ejecutables: <b>{len(aprobadas)}</b>",
        "",
    ]
    for i, op in enumerate(aprobadas[:12], start=1):
        cfg = CONFIG_CATEGORIAS[op["categoria"]]
        lineas.append(
            f"{i}. {cfg['icono']} <b>{op['hora_et_str']}</b> | {op['liga']} | "
            f"{op['partido']} → <b>{op['mercado_label']}: {op['seleccion_label']}</b> "
            f"@ {op['cuota_am']} (dec {op['cuota_dec']:.2f}) | "
            f"<b>+{op['ev_est']*100:.1f}%</b> | <b>${op['stake_usd']:.2f}</b>"
        )
    if creditos is not None:
        lineas.append(f"\n🔋 Créditos API restantes: {creditos}")
    return "\n".join(lineas)


# ==============================================================================
# 11. SERVIDOR HEALTH CHECK PARA RENDER + BUCLE PRINCIPAL 24/7
# ==============================================================================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Real Harvest Radar V4.0 ELITE OK")

    def log_message(self, format, *args):
        return


def iniciar_servidor_health() -> None:
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"[RENDER HEALTH] Escuchando en puerto {port}")
    server.serve_forever()


def main() -> None:
    if not ODDS_API_KEY:
        raise RuntimeError("Falta configurar la variable de entorno ODDS_API_KEY en Render.")

    threading.Thread(target=iniciar_servidor_health, daemon=True).start()

    historial_alertas: Dict[Tuple, Tuple[float, float]] = {}
    print(f"[INICIO] Real Harvest Radar V4.0 ÉLITE activo | Bankroll: ${BANKROLL:.2f}")

    while True:
        modo, intervalo = obtener_modo_horario()
        try:
            aprobadas, creditos = escanear_ciclo()
            ahora_ts = time.time()
            nuevas_alertas = []

            for op in aprobadas:
                clave = (op["event_id"], op["mercado_key"], op["seleccion_id"])
                prev = historial_alertas.get(clave)
                # Enviar si es nueva, pasó el cooldown o el EV subió >= 1.5%
                if (
                    prev is None
                    or (ahora_ts - prev[0]) >= ALERT_COOLDOWN_SEC
                    or (op["ev_est"] - prev[1]) >= 0.015
                ):
                    nuevas_alertas.append(op)
                    historial_alertas[clave] = (ahora_ts, op["ev_est"])
                    registrar_csv(op)

            if nuevas_alertas:
                # 1ro: Mapa cronológico completo para priorizar en 2 segundos
                enviar_telegram(formatear_resumen(nuevas_alertas, modo, creditos))
                # 2do: Tarjetas individuales en orden cronológico ascendente
                for op in nuevas_alertas:
                    enviar_telegram(formatear_alerta(op))
                    time.sleep(0.4)

            print(
                f"[{now_et().strftime('%H:%M:%S ET')}] Ciclo [{modo}] completado | "
                f"Aprobadas: {len(aprobadas)} | Enviadas: {len(nuevas_alertas)} | "
                f"Créditos: {creditos} | Próximo en {intervalo}s"
            )
        except Exception as exc:
            print(f"[ERROR CICLO] {exc}")

        time.sleep(intervalo)


if __name__ == "__main__":
    main()
