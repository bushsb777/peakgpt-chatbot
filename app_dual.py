# app_dual.py — AlphaBot with dual scoring engines (Momentum & Mean-Value)
# PATCHED: Added mean-value engine with toggle, rebound gate, and engine field in output.
# Original momentum logic remains unchanged.

# Notes:
# - Momentum uses risk‑adjusted 12‑1 multi‑horizon (3/6/12 months, skip last 21 days).
# - Mean-value is long-only mean reversion (z-score, RSI, Bollinger Bands, volatility normalization).
# - Required keys in .env: FMP_API_KEY, FINNHUB_API_KEY, SERPAPI_API_KEY
# - Optional keys: POLYGON_API_KEY, TIINGO_API_KEY

import os
import re
import json
import time
import hashlib
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from uuid import uuid4
from typing import List, Dict

# ──────────────────────────────────────────────────────────────────────────────

# OpenAI client (guarded)
try:
    from openai import OpenAI  # v1.x style
    _openai_import_ok = True
except Exception:
    OpenAI = None
    _openai_import_ok = False

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional parser for URL summaries
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# ──────────────────────────────────────────────────────────────────────────────

# Keys / clients
FMP_API_KEY = os.getenv("FMP_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Optional extra vendors
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
TIINGO_API_KEY  = os.getenv("TIINGO_API_KEY", "")

if _openai_import_ok:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = None
else:
    client = None

DISCLAIMER = "_Educational content only. Not investment advice._"
RULES_VERSION = "2025-08-14-v6"

# Where to store audit records
LEDGER_DIR = os.getenv("LEDGER_DIR", "audit/ledger")

st.set_page_config(page_title="AlphaGPT – Financial Intelligence Bot", page_icon="🧠", layout="wide")
st.title("🧠 AlphaGPT – Financial Intelligence Bot")
st.caption("AlphaBot: Daily Buy, Sell, or Hold Signals.")

# Sidebar toggle for selecting scoring engine
engine_choice = st.sidebar.radio("Scoring Engine:", ["Momentum", "Mean Value"], index=0)

# Streamlit toggle shim (older versions may not have sidebar.toggle)
def sidebar_toggle(label: str, value: bool = False):
    if hasattr(st.sidebar, "toggle"):
        return st.sidebar.toggle(label, value=value)
    return st.sidebar.checkbox(label, value=value)

show_diag = sidebar_toggle("🔧 Diagnostics", value=False)
show_deep = sidebar_toggle("🔎 Deep Dive (detailed)", value=True)
# NEW: force‑live toggle (bypass nightly precompute)
force_live = sidebar_toggle("🚫 Use nightly precompute (serve live data)", value=False)

_diag = []

def dlog(tag, **kw):
    if show_diag:
        _diag.append({"tag": tag, **kw})

TICKER_RE = re.compile(r"\b[A-Za-z]{1,6}\b")
URL_RE = re.compile(r"https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+", re.IGNORECASE)

CRYPTO_GUESS = {
    "BTC","ETH","SOL","ADA","BNB","XRP","DOGE","MATIC","DOT","LTC",
    "AVAX","LINK","ATOM","TRX","XLM","BCH","ETC","APT","ARB","OP",
    "SUI","TON","NEAR","ALGO","FIL","ICP","HBAR","EGLD","AAVE","UNI"
}

# Precomputed cache reader (instant lookups)
PRECOMP_PATH = Path("precomputed/latest.json")

@st.cache_data(ttl=15*60, show_spinner=False)
def load_precomputed_index():
    try:
        with open(PRECOMP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def precomputed_lookup(symbol: str) -> dict:
    idx = load_precomputed_index() or {}
    return idx.get((symbol or "").upper(), {})

# Markdown safety helpers
def md_escape(s: str) -> str:
    if not s:
        return ""
    bs = chr(92)
    s = str(s).replace(bs, bs + bs)
    s = s.replace("*", bs + "*").replace("_", bs + "_")
    lines = s.splitlines()
    clean = []
    for ln in lines:
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        ln = re.sub(r"^[•·]\s*", "- ", ln)
        ln = re.sub(r"^\d+\.\s+", "- ", ln)
        clean.append(ln)
    return "\n".join(clean)

# HTTP helpers (with simple circuit breaker)
def _get_json(url, params=None, headers=None, tries=2, timeout=20, soft=False, provider="fmp"):
    key = f"circuit_{provider}_until"
    until = st.session_state.get(key, 0)
    now = time.time()
    if until and now < until:
        return {} if soft else {}
    last_exc = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                st.session_state[key] = time.time() + 60
                if soft:
                    return {}
            r.raise_for_status()
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * (i + 1) + np.random.rand() * 0.2)
    if soft:
        return {}
    raise last_exc

# FMP / Finnhub wrappers
@st.cache_data(ttl=15*60)
def fmp_quote(symbol: str):
    if not FMP_API_KEY:
        return {}
    try:
        js = _get_json(
            f"https://financialmodelingprep.com/api/v3/quote/{symbol.upper()}",
            {"apikey": FMP_API_KEY}, soft=True, provider="fmp"
        )
        return js[0] if isinstance(js, list) and js else {}
    except Exception:
        return {}

@st.cache_data(ttl=6*3600)
def fmp_profile(symbol: str):
    if not FMP_API_KEY:
        return {}
    try:
        js = _get_json(
            f"https://financialmodelingprep.com/api/v3/profile/{symbol.upper()}",
            {"apikey": FMP_API_KEY}, soft=True, provider="fmp"
        )
        return js[0] if isinstance(js, list) and js else {}
    except Exception:
        return {}

@st.cache_data(ttl=24*3600)
def fmp_history(symbol: str, days: int = 400):
    if not FMP_API_KEY:
        return []
    try:
        js = _get_json(
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol.upper()}",
            {"timeseries": days, "apikey": FMP_API_KEY}, soft=True, provider="fmp"
        )
        hist = js.get("historical") if isinstance(js, dict) else None
        if not hist:
            return []
        hist = list(reversed(hist))
        return [{"date": h.get("date"), "close": float(h.get("close"))} for h in hist if h.get("close") is not None]
    except Exception:
        return []

@st.cache_data(ttl=6*3600)
def fmp_filings(symbol: str, limit: int = 8):
    if not FMP_API_KEY:
        return []
    try:
        js = _get_json(
            f"https://financialmodelingprep.com/api/v3/sec_filings/{symbol.upper()}",
            {"limit": limit, "apikey": FMP_API_KEY}, soft=True, provider="fmp"
        )
        return js if isinstance(js, list) else []
    except Exception:
        return []

@st.cache_data(ttl=6*3600)
def finnhub_profile(symbol: str):
    if not FINNHUB_API_KEY:
        return {}
    try:
        url = "https://finnhub.io/api/v1/stock/profile2"
        js = _get_json(url, params={"symbol": symbol.upper(), "token": FINNHUB_API_KEY}, soft=True, provider="finnhub")
        return js if isinstance(js, dict) else {}
    except Exception:
        return {}

@st.cache_data(ttl=5*60)
def finnhub_quote(symbol: str):
    if not FINNHUB_API_KEY:
        return {}
    try:
        url = "https://finnhub.io/api/v1/quote"
        js = _get_json(url, params={"symbol": symbol.upper(), "token": FINNHUB_API_KEY}, soft=True, provider="finnhub")
        if not isinstance(js, dict):
            return {}
        return {"price": js.get("c"), "previousClose": js.get("pc"), "name": symbol.upper()}
    except Exception:
        return {}

# Finnhub daily candles (stocks)
@st.cache_data(ttl=24*3600)
def finnhub_stock_candles(symbol: str, days: int = 400) -> List[Dict]:
    if not FINNHUB_API_KEY:
        return []
    try:
        now = int(time.time())
        frm = now - int(days * 86400 * 1.3)
        url = "https://finnhub.io/api/v1/stock/candle"
        js = _get_json(url, params={
            "symbol": symbol.upper(),
            "resolution": "D",
            "from": frm,
            "to": now,
            "token": FINNHUB_API_KEY
        }, soft=True, provider="finnhub")
        if not isinstance(js, dict) or js.get("s") != "ok":
            return []
        closes, times = js.get("c") or [], js.get("t") or []
        out = []
        for t, c in zip(times, closes):
            try:
                out.append({"date": datetime.utcfromtimestamp(int(t)).strftime("%Y-%m-%d"), "close": float(c)})
            except Exception:
                pass
        return out
    except Exception:
        return []

# Finnhub daily candles (crypto via BINANCE:SYMBOLUSDT)
@st.cache_data(ttl=24*3600)
def finnhub_crypto_candles(symbol: str, days: int = 400) -> List[Dict]:
    if not FINNHUB_API_KEY:
        return []
    try:
        base = symbol.upper()
        pair = f"BINANCE:{base}USDT"
        now = int(time.time())
        frm = now - int(days * 86400 * 1.3)
        url = "https://finnhub.io/api/v1/crypto/candle"
        js = _get_json(url, params={
            "symbol": pair,
            "resolution": "D",
            "from": frm,
            "to": now,
            "token": FINNHUB_API_KEY
        }, soft=True, provider="finnhub")
        if not isinstance(js, dict) or js.get("s") != "ok":
            return []
        closes, times = js.get("c") or [], js.get("t") or []
        out = []
        for t, c in zip(times, closes):
            try:
                out.append({"date": datetime.utcfromtimestamp(int(t)).strftime("%Y-%m-%d"), "close": float(c)})
            except Exception:
                pass
        return out
    except Exception:
        return []

# Tiingo & Polygon (EOD) — new helpers
@st.cache_data(ttl=24*3600)
def tiingo_history(symbol: str, days: int = 400) -> list[dict]:
    if not TIINGO_API_KEY:
        return []
    try:
        end = datetime.utcnow().date()
        start = end - pd.tseries.offsets.BDay(int(days*1.2))
        url = f"https://api.tiingo.com/tiingo/daily/{symbol.upper()}/prices"
        js = _get_json(url, params={
            "startDate": str(start.date()),
            "endDate": str(end),
            "token": TIINGO_API_KEY
        }, soft=True, provider="tiingo")
        if not isinstance(js, list):
            return []
        out = []
        for row in js:
            c = row.get("close"); d = (row.get("date") or "")[:10]
            if c is not None and d:
                out.append({"date": d, "close": float(c)})
        return out
    except Exception:
        return []

@st.cache_data(ttl=24*3600)
def polygon_history(symbol: str, days: int = 400) -> list[dict]:
    if not POLYGON_API_KEY:
        return []
    try:
        end = datetime.utcnow().date()
        start = end - pd.tseries.offsets.BDay(int(days*1.2))
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start.date()}/{end}"
        js = _get_json(url, params={"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY}, soft=True, provider="polygon")
        results = (js or {}).get("results") or []
        out = []
        for r in results:
            t = r.get("t"); c = r.get("c")
            if t is None or c is None:
                continue
            d = datetime.utcfromtimestamp(int(t)/1000).strftime("%Y-%m-%d")
            out.append({"date": d, "close": float(c)})
        return out
    except Exception:
        return []

# Unified history fallback (FMP → Finnhub crypto → Finnhub stock → Polygon → Tiingo)
@st.cache_data(ttl=24*3600)
def history_any(symbol: str, days: int = 400) -> list[dict]:
    sym = (symbol or "").upper()
    # 1) FMP
    h = fmp_history(sym, days=days)
    if len(h) >= 280:
        dlog("hist_vendor", vendor="fmp", len=len(h))
        return h
    # 2) Crypto route first if looks like crypto
    if sym in CRYPTO_GUESS:
        hc = finnhub_crypto_candles(sym, days=days)
        if len(hc) >= 280:
            dlog("hist_vendor", vendor="finnhub_crypto", len=len(hc))
            return hc
    # 3) Finnhub stock/ETF
    hs = finnhub_stock_candles(sym, days=days)
    if len(hs) >= 280:
        dlog("hist_vendor", vendor="finnhub_stock", len=len(hs))
        return hs
    # 4) Polygon
    hp = polygon_history(sym, days=days)
    if len(hp) >= 280:
        dlog("hist_vendor", vendor="polygon", len=len(hp))
        return hp
    # 5) Tiingo
    ht = tiingo_history(sym, days=days)
    if len(ht) >= 280:
        dlog("hist_vendor", vendor="tiingo", len=len(ht))
        return ht
    # best effort; diagnostics will warn if short
    best = max([h, hs, hp, ht], key=lambda z: len(z) if z else 0) or []
    dlog("hist_vendor", vendor="best_effort", len=len(best))
    return best

# Headlines (SerpAPI) — with Diagnostics logs
@st.cache_data(ttl=3600)
def news_search(symbol: str, is_crypto: bool, n: int = 6):
    if not SERPAPI_API_KEY:
        dlog("serpapi_skip", reason="no_key")
        return []
    q = f"{symbol} crypto" if is_crypto else f"{symbol} stock"
    try:
        js = _get_json("https://serpapi.com/search.json", {
            "q": q,
            "engine": "google_news",
            "num": n,
            "api_key": SERPAPI_API_KEY,
        }, provider="serpapi")
        dlog("serpapi_ok", query=q, counted=True)
    except Exception as e:
        dlog("serpapi_err", query=q, err=str(e))
        return []
    out = []
    for it in (js or {}).get("news_results", [])[:n]:
        out.append({
            "title": it.get("title"),
            "link": it.get("link"),
            "snippet": " ".join(str(it.get("snippet") or "").split()),
        })
    return out

@st.cache_data(ttl=1800)
def sentiment_from_headlines(symbol: str, is_crypto: bool):
    try:
        heads = news_search(symbol, is_crypto, n=6) or []
    except Exception:
        heads = []
    if not heads:
        return "Neutral", 0
    pos = {"beat","beats","record","surge","surges","rally","rallies","up","soar","soars",
           "growth","raises","upgrade","upgraded","profitable","profit","tops","outperform",
           "bull","bullish","strong","expands","accelerates","buyback","guidance raised"}
    neg = {"miss","misses","cuts","cut","downgrade","downgraded","fall","falls","plunge","plunges",
           "drop","drops","lawsuit","probe","fraud","weak","warning","warns","bear","bearish",
           "loss","losses","decline","slump","recall","investigation","guidance cut","guidance lowered"}
    score = 0
    for h in heads:
        txt = f"{h.get('title','')} {h.get('snippet','')}".lower()
        score += sum(1 for w in pos if w in txt)
        score -= sum(1 for w in neg if w in txt)
    if score > 1:
        return "Bullish", len(heads)
    if score < -1:
        return "Bearish", len(heads)
    return "Neutral", len(heads)

# URL summarizer (paywall‑aware)
@st.cache_data(ttl=3600)
def fetch_url_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (AlphaBot)"}
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    html = r.text
    if BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        container = soup.find("article") or soup.find("main") or soup.find("div", attrs={"role": "main"}) or soup
        paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
        text = " ".join(paragraphs)
    else:
        text = html
    return text[:50000]

try:
    def summarize_text(text: str, question: str = "") -> str:
        if not text:
            return "No content fetched from the URL."
        if client is None:
            return "Summary unavailable (LLM not configured)."
        chunks = [text[i:i + 6000] for i in range(0, len(text), 6000)]
        partials = []
        for c in chunks[:5]:
            try:
                res = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You summarize web pages crisply for financial users. Be factual, concise, and include key numbers where available."},
                        {"role": "user", "content": (question + "\n\n" if question else "") + c},
                    ],
                    temperature=0.2,
                )
                partials.append(res.choices[0].message.content)
            except Exception:
                partials.append("")
        try:
            final = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Synthesize these notes into bullets plus a one‑line takeaway."},
                    {"role": "user", "content": "\n\n".join(partials)},
                ],
                temperature=0.2,
            )
            return final.choices[0].message.content
        except Exception:
            return "\n".join([p for p in partials if p]) or "Summary unavailable."
except Exception:
    pass

# Scoring engine (with breakdown & confidence)
def _safe_pct(a, b):
    try:
        return (a - b) / b
    except Exception:
        return 0.0

def _blend_momentum_12_1_multihorizon(closes: np.ndarray, skip_days: int = 21):
    if closes is None or len(closes) < (252 + skip_days + 1):
        return 0.0, 0.0, 0.0, 0.0, 50.0
    t = len(closes) - 1 - skip_days
    if t - 252 < 0:
        return 0.0, 0.0, 0.0, 0.0, 50.0
    def k_return(k_days: int) -> float:
        try:
            return _safe_pct(closes[t], closes[max(0, t - k_days)])
        except Exception:
            return 0.0
    r3 = k_return(63)
    r6 = k_return(126)
    r12 = k_return(252)
    try:
        window = closes[max(0, t - 252): t + 1]
        rets = np.diff(window) / window[:-1]
        sigma = float(np.std(rets, ddof=1) * np.sqrt(252))
        if not np.isfinite(sigma) or sigma <= 1e-6:
            sigma = 1e-6
    except Exception:
        sigma = 1e-6
    s3, s6, s12 = r3 / sigma, r6 / sigma, r12 / sigma
    raw = 0.2 * s3 + 0.3 * s6 + 0.5 * s12
    scaled = float(np.clip(50 + 25 * raw, 0, 100))
    return float(r3), float(r6), float(r12), float(raw), scaled

def compute_momentum_breakdown(prices: list[dict]):
    need = 252 + 21 + 1
    have = len(prices or [])
    if have < need:
        dlog("insufficient_history", have=have, need=need)
        return {
            "score": 50.0,
            "raw_score": 0.0,
            "r3m": 0.0,
            "r6m": 0.0,
            "r12m": 0.0,
            "skip_days": 21,
            "cash_score": None,
            "cash_raw": None,
            "warning": f"Insufficient history: have {have}, need {need}"
        }
    closes = np.array([p["close"] for p in prices], dtype=float)
    r3, r6, r12, raw, scaled = _blend_momentum_12_1_multihorizon(closes, skip_days=21)
    return {
        "score": round(scaled, 1),
        "raw_score": round(raw, 6),
        "r3m": round(r3, 6),
        "r6m": round(r6, 6),
        "r12m": round(r12, 6),
        "skip_days": 21,
        "cash_score": None,
        "cash_raw": None,
    }

def compute_mean_value_breakdown(prices: list[dict]):
    have = len(prices or [])
    need = 20
    if have < need:
        dlog("insufficient_history", have=have, need=need)
        return {
            "score": 50.0,
            "rsi": 0.0,
            "zscore": 0.0,
            "bandwidth": 0.0,
            "warning": f"Insufficient history: have {have}, need {need}"
        }
    closes = np.array([p["close"] for p in prices], dtype=float)
    period = 14
    rsi = None
    if len(closes) >= period + 1:
        diffs = np.diff(closes)
        last_diffs = diffs[-period:]
        gains = [d for d in last_diffs if d > 0]
        losses = [abs(d) for d in last_diffs if d < 0]
        avg_gain = np.mean(gains) if gains else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
    window = 20
    zscore = None
    bandwidth = None
    if len(closes) >= window:
        segment = closes[-window:]
        ma20 = np.mean(segment)
        std20 = np.std(segment)
        if std20 == 0:
            zscore = 0.0
        else:
            zscore = (closes[-1] - ma20) / std20
        if ma20 > 0:
            bandwidth = (4 * std20 / ma20) * 100.0
    if rsi is None or zscore is None:
        score = 50.0
    else:
        score = 50.0 + (-zscore * 12.0) + (0.2 * (50.0 - rsi))
        score = float(np.clip(score, 0, 100))
    return {
        "score": round(score, 1),
        "rsi": round(rsi, 1) if rsi is not None else 0.0,
        "zscore": round(zscore, 2) if zscore is not None else 0.0,
        "bandwidth": round(bandwidth, 2) if bandwidth is not None else 0.0
    }

def map_signal(score: float, sentiment: str, above_cash: bool = True) -> str:
    if not above_cash:
        return "Sell"
    if score >= 70 and sentiment != "Bearish":
        return "Buy"
    if score < 40 or sentiment == "Bearish":
        return "Sell"
    return "Hold"

def confidence_meter(score: float, sentiment: str, has_hist: bool, headlines_n: int, has_quote: bool, above_cash: bool = True) -> int:
    base = 0
    base += 40 if has_hist else 10
    base += 30 if headlines_n >= 3 else (15 if headlines_n >= 1 else 5)
    base += 30 if has_quote else 10
    bullish_mo = score >= 70
    bearish_mo = score < 40
    if (bullish_mo and sentiment == "Bullish") or (bearish_mo and sentiment == "Bearish"):
        base += 15
    elif (bullish_mo and sentiment == "Bearish") or (bearish_mo and sentiment == "Bullish"):
        base -= 15
    base += 10 if above_cash else -10
    return int(max(0, min(100, base)))

@st.cache_data(ttl=24*3600)
def latest_filing_summary(symbol: str) -> dict:
    try:
        files = fmp_filings(symbol, limit=8)
        if not files:
            return {}
        prio = {"10-Q": 3, "10-K": 2, "8-K": 1}
        files = sorted(files, key=lambda x: (prio.get(x.get("form"), 0), x.get("acceptedDate","")), reverse=True)
        f = files[0]
        link = f.get("finalLink") or f.get("link")
        if not link:
            return {}
        text = fetch_url_text(link)
        return {"form": f.get("form"), "date": f.get("acceptedDate"), "link": link, "summary": text[:1200]}
    except Exception:
        return {}

# 72h‑cached analysis (multi-engine)
@st.cache_data(ttl=72*3600, show_spinner=False)
def get_analysis(symbol: str, rules_version: str, engine: str = "Momentum"):
    sym = symbol.upper()
    # Precompute fast path (Momentum only, unless force_live)
    if engine == "Momentum":
        pc = precomputed_lookup(sym)
        if pc and not force_live:
            pc["engine"] = engine
            return pc
    is_crypto = sym in CRYPTO_GUESS
    # Quotes / profiles with fallback
    quote = fmp_quote(sym) or finnhub_quote(sym)
    profile = fmp_profile(sym) or finnhub_profile(sym)
    # Price history
    hist = history_any(sym, days=400)
    dlog("fetched", symbol=sym, quote=bool(quote), profile=bool(profile), hist=len(hist))
    if engine == "Momentum":
        brk = compute_momentum_breakdown(hist)
    else:
        brk = compute_mean_value_breakdown(hist)
    if brk.get("warning"):
        dlog("history_warning", msg=brk["warning"])
    if engine == "Momentum":
        # Compare against cash benchmark (BIL)
        if "_bil_brk" not in st.session_state:
            try:
                bil_hist = history_any("BIL", days=400)
                st.session_state["_bil_brk"] = compute_momentum_breakdown(bil_hist)
            except Exception:
                st.session_state["_bil_brk"] = None
        bil_brk = st.session_state.get("_bil_brk")
        try:
            above_cash = (brk.get("raw_score", 0.0) > (bil_brk.get("raw_score", 0.0) if bil_brk else 0.0))
            if bil_brk:
                brk["cash_score"] = bil_brk.get("score")
                brk["cash_raw"] = bil_brk.get("raw_score")
        except Exception:
            above_cash = True
    else:
        above_cash = True
    sentiment, headlines_n = sentiment_from_headlines(sym, is_crypto)
    if engine == "Momentum":
        signal = map_signal(brk["score"], sentiment, above_cash=above_cash)
    else:
        rebound_ok = True
        if brk["score"] >= 70 and len(hist) >= 2:
            if isinstance(hist[-1], dict) and isinstance(hist[-2], dict):
                if hist[-1]["close"] <= hist[-2]["close"]:
                    rebound_ok = False
        if brk["score"] >= 70 and sentiment != "Bearish" and rebound_ok:
            signal = "Buy"
        elif brk["score"] < 40 or sentiment == "Bearish":
            signal = "Sell"
        else:
            signal = "Hold"
    conf = confidence_meter(brk["score"], sentiment, has_hist=len(hist)>=260, headlines_n=headlines_n, has_quote=bool(quote), above_cash=above_cash)
    name = (quote.get("name") if isinstance(quote, dict) else None) or (profile.get("companyName") if isinstance(profile, dict) else None) or sym
    price = (quote.get("price") if isinstance(quote, dict) else None) or (quote.get("previousClose") if isinstance(quote, dict) else None)
    fundamentals = {
        "Market Cap": (quote.get("marketCap") if isinstance(quote, dict) else None) or (profile.get("mktCap") if isinstance(profile, dict) else None),
        "P/E (ttm)": quote.get("pe") if isinstance(quote, dict) else None,
        "EPS (ttm)": quote.get("eps") if isinstance(quote, dict) else None,
        "Beta": profile.get("beta") if isinstance(profile, dict) else None,
        "Sector": profile.get("sector") if isinstance(profile, dict) else None,
        "Industry": profile.get("industry") if isinstance(profile, dict) else None,
    }
    headlines = news_search(sym, is_crypto, n=6)
    result = {
        "symbol": sym,
        "name": name,
        "kind": "Crypto" if is_crypto else "Asset",
        "price": price,
        "engine": engine,
        "sentiment": sentiment,
        "signal": signal,
        "confidence": conf,
        "quote": quote,
        "profile": profile,
        "fundamentals": fundamentals,
        "history": hist,
        "filing": {},
        "headlines": headlines,
        "rules": rules_version,
        "cached_at": datetime.utcnow().isoformat()+"Z"
    }
    if engine == "Momentum":
        result["momentum"] = brk
    else:
        result["mean_value"] = brk
    return result

# Explainability + ledger
def rule_based_rationale(a: dict) -> str:
    engine = a.get("engine", "Momentum")
    if engine == "Momentum":
        m = a.get("momentum", {}) or {}
        sig = a.get("signal") or "Hold"
        sent = a.get("sentiment") or "Neutral"
        parts = [f"Recommendation: {sig}."]
        parts.append(
            " Momentum score {score} (risk‑adjusted 12‑1, multi‑horizon: 3/6/12 months, skip last 21 trading days). "
            "Horizon returns → R3m {r3m:.2%}, R6m {r6m:.2%}, R12m {r12m:.2%}. "
            "Raw composite is volatility‑adjusted (12m sigma). Cash benchmark (BIL) score {cash}.".format(score=m.get("score",0), r3m=m.get("r3m",0), r6m=m.get("r6m",0), r12m=m.get("r12m",0), cash=m.get("cash_score"))
        )
        parts.append(f" Headline sentiment is {sent}.")
        return "".join(parts)
    elif engine == "Mean Value":
        m = a.get("mean_value", {}) or {}
        sig = a.get("signal") or "Hold"
        sent = a.get("sentiment") or "Neutral"
        parts = [f"Recommendation: {sig}."]
        parts.append(
            " Mean-value score {score} (long-only mean reversion; uses RSI, Bollinger Bands, and volatility normalization). "
            "Price is {z:.2f}σ from its 20-day average; 14-day RSI is {rsi:.1f}. "
            "A rebound confirmation is required before a Buy signal to avoid catching falling knives.".format(score=m.get("score",0), z=m.get("zscore",0), rsi=m.get("rsi",0))
        )
        parts.append(f" Headline sentiment is {sent}.")
        return "".join(parts)
    else:
        return ""

@st.cache_data(ttl=1)
def write_ledger_entry_quick(a: dict, rationale: str) -> str:
    Path(LEDGER_DIR).mkdir(parents=True, exist_ok=True)
    decision_id = str(uuid4())
    ts = datetime.utcnow()
    outdir = Path(LEDGER_DIR) / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%d}"
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{ts:%H%M%S}_{decision_id}.json"
    fpath = outdir / fname
    data = {
        "symbol": a.get("symbol"),
        "engine": a.get("engine"),
        "sentiment": a.get("sentiment"),
        "signal": a.get("signal"),
        "rationale": rationale
    }
    if a.get("engine") == "Momentum":
        data["momentum"] = a.get("momentum")
    elif a.get("engine") == "Mean Value":
        data["mean_value"] = a.get("mean_value")
    with open(fpath, "x", encoding="utf-8") as f:
        json.dump(data, f)
    return str(fpath)

# Render helpers
def render_card(a: dict) -> str:
    if not a or a.get("error"):
        return "I couldn't fetch data for that ticker. Try another one."
    lines = [f"### ✅ {a['symbol']} — {a.get('name','').strip()}"]
    if a.get("price") is not None:
        lines.append(f"**Price (snapshot):** {a['price']}")
    lines.append(f"**Signal:** {a['signal']}")
    lines.append(f"**Sentiment:** {a['sentiment']}")
    if a.get("engine") == "Mean Value":
        lines.append(f"**Mean-Value Score:** {a['mean_value']['score']}")
    else:
        lines.append(f"**Momentum Score:** {a['momentum']['score']}")
    lines.append(f"**Confidence (72h):** {a['confidence']}%")
    return "\n\n".join(lines)

def render_deep(a: dict) -> str:
    if not a or a.get("error"):
        return ""
    if a.get("engine") == "Mean Value":
        m = a["mean_value"]
    else:
        m = a["momentum"]
    out = []
    if m.get("warning"):
        out.append("**⚠️ Data warning:** " + md_escape(m["warning"]))
        out.append("")
    if a.get("engine") == "Mean Value":
        rows = pd.DataFrame({
            "Component": ["RSI (14d)", "20d Z-score", "20d Bollinger band width (%)", "Mean-Value Score"],
            "Value": [m.get("rsi"), m.get("zscore"), m.get("bandwidth"), m.get("score")]
        })
    else:
        rows = pd.DataFrame({
            "Component": ["R_3m (ex-1m)", "R_6m (ex-1m)", "R_12m (ex-1m)", "Risk-adj composite"],
            "Value": [m.get("r3m"), m.get("r6m"), m.get("r12m"), m.get("raw_score")]
        })
    out += ["### 🔎 Deep Dive", "", rows.to_markdown(index=False)]
    f = {k: v for k, v in a.get("fundamentals", {}).items() if v not in (None, "", "nan")}
    if f:
        f_rows = pd.DataFrame({"Metric": list(f.keys()), "Value": list(f.values())})
        out += ["", "#### Key fundamentals", f_rows.to_markdown(index=False)]
    heads = a.get("headlines") or []
    if heads:
        lines = ["", "#### Latest headlines", ""]
        for h in heads:
            title = md_escape(h.get("title") or h.get("link") or "")
            snip = md_escape(h.get("snippet", ""))
            link = h.get("link")
            lines.append(f"- [{title}]({link})\n\n    {snip}")
        out += lines
    return "\n\n".join(out)

def draw_chart(a: dict):
    hist = a.get("history") or []
    if len(hist) < 30:
        return
    closes = np.array([h["close"] for h in hist], dtype=float)
    dates = [h["date"] for h in hist]
    s = pd.Series(closes)
    ma20 = s.rolling(20).mean().to_numpy()
    ma50 = s.rolling(50).mean().to_numpy()
    fig = plt.figure(figsize=(7.5,2.6), dpi=150)
    plt.plot(dates, closes, label="Close")
    plt.plot(dates, ma20, label="20‑DMA")
    plt.plot(dates, ma50, label="50‑DMA")
    plt.xticks(rotation=0, ticks=[dates[i] for i in range(0, len(dates), max(1,len(dates)//6))])
    plt.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

# Chat flow
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type one ticker (e.g., AAPL, TSLL, ETH). Add 'news' or paste a URL to summarize.")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    url_match = URL_RE.search(prompt)
    if url_match:
        url = url_match.group(0)
        with st.spinner(f"Fetching and summarizing: {url}"):
            try:
                text = fetch_url_text(url)
                summary = summarize_text(text, question=f"Summarize this page for a financial audience. URL: {url}")
                final = f"### 📰 Summary of {url}\n\n{md_escape(summary)}\n\n" + DISCLAIMER
            except Exception as e:
                final = f"Could not fetch that URL. Error: {e}\n\n" + DISCLAIMER

        st.session_state.messages.append({"role": "assistant", "content": final})
        with st.chat_message("assistant"):
            st.markdown(final)
    else:
        raw_syms = [s.upper() for s in TICKER_RE.findall(prompt)]
        stop = {"NEWS", "DEEP", "DIVE"}
        syms = [s for s in raw_syms if s not in stop]
        dlog("tickers_in_prompt", raw=raw_syms, accepted=syms[:1])
        if not syms:
            final = "I work one ticker at a time. Try **AAPL**, **TSLL**, or **ETH**.\n\n" + DISCLAIMER
            st.session_state.messages.append({"role": "assistant", "content": final})
            with st.chat_message("assistant"):
                st.markdown(final)
        else:
            sym = syms[0]
            with st.spinner(f"Analyzing {sym}…"):
                a = get_analysis(sym, RULES_VERSION, engine_choice)

            fallback = rule_based_rationale(a)
            rationale = fallback
            ledger_path = write_ledger_entry_quick(a, rationale)

            body = [render_card(a)]
            body.append(f"**Why this is a {a['signal']}**\n\n{rationale}\n\n_Audit ref_: `{ledger_path}`")
            if show_deep:
                body.append(render_deep(a))
            final = "\n\n---\n\n".join([b for b in body if b]) + "\n\n" + DISCLAIMER

            st.session_state.messages.append({"role": "assistant", "content": final})
            with st.chat_message("assistant"):
                st.markdown(final)
                draw_chart(a)

if show_diag:
    st.sidebar.markdown("### Diagnostics (this session)")
    st.sidebar.code(json.dumps(_diag, indent=2))
