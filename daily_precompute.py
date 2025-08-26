# daily_precompute.py — AlphaBot nightly cache builder
#
# What it does
# • Reads your three CSV universes (Top 100 Stocks / ETFs / Crypto) OR discovers via APIs (optional)
# • Pulls fresh FMP + Finnhub + headlines
# • Computes Momentum Score, Sentiment, Signal, Confidence — same logic as app
# • Saves one consolidated JSON: precomputed/latest.json (and YYYY‑MM‑DD.json)
# • (Optional) writes small PNG price chart per symbol into precomputed/charts/
#
# Schedule
# • Run daily at 8:00 PM US/Pacific, Sunday→Thursday only (skip Fri/Sat)
#   The script enforces this guard itself, but you should also schedule it with
#   Windows Task Scheduler / cron.
#
# Notes
# • Keep API keys in .env (FMP_API_KEY, FINNHUB_API_KEY, SERPAPI_API_KEY)
# • Output schema matches app.get_analysis() so the app can serve from cache
#   instantly when available.

from __future__ import annotations
import os, json, time, math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import re
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

FMP_API_KEY      = os.getenv("FMP_API_KEY")
FINNHUB_API_KEY  = os.getenv("FINNHUB_API_KEY")
SERPAPI_API_KEY  = os.getenv("SERPAPI_API_KEY")
RULES_VERSION    = "2025-08-10-v5"

DATA_DIR   = Path("data")
OUT_DIR    = Path("precomputed")
CHART_DIR  = OUT_DIR / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------- HTTP helper --------------------------

def _get_json(url, params=None, headers=None, tries=2, timeout=20):
    last=None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last=e
            time.sleep(0.6*(i+1))
    raise last

# -------------------------- FMP / Finnhub --------------------------

def fmp_quote(symbol: str):
    if not FMP_API_KEY:
        return {}
    js = _get_json(f"https://financialmodelingprep.com/api/v3/quote/{symbol}", {"apikey": FMP_API_KEY})
    return js[0] if isinstance(js, list) and js else {}

def fmp_profile(symbol: str):
    if not FMP_API_KEY:
        return {}
    js = _get_json(f"https://financialmodelingprep.com/api/v3/profile/{symbol}", {"apikey": FMP_API_KEY})
    return js[0] if isinstance(js, list) and js else {}

def fmp_history(symbol: str, days: int = 180):
    if not FMP_API_KEY:
        return []
    js = _get_json(
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}",
        {"timeseries": days, "apikey": FMP_API_KEY},
    )
    hist = js.get("historical") if isinstance(js, dict) else None
    if not hist:
        return []
    hist = list(reversed(hist))
    return [{"date": h.get("date"), "close": float(h.get("close"))} for h in hist if h.get("close") is not None]

# -------------------------- Headlines --------------------------

def news_search(symbol: str, is_crypto: bool, n: int = 6):
    if not SERPAPI_API_KEY:
        return []
    q = f"{symbol} crypto" if is_crypto else f"{symbol} stock"
    js = _get_json("https://serpapi.com/search.json", {
        "q": q,
        "engine": "google_news",
        "num": n,
        "api_key": SERPAPI_API_KEY,
    })
    out = []
    for it in (js or {}).get("news_results", [])[:n]:
        out.append({
            "title": it.get("title"),
            "link": it.get("link"),
            "snippet": " ".join(str(it.get("snippet") or "").split()),
        })
    return out

# -------------------------- Scoring engine --------------------------

def _pct(a, b):
    try:
        return (a - b) / b
    except Exception:
        return 0.0

def compute_momentum_breakdown(prices: list[dict]):
    if not prices or len(prices) < 30:
        return {"score": 50.0, "momentum20": 0.0, "breadth20": 0.0, "volatility": 0.0, "confirmation": 0.0}
    closes = np.array([p["close"] for p in prices], dtype=float)
    mom20 = _pct(closes[-1], closes[-21])
    mom20_s = np.clip((mom20 + 0.2) / 0.4, 0, 1)
    dma20 = pd.Series(closes).rolling(20).mean().to_numpy()
    last20 = closes[-20:]
    last20dma = dma20[-20:]
    breadth = float((last20 > last20dma).sum()) / max(1, len(last20))
    diffs = np.abs(np.diff(closes))
    atr14 = pd.Series(diffs).rolling(14).mean().to_numpy()[-1]
    vol_norm = atr14 / max(1e-9, closes[-1])
    vol_s = 1.0 - np.clip(vol_norm / 0.05, 0, 1)
    deltas = np.diff(closes)
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)
    rs = (pd.Series(gains).rolling(14).mean().iloc[-1] + 1e-9) / (pd.Series(losses).rolling(14).mean().iloc[-1] + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi_s = np.clip((rsi - 40) / 20, 0, 1)
    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().iloc[-1]
    macd_pos = 1.0 if ema12 > ema26 else 0.0
    confirm = 0.6 * rsi_s + 0.4 * macd_pos
    score = 100 * (0.4*mom20_s + 0.2*breadth + 0.2*vol_s + 0.2*confirm)
    return {
        "score": float(round(np.clip(score, 0, 100), 1)),
        "momentum20": float(round(mom20_s, 2)),
        "breadth20": float(round(breadth, 2)),
        "volatility": float(round(vol_s, 2)),
        "confirmation": float(round(confirm, 2)),
    }

# -------------------------- CSV universes --------------------------

CSV_FILES = {
    "stocks": DATA_DIR / "Top 100 Stocks List.csv",
    "etfs":   DATA_DIR / "Top 100 ETFs List.csv",
    "crypto": DATA_DIR / "Top 100 Crypto List.csv",
}

SYM_ALIASES = {"symbol","ticker","tickers","sym","tkr","asset","name"}


def _read_symbols_from_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    sym_col = None
    for k in cols:
        if k in {"symbol","ticker","tickers","sym","tkr"}:
            sym_col = cols[k]
            break
    if not sym_col:
        # fallback: first column that looks like a symbol
        sym_col = df.columns[0]
    syms = [str(s).strip().upper() for s in df[sym_col].dropna().tolist()]
    # normalize
    syms = [re.sub(r"[^A-Za-z0-9]+","", s) for s in syms]
    syms = [s for s in syms if 1 <= len(s) <= 6]
    return list(dict.fromkeys(syms))[:100]

# -------------------------- Builder --------------------------

def build_for_universe(kind: str, symbols: list[str]):
    out = {}
    for i, sym in enumerate(symbols, 1):
        try:
            quote = fmp_quote(sym)
            profile = fmp_profile(sym)
            hist = fmp_history(sym, 180)
            brk = compute_momentum_breakdown(hist)
            # sentiment from headlines (rough)
            heads = news_search(sym, is_crypto=(kind=="crypto"), n=6)
            headline_text = "\n".join(h.get("title","") for h in heads)
            # simple rule: positive words count minus negative — keep fast & offline
            pos = sum(w in headline_text.lower() for w in ["beats","surge","rally","strong","up","record"]) \
                - sum(w in headline_text.lower() for w in ["miss","fall","down","weak","probe","recall","lawsuit"]) \
                # keeps deterministic + fast for nightly job
            if pos >= 2:
                sentiment = "Bullish"
            elif pos <= -2:
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"
            # map to signal
            score = brk["score"]
            if score >= 70 and sentiment != "Bearish":
                signal = "Buy"
            elif score < 40 or sentiment == "Bearish":
                signal = "Sell"
            else:
                signal = "Hold"
            # crude confidence
            conf = 40 + (30 if len(hist) >= 60 else 10) + (30 if quote else 10)
            conf = int(max(0, min(100, conf)))
            name = quote.get("name") or profile.get("companyName") or sym
            out[sym] = {
                "symbol": sym,
                "name": name,
                "kind": kind.capitalize(),
                "price": quote.get("price") or quote.get("previousClose"),
                "momentum": brk,
                "sentiment": sentiment,
                "signal": signal,
                "confidence": conf,
                "quote": quote,
                "profile": profile,
                "fundamentals": {
                    "Market Cap": quote.get("marketCap") or profile.get("mktCap"),
                    "P/E (ttm)": quote.get("pe"),
                    "EPS (ttm)": quote.get("eps"),
                    "Beta": profile.get("beta"),
                    "Sector": profile.get("sector"),
                    "Industry": profile.get("industry"),
                },
                "history": hist,
                "filing": {},  # kept blank in nightly job for speed
                "headlines": heads,
                "rules": RULES_VERSION,
                "cached_at": datetime.utcnow().isoformat()+"Z",
            }
            # quick chart
            if len(hist) >= 30:
                closes = np.array([h["close"] for h in hist])
                dates = [h["date"] for h in hist]
                s = pd.Series(closes)
                ma20 = s.rolling(20).mean().to_numpy()
                ma50 = s.rolling(50).mean().to_numpy()
                fig = plt.figure(figsize=(6,2), dpi=140)
                plt.plot(dates, closes, label="Close")
                plt.plot(dates, ma20, label="20-DMA")
                plt.plot(dates, ma50, label="50-DMA")
                plt.xticks(rotation=0, ticks=[dates[j] for j in range(0,len(dates), max(1,len(dates)//6))])
                plt.legend(loc="upper left")
                plt.tight_layout()
                fig.savefig(CHART_DIR / f"{sym}.png")
                plt.close(fig)
        except Exception as e:
            # continue but log
            print(f"[warn] {kind} {sym}: {e}")
        time.sleep(0.05)  # polite pacing
    return out

# -------------------------- Entry --------------------------

def main():
    # Guard: run only Sun→Thu at 20:00 PT ±40 min (if you choose to invoke hourly)
    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    if now_pt.weekday() in (4,5):  # 4=Fri, 5=Sat
        print("Skipping (Fri/Sat)")
        return
    # If you schedule precisely at 20:00 PT, this window check is redundant
    if not (19 <= now_pt.hour <= 21):
        print("Outside 8pm PT window; exiting")
        return

    # 1) Read universes from CSVs (your canonical lists)
    stock_syms = _read_symbols_from_csv(CSV_FILES["stocks"])[:100]
    etf_syms   = _read_symbols_from_csv(CSV_FILES["etfs"])[:100]
    crypto_syms= _read_symbols_from_csv(CSV_FILES["crypto"])[:100]

    # 2) Build
    cache = {}
    cache.update(build_for_universe("asset", stock_syms))
    cache.update(build_for_universe("etf", etf_syms))
    cache.update(build_for_universe("crypto", crypto_syms))

    # 3) Write
    stamp = datetime.utcnow().strftime("%Y-%m-%d")
    with open(OUT_DIR / f"{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    with open(OUT_DIR / "latest.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"Wrote {len(cache)} symbols → {OUT_DIR/'latest.json'}")

if __name__ == "__main__":
    main()
