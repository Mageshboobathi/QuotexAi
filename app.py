import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Quotex Remix AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at top, #101a31 0%, #070b14 55%, #05070c 100%); color:#eef3ff;}
[data-testid="stMetric"] {background:#101827; border:1px solid #26354d; padding:14px; border-radius:14px;}
.engine {background:#0d1523;border:1px solid #26354d;border-radius:14px;padding:14px;min-height:112px}
.small {color:#9eb0ca;font-size:12px}
</style>
""", unsafe_allow_html=True)

st.title("⚡ QUOTEX REMIX AI")
st.caption("Six-rule market analysis dashboard • Live market data • Educational tool, not a guaranteed trading signal")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Twelve Data API key", type="password",
                            value=os.getenv("TWELVE_DATA_API_KEY", ""))
    symbol = st.text_input("Market symbol", "EUR/USD")
    interval = st.selectbox("Analysis timeframe", ["1min", "5min", "15min"], index=0)
    outputsize = st.slider("Candles to load", 80, 300, 180, 20)
    refresh = st.selectbox("Refresh interval", ["Manual", "15 seconds", "30 seconds", "60 seconds"])
    run = st.button("🔄 Analyze market", type="primary", use_container_width=True)
    st.caption("Use a Twelve Data-supported symbol. Data availability and delay depend on your API plan.")

def fetch_data(key: str, sym: str, iv: str, size: int) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": sym.strip(),
        "interval": iv,
        "outputsize": size,
        "apikey": key.strip(),
        "format": "JSON",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if "values" not in payload:
        msg = payload.get("message") or payload.get("code") or str(payload)
        raise RuntimeError(f"Twelve Data response: {msg}")
    df = pd.DataFrame(payload["values"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df.sort_values("datetime").reset_index(drop=True)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c, h, l = x.close, x.high, x.low
    x["ema9"] = c.ewm(span=9, adjust=False).mean()
    x["ema21"] = c.ewm(span=21, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    x["rsi"] = 100 - (100 / (1 + rs))
    x["rsi"] = x["rsi"].fillna(50)
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    x["vol_mean"] = x["atr"].rolling(20).mean()
    x["body"] = (x.close - x.open).abs()
    x["range"] = (x.high - x.low).replace(0, pd.NA)
    x["upper_wick"] = x.high - x[["open", "close"]].max(axis=1)
    x["lower_wick"] = x[["open", "close"]].min(axis=1) - x.low
    x["body_ratio"] = (x.body / x["range"]).fillna(0)
    x["ret"] = x.close.pct_change()
    x["momentum3"] = x.close.pct_change(3)
    x["momentum8"] = x.close.pct_change(8)
    x["std20"] = x.ret.rolling(20).std()
    return x

def score_engines(df: pd.DataFrame) -> List[Dict]:
    r = df.iloc[-1]
    prev = df.iloc[-2]
    engines = []

    # 1: trend alignment
    s = 1 if r.ema9 > r.ema21 > r.ema50 else (-1 if r.ema9 < r.ema21 < r.ema50 else 0)
    engines.append(("TITAN X", s, "EMA 9/21/50 trend alignment"))

    # 2: multi-timeframe proxy: short and medium momentum agreement
    s = 1 if r.momentum3 > 0 and r.momentum8 > 0 else (-1 if r.momentum3 < 0 and r.momentum8 < 0 else 0)
    engines.append(("NEXUS FUSION", s, "3-candle + 8-candle momentum agreement"))

    # 3: candle anatomy / rejection
    bull_reject = r.lower_wick > r.body * 1.5 and r.close > r.open
    bear_reject = r.upper_wick > r.body * 1.5 and r.close < r.open
    s = 1 if bull_reject else (-1 if bear_reject else 0)
    engines.append(("APEX VISION", s, "Wick rejection and candle structure"))

    # 4: RSI mean-reversion filter
    s = 1 if r.rsi < 32 and r.close > prev.close else (-1 if r.rsi > 68 and r.close < prev.close else 0)
    engines.append(("QUANTUM STRIKE", s, f"RSI {r.rsi:.1f} reversal check"))

    # 5: volatility-aware breakout
    recent_high = df.high.iloc[-21:-1].max()
    recent_low = df.low.iloc[-21:-1].min()
    s = 1 if r.close > recent_high else (-1 if r.close < recent_low else 0)
    engines.append(("OMEGA QX", s, "20-candle range breakout"))

    # 6: final confirmation / candle close location
    candle_up = r.close > r.open and r.close > prev.close
    candle_down = r.close < r.open and r.close < prev.close
    s = 1 if candle_up and r.body_ratio > 0.45 else (-1 if candle_down and r.body_ratio > 0.45 else 0)
    engines.append(("QUOTEX MASTER", s, "Last closed candle confirmation"))

    return [{"name": n, "score": s, "reason": why} for n, s, why in engines]

def final_signal(engines: List[Dict]) -> Tuple[str, int, int]:
    ups = sum(e["score"] == 1 for e in engines)
    downs = sum(e["score"] == -1 for e in engines)
    if ups >= 4 and ups > downs:
        return "CALL ↑", ups, downs
    if downs >= 4 and downs > ups:
        return "PUT ↓", ups, downs
    return "WAIT", ups, downs

if not api_key:
    st.info("Add your Twelve Data API key in the left Settings panel, then press Analyze market.")
    st.markdown("""
    **Quick start**
    1. Create a Twelve Data API key at [twelvedata.com](https://twelvedata.com/).
    2. Paste it into the password field in the sidebar.
    3. Choose a supported symbol such as `EUR/USD` and press **Analyze market**.
    """)
    st.stop()

try:
    data = fetch_data(api_key, symbol, interval, outputsize)
    data = add_indicators(data)
    if len(data) < 55:
        st.error("Not enough candles returned. Increase output size or choose another symbol/timeframe.")
        st.stop()
    # Analyze the most recent completed candle, not the still-forming candle.
    now_utc = pd.Timestamp.now(tz="UTC")
    minutes = {"1min": 1, "5min": 5, "15min": 15}[interval]
    current_bucket = now_utc.floor(f"{minutes}min")
    closed_data = data[data["datetime"] < current_bucket].copy()
    if len(closed_data) < 55:
        st.error("Not enough completed candles yet. Please retry shortly.")
        st.stop()
    engines = score_engines(closed_data)
    signal, ups, downs = final_signal(engines)
    latest = closed_data.iloc[-1]
except Exception as e:
    st.error(str(e))
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Market", symbol.upper())
col2.metric("Timeframe", interval)
col3.metric("Latest close", f"{latest.close:.5f}")
col4.metric("AI consensus", f"{max(ups, downs)}/6")

st.divider()
left, right = st.columns([1.45, 1])
with left:
    st.subheader("Live candlestick chart")
    fig = go.Figure(data=[go.Candlestick(
        x=data.datetime, open=data.open, high=data.high, low=data.low, close=data.close,
        name=symbol
    )])
    fig.update_layout(
        template="plotly_dark", height=470, margin=dict(l=10, r=10, t=25, b=10),
        xaxis_rangeslider_visible=False, paper_bgcolor="#070b14", plot_bgcolor="#070b14"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Final AI signal")
    if signal.startswith("CALL"):
        st.success(signal)
    elif signal.startswith("PUT"):
        st.error(signal)
    else:
        st.warning(signal)
    st.metric("Bullish engines", ups)
    st.metric("Bearish engines", downs)
    st.caption("Consensus is a rule-based score, not a calibrated probability or guaranteed next-candle prediction.")

st.subheader("Powered by 6 analysis engines")
cols = st.columns(3)
for i, engine in enumerate(engines):
    with cols[i % 3]:
        label = "BULLISH" if engine["score"] == 1 else ("BEARISH" if engine["score"] == -1 else "NEUTRAL")
        color = "#27c77a" if engine["score"] == 1 else ("#ff5c70" if engine["score"] == -1 else "#aab4c5")
        st.markdown(
            f'<div class="engine"><b>{engine["name"]}</b><br>'
            f'<span style="color:{color};font-weight:700">{label}</span>'
            f'<div class="small">{engine["reason"]}</div></div>',
            unsafe_allow_html=True
        )

st.subheader("Recent candles")
st.dataframe(data[["datetime", "open", "high", "low", "close", "rsi"]].tail(12),
             use_container_width=True, hide_index=True)

st.caption(f"Last analyzed completed candle (UTC): {latest.datetime}. This app reads market data only; it does not connect to or place trades on Quotex.")
st.warning("Risk note: 1-minute OTC/binary-options trading is extremely risky. Signals can be wrong, data may be delayed, and this tool has not been validated for profitability. Test on a demo account and never treat engine agreement as a win-rate guarantee.")

if refresh != "Manual":
    seconds = {"15 seconds": 15, "30 seconds": 30, "60 seconds": 60}[refresh]
    st.caption(f"Auto-refresh is enabled every {seconds} seconds.")
    time.sleep(seconds)
    st.rerun()
