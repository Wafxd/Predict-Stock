import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import requests

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volume import ChaikinMoneyFlowIndicator, MFIIndicator

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================
# 1) PAGE CONFIG (HARUS SEKALI DI ATAS)
# =========================
st.set_page_config(page_title="PREDIKSI SAHAM", page_icon="🦅", layout="wide")

# =========================
# 2) GLOBAL CSS
# =========================
st.markdown("""
<style>
/* === FIX: Tab labels (Streamlit Tabs) === */
div[data-testid="stTabs"] button {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 10px 14px !important;
    opacity: 1 !important;
}

/* underline / indicator */
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #ff4b4b !important;
}

/* tab list border line */
div[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(255,255,255,0.15) !important;
}

/* selected tab styling */
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ff4b4b !important;
}

/* optional: hover */
div[data-testid="stTabs"] button:hover {
    opacity: 0.9 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 3) HELPERS (ANALYZER)
# =========================
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def parse_input_bandar(text_input):
    """Accept: 389K / 1.2M / 0.5B / 123456"""
    try:
        clean_text = str(text_input).strip().upper()
        if not clean_text:
            return 0.0
        multiplier = 1
        if clean_text.endswith("K"):
            multiplier = 1_000
            number_part = clean_text[:-1]
        elif clean_text.endswith("M"):
            multiplier = 1_000_000
            number_part = clean_text[:-1]
        elif clean_text.endswith("B"):
            multiplier = 1_000_000_000
            number_part = clean_text[:-1]
        else:
            number_part = clean_text
        return float(number_part) * multiplier
    except Exception:
        return 0.0

def format_volume(num):
    try:
        num = float(num)
    except Exception:
        return "0"
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(int(num))

def safe_download(ticker: str, start, end):
    """Robust yfinance download for cloud deploy (fallback history())"""
    try:
        # Pastikan end <= hari ini (biar ga aneh di server)
        today = pd.Timestamp.today().normalize()
        end = pd.Timestamp(end)
        if end > today:
            end = today + pd.Timedelta(days=1)

        start = pd.Timestamp(start)

        # ---- Method 1: yf.download ----
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            actions=False,
            threads=False,
        )

        # kalau kosong, lanjut fallback
        if df is None or df.empty:
            # ---- Method 2: history() (sering lebih tembus) ----
            tk = yf.Ticker(ticker)
            df = tk.history(start=start, end=end, auto_adjust=False, actions=False)

        # kalau masih kosong, coba period max
        if df is None or df.empty:
            tk = yf.Ticker(ticker)
            df = tk.history(period="max", auto_adjust=False, actions=False)

        if df is None or df.empty:
            return None

        # rapihin kolom multiindex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = ["Open", "High", "Low", "Close", "Volume"]
        for c in needed:
            if c not in df.columns:
                return None

        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start, end):
    df = safe_download(ticker, start, end)
    if df is None or df.empty:
        return None

    fx = safe_download("IDR=X", start, end)
    if fx is None or fx.empty or "Close" not in fx.columns:
        df["USD_IDR"] = np.nan
        df["USD_IDR"] = df["USD_IDR"].ffill()
        return df

    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)

    df["USD_IDR"] = fx["Close"].reindex(df.index).ffill()
    return df

def volume_dominance(df: pd.DataFrame, lookback: int = 5):
    """Proxy buy vs sell volume based on candle color (green=buy, red=sell)."""
    d = df.tail(lookback).copy()
    if d.empty or "Open" not in d.columns or "Adj Close" not in d.columns:
        return 0.0, 0.0, 0.5, "N/A"

    buy_vol = d.loc[d["Adj Close"] > d["Open"], "Volume"].sum()
    sell_vol = d.loc[d["Adj Close"] < d["Open"], "Volume"].sum()
    total = buy_vol + sell_vol
    ratio = (buy_vol / total) if total > 0 else 0.5

    if ratio >= 0.60:
        label = "🟢 Dominan BUY"
    elif ratio <= 0.40:
        label = "🔴 Dominan SELL"
    else:
        label = "⚪ Seimbang"

    return float(buy_vol), float(sell_vol), float(ratio), label

def compute_indicators(df: pd.DataFrame):
    """Indicators for AI mode. Will dropna heavily for IPO stocks."""
    d = df.copy()

    d["RSI"] = RSIIndicator(d["Adj Close"], window=14).rsi()
    d["MACD"] = MACD(d["Adj Close"]).macd()
    d["EMA"] = EMAIndicator(d["Adj Close"], window=20).ema_indicator()

    d["CMF"] = ChaikinMoneyFlowIndicator(
        high=d["High"], low=d["Low"], close=d["Adj Close"], volume=d["Volume"], window=20
    ).chaikin_money_flow()

    d["MFI"] = MFIIndicator(
        high=d["High"], low=d["Low"], close=d["Adj Close"], volume=d["Volume"], window=14
    ).money_flow_index()

    d["Bandar_Flow"] = d["Volume"] * d["Adj Close"].diff()

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna()
    return d

def build_sequences(scaled: np.ndarray, look_back: int, forecast_offsets: list[int]):
    X, Y = [], []
    max_f = int(max(forecast_offsets))
    for i in range(len(scaled) - look_back - max_f):
        X.append(scaled[i : i + look_back])
        Y.append([scaled[i + look_back + d, 0] for d in forecast_offsets])
    if len(X) == 0:
        return None, None
    return np.array(X), np.array(Y)

def style_small_chart(ax, mode="month"):
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    if mode == "month":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")

def load_broksum_from_csv(file) -> dict | None:
    """
    Expected columns (case-insensitive):
    daily_buy1,daily_buy2,daily_buy3,daily_sell1,daily_sell2,daily_sell3,
    weekly_buy1,weekly_buy2,weekly_buy3,weekly_sell1,weekly_sell2,weekly_sell3
    Optional: ticker,date
    Takes last row.
    """
    try:
        df = pd.read_csv(file)
        if df is None or df.empty:
            return None

        cols = {c.lower(): c for c in df.columns}
        need = [
            "daily_buy1","daily_buy2","daily_buy3","daily_sell1","daily_sell2","daily_sell3",
            "weekly_buy1","weekly_buy2","weekly_buy3","weekly_sell1","weekly_sell2","weekly_sell3",
        ]
        if not all(k in cols for k in need):
            return None

        row = df.iloc[-1]
        out = {k: str(row[cols[k]]) for k in need}
        out["ticker"] = str(row[cols["ticker"]]) if "ticker" in cols else ""
        out["date"] = str(row[cols["date"]]) if "date" in cols else ""
        return out
    except Exception:
        return None

def finite_or_nan(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan

# =========================
# 4) UI: TABS
# =========================
tab1, tab2 = st.tabs(["📈 Sniper Analyzer", "🧮 Kalkulator Rata-rata"])

# =========================================================
# TAB 1: SNIPER ANALYZER (FULL, LANGSUNG JALAN)
# =========================================================
with tab1:
    st.markdown('<div class="bigtitle">🦅 AI STOCK SNIPER</div>', unsafe_allow_html=True)
    st.caption("Analyzer + AI (kalau data cukup). Kalau saham baru/IPO, AI otomatis nonaktif tapi analisa tetap jalan.")

    with st.expander("⚙️ KONFIGURASI & INPUT DATA (Klik untuk Buka/Tutup)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            TICKER_INPUT = st.text_input(
                "Kode Ticker", value="ARCI", help="Contoh: BUMI, GOTO, SUPA"
            ).upper().strip()
            TICKER = TICKER_INPUT + ".JK" if TICKER_INPUT and not TICKER_INPUT.endswith(".JK") else TICKER_INPUT
        with col2:
            START_DATE = st.date_input("Mulai", value=pd.to_datetime("2022-01-01"))
        with col3:
            END_DATE = st.date_input("Sampai", value=pd.to_datetime("2026-12-31"))

        st.markdown("---")
        left, right = st.columns([1, 1])

        with left:
            st.markdown("🧾 **BROKSUM / TOP BROKER (Opsional, buat konfirmasi bandar)**")
            mode = st.radio("Sumber Broksum", ["Manual", "Upload CSV (Auto)"], horizontal=True)

            # default empty
            h_b1 = h_b2 = h_b3 = h_s1 = h_s2 = h_s3 = ""
            m_b1 = m_b2 = m_b3 = m_s1 = m_s2 = m_s3 = ""
            broksum_info = ""

            if mode == "Upload CSV (Auto)":
                file = st.file_uploader("Upload CSV broksum (export dari sumber kamu)", type=["csv"])
                st.caption("Kolom wajib: daily_buy1..3, daily_sell1..3, weekly_buy1..3, weekly_sell1..3 (ambil baris terakhir).")
                if file:
                    payload = load_broksum_from_csv(file)
                    if payload is None:
                        st.warning("CSV tidak sesuai format kolom yang dibutuhkan. Pakai mode Manual atau rapihin CSV.")
                    else:
                        h_b1, h_b2, h_b3 = payload["daily_buy1"], payload["daily_buy2"], payload["daily_buy3"]
                        h_s1, h_s2, h_s3 = payload["daily_sell1"], payload["daily_sell2"], payload["daily_sell3"]
                        m_b1, m_b2, m_b3 = payload["weekly_buy1"], payload["weekly_buy2"], payload["weekly_buy3"]
                        m_s1, m_s2, m_s3 = payload["weekly_sell1"], payload["weekly_sell2"], payload["weekly_sell3"]
                        if payload.get("ticker") or payload.get("date"):
                            broksum_info = f"Auto CSV: {payload.get('ticker','')} {payload.get('date','')}".strip()
                        st.success(f"Broksum ter-load. {broksum_info}".strip())

            if mode == "Manual":
                st.markdown("📊 **Harian (Top 3 Buy/Sell)**")
                ch1, ch2 = st.columns(2)
                h_b1 = ch1.text_input("Buy 1", key="hb1", placeholder="389K")
                h_b2 = ch1.text_input("Buy 2", key="hb2")
                h_b3 = ch1.text_input("Buy 3", key="hb3")
                h_s1 = ch2.text_input("Sell 1", key="hs1", placeholder="596K")
                h_s2 = ch2.text_input("Sell 2", key="hs2")
                h_s3 = ch2.text_input("Sell 3", key="hs3")

                st.markdown("📅 **Mingguan (Top 3 Buy/Sell)**")
                cm1, cm2 = st.columns(2)
                m_b1 = cm1.text_input("Buy 1 ", key="mb1")
                m_b2 = cm1.text_input("Buy 2 ", key="mb2")
                m_b3 = cm1.text_input("Buy 3 ", key="mb3")
                m_s1 = cm2.text_input("Sell 1 ", key="ms1")
                m_s2 = cm2.text_input("Sell 2 ", key="ms2")
                m_s3 = cm2.text_input("Sell 3 ", key="ms3")

        with right:
            st.markdown("⚙️ **PARAMETER MODEL & CHART**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                LOOK_BACK = st.number_input("LOOK_BACK", min_value=20, max_value=120, value=60, step=5)
            with c2:
                EPOCHS_1 = st.number_input("Epochs (Train)", min_value=5, max_value=80, value=30, step=5)
            with c3:
                EPOCHS_2 = st.number_input("Epochs (Fine-tune)", min_value=0, max_value=50, value=10, step=5)
            with c4:
                SEED = st.number_input("Seed", min_value=1, max_value=9999, value=42, step=1)

            st.markdown("---")
            SELL_THRESHOLD = st.number_input(
                "Strong Sell jika AI prediksi turun ≥ (%)",
                min_value=3.0, max_value=20.0, value=9.0, step=0.5
            )

            CHART_DATE_MODE = st.selectbox("Format tanggal chart", ["Bulanan (ringkas)", "Mingguan (hari)"], index=0)
            SHOW_FORWARD_PRED = st.checkbox("Tampilkan garis prediksi forward (H+1/H+3/H+7) di chart Price", value=True)

        st.markdown("---")
        btn_start = st.button("🚀 ANALISA SEKARANG", type="primary", use_container_width=True)

    # ---- Broksum net ----
    net_daily = (parse_input_bandar(h_b1) + parse_input_bandar(h_b2) + parse_input_bandar(h_b3)) - \
                (parse_input_bandar(h_s1) + parse_input_bandar(h_s2) + parse_input_bandar(h_s3))
    net_weekly = (parse_input_bandar(m_b1) + parse_input_bandar(m_b2) + parse_input_bandar(m_b3)) - \
                (parse_input_bandar(m_s1) + parse_input_bandar(m_s2) + parse_input_bandar(m_s3))

    is_bandar_present = (net_daily != 0) or (net_weekly != 0)

    if btn_start:
        if not TICKER_INPUT:
            st.error("Masukkan kode ticker dulu.")
            st.stop()

        set_seeds(int(SEED))

        val_pred_h1 = np.array([])
        val_real_h1 = np.array([])
        h1_dates = []

        ai_tomorrow = np.nan
        ai_chg = np.nan
        predictions = []
        FEATURES = []

        cmf_now = np.nan
        mfi_now = np.nan
        flow_label = "⚪ Flow N/A (Data pendek)"

        with st.status("🔄 Memproses Data...", expanded=True) as status:
            status.write("⬇️ Download data dari Yahoo Finance...")
            raw_df = get_stock_data(TICKER, START_DATE, END_DATE)
            if raw_df is None or raw_df.empty:
                st.error("Data tidak ditemukan / belum tersedia di Yahoo Finance untuk ticker tersebut.")
                st.stop()

            curr = raw_df.iloc[-1]
            prev = raw_df.iloc[-2] if len(raw_df) >= 2 else curr

            chg = float(curr["Adj Close"] - prev["Adj Close"]) if len(raw_df) >= 2 else 0.0
            chg_pct = float((chg / prev["Adj Close"]) * 100) if len(raw_df) >= 2 and prev["Adj Close"] != 0 else 0.0

            vol_ma20 = raw_df["Volume"].rolling(window=20).mean().iloc[-1] if len(raw_df) >= 20 else np.nan
            curr_vol = float(curr["Volume"])
            weekly_vol_sum = float(raw_df["Volume"].iloc[-5:].sum()) if len(raw_df) >= 5 else float(raw_df["Volume"].sum())

            if len(raw_df) >= 20 and np.isfinite(vol_ma20):
                if curr_vol > (vol_ma20 * 1.2):
                    vol_status = "🔥 HIGH (Spike)"
                    vol_desc = "High Activity"
                elif curr_vol < (vol_ma20 * 0.8):
                    vol_status = "❄️ LOW (Dry)"
                    vol_desc = "Low Activity"
                else:
                    vol_status = "⚖️ NORMAL"
                    vol_desc = "Avg Activity"
            else:
                vol_status = "N/A"
                vol_desc = "N/A"

            buy_vol_5, sell_vol_5, buy_ratio_5, vol_dom_label = volume_dominance(raw_df, lookback=5)

            ema20_raw = EMAIndicator(raw_df["Adj Close"], window=20).ema_indicator()
            ema20_now = float(ema20_raw.iloc[-1]) if len(ema20_raw.dropna()) > 0 else np.nan
            trend_label = "⚪ Trend N/A"
            if np.isfinite(ema20_now):
                trend_label = "🟢 Above EMA20" if float(curr["Adj Close"]) >= ema20_now else "🔴 Below EMA20"

            status.write("🧪 Menghitung indikator teknikal (AI mode)...")
            df = compute_indicators(raw_df)

            if df is None or df.empty:
                status.write("⚠️ Data indikator kurang (IPO/saham baru). AI dimatikan, tapi analisa non-AI tetap jalan.")
                is_ai = False
            else:
                cmf_now = finite_or_nan(df["CMF"].iloc[-1])
                mfi_now = finite_or_nan(df["MFI"].iloc[-1])
                if np.isfinite(cmf_now):
                    flow_label = "🟢 Money Flow Masuk" if cmf_now > 0 else "🔴 Money Flow Keluar"
                else:
                    flow_label = "⚪ Flow N/A (Data pendek)"

                FORECAST = [1, 3, 5, 7]
                MIN_ROWS_AFTER = int(LOOK_BACK) + int(max(FORECAST)) + 50
                is_ai = len(df) >= MIN_ROWS_AFTER

                FEATURES = ["Adj Close", "RSI", "MACD", "EMA", "USD_IDR", "Bandar_Flow", "CMF", "MFI"]
                if is_ai:
                    missing = [c for c in FEATURES if c not in df.columns]
                    if missing:
                        status.write(f"⚠️ Fitur AI kurang: {missing}. AI dimatikan.")
                        is_ai = False

            if is_ai:
                status.write("🧠 Training AI Model...")

                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(df[FEATURES].values)
                n_features = scaled.shape[1]

                X, Y = build_sequences(scaled, int(LOOK_BACK), [1, 3, 5, 7])
                if X is None or Y is None or len(X) < 60:
                    status.write("⚠️ Sequence AI terlalu sedikit. AI dimatikan.")
                    is_ai = False
                else:
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = Y[:train_size], Y[train_size:]

                    if len(X_test) < 5:
                        status.write("⚠️ X_test terlalu sedikit. AI dimatikan.")
                        is_ai = False
                    else:
                        model = Sequential(
                            [
                                Bidirectional(LSTM(60, return_sequences=True), input_shape=(int(LOOK_BACK), n_features)),
                                Dropout(0.2),
                                Bidirectional(LSTM(30)),
                                Dense(4),
                            ]
                        )
                        model.compile(optimizer="adam", loss="mse")
                        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

                        model.fit(
                            X_train,
                            y_train,
                            epochs=int(EPOCHS_1),
                            batch_size=32,
                            verbose=0,
                            validation_split=0.1,
                            callbacks=[es],
                        )

                        val_pred = model.predict(X_test, verbose=0)

                        dummy_pred = np.zeros((len(val_pred), n_features))
                        dummy_pred[:, 0] = val_pred[:, 0]
                        val_pred_h1 = scaler.inverse_transform(dummy_pred)[:, 0]

                        dummy_real = np.zeros((len(y_test), n_features))
                        dummy_real[:, 0] = y_test[:, 0]
                        val_real_h1 = scaler.inverse_transform(dummy_real)[:, 0]

                        test_start_i = train_size
                        test_end_i = train_size + len(X_test)
                        h1_dates = [df.index[i + int(LOOK_BACK) + 1] for i in range(test_start_i, test_end_i)]

                        if int(EPOCHS_2) > 0:
                            model.fit(X, Y, epochs=int(EPOCHS_2), batch_size=32, verbose=0)

                        last_batch = scaled[-int(LOOK_BACK):].reshape(1, int(LOOK_BACK), n_features)
                        pred_raw = model.predict(last_batch, verbose=0)

                        dummy_fut = np.zeros((4, n_features))
                        dummy_fut[:, 0] = pred_raw[0]
                        predictions = scaler.inverse_transform(dummy_fut)[:, 0].tolist()

                        ai_tomorrow = float(predictions[0])
                        price_now = float(curr["Adj Close"])
                        ai_chg = ((ai_tomorrow - price_now) / price_now) * 100 if price_now != 0 else np.nan

            status.update(label="Selesai!", state="complete", expanded=False)

        st.divider()
        st.subheader(f"📸 SNAPSHOT ANALISA: {TICKER}")

        col_res_1, col_res_2 = st.columns(2)

        with col_res_1:
            st.markdown("#### 📊 MARKET SUMMARY")
            st.caption(f"📅 {raw_df.index[-1].strftime('%d %b %Y')} | Range: {float(curr['Low']):.0f}-{float(curr['High']):.0f}")

            c1, c2 = st.columns(2)
            c1.metric("Close", f"{float(curr['Adj Close']):.0f}", f"{chg_pct:.2f}%")
            c2.metric("Vol Harian", format_volume(curr_vol), vol_desc)

            st.markdown(
                f"<div style='margin-top:5px; font-size:0.9rem; color:#ddd;'>"
                f"📦 <b>Vol Mingguan:</b> {format_volume(weekly_vol_sum)} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Status:</b> {vol_status}"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='margin-top:5px; font-size:0.9rem; color:#ddd;'>"
                f"🧾 <b>Dominasi Vol (5H):</b> {vol_dom_label} "
                f"(<b>Buy</b> {format_volume(buy_vol_5)} vs <b>Sell</b> {format_volume(sell_vol_5)} | <b>Buy%</b> {buy_ratio_5*100:.0f}%)"
                f"</div>",
                unsafe_allow_html=True,
            )

            cmf_txt = f"{cmf_now:.2f}" if np.isfinite(cmf_now) else "N/A"
            mfi_txt = f"{mfi_now:.0f}" if np.isfinite(mfi_now) else "N/A"
            st.markdown(
                f"<div style='margin-top:5px; font-size:0.9rem; color:#ddd;'>"
                f"💧 <b>Flow:</b> {flow_label} | <b>CMF(20)</b>: {cmf_txt} | <b>MFI(14)</b>: {mfi_txt}"
                f"</div>",
                unsafe_allow_html=True,
            )

            ema_txt = f"{ema20_now:.0f}" if np.isfinite(ema20_now) else "N/A"
            st.markdown(
                f"<div style='margin-top:5px; font-size:0.9rem; color:#ddd;'>"
                f"📈 <b>Trend:</b> {trend_label} | <b>EMA20</b>: {ema_txt}"
                f"</div>",
                unsafe_allow_html=True,
            )

            if broksum_info:
                st.caption(broksum_info)

        with col_res_2:
            st.markdown("#### 🤖 PREDIKSI AI (H+1)")
            if is_ai and predictions and np.isfinite(ai_tomorrow):
                col_c, _ = st.columns(2)
                col_c.metric("Target H+1", f"{ai_tomorrow:.0f}", f"{ai_chg:.2f}%")

                color_h3 = "#4caf50" if predictions[1] > float(curr["Adj Close"]) else "#f44336"
                color_h7 = "#4caf50" if predictions[3] > float(curr["Adj Close"]) else "#f44336"
                st.markdown(
                    f"""
                <div style="font-size:0.95rem; margin-top:5px; background-color: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px;">
                    <span style="color: {color_h3}; font-weight: bold;">H+3: {predictions[1]:.0f}</span> &nbsp;|&nbsp;
                    <span style="color: {color_h7}; font-weight: bold;">H+7: {predictions[3]:.0f}</span>
                </div>""",
                    unsafe_allow_html=True,
                )
                st.caption(f"Fitur: {', '.join(FEATURES)} | LOOK_BACK={LOOK_BACK} | Rows valid={len(df)}")
            else:
                st.info("AI Non-aktif (IPO/data pendek). Tetap ada analisa Market + Volume + Broksum + Trend.")

        st.markdown("#### 📝 ANALISA FINAL")

        def get_stat(net):
            if net == 0:
                return "⚪ NETRAL / NO DATA"
            return "🟢 AKUMULASI" if net > 0 else "🔴 DISTRIBUSI"

        s_daily = get_stat(net_daily)
        s_weekly = get_stat(net_weekly)

        dom_txt = " + Seimbang"
        if vol_dom_label == "🟢 Dominan BUY":
            dom_txt = " + Dominan BUY"
        elif vol_dom_label == "🔴 Dominan SELL":
            dom_txt = " + Dominan SELL"

        flow_txt = ""
        if np.isfinite(cmf_now):
            flow_txt = " + Flow Masuk" if cmf_now > 0 else " + Flow Keluar"
        else:
            flow_txt = " + Flow N/A"

        trend_txt = " + Trend N/A"
        if np.isfinite(ema20_now):
            trend_txt = " + Above EMA20" if float(curr["Adj Close"]) >= ema20_now else " + Below EMA20"

        verdict, desc, bg = "", "", "#444"

        bandar_distribution = is_bandar_present and (net_daily < 0) and (net_weekly < 0)
        bandar_accum = is_bandar_present and (net_daily > 0) and (net_weekly > 0)

        if not is_ai:
            if bandar_accum:
                if vol_dom_label == "🟢 Dominan BUY" or (np.isfinite(cmf_now) and cmf_now > 0) or (np.isfinite(ema20_now) and float(curr["Adj Close"]) >= ema20_now):
                    verdict, bg = "✅ AKUMULASI KUAT (No AI)", "#28a745"
                    desc = f"Broksum akumulasi (H&W){dom_txt}{flow_txt}{trend_txt}."
                else:
                    verdict, bg = "✅ AKUMULASI (No AI)", "#20c997"
                    desc = f"Broksum akumulasi (H&W) tapi konfirmasi teknikal belum kuat{dom_txt}{flow_txt}{trend_txt}."
            elif bandar_distribution:
                verdict, bg = "⚠️ DISTRIBUSI (No AI)", "#ffc107"
                desc = f"Broksum distribusi (H&W){dom_txt}{flow_txt}{trend_txt}. Waspada."
            else:
                if "HIGH" in vol_status and vol_dom_label == "🔴 Dominan SELL" and (not np.isfinite(ema20_now) or float(curr["Adj Close"]) < ema20_now):
                    verdict, bg = "⚠️ DISTRIBUSI KUAT (No AI)", "#dc3545"
                    desc = f"Volume tinggi tapi tekanan jual{dom_txt}{flow_txt}{trend_txt}."
                elif "HIGH" in vol_status and vol_dom_label == "🟢 Dominan BUY" and (not np.isfinite(ema20_now) or float(curr["Adj Close"]) >= ema20_now):
                    verdict, bg = "✅ AKUMULASI KUAT (No AI)", "#28a745"
                    desc = f"Volume tinggi dengan indikasi pembelian{dom_txt}{flow_txt}{trend_txt}."
                else:
                    verdict, bg = "⚠️ SPEKULATIF / MENUNGGU (No AI)", "#6c757d"
                    desc = f"Sinyal campuran{dom_txt}{flow_txt}{trend_txt}."
        else:
            price_now = float(curr["Adj Close"])
            ai_up = np.isfinite(ai_tomorrow) and (ai_tomorrow > price_now)

            if bandar_distribution:
                if np.isfinite(ai_chg) and ai_chg <= -SELL_THRESHOLD:
                    verdict, bg = "❌ STRONG SELL / AVOID", "#dc3545"
                    desc = f"Bandar distribusi (H&W) + AI bearish {ai_chg:.2f}% (≥ {SELL_THRESHOLD:.1f}%)."
                else:
                    verdict, bg = "📉 KOREKSI / DISTRIBUSI SEDANG", "#d63384"
                    desc = f"Bandar distribusi (H&W) + AI bearish ringan {ai_chg:.2f}% (< {SELL_THRESHOLD:.1f}%). Fase koreksi."
            else:
                if not is_bandar_present:
                    if ai_up:
                        if vol_dom_label == "🟢 Dominan BUY" and (not np.isfinite(cmf_now) or cmf_now > 0):
                            verdict, bg = "🚀 AI BULLISH (Confirm Vol/Flow)", "#28a745"
                            desc = f"AI naik + konfirmasi akumulasi{dom_txt}{flow_txt}{trend_txt}."
                        else:
                            verdict, bg = "📈 AI BULLISH (No Bandar)", "#6610f2"
                            desc = f"AI prediksi naik{dom_txt}{flow_txt}{trend_txt}. Tidak ada konfirmasi bandar."
                    else:
                        if np.isfinite(ai_chg) and ai_chg <= -SELL_THRESHOLD and (vol_dom_label == "🔴 Dominan SELL" or (np.isfinite(cmf_now) and cmf_now < 0)):
                            verdict, bg = "❌ STRONG SELL / AVOID", "#dc3545"
                            desc = f"AI bearish {ai_chg:.2f}% (≥ {SELL_THRESHOLD:.1f}%) + tekanan jual{dom_txt}{flow_txt}{trend_txt}."
                        else:
                            verdict, bg = "📉 AI BEARISH (No Bandar)", "#6c757d"
                            desc = f"AI prediksi turun {ai_chg:.2f}%{dom_txt}{flow_txt}{trend_txt}."
                else:
                    if ai_up:
                        if bandar_accum and vol_dom_label == "🟢 Dominan BUY" and (not np.isfinite(cmf_now) or cmf_now > 0):
                            verdict, bg = "🚀 SUPER STRONG BUY", "#28a745"
                            desc = f"AI naik + bandar akumulasi kompak{dom_txt}{flow_txt}{trend_txt}."
                        elif net_weekly < 0:
                            verdict, bg = "⚠️ BULL TRAP (Waspada)", "#ffc107"
                            desc = f"AI naik tapi weekly cenderung distribusi{dom_txt}{flow_txt}{trend_txt}."
                        else:
                            verdict, bg = "✅ BUY (Moderate)", "#20c997"
                            desc = f"AI naik. Bandar campuran{dom_txt}{flow_txt}{trend_txt}."
                    else:
                        if net_daily > 0 and (vol_dom_label == "🟢 Dominan BUY" or (np.isfinite(cmf_now) and cmf_now > 0)):
                            verdict, bg = "💎 BUY ON WEAKNESS", "#17a2b8"
                            desc = f"AI turun tapi ada indikasi nampung (daily net buy){dom_txt}{flow_txt}{trend_txt}."
                        elif net_daily < 0:
                            if np.isfinite(ai_chg) and ai_chg <= -SELL_THRESHOLD:
                                verdict, bg = "❌ STRONG SELL / AVOID", "#dc3545"
                                desc = f"AI bearish {ai_chg:.2f}% (≥ {SELL_THRESHOLD:.1f}%) + bandar daily jualan."
                            else:
                                verdict, bg = "📉 KOREKSI / DISTRIBUSI SEDANG", "#d63384"
                                desc = f"AI bearish ringan {ai_chg:.2f}% (< {SELL_THRESHOLD:.1f}%) + bandar daily jualan. Fase koreksi."
                        else:
                            verdict, bg = "📉 KOREKSI SEHAT", "#d63384"
                            desc = f"Penurunan wajar{dom_txt}{flow_txt}{trend_txt}. Bandar netral."

        st.markdown(
            f"""
        <div style="border: 2px solid {bg}; border-radius: 12px; padding: 15px; background-color: rgba(255,255,255,0.05);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px; color: white;">
                <div><b>Harian:</b> {s_daily} <br><small>Net: {net_daily:,.0f}</small></div>
                <div><b>Mingguan:</b> {s_weekly} <br><small>Net: {net_weekly:,.0f}</small></div>
            </div>
            <hr style="margin: 5px 0; border-color: #444;">
            <div style="text-align: center; margin-top: 10px;">
                <h2 style="color: {bg}; margin: 0;">{verdict}</h2>
                <small style="color: #ccc;">{desc}</small>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("📉 Lihat Grafik Validasi & Tren", expanded=False):
            DATE_MODE = "month" if "Bulanan" in CHART_DATE_MODE else "day"

            show = raw_df.tail(180).copy()
            ema_disp = EMAIndicator(show["Adj Close"], window=20).ema_indicator()

            fig, ax = plt.subplots(figsize=(7.5, 2.6), dpi=140)
            ax.plot(show.index, show["Adj Close"], label="Price", linewidth=1.2)
            ax.plot(show.index, ema_disp, label="EMA(20)", linewidth=1.2)

            if SHOW_FORWARD_PRED and is_ai and predictions:
                last_date = show.index[-1]
                f_dates = pd.bdate_range(last_date, periods=max([1, 3, 5, 7]) + 1)[1:]
                f_map = {1: predictions[0], 3: predictions[1], 5: predictions[2], 7: predictions[3]}
                xs, ys = [], []
                for d in [1, 3, 5, 7]:
                    if d <= len(f_dates):
                        xs.append(f_dates[d - 1])
                        ys.append(f_map[d])
                ax.plot(
                    [last_date] + xs,
                    [float(show["Adj Close"].iloc[-1])] + ys,
                    linestyle="--",
                    linewidth=1.1,
                    label="Forward Pred",
                )

            ax.set_title(f"{TICKER} - Price vs EMA(20)", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            style_small_chart(ax, mode=DATE_MODE)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            if "df" in locals() and df is not None and not df.empty:
                show2 = df.tail(180).copy()

                fig2, ax2 = plt.subplots(figsize=(7.5, 2.1), dpi=140)
                ax2.plot(show2.index, show2["CMF"], label="CMF(20)", linewidth=1.1)
                ax2.axhline(0, linewidth=1, alpha=0.5)
                ax2.set_title("Chaikin Money Flow (20)", fontsize=10)
                ax2.legend(fontsize=8, loc="upper left")
                style_small_chart(ax2, mode=DATE_MODE)
                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)

                fig3, ax3 = plt.subplots(figsize=(7.5, 2.1), dpi=140)
                ax3.plot(show2.index, show2["MFI"], label="MFI(14)", linewidth=1.1)
                ax3.axhline(80, linewidth=1, alpha=0.5)
                ax3.axhline(20, linewidth=1, alpha=0.5)
                ax3.set_title("Money Flow Index (14)", fontsize=10)
                ax3.legend(fontsize=8, loc="upper left")
                style_small_chart(ax3, mode=DATE_MODE)
                fig3.tight_layout()
                st.pyplot(fig3, use_container_width=True)
            else:
                st.info("Chart CMF/MFI tidak tersedia (IPO/data indikator kurang).")

            if is_ai and len(val_real_h1) > 0 and len(val_pred_h1) > 0 and len(h1_dates) == len(val_real_h1):
                df_val = pd.DataFrame(
                    {"Harga Real H+1": val_real_h1, "AI Pred H+1": val_pred_h1},
                    index=pd.to_datetime(h1_dates),
                )
                st.caption("Validasi H+1: prediksi historis (X_test) vs harga real pada tanggal target (t+1).")
                st.line_chart(df_val, height=240)
            else:
                st.info("Validasi AI tidak tersedia (mode IPO / AI nonaktif).")

# =========================================================
# TAB 2: KALKULATOR RATA-RATA (FULL, LANGSUNG JALAN)
# =========================================================
with tab2:
    st.markdown('<div class="bigtitle">🧮 Kalkulator Harga Rata-rata</div>', unsafe_allow_html=True)
    st.caption("Buat hitung avg up / avg down, bisa multi transaksi, plus mode ledger (BUY/SELL) opsional.")

    st.markdown(
        """
        <div class="card">
          <b>Fungsi:</b> hitung rata-rata baru setelah kamu tambah beli (avg up/down).  
          <br><span class="muted">Isi lot & harga, sistem otomatis hitung avg terbaru, total modal, dan target profit.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        lot_size = st.number_input("Ukuran 1 lot (lembar)", min_value=1, max_value=1000, value=100, step=1, key="avg_lot_size")
    with c2:
        include_fee = st.checkbox("Include fee (opsional)", value=False, key="avg_include_fee")
    with c3:
        fee_pct = st.number_input(
            "Fee total (%)", min_value=0.0, max_value=2.0, value=0.0, step=0.01,
            disabled=not include_fee, key="avg_fee_pct"
        )

    st.markdown("### 1) Posisi Awal")
    a1, a2, a3 = st.columns(3)
    with a1:
        avg0 = st.number_input("Harga rata-rata awal", min_value=0.0, value=150.0, step=1.0, key="avg_avg0")
    with a2:
        lot0 = st.number_input("Jumlah lot awal", min_value=0.0, value=10.0, step=1.0, key="avg_lot0")
    with a3:
        saham0 = lot0 * lot_size
        cost0 = avg0 * saham0
        if include_fee and fee_pct > 0:
            cost0 *= (1 + fee_pct / 100)
        st.metric("Total Modal Awal", f"{cost0:,.0f}")

    st.markdown("---")
    st.markdown("### 2) Tambah Pembelian (Avg Up / Avg Down)")
    st.caption("Kamu bisa tambah beberapa kali pembelian. Isi harga & lot, lalu tambah baris.")

    if "avg_rows" not in st.session_state:
        st.session_state.avg_rows = [{"harga": 0.0, "lot": 0.0}]

    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        if st.button("➕ Tambah Baris", key="avg_add_row"):
            st.session_state.avg_rows.append({"harga": 0.0, "lot": 0.0})
    with r2:
        if st.button("🧹 Reset Baris", key="avg_reset_rows"):
            st.session_state.avg_rows = [{"harga": 0.0, "lot": 0.0}]
    with r3:
        st.caption(" ")

    total_add_cost = 0.0
    total_add_shares = 0.0

    for i, row in enumerate(st.session_state.avg_rows):
        cc1, cc2, cc3 = st.columns([1.2, 1.2, 0.6])
        with cc1:
            price = st.number_input(
                f"Harga beli ke-{i+1}", min_value=0.0, value=float(row["harga"]), step=1.0, key=f"avg_price_{i}"
            )
        with cc2:
            lots = st.number_input(
                f"Lot beli ke-{i+1}", min_value=0.0, value=float(row["lot"]), step=1.0, key=f"avg_lot_{i}"
            )
        with cc3:
            if st.button("🗑️", key=f"avg_del_{i}"):
                st.session_state.avg_rows.pop(i)
                st.rerun()

        shares = lots * lot_size
        add_cost = price * shares
        if include_fee and fee_pct > 0:
            add_cost *= (1 + fee_pct / 100)

        total_add_cost += add_cost
        total_add_shares += shares

    st.markdown("---")
    st.markdown("### 3) Hasil")

    total_shares = saham0 + total_add_shares
    total_cost = cost0 + total_add_cost
    avg_new = (total_cost / total_shares) if total_shares > 0 else 0.0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Lembar", f"{total_shares:,.0f}")
    s2.metric("Total Lot", f"{(total_shares/lot_size):,.2f}")
    s3.metric("Total Modal", f"{total_cost:,.0f}")
    s4.metric("Rata-rata Baru", f"{avg_new:,.2f}")

    st.markdown(
        f"""
        <div class="card" style="margin-top:10px;">
          <b>Kesimpulan:</b> dari avg awal <b>{avg0:.2f}</b> dengan <b>{lot0:.0f}</b> lot,
          setelah tambah pembelian, avg kamu jadi <b style="font-size:1.1rem;">{avg_new:.2f}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 4) Target Harga (Break-even / Profit)")

    # --- TARGET VIA PERSEN ---
    t1, t2, t3 = st.columns(3)
    with t1:
        target_pct = st.number_input(
            "Target profit (%)", min_value=-50.0, max_value=200.0, value=5.0, step=0.5, key="avg_target_pct"
        )
    with t2:
        target_price_pct = avg_new * (1 + target_pct / 100)
        st.metric("Target Price (dari %)", f"{target_price_pct:,.2f}")
    with t3:
        unreal_pct = (target_price_pct - avg_new) * total_shares
        st.metric("Estimasi P/L (dari %)", f"{unreal_pct:,.0f}")

    st.markdown("")

    # --- TARGET VIA HARGA ---
    u1, u2, u3, u4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with u1:
        target_price_manual = st.number_input(
            "Target harga (Rp)", min_value=0.0, value=3200.0, step=1.0, key="avg_target_price_manual"
        )
    with u2:
        # % profit berdasarkan target harga
        profit_pct_from_price = ((target_price_manual - avg_new) / avg_new * 100) if avg_new > 0 else 0.0
        st.metric("% Profit (dari harga)", f"{profit_pct_from_price:,.2f}%")
    with u3:
        pl_from_price = (target_price_manual - avg_new) * total_shares
        st.metric("Estimasi P/L (dari harga)", f"{pl_from_price:,.0f}")
    with u4:
        # Break-even = avg_new, jadi jarak ke BE
        dist_to_be = target_price_manual - avg_new
        st.metric("Selisih vs Avg", f"{dist_to_be:,.2f}")

    st.markdown("---")
    st.markdown("### 5) Mode Ledger (Opsional)")
    st.caption("Kalau kamu mau catat transaksi BUY/SELL berkali-kali. Metode: average cost.")
    use_ledger = st.checkbox("Aktifkan mode ledger", value=False, key="avg_use_ledger")

    if use_ledger:
        if "ledger" not in st.session_state:
            st.session_state.ledger = pd.DataFrame(
                [{"Tanggal": pd.Timestamp.today().date(), "Aksi": "BUY", "Harga": 150.0, "Lot": 10.0}],
                columns=["Tanggal", "Aksi", "Harga", "Lot"],
            )

        edited = st.data_editor(
            st.session_state.ledger,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Aksi": st.column_config.SelectboxColumn("Aksi", options=["BUY", "SELL"]),
                "Harga": st.column_config.NumberColumn("Harga", min_value=0.0, step=1.0),
                "Lot": st.column_config.NumberColumn("Lot", min_value=0.0, step=1.0),
            },
            key="avg_ledger_editor",
        )
        st.session_state.ledger = edited

        shares = 0.0
        cost = 0.0
        for _, r in edited.iterrows():
            aksi = str(r["Aksi"]).upper().strip()
            harga = float(r["Harga"]) if np.isfinite(r["Harga"]) else 0.0
            lot = float(r["Lot"]) if np.isfinite(r["Lot"]) else 0.0
            qty = lot * lot_size

            if aksi == "BUY":
                add_cost = harga * qty
                if include_fee and fee_pct > 0:
                    add_cost *= (1 + fee_pct / 100)
                cost += add_cost
                shares += qty

            elif aksi == "SELL":
                if shares <= 0:
                    continue
                avg_now = cost / shares if shares > 0 else 0.0
                sell_qty = min(qty, shares)
                cost -= avg_now * sell_qty
                shares -= sell_qty

        avg_ledger = cost / shares if shares > 0 else 0.0

        st.markdown("#### Ringkasan Ledger")
        k1, k2, k3 = st.columns(3)
        k1.metric("Sisa Lot", f"{shares/lot_size:,.2f}")
        k2.metric("Rata-rata (ledger)", f"{avg_ledger:,.2f}")
        k3.metric("Modal tersisa (avg cost)", f"{cost:,.0f}")

