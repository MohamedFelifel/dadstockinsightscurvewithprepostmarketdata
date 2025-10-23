# app.py
# Streamlit stock movement tracker — full app with fixed 24h x-axis formatting
# Requires: yfinance, pandas, numpy, streamlit, plotly, matplotlib, tzlocal, pytz

import os
import time
from datetime import datetime, timedelta, time as dtime

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator, FuncFormatter

# timezone detection helper
try:
    from tzlocal import get_localzone_name
except Exception:
    get_localzone_name = None

import pytz

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Stock Movement Tracker", layout="wide")

# ----------------------------
# Default tickers & sector mapping
# ----------------------------
STOCKS = [
    "META","TSLA","AMZN","MSFT","NVDA","ORCL","AVGO","GFI","GOOG","GOOGL",
    "AAPL","AMD","PANW","VRT","QCOM","INOD","SMCI","RGTI","QBTS","QUBT",
    "LAES","BBAI","LUMN","IREN","ANET","HIMS","OKTA","MRVL","INTC","SOUN",
    "AGI","AEM","HMY","IAG","NEM","LAC","POET","UNH","JOBY","IONQ","JNJ","LLY"
]

SECTOR_MAPPING = {
    "META": "Tech","TSLA": "Automotive","AMZN": "E-Commerce","MSFT": "Tech","NVDA": "Tech",
    "ORCL": "Tech","AVGO": "Tech","GFI": "Finance","GOOG": "Tech","GOOGL": "Tech",
    "AAPL": "Tech","AMD": "Tech","PANW": "Tech","VRT": "Industrial","QCOM": "Tech",
    "INOD": "Tech","SMCI": "Tech","RGTI": "Tech","QBTS": "Tech","QUBT": "Tech",
    "LAES": "Energy","BBAI": "Tech","LUMN": "Telecom","IREN": "Energy","ANET": "Tech",
    "HIMS": "Healthcare","OKTA": "Tech","MRVL": "Tech","INTC": "Tech","SOUN": "Tech",
    "AGI": "Mining","AEM": "Mining","HMY": "Mining","IAG": "Mining","NEM": "Mining",
    "LAC": "Mining","POET": "Tech","UNH": "Healthcare","JOBY": "Aerospace","IONQ": "Tech"
}

# ----------------------------
# Time windows in hours
# ----------------------------
TIME_WINDOWS = {
    "5 Minutes": 5/60,
    "10 Minutes": 10/60,
    "15 Minutes": 15/60,
    "30 Minutes": 30/60,
    "1 Hour": 1,
    "2 Hours": 2,
    "4 Hours": 4,
    "6 Hours": 6,
    "12 Hours": 12,
    "24 Hours": 24,
    "36 Hours": 36,
    "48 Hours": 48,
    "72 Hours": 72,
    "96 Hours": 96,
    "1 Week": 24 * 7,
    "2 Weeks": 24 * 14,
    "1 Month": 24 * 30,
    "2 Months": 24 * 60,
    "6 Months": 24 * 30 * 6,
    "1 Year": 24 * 365,
    "2 Years": 24 * 365 * 2
}

PRESET_LABELS = list(TIME_WINDOWS.keys())
REQUEST_DELAY_SECONDS = 0.12  # polite pause between yfinance requests

# ----------------------------
# Mapping hours -> yfinance period/interval
# ----------------------------
def map_hours_to_period_interval(hours: float):
    if hours <= 10/60:
        return ("1d", "1m")
    if hours <= 0.25:
        return ("2d", "1m")
    if hours <= 1:
        return ("2d", "2m")
    if hours <= 2:
        return ("5d", "5m")
    if hours <= 4:
        return ("5d", "15m")
    if hours <= 6:
        return ("7d", "30m")
    if hours <= 12:
        return ("7d", "60m")
    if hours <= 24:
        return ("14d", "1h")
    if hours <= 36:
        return ("30d", "1h")
    if hours <= 48:
        return ("30d", "1h")
    if hours <= 72:
        return ("60d", "1d")
    if hours <= 96:
        return ("90d", "1d")
    if hours <= 24 * 21:
        return ("3mo", "1d")
    if hours <= 24 * 62:
        return ("6mo", "1d")
    if hours <= 24 * 200:
        return ("1y", "1d")
    return ("2y", "1d")

# ----------------------------
# Fetch series helpers
# ----------------------------
def fetch_time_series_for_range(ticker: str, period: str, interval: str, cutoff=None):
    """Fetch regular-hours only series (prepost=False). Returns tz-aware Close series or None."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=True, prepost=False)
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            try:
                df.index = df.index.tz_localize("UTC")
            except Exception:
                pass
        if cutoff is not None:
            try:
                df = df.loc[df.index >= cutoff]
            except Exception:
                pass
        if "Close" not in df.columns or df["Close"].shape[0] < 2:
            return None
        return df["Close"]
    except Exception:
        return None

def fetch_time_series_extended_hours(ticker: str, period: str, interval: str, cutoff=None):
    """Fetch minute data including extended hours (prepost=True). Returns tz-aware Close series or None."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=True, prepost=True)
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            try:
                df.index = df.index.tz_localize("UTC")
            except Exception:
                pass
        if cutoff is not None:
            try:
                df = df.loc[df.index >= cutoff]
            except Exception:
                pass
        if "Close" not in df.columns or df["Close"].shape[0] < 2:
            return None
        return df["Close"]
    except Exception:
        return None

def percent_change_series(series: pd.Series):
    if series is None or series.empty:
        return None
    s = series.astype(float)
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return None
    return (s / base - 1.0) * 100.0  # percent

@st.cache_data(ttl=30)
def fetch_percent_changes_for_tickers(tickers, period, interval):
    results = []
    warnings = []
    series_dict = {}
    for t in tickers:
        ser = fetch_time_series_for_range(t, period, interval)
        time.sleep(REQUEST_DELAY_SECONDS)
        if ser is None:
            warnings.append(f"No data for {t}")
            continue
        p = percent_change_series(ser)
        if p is None:
            warnings.append(f"Could not compute percent for {t}")
            continue
        series_dict[t] = p
        try:
            final_pct = float(p.iloc[-1])
        except Exception:
            warnings.append(f"Invalid final pct for {t}")
            continue
        results.append((t, round(final_pct, 4)))
    if not results:
        return pd.DataFrame(columns=["Stock", "Change %"]), warnings, series_dict
    df = pd.DataFrame(results, columns=["Stock", "Change %"])
    df = df.sort_values("Change %", ascending=False).reset_index(drop=True)
    return df, warnings, series_dict

# ----------------------------
# Plot helpers (fixed get_dtick_and_format)
# ----------------------------
def get_dtick_and_format(hours_span):
    """
    Return (dtick, tickformat) for Plotly x-axis.
    For any intraday span (<= 24h) return time-only tick labels ("%H:%M").
    For >24h we return date-based labels.
    """
    if hours_span <= 0.5:
        return 5 * 60 * 1000, "%H:%M"       # 5 min
    if hours_span <= 1:
        return 10 * 60 * 1000, "%H:%M"      # 10 min
    if hours_span <= 6:
        return 30 * 60 * 1000, "%H:%M"      # 30 min
    if hours_span <= 12:
        return 60 * 60 * 1000, "%H:%M"      # 1 hr
    if hours_span <= 24:
        # keep time-only labels on 24h view to avoid full-date labels
        return 2 * 60 * 60 * 1000, "%H:%M"  # 2 hr
    if hours_span <= 24 * 7:
        return "D1", "%b %d"                # 1 day
    else:
        return "D7", "%b %d"                # 1 week

def compute_y_range_for_plot(values_array, min_visible_span=2.0, pad_fraction=0.12):
    values_array = np.asarray(values_array, dtype=float)
    values_array = values_array[~np.isnan(values_array)]
    if values_array.size == 0:
        return -5.0, 5.0, 1.0
    data_min = float(np.min(values_array))
    data_max = float(np.max(values_array))
    raw_span = max(0.0, data_max - data_min)
    if raw_span < min_visible_span:
        visible_span = min_visible_span
    else:
        visible_span = raw_span * (1.0 + pad_fraction)
    center = (data_max + data_min) / 2.0
    y0 = center - visible_span / 2.0
    y1 = center + visible_span / 2.0
    if visible_span <= 5:
        dtick = 0.5
    elif visible_span <= 10:
        dtick = 1.0
    else:
        dtick = 2.0
    return y0, y1, dtick

# ----------------------------
# Timezone detection & selection
# ----------------------------
def detect_local_tz():
    tz_name = None
    if get_localzone_name is not None:
        try:
            tz_name = get_localzone_name()
        except Exception:
            tz_name = None
    if tz_name is None:
        tz_env = os.environ.get("TZ")
        if tz_env:
            tz_name = tz_env
    if tz_name is None:
        tz_name = "UTC"
    try:
        pytz.timezone(tz_name)
    except Exception:
        tz_name = "UTC"
    return tz_name

detected_tz = detect_local_tz()
all_timezones = ["UTC"] + sorted([t for t in pytz.all_timezones if "/" in t])
display_tz = st.sidebar.selectbox("Display timezone (chart labels):", options=all_timezones, index=(all_timezones.index(detected_tz) if detected_tz in all_timezones else 0))
st.sidebar.write(f"Detected tz: **{detected_tz}** — charts display times in **{display_tz}**. (Market session windows still use America/New_York.)")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Stock Movements", "Sector Percent Trends", "Pre/Post Market"])

# ---------- Tab 1: Stock Movements ----------
with tab1:
    st.title("📊 Stock Movement Tracker")
    st.write("Pick a timeframe (presets match the detailed hour-based windows). The static bar chart appears above the interactive chart.")

    col_left, col_right = st.columns([3,1])
    with col_left:
        preset_label = st.selectbox("Select Timeframe", PRESET_LABELS, index=3)
        hours = TIME_WINDOWS[preset_label]
        period, interval = map_hours_to_period_interval(hours)
        chosen_label = preset_label

        tickers_text = st.text_area("Tickers (comma separated) — leave blank to use defaults:", value=", ".join(STOCKS), height=120)
        tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
        if not tickers:
            tickers = STOCKS.copy()

    with col_right:
        view_mode = st.radio("View mode", ["Percent change (normalized)", "Raw price series"])
        fetch_btn = st.button("Fetch & Show")

    if fetch_btn:
        st.info(f"Fetching data for {chosen_label} (period={period}, interval={interval}) — this may take a few seconds.")
        df, warnings, series_dict = fetch_percent_changes_for_tickers(tickers, period, interval)

        if df.empty and not series_dict:
            st.warning("No series data retrieved.")
        else:
            # Matplotlib Bar Chart
            if not df.empty:
                labels = df["Stock"].tolist()
                values = df["Change %"].tolist()
                fig_w = max(10, len(labels)*0.45)
                fig_h = 5.5
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                colors = ["green" if v>=0 else "red" for v in values]
                ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.3)
                ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
                ylim_padding = max(abs(max(values)), abs(min(values)))*0.15
                ax.set_ylim(min(values)-ylim_padding, max(values)+ylim_padding)
                ax.yaxis.set_major_locator(AutoLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{y:.2f}%"))
                ax.grid(axis="y", which="major", linestyle="-", linewidth=0.8, alpha=0.7)
                ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
                ax.set_ylabel("Change %")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.subheader("📊 Full Stock Movement (Bar Chart)")
                st.pyplot(fig)
                plt.close(fig)

            # Plotly Interactive Chart
            st.subheader("📈 Interactive Chart")
            fig = go.Figure()
            all_vals = []
            valid_series = {}

            for t,s in series_dict.items():
                if s is None or len(s)==0:
                    continue
                s = s.dropna()
                if len(s)==0:
                    continue

                # convert the series to display timezone for plotting & labels
                try:
                    s_display = s.tz_convert(display_tz)
                except Exception:
                    s_display = s

                all_vals.append(s_display.values)
                valid_series[t] = s_display

                fig.add_trace(go.Scatter(
                    x=s_display.index,
                    y=s_display.values,
                    mode="lines",
                    name=t,
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}%<extra>%{fullData.name}</extra>"
                ))

            if all_vals:
                combined_vals = np.concatenate(all_vals)
                y0, y1, dtick = compute_y_range_for_plot(combined_vals, min_visible_span=2.0, pad_fraction=0.12)
                fig.update_yaxes(range=[y0, y1], tickformat=".2f", ticks="outside", dtick=dtick, fixedrange=False)
            else:
                fig.update_yaxes(range=[-5,5], tickformat=".2f", ticks="outside", dtick=1.0, fixedrange=False)

            if valid_series:
                combined_index = next(iter(valid_series.values())).index
                total_seconds = (combined_index[-1]-combined_index[0]).total_seconds()
                total_hours = total_seconds/3600.0

                dtick_x, tickformat = get_dtick_and_format(total_hours)
                x_tickformat = tickformat if isinstance(tickformat, str) else "%H:%M"

                fig.update_xaxes(title=f"Time ({display_tz})", dtick=dtick_x, tickformat=x_tickformat, rangeslider_visible=True)
                fig.add_shape(type="line", x0=combined_index[0], x1=combined_index[-1], y0=0, y1=0,
                              line=dict(color="gray", width=1, dash="dash"))

            fig.update_layout(title=f"Percent Change — {chosen_label}", hovermode="x unified", height=600,
                              dragmode="pan",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig, use_container_width=True)

            if warnings:
                st.subheader("Warnings / Errors")
                for w in warnings:
                    st.write("•", w)

# ---------- Tab 2: Sector Percent Trends ----------
with tab2:
    st.title("📈 Sector Percent Trends")
    sectors = sorted(list(set(SECTOR_MAPPING.values())))
    selected_sector = st.selectbox("Choose sector", sectors)
    sector_tickers = [t for t,s in SECTOR_MAPPING.items() if s==selected_sector]
    sector_range_label = st.selectbox("Range", PRESET_LABELS, index=3)
    show_btn = st.button("Show Sector Percent Chart")

    if show_btn and sector_tickers:
        hours = TIME_WINDOWS[sector_range_label]
        period, interval = map_hours_to_period_interval(hours)
        st.info(f"Fetching {len(sector_tickers)} tickers for {sector_range_label}...")
        pct_series = {}
        warnings = []
        for t in sector_tickers:
            s = fetch_time_series_for_range(t, period, interval)
            time.sleep(REQUEST_DELAY_SECONDS)
            p = percent_change_series(s)
            if p is not None:
                pct_series[t] = p
            else:
                warnings.append(f"{t} skipped")

        fig = go.Figure()
        all_vals = []
        valid_series = {}

        for t,s in pct_series.items():
            if s is None or len(s)==0:
                continue
            s = s.dropna()
            if len(s)==0:
                continue

            try:
                s_display = s.tz_convert(display_tz)
            except Exception:
                s_display = s

            all_vals.append(s_display.values)
            valid_series[t] = s_display
            fig.add_trace(go.Scatter(x=s_display.index, y=s_display.values, mode="lines", name=t,
                                     hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}%<extra>%{fullData.name}</extra>"))

        if all_vals:
            combined_vals = np.concatenate(all_vals)
            y0, y1, dtick = compute_y_range_for_plot(combined_vals, min_visible_span=2.0, pad_fraction=0.12)
            fig.update_yaxes(range=[y0, y1], tickformat=".2f", ticks="outside", dtick=dtick, fixedrange=False)
        else:
            fig.update_yaxes(range=[-5,5], tickformat=".2f", ticks="outside", dtick=1.0, fixedrange=False)

        if valid_series:
            combined_index = next(iter(valid_series.values())).index
            total_seconds = (combined_index[-1]-combined_index[0]).total_seconds()
            total_hours = total_seconds/3600.0
            dtick_x, tickformat = get_dtick_and_format(total_hours)
            x_tickformat = tickformat if isinstance(tickformat, str) else "%H:%M"
            fig.update_xaxes(title=f"Time ({display_tz})", dtick=dtick_x, tickformat=x_tickformat, rangeslider_visible=True)
            fig.add_shape(type="line", x0=combined_index[0], x1=combined_index[-1], y0=0, y1=0, line=dict(color="gray", width=1, dash="dash"))

        fig.update_layout(title=f"{selected_sector} — Percent Change ({sector_range_label})", yaxis_title="Change %", hovermode="x unified", height=600, dragmode="pan",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True)

        if warnings:
            st.subheader("Warnings")
            for w in warnings:
                st.write("•", w)

# ---------- Tab 3: Pre/Post Market (fully expanded, display tz applied) ----------
with tab3:
    st.title("🌅 Pre/Post Market (Extended Hours)")

    # Controls: single ticker plot and multi-ticker summary choices
    left_col, mid_col, right_col = st.columns([2, 1, 1])

    with left_col:
        pm_ticker_text = st.text_input("Ticker (single) for plotting:", value="AAPL").strip().upper()

    with mid_col:
        session_choice = st.selectbox("Session to inspect", ["Premarket (04:00–09:30 ET)", "After-hours (16:00–20:00 ET)", "Both (04:00–20:00 ET)"])

    with right_col:
        chosen_date = st.date_input("Date (US Eastern day):", value=datetime.utcnow().date())

    st.markdown("---")

    # Multi-ticker premarket summary controls
    st.subheader("Multi-ticker Pre/Post Market Summary")
    st.write("Enter a comma-separated list of tickers to get a quick pre/post summary (last extended price and % vs previous regular close).")
    multi_tickers_text = st.text_area("Tickers (summary):", value="AAPL,MSFT,TSLA,AMD,NVDA", height=80)
    multi_fetch_btn = st.button("Fetch Summary")

    st.markdown("---")

    # Plot display toggle for the single-ticker chart
    st.subheader("Single-ticker Extended Session Plot")
    display_choice = st.radio("Display", ["Price + %", "Price only", "% only"], index=0)
    pm_fetch_btn = st.button("Fetch & Plot Single Ticker")

    # Helper: compute session UTC bounds from chosen_date and session choice
    def get_session_time_range_for_date(trade_date, session):
        ny_tz = "America/New_York"
        if session == "premarket":
            start_local = datetime.combine(trade_date, dtime(hour=4, minute=0))
            end_local = datetime.combine(trade_date, dtime(hour=9, minute=30))
            label = "Premarket"
        elif session == "afterhours":
            start_local = datetime.combine(trade_date, dtime(hour=16, minute=0))
            end_local = datetime.combine(trade_date, dtime(hour=20, minute=0))
            label = "After-hours"
        else:
            start_local = datetime.combine(trade_date, dtime(hour=4, minute=0))
            end_local = datetime.combine(trade_date, dtime(hour=20, minute=0))
            label = "Extended (Pre + Post)"
        try:
            start_local = pd.Timestamp(start_local).tz_localize(ny_tz)
            end_local = pd.Timestamp(end_local).tz_localize(ny_tz)
            start_utc = start_local.tz_convert("UTC")
            end_utc = end_local.tz_convert("UTC")
        except Exception:
            start_utc = pd.Timestamp(start_local).tz_localize("UTC")
            end_utc = pd.Timestamp(end_local).tz_localize("UTC")
        return start_utc, end_utc, label

    # ----------------------------
    # Multi-ticker summary action
    # ----------------------------
    if multi_fetch_btn:
        tickers_list = [t.strip().upper() for t in multi_tickers_text.split(",") if t.strip()]
        if not tickers_list:
            st.error("Please enter at least one ticker for the summary.")
        else:
            st.info(f"Fetching extended session summary for {len(tickers_list)} tickers...")
            start_utc, end_utc, label = get_session_time_range_for_date(chosen_date,
                                                                        "premarket" if session_choice.startswith("Premarket") else ("afterhours" if session_choice.startswith("After-hours") else "both"))
            summary_rows = []
            for t in tickers_list:
                try:
                    ser = fetch_time_series_extended_hours(t, "2d", "1m")
                    time.sleep(REQUEST_DELAY_SECONDS)
                    if ser is None or len(ser) == 0:
                        summary_rows.append({"Ticker": t, "Status": "no extended data", "Last Extended Price": None, "% vs prev close": None})
                        continue
                    try:
                        ser_utc = ser.tz_convert("UTC")
                    except Exception:
                        ser_utc = ser
                    try:
                        ser_session = ser_utc.loc[(ser_utc.index >= start_utc) & (ser_utc.index <= end_utc)]
                    except Exception:
                        try:
                            ser_ny = ser.tz_convert("America/New_York")
                            start_ny = start_utc.tz_convert("America/New_York")
                            end_ny = end_utc.tz_convert("America/New_York")
                            ser_session = ser_ny.loc[(ser_ny.index >= start_ny) & (ser_ny.index <= end_ny)]
                        except Exception:
                            ser_session = pd.Series(dtype=float)
                    if ser_session is None or len(ser_session) == 0:
                        summary_rows.append({"Ticker": t, "Status": "no data in session", "Last Extended Price": None, "% vs prev close": None})
                        continue
                    last_price = float(ser_session.iloc[-1])
                    # fetch previous regular close
                    tk = yf.Ticker(t)
                    daily = tk.history(period="7d", interval="1d", auto_adjust=True)
                    time.sleep(REQUEST_DELAY_SECONDS)
                    prev_close_val = None
                    if daily is not None and not daily.empty:
                        try:
                            daily_idx = daily.index
                            if daily_idx.tz is None:
                                daily_idx = daily_idx.tz_localize("UTC")
                            daily_ny = daily_idx.tz_convert("America/New_York")
                            found_val = None
                            for ts in reversed(daily_ny):
                                if ts.date() <= chosen_date:
                                    ts_utc = ts.tz_convert("UTC")
                                    try:
                                        found_val = daily.loc[ts_utc]["Close"]
                                    except Exception:
                                        found_val = daily["Close"].iloc[-1]
                                    break
                            if found_val is not None:
                                prev_close_val = float(found_val)
                        except Exception:
                            try:
                                prev_close_val = float(daily["Close"].iloc[-1])
                            except Exception:
                                prev_close_val = None
                    # fallback: minute series pre-09:30
                    if prev_close_val is None:
                        try:
                            ser_ny = ser.tz_convert("America/New_York")
                            cutoff_time = pd.Timestamp(datetime.combine(chosen_date, dtime(hour=9, minute=30))).tz_localize("America/New_York")
                            prev_candidates = ser_ny[ser_ny.index < cutoff_time]
                            if len(prev_candidates) > 0:
                                prev_close_val = float(prev_candidates.iloc[-1])
                        except Exception:
                            prev_close_val = None
                    if prev_close_val is None:
                        pct_vs_prev = None
                        status = "no prev close"
                    else:
                        pct_vs_prev = (last_price / prev_close_val - 1.0) * 100.0
                        status = "ok"
                    summary_rows.append({"Ticker": t, "Status": status, "Last Extended Price": last_price, "% vs prev close": round(pct_vs_prev, 3) if pct_vs_prev is not None else None})
                except Exception as e:
                    summary_rows.append({"Ticker": t, "Status": f"error: {e}", "Last Extended Price": None, "% vs prev close": None})
            summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
            st.subheader(f"{label} Summary ({chosen_date.isoformat()})")
            st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")

    # ----------------------------
    # Single-ticker plot action
    # ----------------------------
    if pm_fetch_btn:
        ticker = pm_ticker_text
        if not ticker:
            st.error("Please provide a ticker symbol.")
        else:
            period = "2d"
            interval = "1m"
            st.info(f"Fetching minute data (extended hours) for {ticker} — this may take a few seconds.")
            series = fetch_time_series_extended_hours(ticker, period, interval)
            time.sleep(REQUEST_DELAY_SECONDS)
            if series is None or len(series) == 0:
                st.warning("No minute data (extended hours) available for this ticker. yfinance may not provide extended-minute data for all symbols.")
            else:
                try:
                    series_display = series.tz_convert(display_tz)
                except Exception:
                    series_display = series
                try:
                    series_eastern = series.tz_convert("America/New_York")
                except Exception:
                    series_eastern = series
                start_utc, end_utc, label = get_session_time_range_for_date(chosen_date,
                                                                            "premarket" if session_choice.startswith("Premarket") else ("afterhours" if session_choice.startswith("After-hours") else "both"))
                try:
                    series_filtered = series.loc[(series.index >= start_utc) & (series.index <= end_utc)]
                except Exception:
                    try:
                        start_ny = start_utc.tz_convert("America/New_York")
                        end_ny = end_utc.tz_convert("America/New_York")
                        series_filtered = series_eastern.loc[(series_eastern.index >= start_ny) & (series_eastern.index <= end_ny)]
                        try:
                            series_filtered = series_filtered.tz_convert("UTC")
                        except Exception:
                            pass
                    except Exception:
                        series_filtered = pd.Series(dtype=float)
                if series_filtered is None or len(series_filtered) == 0:
                    st.warning("No extended-hours minute data for the chosen session/date. Try a different date or symbol.")
                else:
                    tk = yf.Ticker(ticker)
                    daily = tk.history(period="7d", interval="1d", auto_adjust=True)
                    prev_close_val = None
                    if daily is not None and not daily.empty:
                        try:
                            daily_idx = daily.index
                            if daily_idx.tz is None:
                                daily_idx = daily_idx.tz_localize("UTC")
                            daily_ny = daily_idx.tz_convert("America/New_York")
                            found_val = None
                            for ts in reversed(daily_ny):
                                if ts.date() <= chosen_date:
                                    ts_utc = ts.tz_convert("UTC")
                                    try:
                                        found_val = daily.loc[ts_utc]["Close"]
                                    except Exception:
                                        found_val = daily["Close"].iloc[-1]
                                    break
                            if found_val is not None:
                                prev_close_val = float(found_val)
                        except Exception:
                            try:
                                prev_close_val = float(daily["Close"].iloc[-1])
                            except Exception:
                                prev_close_val = None
                    if prev_close_val is None:
                        try:
                            ser_ny = series.tz_convert("America/New_York")
                            cutoff_time = pd.Timestamp(datetime.combine(chosen_date, dtime(hour=9, minute=30))).tz_localize("America/New_York")
                            prev_candidates = ser_ny[ser_ny.index < cutoff_time]
                            if len(prev_candidates) > 0:
                                prev_close_val = float(prev_candidates.iloc[-1])
                        except Exception:
                            prev_close_val = None
                    if prev_close_val is None:
                        pct_series = None
                        st.warning("Could not determine previous close for percent-change baseline. Plotting raw prices instead.")
                    else:
                        try:
                            pct_series = (series_filtered.astype(float) / prev_close_val - 1.0) * 100.0
                        except Exception:
                            pct_series = None
                    try:
                        series_filtered_display = series_filtered.tz_convert(display_tz)
                    except Exception:
                        series_filtered_display = series_filtered
                    if pct_series is not None:
                        try:
                            pct_series_display = pct_series.tz_convert(display_tz)
                        except Exception:
                            pct_series_display = pct_series
                    fig = go.Figure()
                    if display_choice in ("Price + %", "Price only"):
                        fig.add_trace(go.Scatter(
                            x=series_filtered_display.index,
                            y=series_filtered_display.values,
                            mode="lines",
                            name=f"{ticker} Price (extended)",
                            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra>Price</extra>"
                        ))
                    if display_choice in ("Price + %", "% only") and (pct_series is not None):
                        fig.add_trace(go.Scatter(
                            x=pct_series_display.index,
                            y=pct_series_display.values,
                            mode="lines",
                            name=f"{ticker} % vs prev close",
                            yaxis="y2",
                            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}%<extra>Pct</extra>"
                        ))
                    try:
                        session_start_display = start_utc.tz_convert(display_tz)
                        session_end_display = end_utc.tz_convert(display_tz)
                        fig.add_vrect(x0=session_start_display, x1=session_end_display,
                                      fillcolor="LightSalmon", opacity=0.15, layer="below", line_width=0,
                                      annotation_text=label, annotation_position="top left")
                    except Exception:
                        pass
                    price_vals = series_filtered.values.astype(float)
                    if price_vals.size > 0:
                        price_span = max((price_vals.max() - price_vals.min()), price_vals.max()*0.001 if price_vals.max() != 0 else 1.0)
                        p_y0, p_y1, _ = compute_y_range_for_plot(price_vals, min_visible_span=price_span if price_span > 0 else 1.0, pad_fraction=0.08)
                    else:
                        p_y0, p_y1 = None, None
                    if pct_series is not None:
                        pct_vals = pct_series.values.astype(float)
                        if pct_vals.size > 0:
                            y0_pct, y1_pct, dtick_pct = compute_y_range_for_plot(pct_vals, min_visible_span=1.0, pad_fraction=0.12)
                        else:
                            y0_pct, y1_pct, dtick_pct = -5, 5, 1.0
                    else:
                        y0_pct, y1_pct, dtick_pct = -5, 5, 1.0
                    layout_yaxis = dict(title="Price", fixedrange=False)
                    if p_y0 is not None:
                        layout_yaxis["range"] = [p_y0, p_y1]
                    layout_yaxis2 = dict(title="% vs prev close", overlaying="y", side="right", fixedrange=False, tickformat=".2f", range=[y0_pct, y1_pct])
                    fig.update_layout(yaxis=layout_yaxis, yaxis2=layout_yaxis2, hovermode="x unified", dragmode="pan", height=650,
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                      title=f"{ticker} — {label} on {chosen_date.isoformat()} (display tz: {display_tz})")
                    try:
                        total_seconds = (series_filtered_display.index[-1] - series_filtered_display.index[0]).total_seconds()
                        total_hours = total_seconds / 3600.0
                    except Exception:
                        total_hours = 6.0
                    dtick_x, tickfmt = get_dtick_and_format(total_hours)
                    x_tickformat = tickfmt if isinstance(tickfmt, str) else "%H:%M"
                    fig.update_xaxes(title=f"Time ({display_tz})", rangeslider_visible=True, tickformat=x_tickformat)
                    st.subheader(f"{ticker} — {label} on {chosen_date.isoformat()}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("Summary:")
                    st.write(f"• Data points: {len(series_filtered)}")
                    if prev_close_val is not None:
                        last_price = float(series_filtered.iloc[-1])
                        last_pct = (last_price / prev_close_val - 1.0) * 100.0
                        st.write(f"• Previous regular close (baseline): {prev_close_val:.2f}")
                        st.write(f"• Last extended price in session: {last_price:.2f} ({last_pct:.2f}%)")
                    else:
                        st.write("• Previous close baseline unavailable — showing raw prices.")

# EOF



