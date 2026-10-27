
import os
import hmac
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import text

from engine import db, init, read, run, prices, score_frame, WATCH, START_CASH
import strategy_lab as lab

st.set_page_config(page_title="AI Investment Research Lab", page_icon="📈", layout="wide")
st.title("📈 AI Investment Research Lab")
st.caption("Educational paper-trading models • No brokerage orders • Institutional-inspired models are not affiliated with those institutions")

try:
    secret_password = str(st.secrets["APP_PASSWORD"])
    database_url = str(st.secrets["DATABASE_URL"])
except Exception:
    st.error("Set APP_PASSWORD and DATABASE_URL in Streamlit Secrets.")
    st.stop()

password = st.text_input("App password", type="password")
if not hmac.compare_digest(password, secret_password):
    st.stop()
os.environ["DATABASE_URL"] = database_url

try:
    engine = db()
    init(engine)
    lab.initialize(engine)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS personal_holdings (
                ticker TEXT PRIMARY KEY,
                shares DOUBLE PRECISION NOT NULL,
                cost_basis DOUBLE PRECISION
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS personal_snapshots (
                snapshot_date DATE PRIMARY KEY,
                market_value DOUBLE PRECISION NOT NULL,
                priced_count INTEGER NOT NULL,
                missing_count INTEGER NOT NULL
            )
        """))
except Exception as exc:
    st.error(f"Database setup failed ({type(exc).__name__}). Check Supabase logs.")
    st.stop()

ET = ZoneInfo("America/New_York")

def market_ready():
    now = datetime.now(ET)
    return now.weekday() < 5 and (now.hour, now.minute) >= (16, 20)


def read_tables():
    return lab.load(engine, "lab_trades"), lab.load(engine, "lab_snapshots")


def personal_holdings():
    return pd.read_sql_query(text("SELECT * FROM personal_holdings ORDER BY ticker"), engine)


def personal_snapshots():
    return pd.read_sql_query(text("SELECT * FROM personal_snapshots ORDER BY snapshot_date"), engine)


def save_holdings(frame):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM personal_holdings"))
        for _, row in frame.iterrows():
            cost = row.get("cost_basis")
            conn.execute(text("""
                INSERT INTO personal_holdings (ticker, shares, cost_basis)
                VALUES (:ticker, :shares, :cost)
            """), {
                "ticker": row["ticker"],
                "shares": float(row["shares"]),
                "cost": None if pd.isna(cost) else float(cost)
            })


def value_personal(record=True):
    frame = personal_holdings()
    if frame.empty:
        return frame, [], None
    history = lab.fetch_history(frame["ticker"].tolist(), "1mo")
    quotes = {}
    for ticker in frame["ticker"]:
        if ticker in history.columns:
            series = history[ticker].dropna()
            if not series.empty and float(series.iloc[-1]) > 0:
                quotes[ticker] = float(series.iloc[-1])
    frame["last_price"] = frame["ticker"].map(quotes)
    frame["market_value"] = frame["shares"] * frame["last_price"]
    missing = frame.loc[frame["last_price"].isna(), "ticker"].tolist()
    total = float(frame["market_value"].sum())
    # Do not write misleading snapshots if a holding cannot be priced.
    if record and not missing:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO personal_snapshots
                (snapshot_date, market_value, priced_count, missing_count)
                VALUES (:d, :v, :p, 0)
                ON CONFLICT (snapshot_date) DO UPDATE SET
                    market_value = EXCLUDED.market_value,
                    priced_count = EXCLUDED.priced_count,
                    missing_count = 0
            """), {
                "d": datetime.now(ET).date(),
                "v": total,
                "p": len(frame)
            })
    return frame, missing, total


def update_due_models():
    if not market_ready():
        return []
    _, snapshots = read_tables()
    today = datetime.now(ET).date()
    completed = set()
    if not snapshots.empty:
        dates = pd.to_datetime(snapshots["run_date"]).dt.date
        completed = set(snapshots.loc[dates == today, "strategy"])
    results = []
    for name in lab.MODELS:
        if name not in completed:
            try:
                results.append(lab.run_strategy(engine, name))
            except Exception as exc:
                results.append({"strategy": name, "error": type(exc).__name__})
    return results


page = st.sidebar.radio("Navigation", [
    "Strategy Lab", "Research", "Original AI Strategy", "Daily Decisions",
    "Trade Log", "My Portfolio", "Methodology"
])

if st.sidebar.button("Refresh data"):
    st.rerun()

# Opening the dashboard after the market closes runs missing strategy dates.
# It does not recreate trades for past dates on which no run occurred.
if "checked_today" not in st.session_state or st.session_state.checked_today != datetime.now(ET).date():
    if market_ready():
        with st.spinner("Checking today's strategy updates..."):
            auto_results = update_due_models()
        st.session_state.checked_today = datetime.now(ET).date()
        if auto_results:
            st.sidebar.info("Checked today's model updates. See Strategy Lab for results.")

if page == "Strategy Lab":
    st.header("Strategy Lab — All Portfolios")
    st.write("Compare simulated portfolios and your actual holdings on one interactive chart. Click a legend item to hide or show it.")
    st.caption("Models are valued at their saved run dates. Your real holdings use recorded daily valuations and are not traded by the app.")

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Run missing strategies for today", type="primary"):
            if not market_ready():
                st.warning("Automatic paper-trade runs are limited to weekdays after 4:20 p.m. Eastern. This avoids using incomplete trading-day prices.")
            else:
                result = update_due_models()
                if result:
                    st.dataframe(pd.DataFrame(result), hide_index=True)
                else:
                    st.success("All strategies already have today's snapshot.")
    with c2:
        st.info("To run while the app is closed, install the GitHub Actions workflow provided below. The page alone cannot run while Streamlit is asleep.")

    trades, snapshots = read_tables()
    my_history = personal_snapshots()
    all_names = list(lab.MODELS) + ["My Portfolio"]
    selected = st.multiselect("Show portfolios", all_names, default=all_names)
    view = st.radio("Comparison metric", ["Growth of $10,000 (normalized)", "Actual portfolio value ($)"], horizontal=True)
    fig = go.Figure()
    for name in selected:
        if name == "My Portfolio":
            if my_history.empty:
                continue
            rows = my_history.rename(columns={"snapshot_date": "date", "market_value": "value"}).sort_values("date")
        else:
            if snapshots.empty:
                continue
            rows = snapshots[snapshots["strategy"] == name].rename(columns={"run_date": "date", "equity": "value"}).sort_values("date")
        if rows.empty:
            continue
        values = rows["value"].astype(float)
        if view.startswith("Growth"):
            # Simulated portfolios start with $10k. Real portfolio starts at its first recorded valuation.
            baseline = 10000.0 if name != "My Portfolio" else float(values.iloc[0])
            if baseline <= 0:
                continue
            values = values / baseline * 10000
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(rows["date"]), y=values, mode="lines+markers", name=name,
            line=dict(width=4 if name == "My Portfolio" else 2, dash="dash" if name == "My Portfolio" else "solid"),
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra>%{fullData.name}</extra>"
        ))
    fig.update_layout(
        height=570, hovermode="x unified", xaxis_title="Date",
        yaxis_title="Normalized value ($)" if view.startswith("Growth") else "Portfolio value ($)",
        legend=dict(orientation="h", y=-0.25), margin=dict(b=120)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A model with only one recorded date appears as a point. Missing days are not invented. Your personal line is only complete if all holdings could be priced; dividends, deposits and withdrawals are not adjusted.")

    st.subheader("Model leaderboard")
    if not snapshots.empty:
        latest = snapshots.sort_values("run_date").groupby("strategy").tail(1).copy()
        latest["return_pct"] = (latest["equity"] / lab.STARTING_CASH - 1) * 100
        st.dataframe(latest[["strategy", "run_date", "equity", "cash", "return_pct"]].sort_values("return_pct", ascending=False), hide_index=True, use_container_width=True)
    else:
        st.info("No strategy snapshots yet.")

    st.subheader("Individual strategy explorer")
    chosen = st.selectbox("Choose a model", list(lab.MODELS))
    cfg = lab.MODELS[chosen]
    st.write("**Goal:**", cfg["description"])
    st.write("**How it chooses assets:**", cfg["method"])
    st.write("**Universe:**", ", ".join(cfg["tickers"]))
    model_rows = snapshots[snapshots["strategy"] == chosen].sort_values("run_date") if not snapshots.empty else pd.DataFrame()
    if not model_rows.empty:
        current = model_rows.iloc[-1]
        a, b, c = st.columns(3)
        a.metric("Portfolio value", f"${current['equity']:,.2f}")
        b.metric("Cash", f"${current['cash']:,.2f}")
        c.metric("Total return", f"{(current['equity']/10000-1)*100:+.2f}%")
        positions = json.loads(current["positions_json"])
        if positions:
            st.subheader("Current positions")
            st.dataframe(pd.DataFrame([{"Ticker": t, "Shares": q} for t, q in positions.items()]), hide_index=True, use_container_width=True)
            st.caption("Position selection is based on the model's published rules above, not a live institutional signal.")
    model_trades = trades[trades["strategy"] == chosen].sort_values("executed_at", ascending=False) if not trades.empty else pd.DataFrame()
    st.subheader("Recent transactions and reasons")
    if model_trades.empty:
        st.info("No transactions recorded yet.")
    else:
        st.dataframe(model_trades, hide_index=True, use_container_width=True)
        for _, t in model_trades.head(6).iterrows():
            with st.expander(f"{t['trade_date']} · {t['side']} {t['ticker']}"):
                st.write(t["reason"])
                st.write(f"{t['shares']:.5f} shares at ${t['price']:.2f}; simulated fee ${t['fee']:.2f}.")
    st.subheader("All model transactions")
    if not trades.empty:
        st.dataframe(trades.sort_values("executed_at", ascending=False), hide_index=True, use_container_width=True)
        st.download_button("Export all trades", trades.to_csv(index=False), "all_model_trades.csv", "text/csv")
    else:
        st.info("No trades yet.")

elif page == "Research":
    st.header("Compare stocks, ETFs and your original AI strategy")
    if "research_list" not in st.session_state:
        st.session_state.research_list = ["NVDA", "AAPL", "MSFT", "VOO", "QQQ"]
    new_ticker = st.text_input("Add Yahoo Finance ticker (for example PLTR or BTC-USD)")
    if st.button("Add ticker"):
        ticker = new_ticker.strip().upper()
        if ticker and ticker not in st.session_state.research_list:
            st.session_state.research_list.append(ticker)
            st.rerun()
    choices = list(dict.fromkeys(st.session_state.research_list + list(WATCH)))
    selected = st.multiselect("Displayed assets", choices, default=st.session_state.research_list)
    st.session_state.research_list = selected
    period = st.selectbox("Lookback", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    show_ai = st.checkbox("Show original AI strategy", True)
    fig = go.Figure()
    if selected:
        try:
            history = lab.fetch_history(selected, period)
            for ticker in selected:
                if ticker in history.columns:
                    series = history[ticker].dropna()
                    if not series.empty:
                        fig.add_trace(go.Scatter(x=series.index, y=series / series.iloc[0] * 10000, name=ticker, mode="lines"))
        except Exception as exc:
            st.warning(f"Market data unavailable ({type(exc).__name__}).")
    if show_ai:
        original = read(engine, "snapshots")
        if not original.empty:
            original = original.sort_values("run_date")
            fig.add_trace(go.Scatter(x=original["run_date"], y=original["equity"], name="Original AI Strategy", mode="lines+markers", line=dict(width=4, dash="dash")))
    fig.update_layout(height=540, legend=dict(orientation="h", y=-0.3), margin=dict(b=100), hovermode="x unified", yaxis_title="Growth of $10,000", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("What this graph means", expanded=True):
        st.write("Each asset begins at $10,000 for visual comparison; these are price-only returns. The AI line uses saved simulated portfolio values. Legend clicks hide or show individual lines. The AI line may begin later than the stock histories.")

elif page == "Original AI Strategy":
    st.header("Original AI Strategy")
    original = read(engine, "snapshots")
    if original.empty:
        st.info("No original strategy snapshots yet.")
    else:
        st.line_chart(original.sort_values("run_date").set_index("run_date")[["equity", "benchmark_voo", "benchmark_qqq"]])
        st.dataframe(original, hide_index=True, use_container_width=True)

elif page == "Daily Decisions":
    st.header("Original strategy decisions")
    try:
        st.dataframe(score_frame(prices(WATCH)), hide_index=True, use_container_width=True)
        if st.button("Run original strategy"):
            st.write(run(engine))
    except Exception as exc:
        st.error(f"Could not calculate signals: {type(exc).__name__}")

elif page == "Trade Log":
    st.header("Original strategy trade history")
    original_trades = read(engine, "trades")
    st.dataframe(original_trades, hide_index=True, use_container_width=True)

elif page == "My Portfolio":
    st.header("My actual holdings")
    st.write("Upload ticker,shares and optional cost_basis. The holdings are stored in your private Supabase database so you don't have to upload them each time. This does not connect to a broker or detect your real trades.")
    uploaded = st.file_uploader("Holdings CSV", type="csv")
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            frame.columns = frame.columns.str.strip().str.lower()
            if not {"ticker", "shares"}.issubset(frame.columns):
                st.error("CSV requires ticker and shares columns.")
            else:
                frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
                frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce")
                if "cost_basis" not in frame.columns:
                    frame["cost_basis"] = None
                frame["cost_basis"] = pd.to_numeric(frame["cost_basis"], errors="coerce")
                frame = frame.dropna(subset=["ticker", "shares"])
                frame = frame[(frame["ticker"] != "") & (frame["shares"] >= 0)]
                frame = frame.groupby("ticker", as_index=False).agg({"shares": "sum", "cost_basis": "mean"})
                if len(frame) == 0 or len(frame) > 100:
                    st.error("Upload between 1 and 100 valid holdings.")
                elif st.button("Save / replace my holdings", type="primary"):
                    save_holdings(frame)
                    st.success("Holdings saved in Supabase.")
                    st.rerun()
                st.dataframe(frame, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"Invalid CSV: {type(exc).__name__}")
    current = personal_holdings()
    if current.empty:
        st.info("Upload your holdings CSV to add My Portfolio to the Strategy Lab chart.")
    else:
        if st.button("Update my portfolio valuation"):
            st.rerun()
        try:
            valued, missing, total = value_personal(record=True)
            st.metric("Priced market value", f"${total:,.2f}")
            st.dataframe(valued, hide_index=True, use_container_width=True)
            if missing:
                st.warning("Not priced: " + ", ".join(missing) + ". Total is incomplete and no daily snapshot was saved.")
            else:
                st.plotly_chart(px.pie(valued, names="ticker", values="market_value", title="Holdings allocation"), use_container_width=True)
            st.subheader("Saved daily valuations")
            st.dataframe(personal_snapshots(), hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"Valuation failed: {type(exc).__name__}")

else:
    st.header("How the strategies work")
    st.markdown("""
**AI Goldman Sachs:** Quality and lower-volatility proxy; ranks positive-trend securities using momentum minus a volatility penalty. Not Goldman Sachs research or actual trades.

**AI Pelosi Disclosure-Inspired:** Fixed technology-heavy basket. It does **not** ingest congressional disclosures, and it does not mirror Nancy Pelosi's trades.

**AI JPMorgan Chase:** Diversified ETF basket of equities, bonds and gold. Not JPMorgan's proprietary allocation.

**AI Momentum:** Chooses the strongest recent performers that pass a moving-average trend filter.

**AI Defensive Income:** Equal-weight defensive dividend, bond and gold ETFs.

**AI Equal-Weight Growth:** Equal-weight basket of large growth companies.

**Momentum** measures past price changes. **Moving averages** smooth recent prices to help assess trend. **Volatility** measures variation in returns. **Rebalancing** adjusts positions toward target weights.

The simulated models begin at $10,000, generally target 90% invested and 10% cash, and charge 0.05% per simulated transaction. Rebalances happen on the first run and Mondays. Other weekday runs update valuations only. The model uses observed prices and does not model slippage, taxes, dividends or real brokerage fills.

**Important:** Market holidays, historical missing runs, and broker cash flows are not automatically reconstructed. GitHub Actions scheduling requires its own workflow and a valid DATABASE_URL secret. The original strategy is separate and is not automatically run by the new scheduler.
""")
