
import os
import hmac
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf
from sqlalchemy import text

import strategy_lab as lab
from engine import (
    db, init, read, run, prices,
    score_frame, holdings, WATCH, START_CASH
)

st.set_page_config(
    page_title="AI Investment Research Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Investment Research Lab")
st.caption(
    "Market research • Simulated investment strategies • "
    "Personal portfolio • Educational use only"
)

# ============================================================
# LOGIN
# ============================================================

try:
    password_secret = str(st.secrets["APP_PASSWORD"])
    database_url = str(st.secrets["DATABASE_URL"])
except Exception:
    st.error(
        "APP_PASSWORD and DATABASE_URL must be "
        "configured in Streamlit Secrets."
    )
    st.stop()

entered = st.text_input("App password", type="password")

if not hmac.compare_digest(entered, password_secret):
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
    st.error(
        "Database initialization failed: "
        + type(exc).__name__
    )
    st.stop()

ET = ZoneInfo("America/New_York")

# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_history(ticker, period="1y"):
    """
    Download a single ticker independently.

    This prevents the selected chart components
    from affecting the price history of other stocks.
    """
    ticker = str(ticker).upper().strip()

    raw = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        series = raw["Close"][ticker]
    else:
        series = raw["Close"]

    series = pd.to_numeric(
        series, errors="coerce"
    ).dropna()

    series = series[series > 0]

    idx = pd.to_datetime(series.index, utc=True)
    series.index = idx.tz_localize(None).normalize()

    series = series[
        ~series.index.duplicated(keep="last")
    ]

    series.name = ticker

    return series.sort_index()


def normalized(series):
    series = series.dropna()

    if len(series) < 2:
        return pd.Series(dtype=float)

    return series / float(series.iloc[0]) * 10000


def add_line(
    fig, series, name,
    color=None, width=2, dash="solid"
):
    if series.empty:
        return

    line = {"width": width, "dash": dash}

    if color:
        line["color"] = color

    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=name,
        line=line,
        connectgaps=False,
        hovertemplate=(
            "%{x|%b %d, %Y}<br>"
            "$%{y:,.2f}"
            "<extra>%{fullData.name}</extra>"
        )
    ))


def format_chart(fig, title):
    fig.update_layout(
        title=title,
        height=600,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Growth of $10,000",
        legend=dict(
            orientation="h",
            y=-0.24,
            x=0
        ),
        margin=dict(t=70, b=140)
    )
    fig.update_yaxes(
        tickprefix="$",
        tickformat=",.0f"
    )


# ============================================================
# PERSONAL HOLDINGS
# ============================================================

def load_holdings():
    return pd.read_sql_query(
        text("""
            SELECT ticker, shares, cost_basis
            FROM personal_holdings
            ORDER BY ticker
        """),
        engine
    )


def save_holdings(frame):
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM personal_holdings")
        )

        for row in frame.itertuples():
            conn.execute(text("""
                INSERT INTO personal_holdings
                (ticker, shares, cost_basis)
                VALUES (:ticker, :shares, :cost)
            """), {
                "ticker": row.ticker,
                "shares": float(row.shares),
                "cost": (
                    None
                    if pd.isna(row.cost_basis)
                    else float(row.cost_basis)
                )
            })


def portfolio_weights():
    frame = load_holdings()

    if frame.empty:
        return {}, 0, ["No saved holdings"]

    weights = {}
    missing = []

    for row in frame.itertuples():
        ticker = str(row.ticker)
        history = get_history(ticker, "1mo")

        if history.empty:
            missing.append(ticker)
            continue

        price = float(history.iloc[-1])
        value = float(row.shares) * price

        if value > 0:
            weights[ticker] = value

    total = sum(weights.values())

    if total <= 0:
        return {}, 0, missing

    weights = {
        ticker: value / total
        for ticker, value in weights.items()
    }

    return weights, total, missing


# ============================================================
# NEW: FULL-PERIOD PERSONAL PORTFOLIO SIMULATION
# ============================================================

def simulate_portfolio_full_period(weights, period):
    """
    Historical simulation starting at the beginning
    of the selected market-data period.

    Each holding receives its current portfolio weight
    multiplied by $10,000.

    If a holding has no price at the starting date,
    its allocation stays in cash until the first
    available valid trading price.

    Once purchased, shares are held for the remainder
    of the simulation.

    This is a hypothetical cash-until-available
    methodology, not actual historical transactions.
    """

    if not weights:
        return pd.Series(dtype=float), []

    histories = {}
    unavailable = []

    for ticker in weights:
        series = get_history(ticker, period)

        if series.empty:
            unavailable.append(ticker)
        else:
            histories[ticker] = series

    if not histories:
        return pd.Series(dtype=float), unavailable

    # Use the full union of available trading dates.
    # Do not require every asset to have prices
    # on the first date.
    all_dates = sorted(set().union(
        *(series.index for series in histories.values())
    ))

    index = pd.DatetimeIndex(all_dates)

    if len(index) < 2:
        return pd.Series(dtype=float), unavailable

    # Each position starts with a fixed dollar budget.
    # That budget remains in cash until the security
    # has its first valid historical observation.
    cash_allocations = {
        ticker: 10000.0 * weight
        for ticker, weight in weights.items()
    }

    # Unavailable holdings stay in cash throughout.
    for ticker in unavailable:
        cash_allocations[ticker] = (
            10000.0 * weights[ticker]
        )

    shares = {
        ticker: 0.0
        for ticker in weights
    }

    purchased = {
        ticker: False
        for ticker in weights
    }

    aligned = {}

    for ticker, series in histories.items():
        aligned[ticker] = series.reindex(index).ffill()

    values = []

    for date in index:

        # Buy each holding on its first valid date.
        for ticker in histories:

            price = aligned[ticker].loc[date]

            if (
                not purchased[ticker]
                and pd.notna(price)
                and float(price) > 0
            ):
                shares[ticker] = (
                    cash_allocations[ticker]
                    / float(price)
                )

                cash_allocations[ticker] = 0.0
                purchased[ticker] = True

        total_cash = sum(cash_allocations.values())
        invested_value = 0.0

        for ticker in histories:

            price = aligned[ticker].loc[date]

            if purchased[ticker] and pd.notna(price):
                invested_value += (
                    shares[ticker] * float(price)
                )

        values.append(total_cash + invested_value)

    result = pd.Series(
        values,
        index=index,
        name="My Portfolio ($10K)",
        dtype=float
    )

    return result, unavailable


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Pages",
    [
        "Dashboard",
        "Research",
        "Strategy Lab",
        "Daily Decisions",
        "Trade Log",
        "My Portfolio",
        "Data & Methodology"
    ]
)

if st.sidebar.button("Refresh market data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    "Market prices are recent available daily closes. "
    "Simulated trades require a completed model run."
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Investment Dashboard")

    snapshots = read(engine, "snapshots")

    if snapshots.empty:
        st.info(
            "No original AI strategy snapshots yet."
        )
    else:
        snapshots = snapshots.sort_values("run_date")
        latest = snapshots.iloc[-1]

        a, b, c = st.columns(3)

        a.metric(
            "AI Strategy",
            f"${float(latest['equity']):,.2f}"
        )

        b.metric(
            "VOO Benchmark",
            f"${float(latest['benchmark_voo']):,.2f}"
        )

        c.metric(
            "QQQ Benchmark",
            f"${float(latest['benchmark_qqq']):,.2f}"
        )

        chart = snapshots.set_index("run_date")[
            ["equity", "benchmark_voo", "benchmark_qqq"]
        ]

        st.line_chart(chart)

    st.info(
        "The original AI strategy and the additional "
        "Strategy Lab models are separate simulated "
        "investment experiments."
    )


# ============================================================
# RESEARCH
# ============================================================

elif page == "Research":

    st.header("🔬 Investment Research")

    st.write(
        "Compare stocks, ETFs, your personal portfolio, "
        "and the original AI strategy using historical data."
    )

    if "research_symbols" not in st.session_state:
        st.session_state.research_symbols = [
            "NVDA", "AAPL", "MSFT", "VOO", "QQQ"
        ]

    new_ticker = st.text_input(
        "Add a stock, ETF or cryptocurrency",
        placeholder="PLTR, TSLA, BTC-USD"
    )

    if st.button("Add ticker"):
        ticker = new_ticker.strip().upper()

        if (
            ticker
            and ticker not in st.session_state.research_symbols
        ):
            st.session_state.research_symbols.append(ticker)
            st.rerun()

    available = list(dict.fromkeys(
        st.session_state.research_symbols
        + list(WATCH)
    ))

    selected = st.multiselect(
        "Stocks and ETFs to compare",
        options=available,
        default=st.session_state.research_symbols
    )

    st.session_state.research_symbols = selected

    period = st.selectbox(
        "Historical period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

    a, b = st.columns(2)

    with a:
        show_personal = st.checkbox(
            "My Portfolio ($10K)",
            value=True
        )

    with b:
        show_ai = st.checkbox(
            "Original AI Strategy",
            value=True
        )

    fig = go.Figure()
    results = []

    # ------------------------------------------
    # INDIVIDUAL STOCKS
    # ------------------------------------------

    for ticker in selected:

        try:
            history = get_history(ticker, period)
            series = normalized(history)

            if series.empty:
                st.warning(
                    f"No usable historical data for {ticker}."
                )
                continue

            add_line(
                fig,
                series,
                ticker,
                width=2
            )

            results.append({
                "Investment": ticker,
                "Start Date": series.index[0].date(),
                "End Date": series.index[-1].date(),
                "Ending Value ($)": float(series.iloc[-1]),
                "Return (%)": (
                    float(series.iloc[-1]) / 10000 - 1
                ) * 100
            })

        except Exception as exc:
            st.warning(
                f"{ticker}: {type(exc).__name__}"
            )

    # ------------------------------------------
    # PERSONAL PORTFOLIO — FULL PERIOD
    # ------------------------------------------

    if show_personal:

        try:
            weights, actual_value, missing = (
                portfolio_weights()
            )

            if not weights:
                st.warning(
                    "No priced portfolio holdings available. "
                    "Upload your holdings on My Portfolio."
                )

            else:
                simulation, historical_missing = (
                    simulate_portfolio_full_period(
                        weights, period
                    )
                )

                if not simulation.empty:

                    add_line(
                        fig,
                        simulation,
                        "My Portfolio ($10K)",
                        color="#F59E0B",
                        width=5
                    )

                    results.append({
                        "Investment": "My Portfolio ($10K)",
                        "Start Date": simulation.index[0].date(),
                        "End Date": simulation.index[-1].date(),
                        "Ending Value ($)": float(
                            simulation.iloc[-1]
                        ),
                        "Return (%)": (
                            float(simulation.iloc[-1])
                            / 10000 - 1
                        ) * 100
                    })

                if missing:
                    st.warning(
                        "These saved holdings could not be "
                        "priced and are excluded from the "
                        "allocation: "
                        + ", ".join(missing)
                    )

                if historical_missing:
                    st.info(
                        "These holdings remained in cash "
                        "because historical prices were "
                        "unavailable: "
                        + ", ".join(historical_missing)
                    )

        except Exception as exc:
            st.error(
                "Portfolio simulation error: "
                + str(exc)[:250]
            )

    # ------------------------------------------
    # ORIGINAL AI STRATEGY
    # ------------------------------------------

    if show_ai:

        try:
            snapshots = read(engine, "snapshots")

            if not snapshots.empty:

                snapshots = snapshots.sort_values(
                    "run_date"
                )

                series = pd.Series(
                    snapshots["equity"].astype(float).values,
                    index=pd.to_datetime(
                        snapshots["run_date"]
                    )
                )

                add_line(
                    fig,
                    series,
                    "Original AI Strategy",
                    color="#A855F7",
                    width=4,
                    dash="dash"
                )

        except Exception:
            st.warning(
                "Original AI strategy snapshots unavailable."
            )

    format_chart(
        fig,
        "Growth of $10,000 — Research Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Each individual stock is independently normalized "
        "to $10,000. The orange line uses current personal "
        "portfolio weights and holds allocations in cash "
        "until their historical securities become available. "
        "The purple line shows recorded original strategy "
        "equity, not a reconstructed backtest."
    )

    # ------------------------------------------
    # RESULTS
    # ------------------------------------------

    st.subheader("Historical Performance")

    if results:

        table = pd.DataFrame(results).sort_values(
            "Return (%)",
            ascending=False
        )

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Compare starting and ending dates before "
            "ranking returns. Securities may have "
            "different historical coverage."
        )

    # ------------------------------------------
    # PERSONAL ALLOCATION
    # ------------------------------------------

    if show_personal:

        try:
            weights, actual_value, missing = (
                portfolio_weights()
            )

            if weights:

                st.subheader(
                    "My Portfolio — $10,000 Allocation"
                )

                allocation = pd.DataFrame([
                    {
                        "Ticker": ticker,
                        "Weight (%)": weight * 100,
                        "Starting Allocation ($)": weight * 10000
                    }
                    for ticker, weight in weights.items()
                ])

                st.dataframe(
                    allocation.sort_values(
                        "Weight (%)",
                        ascending=False
                    ),
                    hide_index=True,
                    use_container_width=True
                )

                st.metric(
                    "Current Priced Portfolio Value",
                    f"${actual_value:,.2f}"
                )

        except Exception:
            pass

    with st.expander(
        "📚 How the historical simulation works",
        expanded=True
    ):

        st.markdown("""
### Why your orange line now goes back farther

Previously, your portfolio waited until every
holding had historical price data.

The new method begins at the earliest available
date in the requested period.

### How $10,000 is invested

Your current portfolio allocation determines
the initial investment assigned to each holding.

For example:

- 50% NVDA = $5,000
- 30% VOO = $3,000
- 20% PLTR = $2,000

### What happens to newer stocks?

If an investment does not have historical prices
at the beginning of the period, its allocation
remains in cash.

When its first historical price becomes available,
the simulation purchases hypothetical shares.

Those shares are held afterward.

### Why this matters

The orange portfolio line can now extend back
to the selected historical period without
fabricating stock prices.

### Important assumptions

- Current portfolio weights are used retroactively.
- Missing historical prices are not invented.
- Unavailable investments are initially held as cash.
- There are no simulated taxes or trading fees.
- No rebalancing occurs after the initial purchases.
- The model does not recreate your actual trades.
- Cash is assumed to earn zero interest.
- Historical results do not predict future returns.
""")


# ============================================================
# STRATEGY LAB
# ============================================================

elif page == "Strategy Lab":

    st.header("🏦 Multi-Strategy Investment Lab")

    st.caption(
        "Goldman Sachs, JPMorgan Chase and Pelosi-inspired "
        "strategies are independent approximations, "
        "not actual institutional or congressional portfolios."
    )

    if st.button(
        "Run All Paper Strategies",
        type="primary"
    ):
        try:
            messages = lab.run_daily(engine)

            st.success(
                "Paper strategy update completed."
            )

            for message in messages:
                st.write(message)

        except Exception as exc:
            st.error(
                "Strategy run failed: "
                + str(exc)[:200]
            )

    @st.fragment(run_every="60s")
    def strategy_dashboard():

        snapshots = lab.data(
            engine, "lab_snapshots"
        )

        trades = lab.data(
            engine, "lab_trades"
        )

        positions = lab.data(
            engine, "lab_positions"
        )

        accounts = lab.data(
            engine, "lab_models"
        )

        st.subheader("Strategy Performance Comparison")

        if not snapshots.empty:

            comparison = snapshots.copy()

            comparison["day"] = pd.to_datetime(
                comparison["day"]
            )

            comparison = comparison.sort_values("day")

            comparison["Normalized $10K"] = (
                comparison.groupby("model")["equity"]
                .transform(
                    lambda s: (
                        s / s.iloc[0] * 10000
                        if s.iloc[0] > 0
                        else np.nan
                    )
                )
            )

            chart = px.line(
                comparison,
                x="day",
                y="Normalized $10K",
                color="model",
                markers=True,
                title="AI Investment Strategies — Growth of $10,000"
            )

            chart.update_layout(
                height=550,
                hovermode="x unified"
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

            latest = (
                snapshots.sort_values("day")
                .drop_duplicates("model", keep="last")
                .copy()
            )

            st.subheader("Strategy Leaderboard")

            latest["Return (%)"] = (
                latest["equity"] / lab.START - 1
            ) * 100

            st.dataframe(
                latest[
                    ["model", "day", "equity", "cash", "Return (%)"]
                ],
                hide_index=True,
                use_container_width=True
            )

        else:
            st.info(
                "No saved strategy snapshots yet."
            )

        st.subheader("Daily Transactions")

        if not trades.empty:

            st.dataframe(
                trades.sort_values(
                    "executed_at",
                    ascending=False
                ),
                hide_index=True,
                use_container_width=True
            )

            st.download_button(
                "Download Transactions",
                trades.to_csv(index=False),
                "strategy_trades.csv",
                "text/csv"
            )

        else:
            st.info(
                "No simulated transactions yet."
            )

        st.subheader("Individual Strategy Explorer")

        chosen = st.selectbox(
            "Select investment strategy",
            list(lab.MODELS)
        )

        st.write(
            "**Strategy objective:**",
            lab.MODELS[chosen]
        )

        if not positions.empty:

            selected_positions = positions[
                positions["model"] == chosen
            ]

            st.write("**Current Positions**")

            st.dataframe(
                selected_positions,
                hide_index=True,
                use_container_width=True
            )

        if not trades.empty:

            selected_trades = trades[
                trades["model"] == chosen
            ].sort_values(
                "executed_at",
                ascending=False
            )

            st.write("**Recent Trades and Reasoning**")

            st.dataframe(
                selected_trades,
                hide_index=True,
                use_container_width=True
            )

            for row in selected_trades.head(5).itertuples():

                with st.expander(
                    f"{row.side} {row.ticker}"
                ):

                    st.write(
                        "**Recorded reason:**",
                        row.reason
                    )

                    st.write(
                        f"Shares: {row.shares:.5f}"
                    )

                    st.write(
                        f"Price: ${row.price:,.2f}"
                    )

        st.caption(
            "This page refreshes its database tables "
            "approximately every 60 seconds while open. "
            "New trades require a completed strategy run."
        )

    strategy_dashboard()


# ============================================================
# DAILY DECISIONS
# ============================================================

elif page == "Daily Decisions":

    st.header("📅 Original AI Strategy Decisions")

    try:
        signals = score_frame(prices(WATCH))

        st.dataframe(
            signals,
            hide_index=True,
            use_container_width=True
        )

        if st.button("Run Original AI Strategy"):

            result = run(engine)

            st.success(
                "Original AI Strategy updated."
            )

            st.write(result)

    except Exception as exc:

        st.error(
            "Unable to calculate daily decisions: "
            + type(exc).__name__
        )


# ============================================================
# TRADE LOG
# ============================================================

elif page == "Trade Log":

    st.header("📋 Original AI Trade Log")

    trades = read(engine, "trades")

    if trades.empty:
        st.info("No recorded trades.")
    else:
        st.dataframe(
            trades,
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download Original Trades",
            trades.to_csv(index=False),
            "original_trades.csv",
            "text/csv"
        )


# ============================================================
# MY PORTFOLIO
# ============================================================

elif page == "My Portfolio":

    st.header("💼 My Actual Portfolio")

    st.write(
        "Upload a CSV containing ticker and shares. "
        "An optional cost_basis column is supported."
    )

    uploaded = st.file_uploader(
        "Upload holdings CSV",
        type="csv"
    )

    if uploaded is not None:

        try:
            frame = pd.read_csv(uploaded)

            frame.columns = (
                frame.columns.str.strip().str.lower()
            )

            if not {"ticker", "shares"}.issubset(frame.columns):
                st.error(
                    "CSV must contain ticker and shares."
                )
            else:

                frame["ticker"] = (
                    frame["ticker"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )

                frame["shares"] = pd.to_numeric(
                    frame["shares"],
                    errors="coerce"
                )

                if "cost_basis" not in frame.columns:
                    frame["cost_basis"] = np.nan

                frame["cost_basis"] = pd.to_numeric(
                    frame["cost_basis"],
                    errors="coerce"
                )

                frame = frame.dropna(
                    subset=["ticker", "shares"]
                )

                frame = frame[
                    (frame["ticker"] != "")
                    & (frame["ticker"] != "NAN")
                    & (frame["shares"] >= 0)
                ]

                frame = frame.groupby(
                    "ticker",
                    as_index=False
                ).agg({
                    "shares": "sum",
                    "cost_basis": "mean"
                })

                st.dataframe(
                    frame,
                    hide_index=True,
                    use_container_width=True
                )

                if st.button(
                    "Save / Replace My Holdings",
                    type="primary"
                ):

                    if not 1 <= len(frame) <= 100:
                        st.error(
                            "Provide 1–100 valid holdings."
                        )
                    else:
                        save_holdings(frame)

                        st.success(
                            "Holdings saved in Supabase."
                        )

                        st.rerun()

        except Exception as exc:
            st.error(
                "CSV processing error: "
                + str(exc)[:200]
            )

    frame = load_holdings()

    if frame.empty:
        st.info("No saved holdings.")
    else:

        weights, total, missing = portfolio_weights()

        st.metric(
            "Priced Portfolio Value",
            f"${total:,.2f}"
        )

        st.dataframe(
            frame,
            hide_index=True,
            use_container_width=True
        )

        if missing:
            st.warning(
                "Unpriced holdings: "
                + ", ".join(missing)
            )

        if weights:

            allocation = pd.DataFrame([
                {
                    "Ticker": ticker,
                    "Market Value": weight * total
                }
                for ticker, weight in weights.items()
            ])

            fig = px.pie(
                allocation,
                names="Ticker",
                values="Market Value",
                hole=0.4,
                title="Current Portfolio Allocation"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# METHODOLOGY
# ============================================================

else:

    st.header("📚 Methodology")

    for name, description in lab.MODELS.items():
        st.markdown(
            f"### {name}\n{description}"
        )

    st.markdown("""
### Market Data

Historical adjusted prices are retrieved from
Yahoo Finance.

### Investment Signals

The original model uses price trends, moving
averages, momentum, and volatility.

### Simulated Transactions

All paper trades are simulated.

No real brokerage orders are placed.

### Historical Portfolio Comparison

Your personal portfolio is simulated using
current allocation weights applied retroactively.

If a security lacks historical prices at the
start of the selected period, its allocation
remains in cash until that security becomes
available.

### Automatic Updates

GitHub Actions can run the paper strategies
on a schedule when configured correctly.

The Streamlit interface itself does not run
scheduled jobs while the application is asleep.

### Important Limitations

- Models are rules-based rather than autonomous AI.
- Institutional-inspired strategies are approximations.
- Historical results do not guarantee future returns.
- Prices may be delayed or revised.
- No actual historical personal trades are reconstructed.
- Trading fees and taxes are simplified or excluded.
- Cash is assumed to earn no interest in the backtest.
""")
