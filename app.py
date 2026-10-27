
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
from engine import db, init, read, run, prices, score_frame, WATCH, START_CASH


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Investment Research Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Investment Research Lab")
st.caption(
    "Historical investment research • Six paper-trading models • "
    "Personal portfolio • No real brokerage trading"
)


# ============================================================
# AUTHENTICATION
# ============================================================

try:
    APP_PASSWORD = str(st.secrets["APP_PASSWORD"])
    DATABASE_URL = str(st.secrets["DATABASE_URL"])
except Exception:
    st.error(
        "APP_PASSWORD and DATABASE_URL must be configured "
        "in Streamlit Secrets."
    )
    st.stop()

entered_password = st.text_input(
    "App password",
    type="password"
)

if not hmac.compare_digest(entered_password, APP_PASSWORD):
    st.stop()

os.environ["DATABASE_URL"] = DATABASE_URL

ET = ZoneInfo("America/New_York")


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:
    engine = db()
    init(engine)

    # Correct function used by your existing strategy_lab.py
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
        f"Database initialization failed: "
        f"{type(exc).__name__}: {exc}"
    )
    st.stop()


# ============================================================
# GENERAL HELPERS
# ============================================================

def eastern_now():
    return datetime.now(ET)


def market_ready():
    now = eastern_now()
    return (
        now.weekday() < 5
        and (now.hour, now.minute) >= (16, 20)
    )


def safe_read(table):
    try:
        return read(engine, table)
    except Exception:
        return pd.DataFrame()


def load_lab_table(table):
    try:
        return lab.load(engine, table)
    except Exception as exc:
        st.warning(
            f"Could not load {table}: "
            f"{type(exc).__name__}: {exc}"
        )
        return pd.DataFrame()


def date_column(frame, candidates):
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def normalize_to_10000(series):
    series = pd.to_numeric(
        series, errors="coerce"
    ).dropna()

    series = series[series > 0]

    if series.empty:
        return pd.Series(dtype=float)

    return series / float(series.iloc[0]) * 10000


def add_line(
    fig,
    series,
    name,
    color=None,
    width=2,
    dash="solid"
):
    if series.empty:
        return

    line = {
        "width": width,
        "dash": dash
    }

    if color:
        line["color"] = color

    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        name=name,
        mode="lines",
        line=line,
        connectgaps=False,
        hovertemplate=(
            "%{x|%b %d, %Y}<br>"
            "$%{y:,.2f}"
            "<extra>%{fullData.name}</extra>"
        )
    ))


def format_chart(fig, title, ylabel="Growth of $10,000"):
    fig.update_layout(
        title=title,
        height=600,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=ylabel,
        legend=dict(
            orientation="h",
            y=-0.25,
            x=0
        ),
        margin=dict(
            t=65,
            b=140
        )
    )

    fig.update_yaxes(
        tickprefix="$",
        tickformat=",.0f"
    )


# ============================================================
# HISTORICAL MARKET DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_history(ticker, period="1y"):
    """
    Download adjusted daily historical prices for
    one ticker independently.

    This keeps individual stock chart histories
    unchanged when other chart lines are toggled.
    """

    ticker = str(ticker).upper().strip()

    if not ticker:
        return pd.Series(dtype=float)

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data.empty:
        return pd.Series(dtype=float)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]

        if ticker in close.columns:
            series = close[ticker]
        else:
            series = close.iloc[:, 0]
    else:
        series = data["Close"]

    series = pd.to_numeric(
        series, errors="coerce"
    ).dropna()

    series = series[series > 0]

    if series.empty:
        return pd.Series(dtype=float)

    dates = pd.to_datetime(series.index)

    if dates.tz is not None:
        dates = dates.tz_localize(None)

    series.index = dates.normalize()

    series = series[
        ~series.index.duplicated(keep="last")
    ]

    series.name = ticker

    return series.sort_index()


def historical_reference_index(tickers, period):
    """
    Build a stable comparison calendar from the
    selected benchmark stocks.

    If no benchmarks are selected, use VOO.

    The portfolio simulation uses this same calendar
    and begins at its first date, even if some
    holdings have no historical prices yet.
    """

    symbols = list(dict.fromkeys(
        list(tickers) if tickers else ["VOO"]
    ))

    all_dates = set()

    for ticker in symbols:
        history = get_history(ticker, period)
        all_dates.update(history.index)

    if not all_dates:
        return pd.DatetimeIndex([])

    return pd.DatetimeIndex(sorted(all_dates))


# ============================================================
# PERSONAL PORTFOLIO DATABASE
# ============================================================

def load_personal_holdings():
    return pd.read_sql_query(
        text("""
            SELECT ticker, shares, cost_basis
            FROM personal_holdings
            ORDER BY ticker
        """),
        engine
    )


def save_personal_holdings(frame):
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM personal_holdings")
        )

        for row in frame.itertuples(index=False):
            conn.execute(text("""
                INSERT INTO personal_holdings
                    (ticker, shares, cost_basis)
                VALUES
                    (:ticker, :shares, :cost_basis)
            """), {
                "ticker": row.ticker,
                "shares": float(row.shares),
                "cost_basis": (
                    None
                    if pd.isna(row.cost_basis)
                    else float(row.cost_basis)
                )
            })


def load_personal_snapshots():
    return pd.read_sql_query(
        text("""
            SELECT *
            FROM personal_snapshots
            ORDER BY snapshot_date
        """),
        engine
    )


def get_portfolio_weights():
    """
    Calculate current market-value weights.

    If one holding cannot be priced, return an
    incomplete-price warning. Do not silently
    treat it as a zero-value position.
    """

    frame = load_personal_holdings()

    if frame.empty:
        return {}, 0.0, ["No holdings saved"]

    values = {}
    missing = []

    for row in frame.itertuples():
        ticker = str(row.ticker).upper()
        shares = float(row.shares)

        if shares <= 0:
            continue

        history = get_history(ticker, "1mo")

        if history.empty:
            missing.append(ticker)
            continue

        latest_price = float(history.iloc[-1])

        values[ticker] = shares * latest_price

    total = sum(values.values())

    if total <= 0:
        return {}, 0.0, missing

    weights = {
        ticker: value / total
        for ticker, value in values.items()
        if value > 0
    }

    return weights, total, missing


# ============================================================
# FULL-PERIOD PERSONAL PORTFOLIO BACKTEST
# ============================================================

def simulate_portfolio_full_period(
    weights,
    period,
    reference_index
):
    """
    A hypothetical $10,000 buy-and-hold simulation.

    At the first benchmark date:
      - Each stock receives a cash allocation equal
        to its current portfolio weight.
      - If the stock already has a valid price,
        the simulation buys it.
      - Otherwise, its allocation remains in cash
        until its first available trading price.

    No historical stock prices are fabricated.
    No rebalancing, fees, taxes or cash interest.
    """

    if not weights or len(reference_index) < 2:
        return pd.Series(dtype=float), []

    index = pd.DatetimeIndex(
        sorted(reference_index.unique())
    )

    cash = {
        ticker: 10000.0 * float(weight)
        for ticker, weight in weights.items()
    }

    shares = {
        ticker: 0.0
        for ticker in weights
    }

    purchased = {
        ticker: False
        for ticker in weights
    }

    histories = {}
    unavailable = []

    for ticker in weights:
        series = get_history(ticker, period)

        if series.empty:
            unavailable.append(ticker)
            continue

        # Reindex to the common benchmark calendar.
        # Forward-fill only after the stock has
        # actually started trading.
        aligned = series.reindex(
            index.union(series.index)
        ).sort_index().ffill()

        aligned = aligned.reindex(index)

        histories[ticker] = aligned

    portfolio_values = []

    for date in index:

        for ticker, series in histories.items():

            price = series.loc[date]

            if (
                not purchased[ticker]
                and pd.notna(price)
                and float(price) > 0
            ):
                shares[ticker] = (
                    cash[ticker] / float(price)
                )

                cash[ticker] = 0.0
                purchased[ticker] = True

        invested = 0.0

        for ticker, series in histories.items():
            price = series.loc[date]

            if purchased[ticker] and pd.notna(price):
                invested += (
                    shares[ticker] * float(price)
                )

        portfolio_values.append(
            sum(cash.values()) + invested
        )

    result = pd.Series(
        portfolio_values,
        index=index,
        dtype=float,
        name="My Portfolio ($10K)"
    )

    return result, unavailable


# ============================================================
# PERSONAL PORTFOLIO VALUATION
# ============================================================

def value_personal_portfolio():
    frame = load_personal_holdings()

    if frame.empty:
        return frame, 0.0, []

    prices_by_ticker = {}
    missing = []

    for ticker in frame["ticker"]:
        history = get_history(ticker, "1mo")

        if history.empty:
            missing.append(ticker)
        else:
            prices_by_ticker[ticker] = float(
                history.iloc[-1]
            )

    frame["last_price"] = frame["ticker"].map(
        prices_by_ticker
    )

    frame["market_value"] = (
        frame["shares"] * frame["last_price"]
    )

    total = float(frame["market_value"].sum())

    return frame, total, missing


def record_personal_snapshot(value, priced, missing):
    if missing:
        return

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO personal_snapshots
                (
                    snapshot_date,
                    market_value,
                    priced_count,
                    missing_count
                )
            VALUES
                (:day, :value, :priced, 0)
            ON CONFLICT (snapshot_date)
            DO UPDATE SET
                market_value = EXCLUDED.market_value,
                priced_count = EXCLUDED.priced_count,
                missing_count = 0
        """), {
            "day": eastern_now().date(),
            "value": float(value),
            "priced": int(priced)
        })


# ============================================================
# STRATEGY LAB HELPERS
# ============================================================

def lab_tables():
    trades = load_lab_table("lab_trades")
    snapshots = load_lab_table("lab_snapshots")
    return trades, snapshots


def run_missing_strategies():
    """
    Uses the existing run_strategy function rather
    than the nonexistent run_daily function.
    """

    _, snapshots = lab_tables()

    today = eastern_now().date()
    already_done = set()

    if not snapshots.empty:
        name_col = date_column(
            snapshots, ["strategy", "model"]
        )

        day_col = date_column(
            snapshots, ["run_date", "day"]
        )

        if name_col and day_col:
            snapshot_dates = pd.to_datetime(
                snapshots[day_col],
                errors="coerce"
            ).dt.date

            already_done = set(
                snapshots.loc[
                    snapshot_dates == today,
                    name_col
                ].astype(str)
            )

    results = []

    for name in lab.MODELS:

        if name in already_done:
            results.append({
                "Strategy": name,
                "Status": "Already updated today"
            })
            continue

        try:
            result = lab.run_strategy(
                engine, name
            )

            results.append({
                "Strategy": name,
                "Status": "Run completed",
                "Details": str(result)[:200]
            })

        except Exception as exc:
            results.append({
                "Strategy": name,
                "Status": "Failed",
                "Details": (
                    f"{type(exc).__name__}: {exc}"
                )[:250]
            })

    return pd.DataFrame(results)


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Research",
        "Strategy Lab",
        "Daily Decisions",
        "Trade Log",
        "My Portfolio",
        "Methodology"
    ]
)

if st.sidebar.button("Refresh dashboard"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    "Historical prices may be delayed. "
    "All investment strategies use paper trading."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Investment Dashboard")

    snapshots = safe_read("snapshots")

    if snapshots.empty:
        st.info(
            "No original AI strategy snapshots yet."
        )

    else:
        snapshots = snapshots.sort_values(
            "run_date"
        )

        latest = snapshots.iloc[-1]

        a, b, c = st.columns(3)

        a.metric(
            "Original AI Portfolio",
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
        "Research compares historical prices. "
        "Strategy Lab tracks six independently "
        "simulated investment models."
    )


# ============================================================
# RESEARCH
# ============================================================

elif page == "Research":

    st.header("🔬 Investment Research")

    st.write(
        "Compare individual investments and your "
        "personal portfolio using a consistent "
        "historical calendar."
    )

    if "research_symbols" not in st.session_state:
        st.session_state.research_symbols = [
            "NVDA", "AAPL", "MSFT", "VOO", "QQQ"
        ]

    new_ticker = st.text_input(
        "Add a stock, ETF, or cryptocurrency",
        placeholder="PLTR, TSLA, BTC-USD"
    )

    if st.button("Add ticker"):
        ticker = new_ticker.strip().upper()

        if (
            ticker
            and ticker not in st.session_state.research_symbols
        ):
            st.session_state.research_symbols.append(
                ticker
            )
            st.rerun()

    options = list(dict.fromkeys(
        st.session_state.research_symbols
        + list(WATCH)
    ))

    selected = st.multiselect(
        "Select investments",
        options=options,
        default=[
            ticker
            for ticker in st.session_state.research_symbols
            if ticker in options
        ]
    )

    st.session_state.research_symbols = selected

    period = st.selectbox(
        "Historical period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

    c1, c2 = st.columns(2)

    with c1:
        show_personal = st.checkbox(
            "My Portfolio ($10K)",
            value=True
        )

    with c2:
        show_original = st.checkbox(
            "Original AI Strategy",
            value=True
        )

    fig = go.Figure()
    performance = []

    with st.spinner("Loading historical prices..."):
        reference_index = historical_reference_index(
            selected,
            period
        )

    # --------------------------------------------------------
    # INDIVIDUAL INVESTMENTS
    # --------------------------------------------------------

    for ticker in selected:

        try:
            history = get_history(
                ticker, period
            )

            normalized = normalize_to_10000(
                history
            )

            if normalized.empty:
                st.warning(
                    f"No usable historical data for {ticker}."
                )
                continue

            add_line(
                fig,
                normalized,
                ticker,
                width=2
            )

            performance.append({
                "Investment": ticker,
                "Start Date": normalized.index[0].date(),
                "End Date": normalized.index[-1].date(),
                "Ending Value ($)": float(
                    normalized.iloc[-1]
                ),
                "Return (%)": (
                    float(normalized.iloc[-1])
                    / 10000 - 1
                ) * 100
            })

        except Exception as exc:
            st.warning(
                f"{ticker}: {type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # FULL-PERIOD PERSONAL PORTFOLIO
    # --------------------------------------------------------

    if show_personal:

        try:
            weights, actual_value, missing = (
                get_portfolio_weights()
            )

            if not weights:
                st.warning(
                    "No priced portfolio holdings found. "
                    "Upload holdings on My Portfolio."
                )

            else:
                simulation, unavailable = (
                    simulate_portfolio_full_period(
                        weights,
                        period,
                        reference_index
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

                    performance.append({
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
                        "These holdings could not be "
                        "priced today and are excluded "
                        "from the simulation: "
                        + ", ".join(missing)
                    )

                if unavailable:
                    st.info(
                        "These investments stayed in cash "
                        "because historical prices were "
                        "unavailable: "
                        + ", ".join(unavailable)
                    )

        except Exception as exc:
            st.error(
                "Personal portfolio simulation failed: "
                + str(exc)[:250]
            )

    # --------------------------------------------------------
    # ORIGINAL AI STRATEGY
    # --------------------------------------------------------

    if show_original:

        snapshots = safe_read("snapshots")

        if not snapshots.empty:

            snapshots = snapshots.sort_values(
                "run_date"
            )

            ai_series = pd.Series(
                snapshots["equity"].astype(float).values,
                index=pd.to_datetime(
                    snapshots["run_date"]
                )
            )

            if len(reference_index) > 0:
                ai_series = ai_series.loc[
                    (ai_series.index >= reference_index.min())
                    & (ai_series.index <= reference_index.max())
                ]

            add_line(
                fig,
                ai_series,
                "Original AI Strategy",
                color="#A855F7",
                width=4,
                dash="dash"
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
        "Stocks use independently downloaded historical "
        "adjusted prices. Your personal portfolio begins "
        "on the benchmark calendar's first date and holds "
        "newer-stock allocations as cash until their "
        "first available price. The original AI line "
        "shows recorded paper-trading equity."
    )

    st.subheader("Historical Performance")

    if performance:

        results = pd.DataFrame(
            performance
        ).sort_values(
            "Return (%)",
            ascending=False
        )

        st.dataframe(
            results,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Different starting dates can make returns "
            "non-comparable. Check the Start Date and "
            "End Date columns."
        )

    if show_personal:

        try:
            weights, total, missing = (
                get_portfolio_weights()
            )

            if weights:

                st.divider()
                st.subheader(
                    "💼 My Portfolio — $10,000 Allocation"
                )

                allocation = pd.DataFrame([
                    {
                        "Ticker": ticker,
                        "Weight (%)": weight * 100,
                        "Hypothetical Allocation ($)": (
                            weight * 10000
                        )
                    }
                    for ticker, weight in weights.items()
                ]).sort_values(
                    "Weight (%)",
                    ascending=False
                )

                st.dataframe(
                    allocation,
                    hide_index=True,
                    use_container_width=True
                )

                st.metric(
                    "Current Priced Portfolio Value",
                    f"${total:,.2f}"
                )

        except Exception:
            pass

    with st.expander(
        "How the $10,000 historical comparison works",
        expanded=False
    ):
        st.markdown("""
**Individual stocks:** Each stock uses its own
historical adjusted closing prices, normalized
to a starting investment of $10,000.

**Your portfolio:** The simulation applies your
current market-value allocation weights to a
hypothetical $10,000 starting balance.

**Newer investments:** If a stock lacks historical
prices at the beginning, its assigned money
stays in cash until the first available price.

**Important:** This is a hypothetical historical
simulation, not your actual investment returns.
It does not reconstruct historical transactions.

No cash interest, rebalancing, fees or taxes
are modeled.
""")


# ============================================================
# STRATEGY LAB
# ============================================================

elif page == "Strategy Lab":

    st.header("🏦 Multi-Strategy Investment Lab")

    st.write(
        "Six independent paper-trading models, "
        "each with a $10,000 starting balance."
    )

    st.caption(
        "Institutional-inspired model names are "
        "educational approximations, not actual "
        "institutional portfolios or trade records."
    )

    if st.button(
        "Run Missing Paper Strategies",
        type="primary"
    ):

        if not market_ready():
            st.warning(
                "Paper strategy updates are intended "
                "for weekdays after 4:20 PM Eastern."
            )

        else:
            with st.spinner(
                "Running missing strategy updates..."
            ):
                results = run_missing_strategies()

            st.dataframe(
                results,
                hide_index=True,
                use_container_width=True
            )

    trades, snapshots = lab_tables()

    st.subheader("All Strategy Performance")

    fig = go.Figure()

    if not snapshots.empty:

        strategy_col = date_column(
            snapshots,
            ["strategy", "model"]
        )

        day_col = date_column(
            snapshots,
            ["run_date", "day"]
        )

        if (
            strategy_col
            and day_col
            and "equity" in snapshots.columns
        ):

            for name in lab.MODELS:

                rows = snapshots[
                    snapshots[strategy_col] == name
                ].sort_values(day_col)

                if rows.empty:
                    continue

                series = pd.Series(
                    rows["equity"].astype(float).values,
                    index=pd.to_datetime(
                        rows[day_col]
                    )
                )

                # These strategies start with $10K.
                add_line(
                    fig,
                    series,
                    name,
                    width=3
                )

            format_chart(
                fig,
                "Six Paper-Trading Strategies",
                "Portfolio Value ($)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Strategy Leaderboard")

            latest = (
                snapshots.sort_values(day_col)
                .groupby(strategy_col)
                .tail(1)
                .copy()
            )

            latest["Return (%)"] = (
                latest["equity"].astype(float)
                / lab.STARTING_CASH - 1
            ) * 100

            st.dataframe(
                latest[
                    [
                        strategy_col,
                        day_col,
                        "equity",
                        "cash",
                        "Return (%)"
                    ]
                ].sort_values(
                    "Return (%)",
                    ascending=False
                ),
                hide_index=True,
                use_container_width=True
            )

        else:
            st.warning(
                "Strategy snapshots have unexpected "
                "column names."
            )

    else:
        st.info(
            "No strategy snapshots yet. "
            "Run the models after market close."
        )

    st.divider()
    st.subheader("Individual Strategy Explorer")

    chosen = st.selectbox(
        "Select a strategy",
        list(lab.MODELS)
    )

    config = lab.MODELS[chosen]

    if isinstance(config, dict):

        st.write(
            "**Strategy objective:**",
            config.get(
                "description",
                "No description provided."
            )
        )

        st.write(
            "**Methodology:**",
            config.get(
                "method",
                "Rules-based selection."
            )
        )

        tickers = config.get(
            "tickers", []
        )

        if tickers:
            st.write(
                "**Investment universe:**",
                ", ".join(tickers)
            )

    else:
        st.write(config)

    if not snapshots.empty:

        strategy_col = date_column(
            snapshots,
            ["strategy", "model"]
        )

        day_col = date_column(
            snapshots,
            ["run_date", "day"]
        )

        if strategy_col and day_col:

            selected_snapshots = snapshots[
                snapshots[strategy_col] == chosen
            ].sort_values(day_col)

            if not selected_snapshots.empty:

                latest = selected_snapshots.iloc[-1]

                a, b, c = st.columns(3)

                equity = float(
                    latest["equity"]
                )

                a.metric(
                    "Portfolio Value",
                    f"${equity:,.2f}"
                )

                b.metric(
                    "Cash",
                    f"${float(latest['cash']):,.2f}"
                )

                c.metric(
                    "Return",
                    f"{(equity/lab.STARTING_CASH-1)*100:+.2f}%"
                )

                if (
                    "positions_json"
                    in selected_snapshots.columns
                ):

                    try:
                        positions = json.loads(
                            latest["positions_json"]
                        )

                        st.subheader(
                            "Current Positions"
                        )

                        st.dataframe(
                            pd.DataFrame([
                                {
                                    "Ticker": ticker,
                                    "Shares": shares
                                }
                                for ticker, shares
                                in positions.items()
                            ]),
                            hide_index=True,
                            use_container_width=True
                        )

                    except Exception:
                        pass

    st.divider()
    st.subheader("Strategy Transactions")

    if not trades.empty:

        strategy_col = date_column(
            trades,
            ["strategy", "model"]
        )

        if strategy_col:

            chosen_trades = trades[
                trades[strategy_col] == chosen
            ].copy()

            sort_col = date_column(
                chosen_trades,
                ["executed_at", "trade_date", "day"]
            )

            if sort_col:
                chosen_trades = chosen_trades.sort_values(
                    sort_col,
                    ascending=False
                )

            st.dataframe(
                chosen_trades,
                hide_index=True,
                use_container_width=True
            )

            if (
                "reason" in chosen_trades.columns
                and "ticker" in chosen_trades.columns
            ):

                st.subheader(
                    "Why the Model Made These Trades"
                )

                for _, trade in (
                    chosen_trades.head(8).iterrows()
                ):

                    label = (
                        f"{trade.get('side', 'TRADE')} "
                        f"{trade.get('ticker', '')}"
                    )

                    with st.expander(label):
                        st.write(
                            trade.get(
                                "reason",
                                "No explanation recorded."
                            )
                        )

            st.download_button(
                "Download Selected Strategy Trades",
                chosen_trades.to_csv(index=False),
                "selected_strategy_trades.csv",
                "text/csv"
            )

        st.subheader("All Recorded Strategy Trades")

        st.dataframe(
            trades,
            hide_index=True,
            use_container_width=True
        )

    else:
        st.info(
            "No strategy trades have been recorded."
        )

    st.caption(
        "GitHub Actions can run scheduled paper "
        "strategies when the workflow is configured "
        "and executing successfully."
    )


# ============================================================
# DAILY DECISIONS
# ============================================================

elif page == "Daily Decisions":

    st.header("📅 Original AI Daily Decisions")

    try:
        signals = score_frame(
            prices(WATCH)
        )

        st.dataframe(
            signals,
            hide_index=True,
            use_container_width=True
        )

        if st.button(
            "Run Original AI Strategy",
            type="primary"
        ):

            result = run(engine)

            st.success(
                "Original AI strategy run completed."
            )

            st.write(result)

    except Exception as exc:
        st.error(
            "Unable to calculate daily decisions: "
            + str(exc)[:250]
        )


# ============================================================
# ORIGINAL TRADE LOG
# ============================================================

elif page == "Trade Log":

    st.header("📋 Original AI Trade Log")

    trades = safe_read("trades")

    if trades.empty:
        st.info(
            "No original strategy trades recorded."
        )

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
        "Upload a CSV with ticker and shares columns. "
        "cost_basis is optional."
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

            if not {
                "ticker",
                "shares"
            }.issubset(frame.columns):

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
                    "Save / Replace Holdings",
                    type="primary"
                ):

                    if not 1 <= len(frame) <= 100:
                        st.error(
                            "Upload 1–100 valid holdings."
                        )

                    else:
                        save_personal_holdings(frame)

                        st.success(
                            "Holdings saved to Supabase."
                        )

                        st.rerun()

        except Exception as exc:
            st.error(
                "CSV error: "
                + str(exc)[:250]
            )

    saved = load_personal_holdings()

    if saved.empty:

        st.info(
            "No saved portfolio holdings."
        )

    else:

        valued, total, missing = (
            value_personal_portfolio()
        )

        st.metric(
            "Priced Portfolio Value",
            f"${total:,.2f}"
        )

        st.dataframe(
            valued,
            hide_index=True,
            use_container_width=True
        )

        if missing:

            st.warning(
                "Missing market prices: "
                + ", ".join(missing)
            )

        else:

            record_personal_snapshot(
                total,
                len(valued),
                missing
            )

            pie = px.pie(
                valued,
                names="ticker",
                values="market_value",
                hole=0.4,
                title="Current Portfolio Allocation"
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )

        st.divider()
        st.subheader(
            "Saved Daily Portfolio Valuations"
        )

        history = load_personal_snapshots()

        if not history.empty:

            st.line_chart(
                history.set_index(
                    "snapshot_date"
                )["market_value"]
            )

            st.dataframe(
                history,
                hide_index=True,
                use_container_width=True
            )

        else:
            st.info(
                "No daily portfolio valuations yet."
            )


# ============================================================
# METHODOLOGY
# ============================================================

else:

    st.header("📚 Methodology")

    st.markdown("""
## Historical Price Research

Stock and ETF prices are downloaded separately
using Yahoo Finance adjusted daily closes.

Each selected investment is normalized to a
hypothetical $10,000 initial investment.

Enabling or disabling your personal portfolio
does not change the individual stock data.

## Personal Portfolio Backtest

The personal portfolio uses current market-value
weights based on saved share quantities.

Each holding receives a hypothetical allocation
of the original $10,000.

If a holding has no valid historical price at
the beginning of the comparison, its allocation
stays in cash until its first available price.

The model does not reconstruct actual historical
personal trades.

## Six Strategy Lab Models

Each strategy starts with $10,000 and independently
records simulated positions, trades and valuations.

Institutional-inspired models do not represent
actual Goldman Sachs or JPMorgan Chase portfolios.

The Pelosi-inspired model does not reproduce
actual congressional transactions.

## Strategy Updates

The six strategies use the existing
`strategy_lab.run_strategy()` function.

GitHub Actions can execute scheduled updates
when its workflow and database credentials
are correctly configured.

## Important Limitations

- No actual brokerage trades are executed.
- Backtested results are hypothetical.
- Current weights are applied retroactively.
- Cash earns zero interest in the simulation.
- No portfolio rebalancing is simulated.
- Trading fees and taxes are excluded from
  the historical portfolio comparison.
- Prices can be delayed or unavailable.
- Historical returns do not guarantee future results.
""")
