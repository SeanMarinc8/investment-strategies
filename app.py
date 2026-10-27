
import os
import hmac
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sqlalchemy import text

from engine import (
    db,
    init,
    read,
    run,
    prices,
    score_frame,
    WATCH,
    START_CASH,
)

import strategy_lab as lab


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Investment Research Lab",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Investment Research Lab")

st.caption(
    "Institutional-inspired investment strategies | "
    "Multi-portfolio research | "
    "Automated paper trading | "
    "Personal portfolio tracking"
)


# ============================================================
# LOGIN
# ============================================================

try:
    APP_PASSWORD = str(st.secrets["APP_PASSWORD"])
    DATABASE_URL = str(st.secrets["DATABASE_URL"])

except Exception:
    st.error(
        "APP_PASSWORD or DATABASE_URL is missing "
        "from Streamlit Secrets."
    )
    st.stop()


entered_password = st.text_input(
    "Enter your app password",
    type="password",
)

if not hmac.compare_digest(
    entered_password,
    APP_PASSWORD,
):
    st.stop()


os.environ["DATABASE_URL"] = DATABASE_URL


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

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
# HELPER FUNCTIONS
# ============================================================

def current_eastern_time():
    return datetime.now(ET)


def market_ready():
    now = current_eastern_time()

    return (
        now.weekday() < 5
        and (now.hour, now.minute) >= (16, 20)
    )


def read_strategy_tables():

    trades = lab.load(
        engine,
        "lab_trades",
    )

    snapshots = lab.load(
        engine,
        "lab_snapshots",
    )

    return trades, snapshots


def get_personal_holdings():

    return pd.read_sql_query(
        text("""
            SELECT ticker, shares, cost_basis
            FROM personal_holdings
            ORDER BY ticker
        """),
        engine,
    )


def get_personal_snapshots():

    return pd.read_sql_query(
        text("""
            SELECT *
            FROM personal_snapshots
            ORDER BY snapshot_date
        """),
        engine,
    )


def save_personal_holdings(frame):

    with engine.begin() as conn:

        conn.execute(
            text("DELETE FROM personal_holdings")
        )

        for _, row in frame.iterrows():

            cost = row["cost_basis"]

            conn.execute(text("""
                INSERT INTO personal_holdings
                (ticker, shares, cost_basis)
                VALUES
                (:ticker, :shares, :cost_basis)
            """), {
                "ticker": row["ticker"],
                "shares": float(row["shares"]),
                "cost_basis": (
                    None
                    if pd.isna(cost)
                    else float(cost)
                ),
            })


def download_history(tickers, period="1y"):

    if not tickers:
        return pd.DataFrame()

    history = lab.fetch_history(
        list(dict.fromkeys(tickers)),
        period,
    )

    if history.empty:
        return history

    history.index = pd.to_datetime(
        history.index
    )

    if history.index.tz is not None:
        history.index = history.index.tz_localize(None)

    return history.sort_index()


def get_current_portfolio_weights():

    holdings = get_personal_holdings()

    if holdings.empty:
        return {}, None, "No holdings saved yet."

    tickers = holdings["ticker"].tolist()

    history = download_history(
        tickers,
        "1mo",
    )

    values = {}
    missing = []

    for row in holdings.itertuples():

        ticker = str(row.ticker).upper()
        shares = float(row.shares)

        if (
            ticker not in history.columns
            or history[ticker].dropna().empty
        ):
            missing.append(ticker)
            continue

        latest = float(
            history[ticker].dropna().iloc[-1]
        )

        if not np.isfinite(latest) or latest <= 0:
            missing.append(ticker)
            continue

        values[ticker] = (
            values.get(ticker, 0)
            + shares * latest
        )

    if missing:
        return (
            {},
            None,
            "Missing current prices: "
            + ", ".join(missing),
        )

    total = sum(values.values())

    if total <= 0:
        return {}, None, "Portfolio has no positive value."

    weights = {
        ticker: value / total
        for ticker, value in values.items()
        if value > 0
    }

    return weights, total, None


def simulate_portfolio(history, weights):

    """
    Simulates $10,000 invested using the current
    portfolio's weights at the beginning of the
    available common historical period.

    Shares remain fixed: buy-and-hold simulation.
    """

    if not weights:
        return pd.Series(dtype=float)

    tickers = list(weights.keys())

    missing = [
        ticker
        for ticker in tickers
        if ticker not in history.columns
    ]

    if missing:
        raise ValueError(
            "Missing historical prices: "
            + ", ".join(missing)
        )

    selected_prices = history[tickers].copy()

    selected_prices = selected_prices.apply(
        pd.to_numeric,
        errors="coerce",
    )

    selected_prices = selected_prices.ffill()

    # Never invent prices before an asset begins trading.
    selected_prices = selected_prices.dropna(
        how="any"
    )

    selected_prices = selected_prices[
        (selected_prices > 0).all(axis=1)
    ]

    if len(selected_prices) < 2:
        raise ValueError(
            "Not enough overlapping historical prices."
        )

    start_prices = selected_prices.iloc[0]

    hypothetical_shares = {
        ticker: (
            10000.0
            * weights[ticker]
            / float(start_prices[ticker])
        )
        for ticker in tickers
    }

    portfolio_value = pd.Series(
        0.0,
        index=selected_prices.index,
    )

    for ticker, shares in hypothetical_shares.items():

        portfolio_value = (
            portfolio_value
            + selected_prices[ticker] * shares
        )

    portfolio_value.name = "My Portfolio ($10K)"

    return portfolio_value


def value_personal_portfolio(record=False):

    holdings = get_personal_holdings()

    if holdings.empty:
        return holdings, [], 0.0

    history = download_history(
        holdings["ticker"].tolist(),
        "1mo",
    )

    quotes = {}

    for ticker in holdings["ticker"]:

        if ticker not in history.columns:
            continue

        series = history[ticker].dropna()

        if series.empty:
            continue

        price = float(series.iloc[-1])

        if np.isfinite(price) and price > 0:
            quotes[ticker] = price

    holdings["last_price"] = (
        holdings["ticker"].map(quotes)
    )

    holdings["market_value"] = (
        holdings["shares"]
        * holdings["last_price"]
    )

    missing = holdings.loc[
        holdings["last_price"].isna(),
        "ticker",
    ].tolist()

    total = float(
        holdings["market_value"].sum()
    )

    if record and not missing:

        with engine.begin() as conn:

            conn.execute(text("""
                INSERT INTO personal_snapshots
                (
                    snapshot_date,
                    market_value,
                    priced_count,
                    missing_count
                )
                VALUES (:date, :value, :count, 0)

                ON CONFLICT (snapshot_date)
                DO UPDATE SET
                    market_value = EXCLUDED.market_value,
                    priced_count = EXCLUDED.priced_count,
                    missing_count = 0
            """), {
                "date": current_eastern_time().date(),
                "value": total,
                "count": len(holdings),
            })

    return holdings, missing, total


def update_due_strategies():

    if not market_ready():
        return []

    _, snapshots = read_strategy_tables()

    today = current_eastern_time().date()

    completed = set()

    if not snapshots.empty:

        dates = pd.to_datetime(
            snapshots["run_date"]
        ).dt.date

        completed = set(
            snapshots.loc[
                dates == today,
                "strategy",
            ]
        )

    results = []

    for name in lab.MODELS:

        if name in completed:
            continue

        try:

            result = lab.run_strategy(
                engine,
                name,
            )

            results.append(result)

        except Exception as exc:

            results.append({
                "strategy": name,
                "error": type(exc).__name__,
            })

    return results


def normalized_series(series):

    series = series.dropna()

    if series.empty:
        return series

    start = float(series.iloc[0])

    if start <= 0:
        return pd.Series(dtype=float)

    return series / start * 10000.0


def add_chart_line(
    fig,
    series,
    name,
    color=None,
    width=2,
    dash="solid",
):

    if series.empty:
        return

    line_settings = {
        "width": width,
        "dash": dash,
    }

    if color:
        line_settings["color"] = color

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=name,
            line=line_settings,
            hovertemplate=(
                "%{x|%b %d, %Y}"
                "<br>$%{y:,.2f}"
                "<extra>%{fullData.name}</extra>"
            ),
        )
    )


def format_comparison_chart(fig, title):

    fig.update_layout(
        title=title,
        height=610,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Growth of $10,000",
        legend=dict(
            orientation="h",
            y=-0.25,
            x=0,
        ),
        margin=dict(
            t=70,
            b=140,
        ),
    )

    fig.update_yaxes(
        tickprefix="$",
        tickformat=",.0f",
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Strategy Lab",
        "Research",
        "Original AI Strategy",
        "Daily Decisions",
        "Trade Log",
        "My Portfolio",
        "Methodology",
    ],
)

if st.sidebar.button("Refresh dashboard"):
    st.rerun()


# ============================================================
# AUTOMATIC CHECK WHEN APP OPENS
# ============================================================

today = current_eastern_time().date()

if (
    market_ready()
    and st.session_state.get("last_auto_check") != today
):

    with st.spinner(
        "Checking today's paper-strategy updates..."
    ):

        results = update_due_strategies()

    # Failed strategies can be retried on a later rerun.
    if not any("error" in r for r in results):
        st.session_state["last_auto_check"] = today

    if results:
        st.sidebar.info(
            "Strategy update check completed."
        )


# ============================================================
# STRATEGY LAB
# ============================================================

if page == "Strategy Lab":

    st.header("🏦 Multi-Strategy Investment Lab")

    st.write(
        "Compare six independent paper-trading models "
        "against your real portfolio."
    )

    st.info(
        "Goldman Sachs, JPMorgan Chase and Pelosi-inspired "
        "models are independent approximations. "
        "They do not copy proprietary or live disclosed trades."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Run missing paper strategies",
            type="primary",
        ):

            if not market_ready():

                st.warning(
                    "Runs are limited to weekdays "
                    "after 4:20 PM Eastern."
                )

            else:

                results = update_due_strategies()

                if results:

                    st.dataframe(
                        pd.DataFrame(results),
                        hide_index=True,
                        use_container_width=True,
                    )

                else:

                    st.success(
                        "All models already have "
                        "today's snapshots."
                    )

    with col2:

        st.caption(
            "GitHub Actions can update these models "
            "when Streamlit is closed. "
            "The app also checks for missing updates "
            "when opened after market close."
        )

    trades, snapshots = read_strategy_tables()

    personal_history = get_personal_snapshots()

    st.subheader("📈 All Portfolio Performance")

    all_names = (
        list(lab.MODELS)
        + ["My Portfolio"]
    )

    selected_models = st.multiselect(
        "Choose portfolios to display",
        options=all_names,
        default=all_names,
    )

    comparison_mode = st.radio(
        "Chart display",
        [
            "Normalized $10,000 Growth",
            "Actual Recorded Value",
        ],
        horizontal=True,
    )

    fig = go.Figure()

    for name in selected_models:

        if name == "My Portfolio":

            if personal_history.empty:
                continue

            rows = personal_history.sort_values(
                "snapshot_date"
            )

            series = pd.Series(
                rows["market_value"].astype(float).values,
                index=pd.to_datetime(
                    rows["snapshot_date"]
                ),
            )

            if comparison_mode.startswith("Normalized"):
                series = normalized_series(series)

            add_chart_line(
                fig,
                series,
                name,
                color="#F59E0B",
                width=5,
                dash="dash",
            )

        else:

            if snapshots.empty:
                continue

            rows = snapshots[
                snapshots["strategy"] == name
            ].sort_values("run_date")

            if rows.empty:
                continue

            series = pd.Series(
                rows["equity"].astype(float).values,
                index=pd.to_datetime(
                    rows["run_date"]
                ),
            )

            # All simulated models began with $10,000.
            add_chart_line(
                fig,
                series,
                name,
                width=3,
            )

    format_comparison_chart(
        fig,
        "AI Investment Models vs My Portfolio",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Your real portfolio uses saved daily valuations. "
        "The AI models use recorded simulated portfolio equity. "
        "The personal portfolio line is not adjusted for "
        "deposits, withdrawals or changes in share quantities."
    )

    # ---------------- LEADERBOARD ----------------

    st.subheader("🏆 Strategy Leaderboard")

    if not snapshots.empty:

        latest = (
            snapshots.sort_values("run_date")
            .groupby("strategy")
            .tail(1)
            .copy()
        )

        latest["return_pct"] = (
            latest["equity"]
            / lab.STARTING_CASH
            - 1
        ) * 100

        leaderboard = latest[
            [
                "strategy",
                "run_date",
                "equity",
                "cash",
                "return_pct",
            ]
        ].sort_values(
            "return_pct",
            ascending=False,
        )

        st.dataframe(
            leaderboard,
            hide_index=True,
            use_container_width=True,
        )

    else:

        st.info(
            "No simulated strategy results recorded yet."
        )

    # ---------------- INDIVIDUAL MODEL ----------------

    st.divider()

    st.subheader("🔎 Individual AI Strategy Explorer")

    chosen = st.selectbox(
        "Select an AI model",
        list(lab.MODELS),
    )

    config = lab.MODELS[chosen]

    st.markdown(
        "**Investment objective:** "
        + config["description"]
    )

    st.markdown(
        "**How investments are selected:** "
        + config["method"]
    )

    st.markdown(
        "**Investment universe:** "
        + ", ".join(config["tickers"])
    )

    model_snapshots = (
        snapshots[
            snapshots["strategy"] == chosen
        ].sort_values("run_date")
        if not snapshots.empty
        else pd.DataFrame()
    )

    if not model_snapshots.empty:

        latest = model_snapshots.iloc[-1]

        a, b, c = st.columns(3)

        a.metric(
            "Portfolio Value",
            f"${latest['equity']:,.2f}",
        )

        b.metric(
            "Cash",
            f"${latest['cash']:,.2f}",
        )

        c.metric(
            "Return",
            f"{(latest['equity']/10000-1)*100:+.2f}%",
        )

        positions = json.loads(
            latest["positions_json"]
        )

        st.subheader("Current Positions")

        if positions:

            position_table = pd.DataFrame([
                {
                    "Ticker": ticker,
                    "Shares": shares,
                }
                for ticker, shares in positions.items()
            ])

            st.dataframe(
                position_table,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "These positions follow the model's "
                "selection and allocation rules."
            )

        else:

            st.info(
                "This model currently has no open positions."
            )

    # ---------------- TRADE EXPLANATIONS ----------------

    st.subheader("Recent Trades & Explanations")

    model_trades = (
        trades[
            trades["strategy"] == chosen
        ].sort_values(
            "executed_at",
            ascending=False,
        )
        if not trades.empty
        else pd.DataFrame()
    )

    if model_trades.empty:

        st.info(
            "No transactions recorded for this model."
        )

    else:

        st.dataframe(
            model_trades,
            hide_index=True,
            use_container_width=True,
        )

        for _, trade in model_trades.head(8).iterrows():

            with st.expander(
                f"{trade['trade_date']} — "
                f"{trade['side']} {trade['ticker']}"
            ):

                st.write(
                    "**Why the model made this trade:**"
                )

                st.write(trade["reason"])

                st.write(
                    f"Shares: {trade['shares']:.6f}"
                )

                st.write(
                    f"Execution price: ${trade['price']:.2f}"
                )

                st.write(
                    f"Simulated fee: ${trade['fee']:.2f}"
                )

    # ---------------- ALL TRANSACTIONS ----------------

    st.divider()

    st.subheader("📋 All AI Strategy Transactions")

    if trades.empty:

        st.info("No trades recorded yet.")

    else:

        st.dataframe(
            trades.sort_values(
                "executed_at",
                ascending=False,
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Export All Transactions",
            trades.to_csv(index=False),
            file_name="all_ai_strategy_trades.csv",
            mime="text/csv",
        )


# ============================================================
# RESEARCH — UPDATED WITH PERSONAL PORTFOLIO
# ============================================================

elif page == "Research":

    st.header("📊 Investment Research & Comparison")

    st.write(
        "Compare stocks, ETFs, your original AI strategy, "
        "and a hypothetical $10,000 investment using "
        "the same percentage allocations as your real portfolio."
    )

    # ---------------- SELECT TICKERS ----------------

    if "research_tickers" not in st.session_state:

        st.session_state.research_tickers = [
            "NVDA",
            "AAPL",
            "MSFT",
            "VOO",
            "QQQ",
        ]

    new_ticker = st.text_input(
        "Add a stock, ETF or cryptocurrency",
        placeholder="Example: PLTR, TSLA, BTC-USD",
    )

    if st.button("Add Stock / ETF"):

        ticker = new_ticker.strip().upper()

        if (
            ticker
            and ticker not in st.session_state.research_tickers
        ):

            st.session_state.research_tickers.append(
                ticker
            )

            st.rerun()

    available_tickers = list(dict.fromkeys(
        st.session_state.research_tickers
        + list(WATCH)
    ))

    selected_tickers = st.multiselect(
        "Choose investments to compare",
        options=available_tickers,
        default=st.session_state.research_tickers,
    )

    st.session_state.research_tickers = selected_tickers

    period = st.selectbox(
        "Historical comparison period",
        [
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
        ],
        index=3,
    )

    col1, col2 = st.columns(2)

    with col1:

        show_portfolio = st.checkbox(
            "My Portfolio ($10K)",
            value=True,
        )

    with col2:

        show_original_ai = st.checkbox(
            "Original AI Strategy",
            value=True,
        )

    # ---------------- CURRENT PORTFOLIO WEIGHTS ----------------

    portfolio_weights = {}
    portfolio_value = None

    if show_portfolio:

        try:

            (
                portfolio_weights,
                portfolio_value,
                portfolio_error,
            ) = get_current_portfolio_weights()

            if portfolio_error:

                st.warning(portfolio_error)

        except Exception as exc:

            st.warning(
                "Unable to calculate personal allocation: "
                + type(exc).__name__
            )

    # ---------------- MARKET HISTORY ----------------

    all_tickers = list(dict.fromkeys(
        selected_tickers
        + (
            list(portfolio_weights)
            if show_portfolio
            else []
        )
    ))

    history = pd.DataFrame()

    if all_tickers:

        try:

            with st.spinner(
                "Loading historical prices..."
            ):

                history = download_history(
                    all_tickers,
                    period,
                )

        except Exception as exc:

            st.error(
                "Unable to retrieve historical data: "
                + type(exc).__name__
            )

    fig = go.Figure()

    performance_rows = []

    simulated_personal = pd.Series(dtype=float)

    # ---------------- MY PORTFOLIO ($10K) ----------------

    if (
        show_portfolio
        and portfolio_weights
        and not history.empty
    ):

        try:

            simulated_personal = simulate_portfolio(
                history,
                portfolio_weights,
            )

            add_chart_line(
                fig,
                simulated_personal,
                "My Portfolio ($10K)",
                color="#F59E0B",
                width=5,
            )

            performance_rows.append({
                "Investment": "My Portfolio ($10K)",
                "Start Date": simulated_personal.index[0].date(),
                "End Date": simulated_personal.index[-1].date(),
                "Starting Value": 10000.0,
                "Ending Value": float(
                    simulated_personal.iloc[-1]
                ),
                "Return (%)": (
                    float(simulated_personal.iloc[-1])
                    / 10000.0
                    - 1
                ) * 100,
            })

        except Exception as exc:

            st.warning(
                "Could not simulate your portfolio: "
                + str(exc)
            )

    # ---------------- COMMON COMPARISON WINDOW ----------------

    if not simulated_personal.empty:

        comparison_start = simulated_personal.index[0]
        comparison_end = simulated_personal.index[-1]

    elif not history.empty:

        comparison_start = history.index[0]
        comparison_end = history.index[-1]

    else:

        comparison_start = None
        comparison_end = None

    # ---------------- INDIVIDUAL STOCK LINES ----------------

    if not history.empty:

        for ticker in selected_tickers:

            if ticker not in history.columns:
                continue

            series = pd.to_numeric(
                history[ticker],
                errors="coerce",
            ).dropna()

            if comparison_start is not None:

                series = series.loc[
                    (series.index >= comparison_start)
                    & (series.index <= comparison_end)
                ]

            if len(series) < 2:
                continue

            normalized = normalized_series(series)

            add_chart_line(
                fig,
                normalized,
                ticker,
            )

            performance_rows.append({
                "Investment": ticker,
                "Start Date": normalized.index[0].date(),
                "End Date": normalized.index[-1].date(),
                "Starting Value": 10000.0,
                "Ending Value": float(
                    normalized.iloc[-1]
                ),
                "Return (%)": (
                    float(normalized.iloc[-1])
                    / 10000.0
                    - 1
                ) * 100,
            })

    # ---------------- ORIGINAL AI STRATEGY ----------------

    if show_original_ai:

        try:

            original = read(
                engine,
                "snapshots",
            )

            if not original.empty:

                original = original.sort_values(
                    "run_date"
                )

                ai_series = pd.Series(
                    original["equity"].astype(float).values,
                    index=pd.to_datetime(
                        original["run_date"]
                    ),
                )

                if comparison_start is not None:

                    ai_series = ai_series.loc[
                        (ai_series.index >= comparison_start)
                        & (ai_series.index <= comparison_end)
                    ]

                add_chart_line(
                    fig,
                    ai_series,
                    "Original AI Strategy",
                    color="#A855F7",
                    width=4,
                    dash="dash",
                )

        except Exception as exc:

            st.warning(
                "Original strategy data unavailable: "
                + type(exc).__name__
            )

    # ---------------- MAIN RESEARCH CHART ----------------

    format_comparison_chart(
        fig,
        "Growth of $10,000 — Research Comparison",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Orange = your current portfolio weights applied "
        "to a hypothetical $10,000 historical investment. "
        "Purple dashed = original AI Strategy. "
        "Individual tickers are normalized to $10,000. "
        "Click legend names to hide or show lines."
    )

    # ---------------- PORTFOLIO WEIGHTS ----------------

    if show_portfolio and portfolio_weights:

        st.divider()

        st.subheader(
            "💼 My Portfolio — $10,000 Allocation"
        )

        allocation_table = pd.DataFrame([
            {
                "Ticker": ticker,
                "Portfolio Weight (%)": weight * 100,
                "Hypothetical Investment ($)": weight * 10000,
            }
            for ticker, weight in portfolio_weights.items()
        ])

        allocation_table = allocation_table.sort_values(
            "Portfolio Weight (%)",
            ascending=False,
        )

        st.dataframe(
            allocation_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Portfolio Weight (%)":
                    st.column_config.NumberColumn(
                        "Portfolio Weight (%)",
                        format="%.2f%%",
                    ),
                "Hypothetical Investment ($)":
                    st.column_config.NumberColumn(
                        "Hypothetical Investment ($)",
                        format="$%.2f",
                    ),
            },
        )

        if portfolio_value is not None:

            st.metric(
                "Actual Portfolio Market Value",
                f"${portfolio_value:,.2f}",
            )

    # ---------------- PERFORMANCE TABLE ----------------

    st.divider()

    st.subheader("📊 Historical Performance Comparison")

    if performance_rows:

        performance_table = pd.DataFrame(
            performance_rows
        )

        performance_table = performance_table.sort_values(
            "Return (%)",
            ascending=False,
        )

        st.dataframe(
            performance_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Starting Value":
                    st.column_config.NumberColumn(
                        "Starting Value",
                        format="$%.2f",
                    ),
                "Ending Value":
                    st.column_config.NumberColumn(
                        "Ending Value",
                        format="$%.2f",
                    ),
                "Return (%)":
                    st.column_config.NumberColumn(
                        "Return (%)",
                        format="%.2f%%",
                    ),
            },
        )

        st.caption(
            "Some assets may have different starting dates "
            "because of missing historical prices. "
            "Check the Start Date column before ranking returns."
        )

    else:

        st.info(
            "Select investments to view performance."
        )

    # ---------------- EXPLANATIONS ----------------

    with st.expander(
        "📚 How to Interpret the Research Chart",
        expanded=True,
    ):

        st.markdown("""
### Why every line starts around $10,000

Stocks have different share prices, so comparing
their raw prices can be misleading.

Normalizing investments to $10,000 allows you
to compare their percentage growth.

### How My Portfolio ($10K) works

The program:

1. Reads the shares in your saved personal portfolio.
2. Retrieves their latest available market prices.
3. Calculates each holding's current percentage weight.
4. Allocates a hypothetical $10,000 using those weights.
5. Buys hypothetical shares at the beginning of the
   available historical period.
6. Holds those shares and calculates their changing value.

**Example**

If your portfolio is 50% NVDA, 30% VOO
and 20% PLTR:

- NVDA receives $5,000.
- VOO receives $3,000.
- PLTR receives $2,000.

The chart then calculates what that exact
starting allocation would be worth over time.

### Why this matters

You can compare your diversified portfolio
against owning individual stocks.

This helps you evaluate whether diversification
improved or reduced your historical returns.

### What the Original AI Strategy represents

This line uses actual saved simulated
portfolio values from your original trading model.

It is not reconstructed from historical stock prices.

### Important limitations

- Current portfolio weights are applied retroactively.
- The model does not recreate your real purchase dates.
- The personal portfolio simulation is buy-and-hold.
- Dividends, taxes, fees and cash flows are excluded.
- Historical returns do not predict future performance.
- Different start dates can make comparisons misleading.
""")


# ============================================================
# ORIGINAL AI STRATEGY
# ============================================================

elif page == "Original AI Strategy":

    st.header("🤖 Original AI Trading Strategy")

    snapshots = read(
        engine,
        "snapshots",
    )

    if snapshots.empty:

        st.info(
            "No original AI Strategy snapshots recorded."
        )

    else:

        snapshots = snapshots.sort_values(
            "run_date"
        )

        chart = snapshots.set_index(
            "run_date"
        )[
            [
                "equity",
                "benchmark_voo",
                "benchmark_qqq",
            ]
        ]

        st.line_chart(chart)

        st.dataframe(
            snapshots,
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("How the original strategy works"):

        st.write(
            "The original strategy evaluates price trends, "
            "momentum and volatility, records simulated "
            "trades, and compares its equity against "
            "VOO and QQQ benchmarks."
        )


# ============================================================
# DAILY DECISIONS
# ============================================================

elif page == "Daily Decisions":

    st.header("📅 Original AI Daily Decisions")

    try:

        ranking = score_frame(
            prices(WATCH)
        )

        st.dataframe(
            ranking,
            hide_index=True,
            use_container_width=True,
        )

        if st.button(
            "Run Original AI Strategy"
        ):

            result = run(engine)

            st.success(
                "Original strategy updated."
            )

            st.write(result)

    except Exception as exc:

        st.error(
            "Unable to calculate signals: "
            + type(exc).__name__
        )


# ============================================================
# ORIGINAL TRADE LOG
# ============================================================

elif page == "Trade Log":

    st.header("📋 Original AI Strategy Trade History")

    original_trades = read(
        engine,
        "trades",
    )

    if original_trades.empty:

        st.info("No trades recorded yet.")

    else:

        st.dataframe(
            original_trades,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download Original Trades",
            original_trades.to_csv(index=False),
            file_name="original_ai_trades.csv",
            mime="text/csv",
        )


# ============================================================
# PERSONAL PORTFOLIO
# ============================================================

elif page == "My Portfolio":

    st.header("💼 My Actual Investment Portfolio")

    st.write(
        "Upload your real holdings to track market value "
        "and compare your allocation against the AI models."
    )

    st.caption(
        "Required CSV columns: ticker, shares. "
        "Optional: cost_basis."
    )

    uploaded = st.file_uploader(
        "Upload Portfolio CSV",
        type="csv",
    )

    if uploaded is not None:

        try:

            frame = pd.read_csv(uploaded)

            frame.columns = (
                frame.columns
                .str.strip()
                .str.lower()
            )

            if not {
                "ticker",
                "shares",
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
                    errors="coerce",
                )

                if "cost_basis" not in frame.columns:

                    frame["cost_basis"] = np.nan

                frame["cost_basis"] = pd.to_numeric(
                    frame["cost_basis"],
                    errors="coerce",
                )

                frame = frame.dropna(
                    subset=[
                        "ticker",
                        "shares",
                    ]
                )

                frame = frame[
                    (frame["ticker"] != "")
                    & (frame["shares"] >= 0)
                ]

                frame = frame.groupby(
                    "ticker",
                    as_index=False,
                ).agg({
                    "shares": "sum",
                    "cost_basis": "mean",
                })

                st.dataframe(
                    frame,
                    hide_index=True,
                    use_container_width=True,
                )

                if len(frame) == 0 or len(frame) > 100:

                    st.error(
                        "Upload between 1 and 100 holdings."
                    )

                elif st.button(
                    "Save / Replace My Holdings",
                    type="primary",
                ):

                    save_personal_holdings(frame)

                    st.success(
                        "Holdings saved to Supabase."
                    )

                    st.rerun()

        except Exception as exc:

            st.error(
                "Unable to process CSV: "
                + type(exc).__name__
            )

    current_holdings = get_personal_holdings()

    if current_holdings.empty:

        st.info(
            "Upload your holdings CSV to get started."
        )

    else:

        st.subheader("Current Saved Holdings")

        try:

            valued, missing, total = (
                value_personal_portfolio(
                    record=True
                )
            )

            st.metric(
                "Portfolio Market Value",
                f"${total:,.2f}",
            )

            st.dataframe(
                valued,
                hide_index=True,
                use_container_width=True,
            )

            if missing:

                st.warning(
                    "Some prices are unavailable: "
                    + ", ".join(missing)
                )

                st.caption(
                    "The displayed total is incomplete "
                    "and no valuation snapshot was saved."
                )

            else:

                fig = px.pie(
                    valued,
                    names="ticker",
                    values="market_value",
                    title="Current Portfolio Allocation",
                    hole=0.4,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            st.subheader(
                "Saved Daily Portfolio Valuations"
            )

            st.dataframe(
                get_personal_snapshots(),
                hide_index=True,
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                "Portfolio valuation failed: "
                + type(exc).__name__
            )


# ============================================================
# METHODOLOGY
# ============================================================

else:

    st.header("📚 Investment Model Methodology")

    st.markdown("""
## How the investment models work

Each model begins with a simulated $10,000.

The program downloads historical market prices
and applies its own investment-selection rules.

Selected securities are allocated simulated capital.

Trades, positions and valuations are stored in Supabase.

## AI Goldman Sachs

A quality and lower-volatility factor approximation.

The model seeks positive-trend companies while
penalizing comparatively volatile investments.

This is not Goldman Sachs proprietary research.

## AI Pelosi Disclosure-Inspired

A predefined technology-heavy investment basket.

This model does not receive live congressional
disclosures or copy actual Pelosi transactions.

## AI JPMorgan Chase

A diversified multi-asset allocation proxy.

Combines equities, bonds, international markets
and gold to balance different investment risks.

This is not JPMorgan's proprietary strategy.

## AI Momentum

Ranks securities by recent price momentum.

Only securities passing the moving-average
trend filter are eligible.

## AI Defensive Income

Uses defensive ETFs, dividend-oriented equities,
Treasury bonds and gold.

The goal is a comparatively defensive allocation.

## AI Equal-Weight Growth

Invests across a predefined basket of
large growth companies.

Uses equal starting weights instead of
market-cap weighting.

## Important investment concepts

**Momentum**

The percentage change in price over a recent period.

**Moving Average**

The average price over a specified number of days.

**Volatility**

The variation in daily investment returns.

**Allocation**

The percentage of portfolio capital assigned
to each investment.

**Rebalancing**

Buying and selling securities to restore
target portfolio weights.

**Benchmark**

A reference investment used to evaluate performance.

**Drawdown**

The decline from a previous portfolio peak.

## Automatic updates

GitHub Actions can run the simulated strategies
on weekdays after the US stock market closes.

The app also checks for missing daily updates
when opened after market close.

The original AI Strategy remains separate
from the six additional models.

## Limitations

- All trades are simulated.
- No brokerage orders are placed.
- Market prices can be delayed.
- Trading costs and execution are simplified.
- Models are rules-based, not autonomous AI agents.
- Missed historical trading days are not reconstructed.
- Market holidays are not explicitly handled.
- Personal portfolio holdings require manual updates.
- Past performance does not guarantee future returns.
""")
