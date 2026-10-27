
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
    db, init, read, run, prices,
    score_frame, WATCH, START_CASH
)
import strategy_lab as lab


# ============================================================
# PAGE SETUP AND LOGIN
# ============================================================

st.set_page_config(
    page_title="AI Investment Research Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Investment Research Lab")
st.caption(
    "Investment research • Six paper-trading models • "
    "Personal portfolio comparison • No real brokerage orders"
)

try:
    APP_PASSWORD = str(st.secrets["APP_PASSWORD"])
    DATABASE_URL = str(st.secrets["DATABASE_URL"])
except Exception:
    st.error("Configure APP_PASSWORD and DATABASE_URL in Streamlit Secrets.")
    st.stop()

entered_password = st.text_input(
    "App password", type="password"
)

if not hmac.compare_digest(entered_password, APP_PASSWORD):
    st.stop()

os.environ["DATABASE_URL"] = DATABASE_URL

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
    st.error(f"Database initialization failed: {type(exc).__name__}")
    st.stop()

ET = ZoneInfo("America/New_York")


# ============================================================
# DATABASE HELPERS
# ============================================================

def eastern_now():
    return datetime.now(ET)


def market_ready():
    now = eastern_now()
    return now.weekday() < 5 and (now.hour, now.minute) >= (16, 20)


def strategy_tables():
    return (
        lab.load(engine, "lab_trades"),
        lab.load(engine, "lab_snapshots")
    )


def personal_holdings():
    return pd.read_sql_query(
        text("SELECT ticker, shares, cost_basis FROM personal_holdings ORDER BY ticker"),
        engine
    )


def personal_snapshots():
    return pd.read_sql_query(
        text("SELECT * FROM personal_snapshots ORDER BY snapshot_date"),
        engine
    )


def save_holdings(frame):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM personal_holdings"))

        for row in frame.itertuples():
            conn.execute(text("""
                INSERT INTO personal_holdings
                (ticker, shares, cost_basis)
                VALUES (:ticker, :shares, :cost)
            """), {
                "ticker": row.ticker,
                "shares": float(row.shares),
                "cost": (
                    None if pd.isna(row.cost_basis)
                    else float(row.cost_basis)
                )
            })


# ============================================================
# MARKET DATA
# ============================================================

def fetch_market_history(tickers, period="1y"):
    """
    Downloads historical adjusted prices.

    Ticker data is downloaded separately to prevent
    changing the selected asset list from changing
    the historical comparison window.

    Invalid tickers are reported rather than silently
    substituted with other securities.
    """
    tickers = list(dict.fromkeys(
        str(t).upper().strip()
        for t in tickers if str(t).strip()
    ))

    if not tickers:
        return pd.DataFrame(), []

    series_list = []
    missing = []

    for ticker in tickers:
        try:
            data = lab.fetch_history([ticker], period)

            if ticker not in data.columns:
                missing.append(ticker)
                continue

            s = pd.to_numeric(
                data[ticker], errors="coerce"
            ).dropna()

            s = s[s > 0]

            if s.empty:
                missing.append(ticker)
                continue

            idx = pd.to_datetime(s.index)

            if idx.tz is not None:
                idx = idx.tz_localize(None)

            s.index = idx.normalize()
            s = s[~s.index.duplicated(keep="last")]
            s.name = ticker

            series_list.append(s)

        except Exception:
            missing.append(ticker)

    if not series_list:
        return pd.DataFrame(), missing

    history = pd.concat(
        series_list, axis=1, join="outer"
    ).sort_index()

    return history, missing


def get_portfolio_weights():
    """
    Calculates weights using current share quantities
    multiplied by latest available adjusted prices.
    """
    frame = personal_holdings()

    if frame.empty:
        return {}, 0.0, ["No holdings saved yet."]

    history, unavailable = fetch_market_history(
        frame["ticker"].tolist(), "1mo"
    )

    if unavailable:
        return {}, 0.0, unavailable

    values = {}

    for row in frame.itertuples():
        ticker = str(row.ticker)
        s = history[ticker].dropna()

        if s.empty:
            return {}, 0.0, [ticker]

        values[ticker] = float(row.shares) * float(s.iloc[-1])

    total = sum(values.values())

    if total <= 0:
        return {}, 0.0, ["Portfolio market value is zero."]

    weights = {
        ticker: value / total
        for ticker, value in values.items()
        if value > 0
    }

    return weights, total, []


def simulate_personal_portfolio(history, weights):
    """
    Historical buy-and-hold simulation.

    Applies today's allocation percentages at the
    beginning of the first date when every holding
    has valid historical prices.

    Does not alter individual stock histories.
    """
    if not weights or history.empty:
        return pd.Series(dtype=float)

    tickers = list(weights)

    if any(t not in history.columns for t in tickers):
        return pd.Series(dtype=float)

    data = history[tickers].copy()
    data = data.apply(pd.to_numeric, errors="coerce")

    # Only fill gaps after the first valid observation.
    data = data.ffill().dropna(how="any")
    data = data[(data > 0).all(axis=1)]

    if len(data) < 2:
        return pd.Series(dtype=float)

    starting_prices = data.iloc[0]

    shares = {
        ticker: (
            10000 * weights[ticker] /
            float(starting_prices[ticker])
        )
        for ticker in tickers
    }

    value = pd.Series(
        0.0, index=data.index, dtype=float
    )

    for ticker, quantity in shares.items():
        value += data[ticker] * quantity

    value.name = "My Portfolio ($10K)"
    return value


def value_actual_portfolio(record=False):
    frame = personal_holdings()

    if frame.empty:
        return frame, [], 0.0

    history, unavailable = fetch_market_history(
        frame["ticker"].tolist(), "1mo"
    )

    quotes = {}

    for ticker in frame["ticker"]:
        if ticker in history.columns:
            s = history[ticker].dropna()
            if not s.empty:
                quotes[ticker] = float(s.iloc[-1])

    frame["last_price"] = frame["ticker"].map(quotes)
    frame["market_value"] = (
        frame["shares"] * frame["last_price"]
    )

    missing = sorted(set(
        unavailable +
        frame.loc[
            frame["last_price"].isna(), "ticker"
        ].tolist()
    ))

    total = float(frame["market_value"].sum())

    if record and not missing:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO personal_snapshots
                (snapshot_date, market_value, priced_count, missing_count)
                VALUES (:d, :v, :n, 0)
                ON CONFLICT (snapshot_date)
                DO UPDATE SET
                    market_value = EXCLUDED.market_value,
                    priced_count = EXCLUDED.priced_count,
                    missing_count = 0
            """), {
                "d": eastern_now().date(),
                "v": total,
                "n": len(frame)
            })

    return frame, missing, total


# ============================================================
# CHART HELPERS
# ============================================================

def normalize_to_10000(series):
    s = pd.to_numeric(
        series, errors="coerce"
    ).dropna()

    s = s[s > 0]

    if len(s) < 2:
        return pd.Series(dtype=float)

    return 10000.0 * s / float(s.iloc[0])


def add_line(fig, series, label, color=None,
             width=2, dash="solid"):
    if series.empty:
        return

    line = {"width": width, "dash": dash}

    if color:
        line["color"] = color

    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        name=label,
        mode="lines+markers" if len(series) == 1 else "lines",
        line=line,
        hovertemplate=(
            "%{x|%b %d, %Y}<br>$%{y:,.2f}"
            "<extra>%{fullData.name}</extra>"
        )
    ))


def style_chart(fig, title, ylabel="Growth of $10,000"):
    fig.update_layout(
        title=title,
        height=600,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=ylabel,
        legend=dict(orientation="h", y=-0.24),
        margin=dict(t=65, b=130)
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


# ============================================================
# AUTOMATIC STRATEGY CHECK
# ============================================================

def update_due_strategies():
    if not market_ready():
        return []

    _, snapshots = strategy_tables()
    today = eastern_now().date()

    completed = set()

    if not snapshots.empty:
        dates = pd.to_datetime(
            snapshots["run_date"]
        ).dt.date

        completed = set(
            snapshots.loc[
                dates == today, "strategy"
            ]
        )

    results = []

    for name in lab.MODELS:
        if name in completed:
            continue

        try:
            result = lab.run_strategy(engine, name)
            results.append({
                "strategy": name,
                "result": str(result)[:250]
            })
        except Exception as exc:
            results.append({
                "strategy": name,
                "error": str(exc)[:200]
            })

    return results


# ============================================================
# NAVIGATION
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
        "Methodology"
    ]
)

if st.sidebar.button("Refresh dashboard"):
    st.rerun()

today = eastern_now().date()

if (
    market_ready()
    and st.session_state.get("last_auto_check") != today
):
    with st.spinner("Checking daily strategy updates..."):
        auto_results = update_due_strategies()

    if not any("error" in r for r in auto_results):
        st.session_state["last_auto_check"] = today

    if auto_results:
        st.sidebar.info("Daily strategy update check completed.")


# ============================================================
# STRATEGY LAB
# ============================================================

if page == "Strategy Lab":

    st.header("🏦 Multi-Strategy Investment Lab")

    st.write(
        "Compare six independently simulated investment "
        "strategies and your saved personal portfolio."
    )

    if st.button("Run missing paper strategies", type="primary"):
        if not market_ready():
            st.warning(
                "Strategy updates run on weekdays "
                "after 4:20 PM Eastern."
            )
        else:
            results = update_due_strategies()
            if results:
                st.dataframe(
                    pd.DataFrame(results),
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.success("Today's model snapshots already exist.")

    st.caption(
        "GitHub Actions can update strategies while "
        "Streamlit is closed, provided the scheduled "
        "workflow is configured and running successfully."
    )

    trades, snapshots = strategy_tables()
    my_history = personal_snapshots()

    st.subheader("All Portfolio Performance")

    names = list(lab.MODELS) + ["My Portfolio"]

    selected = st.multiselect(
        "Show strategies",
        names,
        default=names
    )

    display_mode = st.radio(
        "Chart metric",
        ["Normalized $10,000", "Recorded Dollar Value"],
        horizontal=True
    )

    fig = go.Figure()

    for name in selected:
        if name == "My Portfolio":
            if my_history.empty:
                continue

            rows = my_history.sort_values("snapshot_date")
            series = pd.Series(
                rows["market_value"].astype(float).values,
                index=pd.to_datetime(rows["snapshot_date"])
            )

            if display_mode.startswith("Normalized"):
                series = normalize_to_10000(series)

            add_line(
                fig, series, name,
                color="#F59E0B", width=4, dash="dash"
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
                index=pd.to_datetime(rows["run_date"])
            )

            if display_mode.startswith("Normalized"):
                # Models start with $10,000, so preserve
                # their true simulated initial capital.
                pass

            add_line(fig, series, name, width=3)

    ylabel = (
        "Normalized Value ($)"
        if display_mode.startswith("Normalized")
        else "Recorded Portfolio Value ($)"
    )

    style_chart(
        fig, "AI Strategies vs My Portfolio", ylabel
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The simulated strategies start at $10,000. "
        "Your real portfolio's normalized line uses its "
        "first recorded valuation as a baseline. "
        "Cash flows and portfolio changes are not adjusted."
    )

    st.subheader("Strategy Leaderboard")

    if not snapshots.empty:
        latest = (
            snapshots.sort_values("run_date")
            .groupby("strategy")
            .tail(1)
            .copy()
        )

        latest["return_pct"] = (
            latest["equity"] / lab.STARTING_CASH - 1
        ) * 100

        st.dataframe(
            latest[
                ["strategy", "run_date", "equity",
                 "cash", "return_pct"]
            ].sort_values(
                "return_pct", ascending=False
            ),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No strategy snapshots recorded yet.")

    st.divider()
    st.subheader("Individual Strategy Explorer")

    chosen = st.selectbox(
        "Select a model", list(lab.MODELS)
    )

    config = lab.MODELS[chosen]

    st.write("**Investment objective:**", config["description"])
    st.write("**Selection methodology:**", config["method"])
    st.write("**Investment universe:**", ", ".join(config["tickers"]))

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

        a.metric("Portfolio Value", f"${latest['equity']:,.2f}")
        b.metric("Cash", f"${latest['cash']:,.2f}")
        c.metric(
            "Total Return",
            f"{(latest['equity']/lab.STARTING_CASH-1)*100:+.2f}%"
        )

        positions = json.loads(latest["positions_json"])

        st.subheader("Current Positions")

        if positions:
            st.dataframe(
                pd.DataFrame([
                    {"Ticker": ticker, "Shares": shares}
                    for ticker, shares in positions.items()
                ]),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No current positions.")

    model_trades = (
        trades[
            trades["strategy"] == chosen
        ].sort_values("executed_at", ascending=False)
        if not trades.empty
        else pd.DataFrame()
    )

    st.subheader("Recent Transactions and Explanations")

    if model_trades.empty:
        st.info("No trades recorded for this model.")
    else:
        st.dataframe(
            model_trades,
            hide_index=True,
            use_container_width=True
        )

        for _, trade in model_trades.head(8).iterrows():
            with st.expander(
                f"{trade['trade_date']} — "
                f"{trade['side']} {trade['ticker']}"
            ):
                st.write("**Why this trade was selected:**")
                st.write(trade["reason"])
                st.write(
                    f"Shares: {float(trade['shares']):.5f}"
                )
                st.write(
                    f"Price: ${float(trade['price']):,.2f}"
                )
                st.write(
                    f"Simulated fee: ${float(trade['fee']):,.2f}"
                )

    st.divider()
    st.subheader("All Strategy Transactions")

    if not trades.empty:
        st.dataframe(
            trades.sort_values(
                "executed_at", ascending=False
            ),
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Export All Trades",
            trades.to_csv(index=False),
            "all_strategy_trades.csv",
            "text/csv"
        )
    else:
        st.info("No trades yet.")


# ============================================================
# RESEARCH — FIXED HISTORICAL COMPARISON
# ============================================================

elif page == "Research":

    st.header("📊 Investment Research & Comparison")

    st.write(
        "Compare historical performance using a fixed "
        "lookback period. Turning My Portfolio on or off "
        "will not change the individual stocks' baselines."
    )

    if "research_options" not in st.session_state:
        st.session_state.research_options = [
            "NVDA", "AAPL", "MSFT", "VOO", "QQQ"
        ]

    new_ticker = st.text_input(
        "Add stock, ETF, or crypto ticker",
        placeholder="PLTR, TSLA, BTC-USD"
    )

    if st.button("Add ticker"):
        ticker = new_ticker.upper().strip()

        if ticker and ticker not in st.session_state.research_options:
            st.session_state.research_options.append(ticker)
            st.rerun()

    available = list(dict.fromkeys(
        st.session_state.research_options + list(WATCH)
    ))

    selected_tickers = st.multiselect(
        "Choose investments",
        available,
        default=[
            t for t in st.session_state.research_options
            if t in available
        ]
    )

    period = st.selectbox(
        "Historical comparison period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

    c1, c2 = st.columns(2)

    with c1:
        show_portfolio = st.checkbox(
            "My Portfolio ($10K)", value=True
        )

    with c2:
        show_original = st.checkbox(
            "Original AI Strategy", value=True
        )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # Individual stocks are downloaded and normalized
    # independently of the personal portfolio.
    #
    # No portfolio dates are used to reset their baselines.
    # --------------------------------------------------------

    fig = go.Figure()
    performance_rows = []

    with st.spinner("Loading historical stock prices..."):
        stock_history, missing_stocks = fetch_market_history(
            selected_tickers, period
        )

    if missing_stocks:
        st.warning(
            "Unavailable market data: "
            + ", ".join(missing_stocks)
        )

    # Establish a stable reference window from the
    # selected stocks only, not the personal portfolio.
    if not stock_history.empty:
        reference_start = stock_history.index.min()
        reference_end = stock_history.index.max()
    else:
        reference_start = None
        reference_end = None

    # Each stock uses its own first valid observation
    # within the selected historical lookback.
    # Adding another chart line cannot alter its return.
    for ticker in selected_tickers:
        if ticker not in stock_history.columns:
            continue

        series = stock_history[ticker].dropna()
        normalized = normalize_to_10000(series)

        if normalized.empty:
            continue

        add_line(fig, normalized, ticker, width=2)

        performance_rows.append({
            "Investment": ticker,
            "Start Date": normalized.index[0].date(),
            "End Date": normalized.index[-1].date(),
            "Starting Value": 10000.0,
            "Ending Value": float(normalized.iloc[-1]),
            "Return (%)": (
                float(normalized.iloc[-1]) / 10000 - 1
            ) * 100
        })

    # --------------------------------------------------------
    # PERSONAL PORTFOLIO:
    # Added separately, without modifying stock histories.
    # --------------------------------------------------------

    weights = {}
    actual_value = 0.0

    if show_portfolio:
        try:
            weights, actual_value, missing = (
                get_portfolio_weights()
            )

            if missing:
                if missing == ["No holdings saved yet."]:
                    st.warning(
                        "No holdings saved yet. Upload your "
                        "CSV on the My Portfolio page."
                    )
                else:
                    st.warning(
                        "Cannot calculate the complete personal "
                        "portfolio. Missing data: "
                        + ", ".join(missing)
                    )

            elif weights:
                portfolio_history, missing_history = (
                    fetch_market_history(list(weights), period)
                )

                if missing_history:
                    st.warning(
                        "Missing historical portfolio prices: "
                        + ", ".join(missing_history)
                    )
                else:
                    simulated = simulate_personal_portfolio(
                        portfolio_history, weights
                    )

                    if simulated.empty:
                        st.warning(
                            "Not enough overlapping historical "
                            "data for your portfolio."
                        )
                    else:
                        add_line(
                            fig,
                            simulated,
                            "My Portfolio ($10K)",
                            color="#F59E0B",
                            width=5
                        )

                        performance_rows.append({
                            "Investment": "My Portfolio ($10K)",
                            "Start Date": simulated.index[0].date(),
                            "End Date": simulated.index[-1].date(),
                            "Starting Value": 10000.0,
                            "Ending Value": float(simulated.iloc[-1]),
                            "Return (%)": (
                                float(simulated.iloc[-1]) /
                                10000 - 1
                            ) * 100
                        })

                        if reference_start is not None:
                            if simulated.index[0] > reference_start:
                                st.info(
                                    "Your personal portfolio begins on "
                                    f"{simulated.index[0].date()} because "
                                    "not all holdings have earlier usable "
                                    "prices. Individual stock lines keep "
                                    "their original starting dates."
                                )

        except Exception as exc:
            st.warning(
                "Personal portfolio simulation unavailable: "
                + str(exc)[:200]
            )

    # --------------------------------------------------------
    # ORIGINAL AI STRATEGY:
    # Uses its own saved equity history, not a fabricated
    # year of returns.
    # --------------------------------------------------------

    if show_original:
        try:
            original = read(engine, "snapshots")

            if not original.empty:
                original = original.sort_values("run_date")

                ai_series = pd.Series(
                    original["equity"].astype(float).values,
                    index=pd.to_datetime(
                        original["run_date"]
                    )
                )

                if ai_series.index.tz is not None:
                    ai_series.index = (
                        ai_series.index.tz_localize(None)
                    )

                if reference_start is not None:
                    ai_series = ai_series.loc[
                        (ai_series.index >= reference_start)
                        & (ai_series.index <= reference_end)
                    ]

                add_line(
                    fig,
                    ai_series,
                    "Original AI Strategy",
                    color="#A855F7",
                    width=4,
                    dash="dash"
                )

        except Exception as exc:
            st.warning(
                "Original AI Strategy data unavailable: "
                + type(exc).__name__
            )

    style_chart(
        fig,
        "Growth of $10,000 — Research Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Each individual stock is normalized using its "
        "own historical price data for the selected period. "
        "My Portfolio is a separate historical buy-and-hold "
        "simulation using your current allocation. "
        "Click a legend name to hide or show its line."
    )

    # --------------------------------------------------------
    # ALLOCATION
    # --------------------------------------------------------

    if show_portfolio and weights:
        st.divider()
        st.subheader("💼 My Portfolio — $10,000 Allocation")

        allocation = pd.DataFrame([
            {
                "Ticker": ticker,
                "Weight (%)": weight * 100,
                "Hypothetical Investment ($)": weight * 10000
            }
            for ticker, weight in weights.items()
        ]).sort_values(
            "Weight (%)", ascending=False
        )

        st.dataframe(
            allocation,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Weight (%)": st.column_config.NumberColumn(
                    "Weight (%)", format="%.2f%%"
                ),
                "Hypothetical Investment ($)":
                    st.column_config.NumberColumn(
                        "Hypothetical Investment ($)",
                        format="$%.2f"
                    )
            }
        )

        st.metric(
            "Current Actual Portfolio Value",
            f"${actual_value:,.2f}"
        )

    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    st.divider()
    st.subheader("📊 Historical Performance")

    if performance_rows:
        table = pd.DataFrame(performance_rows)
        table = table.sort_values(
            "Return (%)", ascending=False
        )

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Starting Value": st.column_config.NumberColumn(
                    "Starting Value", format="$%.2f"
                ),
                "Ending Value": st.column_config.NumberColumn(
                    "Ending Value", format="$%.2f"
                ),
                "Return (%)": st.column_config.NumberColumn(
                    "Return (%)", format="%.2f%%"
                )
            }
        )

        st.caption(
            "Compare Start Date and End Date before "
            "ranking investments. Different available "
            "histories mean different measurement periods."
        )
    else:
        st.info("Select investments to display performance.")

    # --------------------------------------------------------
    # EXPLANATIONS
    # --------------------------------------------------------

    with st.expander(
        "📚 How the chart works and why it is accurate",
        expanded=True
    ):
        st.markdown("""
### 1. Historical stock prices

The app downloads historical adjusted prices
for each selected investment.

Adjusted prices account for stock splits and,
when supplied by the data provider, dividends.

### 2. The $10,000 comparison

Each investment begins at a hypothetical $10,000.

For example, if a stock gains 20% during its
selected historical period:

**$10,000 becomes $12,000.**

### 3. Fixed historical comparison logic

Enabling your personal portfolio does not alter
the start date or historical return of MSFT,
NVDA, AAPL, VOO, QQQ, or any other stock.

Individual stock data is processed independently.

### 4. Your personal portfolio

The app retrieves your saved holdings, calculates
their current percentage weights, and applies
those weights to a hypothetical $10,000 investment.

It purchases hypothetical shares at the first
common historical date for all your holdings.

Those shares are then held without rebalancing.

### 5. Why your portfolio may start later

Some holdings may have a shorter price history.

Rather than silently changing the starting dates
of all other investments, the app displays your
personal portfolio over its available period.

### 6. Original AI Strategy

The original AI Strategy line uses saved paper
trading snapshots. It does not invent earlier
performance data.

### 7. Important limitations

- Historical data depends on the market-data provider.
- Different start dates make return rankings less comparable.
- Your current weights are applied retroactively.
- Your real historical trades are not reconstructed.
- The portfolio simulation excludes trading fees and taxes.
- The portfolio simulation is not a brokerage performance statement.
- Past performance does not predict future results.
""")


# ============================================================
# ORIGINAL AI STRATEGY
# ============================================================

elif page == "Original AI Strategy":

    st.header("🤖 Original AI Strategy")

    snapshots = read(engine, "snapshots")

    if snapshots.empty:
        st.info("No original strategy snapshots yet.")
    else:
        snapshots = snapshots.sort_values("run_date")

        st.line_chart(
            snapshots.set_index("run_date")[
                ["equity", "benchmark_voo", "benchmark_qqq"]
            ]
        )

        st.dataframe(
            snapshots,
            hide_index=True,
            use_container_width=True
        )

    with st.expander("How this strategy works"):
        st.write(
            "The original rules-based model evaluates "
            "momentum, price trends, and volatility, "
            "then records simulated investment decisions. "
            "VOO and QQQ are comparison benchmarks."
        )


# ============================================================
# DAILY DECISIONS
# ============================================================

elif page == "Daily Decisions":

    st.header("📅 Original AI Daily Decisions")

    try:
        signals = score_frame(prices(WATCH))

        st.dataframe(
            signals,
            hide_index=True,
            use_container_width=True
        )

        if st.button("Run Original AI Strategy"):
            result = run(engine)
            st.success("Original strategy run completed.")
            st.write(result)

    except Exception as exc:
        st.error(
            "Could not calculate signals: "
            + type(exc).__name__
        )


# ============================================================
# ORIGINAL TRADE LOG
# ============================================================

elif page == "Trade Log":

    st.header("📋 Original AI Strategy Trade Log")

    trades = read(engine, "trades")

    if trades.empty:
        st.info("No original strategy trades recorded.")
    else:
        st.dataframe(
            trades,
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Export Original Trades",
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
        "Upload your holdings and save them to Supabase. "
        "Required CSV columns: ticker, shares. "
        "Optional column: cost_basis."
    )

    uploaded = st.file_uploader(
        "Upload holdings CSV", type="csv"
    )

    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)

            frame.columns = (
                frame.columns.str.strip().str.lower()
            )

            if not {"ticker", "shares"}.issubset(frame.columns):
                st.error(
                    "CSV must contain ticker and shares columns."
                )
            else:
                frame["ticker"] = (
                    frame["ticker"].astype(str)
                    .str.upper().str.strip()
                )

                frame["shares"] = pd.to_numeric(
                    frame["shares"], errors="coerce"
                )

                if "cost_basis" not in frame.columns:
                    frame["cost_basis"] = np.nan

                frame["cost_basis"] = pd.to_numeric(
                    frame["cost_basis"], errors="coerce"
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
                    "ticker", as_index=False
                ).agg({
                    "shares": "sum",
                    "cost_basis": "mean"
                })

                st.dataframe(
                    frame,
                    hide_index=True,
                    use_container_width=True
                )

                if not 1 <= len(frame) <= 100:
                    st.error(
                        "Upload between 1 and 100 valid holdings."
                    )
                elif st.button(
                    "Save / Replace My Holdings",
                    type="primary"
                ):
                    save_holdings(frame)
                    st.success("Holdings saved in Supabase.")
                    st.rerun()

        except Exception as exc:
            st.error(
                "CSV processing error: " + str(exc)[:200]
            )

    saved = personal_holdings()

    if saved.empty:
        st.info("No holdings saved yet.")
    else:
        st.subheader("Saved Holdings")

        try:
            valued, missing, total = (
                value_actual_portfolio(record=True)
            )

            st.metric(
                "Priced Market Value",
                f"${total:,.2f}"
            )

            st.dataframe(
                valued,
                hide_index=True,
                use_container_width=True
            )

            if missing:
                st.warning(
                    "Missing prices: " + ", ".join(missing)
                )
                st.caption(
                    "The total is incomplete. No complete "
                    "daily valuation was saved."
                )
            else:
                chart = px.pie(
                    valued,
                    names="ticker",
                    values="market_value",
                    hole=0.4,
                    title="Current Portfolio Allocation"
                )
                st.plotly_chart(
                    chart, use_container_width=True
                )

            st.subheader("Saved Daily Valuations")

            st.dataframe(
                personal_snapshots(),
                hide_index=True,
                use_container_width=True
            )

        except Exception as exc:
            st.error(
                "Portfolio valuation error: "
                + str(exc)[:200]
            )


# ============================================================
# METHODOLOGY
# ============================================================

else:

    st.header("📚 Investment Strategy Methodology")

    st.markdown("""
## AI Goldman Sachs

A quality and lower-volatility factor approximation.

It evaluates securities using trend, momentum,
and volatility-based rules.

It is not affiliated with Goldman Sachs.

## AI Pelosi Disclosure-Inspired

A predefined technology-heavy investment basket.

It does not follow live congressional disclosures
or reproduce Nancy Pelosi's actual transactions.

## AI JPMorgan Chase

A diversified multi-asset investment approximation.

It combines exposure to different asset classes.

It is not JPMorgan Chase's proprietary strategy.

## AI Momentum

Selects investments using recent momentum and
moving-average trend filters.

## AI Defensive Income

Focuses on defensive and income-oriented assets.

## AI Equal-Weight Growth

Uses a predefined basket of growth companies
with equal-weight allocation rules.

## Key Financial Concepts

**Momentum:** How much an investment's price
has changed over a recent period.

**Moving average:** A smoothed average of
historical prices.

**Volatility:** How widely investment returns vary.

**Rebalancing:** Adjusting investments to restore
target allocations.

**Drawdown:** The decline from a previous peak.

**Benchmark:** A reference investment used to
evaluate performance.

## Daily Updates

The six Strategy Lab models can run through
GitHub Actions on weekdays after market close.

Successful results are stored in Supabase.

The original AI Strategy has a separate run process.

## Important Limitations

- All model trades are simulated.
- No real brokerage orders are placed.
- Models are rules-based, not autonomous AI agents.
- Missed historical trades are not reconstructed.
- Market holidays are not explicitly handled.
- Transaction fees and execution are simplified.
- Personal holdings require manual updates.
- Historical performance does not guarantee future returns.
""")
