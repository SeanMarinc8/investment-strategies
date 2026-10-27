
import os
import hmac
import json

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

from engine import (
    db, init, read, run, prices,
    score_frame, holdings, WATCH, START_CASH
)
import strategy_lab as lab

st.set_page_config(
    page_title="AI Investment Research Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Investment Research Lab")
st.caption(
    "Multi-strategy paper trading, investment research "
    "and personal portfolio tracking."
)

# ---------------- SECURITY ----------------

try:
    password = st.secrets["APP_PASSWORD"]
    database_url = st.secrets["DATABASE_URL"]
except Exception:
    st.error("Configure APP_PASSWORD and DATABASE_URL in Streamlit Secrets.")
    st.stop()

entered = st.text_input("App password", type="password")

if not hmac.compare_digest(entered, str(password)):
    st.stop()

os.environ["DATABASE_URL"] = str(database_url)

# ---------------- DATABASE ----------------

try:
    engine = db()
    init(engine)
    lab.initialize(engine)
except Exception as exc:
    st.error(f"Database initialization failed: {type(exc).__name__}")
    st.stop()


@st.cache_data(ttl=60)
def get_data():
    return (
        lab.load(engine, "lab_trades"),
        lab.load(engine, "lab_snapshots")
    )


@st.cache_data(ttl=300)
def get_prices(tickers, period):
    return lab.fetch_history(tuple(tickers), period)


def reset_cache():
    get_data.clear()
    get_prices.clear()


def strategy_equity_chart(snapshots, selected):
    if snapshots.empty:
        st.info("Run the models to begin recording performance.")
        return

    fig = go.Figure()

    for name in selected:
        rows = snapshots[
            snapshots["strategy"] == name
        ].sort_values("run_date")

        if rows.empty:
            continue

        # All simulated strategies begin at $10,000.
        fig.add_trace(go.Scatter(
            x=rows["run_date"],
            y=rows["equity"],
            mode="lines+markers",
            name=name,
            hovertemplate="%{x}<br>$%{y:,.2f}<extra>%{fullData.name}</extra>"
        ))

    fig.update_layout(
        title="Simulated Portfolio Values",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend=dict(
            orientation="h",
            y=-0.3
        ),
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def explain(name):
    if name in lab.MODELS:
        config = lab.MODELS[name]
        st.markdown(f"**Objective:** {config['description']}")
        st.markdown(f"**Selection method:** {config['method']}")

        st.info(
            "These are independent, rules-based models. "
            "They do not represent actual trades by "
            "Goldman Sachs, JPMorgan or Nancy Pelosi."
        )


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
    reset_cache()
    st.rerun()

# ==================================================
# STRATEGY LAB
# ==================================================

if page == "Strategy Lab":

    st.header("Multi-Strategy Investment Lab")

    st.write(
        "Compare six independently simulated strategies, "
        "each starting with $10,000. Their trades and "
        "daily valuations are stored in Supabase."
    )

    if st.button(
        "Run all paper strategies",
        type="primary"
    ):
        with st.spinner("Analyzing markets and updating models..."):
            results = lab.run_all(engine)

        reset_cache()
        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True
        )

    trades, snapshots = get_data()

    st.subheader("Performance comparison")

    selected = st.multiselect(
        "Show strategies on chart",
        list(lab.MODELS),
        default=list(lab.MODELS)
    )

    strategy_equity_chart(snapshots, selected)

    st.caption(
        "Each model begins with $10,000. "
        "Lines contain only recorded paper-trading dates. "
        "This is not a historical backtest."
    )

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

        st.subheader("Strategy leaderboard")

        st.dataframe(
            latest[
                ["strategy", "run_date", "equity", "cash", "return_pct"]
            ].sort_values("return_pct", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.subheader("Individual model explorer")

    chosen = st.selectbox(
        "Select a strategy",
        list(lab.MODELS)
    )

    explain(chosen)

    model_snapshots = (
        snapshots[snapshots["strategy"] == chosen]
        if not snapshots.empty else pd.DataFrame()
    )

    if not model_snapshots.empty:
        current = model_snapshots.sort_values(
            "run_date"
        ).iloc[-1]

        a, b, c = st.columns(3)
        a.metric("Portfolio value", f"${current['equity']:,.2f}")
        b.metric("Cash", f"${current['cash']:,.2f}")
        c.metric(
            "Return since inception",
            f"{(current['equity']/10000-1)*100:+.2f}%"
        )

        positions = json.loads(current["positions_json"])

        if positions:
            holdings_df = pd.DataFrame([
                {"ticker": t, "shares": float(q)}
                for t, q in positions.items()
            ])

            st.subheader("Current positions")
            st.dataframe(
                holdings_df,
                use_container_width=True,
                hide_index=True
            )

            st.write(
                "**Why these positions?** "
                + lab.MODELS[chosen]["method"]
            )

    model_trades = (
        trades[trades["strategy"] == chosen]
        if not trades.empty else pd.DataFrame()
    )

    st.subheader("Trade history")

    if model_trades.empty:
        st.info("No trades recorded for this model yet.")
    else:
        st.dataframe(
            model_trades.sort_values(
                "executed_at",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Recent trade explanations")

        for _, trade in model_trades.sort_values(
            "executed_at",
            ascending=False
        ).head(8).iterrows():

            with st.expander(
                f"{trade['side']} {trade['ticker']} "
                f"— {trade['trade_date']}"
            ):
                st.write(trade["reason"])
                st.write(
                    f"Shares: {trade['shares']:.6f} | "
                    f"Price: ${trade['price']:.2f} | "
                    f"Fee: ${trade['fee']:.2f}"
                )

    st.divider()
    st.subheader("All models — daily transactions")

    if trades.empty:
        st.info("No transactions yet.")
    else:
        filtered = trades[
            trades["strategy"].isin(selected)
        ]

        st.dataframe(
            filtered.sort_values(
                "executed_at",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Export all strategy trades",
            trades.to_csv(index=False),
            file_name="strategy_trades.csv",
            mime="text/csv"
        )

    st.caption(
        "Refresh the page to retrieve the latest database results. "
        "New trades are generated only when a model run executes."
    )

# ==================================================
# RESEARCH
# ==================================================

elif page == "Research":

    st.header("Multi-Asset Research")

    st.write(
        "Add or remove investments and compare them "
        "on one chart. Prices are normalized to a "
        "hypothetical $10,000 investment."
    )

    default_symbols = ["NVDA", "AAPL", "MSFT", "VOO", "QQQ"]

    if "research_symbols" not in st.session_state:
        st.session_state.research_symbols = default_symbols

    new_symbol = st.text_input(
        "Add ticker (example: PLTR or BTC-USD)"
    )

    if st.button("Add ticker"):
        symbol = new_symbol.upper().strip()

        if symbol and symbol not in st.session_state.research_symbols:
            st.session_state.research_symbols.append(symbol)
            st.rerun()

    selected_symbols = st.multiselect(
        "Stocks and ETFs to display",
        options=list(dict.fromkeys(
            st.session_state.research_symbols + list(WATCH)
        )),
        default=st.session_state.research_symbols
    )

    st.session_state.research_symbols = selected_symbols

    period = st.selectbox(
        "Historical period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

    include_strategy = st.checkbox(
        "Include original AI Strategy",
        value=True
    )

    if selected_symbols:
        try:
            history = get_prices(selected_symbols, period)
            fig = go.Figure()

            for ticker in selected_symbols:
                if ticker not in history.columns:
                    continue

                series = history[ticker].dropna()

                if series.empty:
                    continue

                normalized = series / series.iloc[0] * 10000

                fig.add_trace(go.Scatter(
                    x=normalized.index,
                    y=normalized,
                    name=ticker,
                    mode="lines"
                ))

            if include_strategy:
                original = read(engine, "snapshots")

                if not original.empty:
                    original = original.sort_values("run_date")

                    fig.add_trace(go.Scatter(
                        x=original["run_date"],
                        y=original["equity"],
                        name="Original AI Strategy",
                        mode="lines+markers",
                        line=dict(width=4, dash="dash")
                    ))

            fig.update_layout(
                title="Growth of $10,000 — Asset Comparison",
                xaxis_title="Date",
                yaxis_title="Hypothetical Value ($)",
                legend=dict(orientation="h", y=-0.3),
                hovermode="x unified",
                height=550
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.error(
                f"Research data unavailable: {type(exc).__name__}"
            )

    with st.expander("How to read this chart", expanded=True):
        st.markdown("""
**Why normalize prices?**

One share of a $500 stock isn't necessarily a better
investment than a $50 stock. By starting every line
at $10,000, you can compare percentage growth.

**What does the legend do?**

Click a ticker in the legend to hide or show its line.

**What is the original AI Strategy?**

It uses recorded virtual portfolio values rather than
a single stock price. Its history starts on the first
recorded simulation date, not the start of the chart.

**Important:** This is a price-return comparison.
It does not fully account for dividends, taxes,
transaction costs or portfolio cash flows.
""")

# ==================================================
# ORIGINAL STRATEGY
# ==================================================

elif page == "Original AI Strategy":

    st.header("Original AI Strategy")

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
            snapshots.tail(30),
            use_container_width=True,
            hide_index=True
        )

# ==================================================
# DAILY DECISIONS
# ==================================================

elif page == "Daily Decisions":

    st.header("Original Strategy Decisions")

    try:
        ranking = score_frame(prices(WATCH))

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Run original AI strategy"):
            result = run(engine)
            st.success("Paper strategy updated.")
            st.write(result)

    except Exception as exc:
        st.error(f"Strategy error: {type(exc).__name__}")

# ==================================================
# TRADE LOG
# ==================================================

elif page == "Trade Log":

    st.header("Original Strategy Trade Log")

    trades = read(engine, "trades")

    if trades.empty:
        st.info("No trades recorded.")
    else:
        st.dataframe(
            trades,
            use_container_width=True,
            hide_index=True
        )

# ==================================================
# MY PORTFOLIO
# ==================================================

elif page == "My Portfolio":

    st.header("My Real Portfolio")

    st.write(
        "Upload a CSV containing ticker and shares. "
        "This section values your actual holdings "
        "without placing trades or saving holdings to Supabase."
    )

    uploaded = st.file_uploader(
        "Upload holdings CSV",
        type="csv"
    )

    if uploaded:
        try:
            portfolio = pd.read_csv(uploaded)
            portfolio.columns = portfolio.columns.str.strip().str.lower()

            if not {"ticker", "shares"}.issubset(portfolio.columns):
                st.error("CSV needs ticker and shares columns.")
                st.stop()

            portfolio["ticker"] = (
                portfolio["ticker"].astype(str).str.upper().str.strip()
            )

            portfolio["shares"] = pd.to_numeric(
                portfolio["shares"],
                errors="coerce"
            )

            portfolio = portfolio.dropna(
                subset=["ticker", "shares"]
            )

            portfolio = portfolio[
                portfolio["shares"] >= 0
            ].copy()

            tickers = portfolio["ticker"].unique().tolist()
            history = lab.fetch_history(tickers, "1mo")

            quotes = {}

            for ticker in tickers:
                if ticker in history.columns:
                    series = history[ticker].dropna()
                    if not series.empty:
                        quotes[ticker] = float(series.iloc[-1])

            portfolio["last_price"] = portfolio["ticker"].map(quotes)
            portfolio["market_value"] = (
                portfolio["shares"] * portfolio["last_price"]
            )

            priced = portfolio.dropna(
                subset=["market_value"]
            )

            st.metric(
                "Value of priced holdings",
                f"${priced['market_value'].sum():,.2f}"
            )

            st.dataframe(
                portfolio,
                use_container_width=True,
                hide_index=True
            )

            if not priced.empty:
                fig = px.pie(
                    priced,
                    names="ticker",
                    values="market_value",
                    title="Portfolio Allocation"
                )

                st.plotly_chart(fig, use_container_width=True)

            missing = portfolio[
                portfolio["last_price"].isna()
            ]

            if not missing.empty:
                st.warning(
                    "Some holdings could not be priced. "
                    "Your total is incomplete."
                )
                st.dataframe(missing)

        except Exception as exc:
            st.error(
                f"Unable to process portfolio: {type(exc).__name__}"
            )

# ==================================================
# METHODOLOGY
# ==================================================

else:

    st.header("Strategy Methodology")

    st.markdown("""
### How the system works

1. Yahoo Finance supplies recent historical market prices.
2. Each model applies its own rules to select investments.
3. The program calculates target allocations.
4. Simulated BUY and SELL transactions are recorded.
5. Portfolio values and cash are stored in Supabase.
6. The dashboard compares saved results across models.

### Understanding the indicators

**Momentum:** How much an investment has risen or fallen
over a recent period. Higher momentum can indicate
relative strength, but may reverse.

**Moving average:** The average price over a set number
of trading days. A price above its moving average can
indicate a positive trend.

**Volatility:** How much daily returns fluctuate.
Higher volatility generally indicates greater price risk.

**Allocation:** The percentage of a portfolio assigned
to each investment.

**Drawdown:** The percentage decline from a previous
portfolio peak.

**Benchmark:** A reference investment used to judge
relative performance, such as VOO or QQQ.

### Trading limitations

- All strategies are simulated.
- Models start with $10,000.
- A 0.05% fee is charged per simulated trade.
- The system does not place brokerage orders.
- Prices may be delayed.
- Execution assumes observed prices are available.
- No model guarantees positive returns.
- Institutional names describe conceptual approximations.
- Congressional disclosure data is not connected.
- Model results are not a validated historical backtest.

### Live updates

The dashboard reads Supabase when refreshed.
Running the models generates and stores new transactions
when their rebalance rules permit.

For unattended daily execution, a separate scheduled
GitHub Actions workflow is required.
""")
