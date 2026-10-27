
import os
import logging
import hmac

import streamlit as st
import pandas as pd
import plotly.express as px

from engine import (
    db, init, read, run, prices, score_frame,
    holdings, WATCH, START_CASH
)

# ==================================================
# PAGE SETUP
# ==================================================

st.set_page_config(
    page_title="AI Investment Research Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Investment Research Lab")
st.caption(
    "Research + $10,000 virtual investment strategy "
    "| Educational only | No real trades"
)

# ==================================================
# SECURITY
# ==================================================

try:
    database_url = st.secrets.get("DATABASE_URL")
    app_password = st.secrets.get("APP_PASSWORD")
except (FileNotFoundError, KeyError):
    database_url = None
    app_password = None

if not app_password:
    st.error("APP_PASSWORD is missing from Streamlit Secrets.")
    st.stop()

password_input = st.text_input(
    "App access password",
    type="password"
)

if not hmac.compare_digest(
    password_input,
    str(app_password)
):
    st.stop()

if not database_url:
    st.error("DATABASE_URL is missing from Streamlit Secrets.")
    st.stop()

os.environ["DATABASE_URL"] = str(database_url)

# ==================================================
# DATABASE
# ==================================================

try:
    engine = db()
    init(engine)

    sn = read(engine, "snapshots")
    tr = read(engine, "trades")
    de = read(engine, "decisions")

except Exception as exc:
    st.error("Database connection or initialization failed.")
    st.write("Error type:", type(exc).__name__)

    error_text = str(exc).lower()

    if "password authentication failed" in error_text:
        st.warning("Supabase rejected the database password.")

    elif "tenant or user not found" in error_text:
        st.warning("Check the Supabase pooler username and hostname.")

    elif "timeout" in error_text:
        st.warning("Supabase connection timed out.")

    elif "could not translate host name" in error_text:
        st.warning("Supabase hostname could not be resolved.")

    else:
        st.info(
            "Check your Supabase connection settings "
            "and Streamlit Secrets."
        )

    logging.error(
        "Database error category: %s",
        type(exc).__name__
    )
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================

page = st.sidebar.radio(
    "Pages",
    [
        "Dashboard",
        "Daily Decisions",
        "Research",
        "Trade Log",
        "My Portfolio Comparison",
        "Data & Methodology"
    ]
)

with st.sidebar:
    if st.button("Refresh latest quotes / signals"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Market data may be delayed. "
        "Refreshing does not place trades."
    )

# ==================================================
# MARKET DATA
# ==================================================

@st.cache_data(ttl=300)
def market():
    return prices(WATCH)


@st.cache_data(ttl=300)
def signals():
    return score_frame(market())


@st.cache_data(ttl=300)
def get_portfolio_prices(tickers):
    """
    Retrieve prices for portfolio symbols.
    First try a batch download, then retry
    missing symbols individually.
    """

    import yfinance as yf

    tickers = list(tickers)
    found = {}
    errors = {}

    if not tickers:
        return found, errors

    try:
        data = yf.download(
            tickers,
            period="1mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:
                close = data["Close"]

            if isinstance(close, pd.Series):
                close = close.to_frame(name=tickers[0])

            for ticker in tickers:
                if ticker in close.columns:
                    series = pd.to_numeric(
                        close[ticker],
                        errors="coerce"
                    ).dropna()

                    if not series.empty:
                        price = float(series.iloc[-1])

                        if price > 0:
                            found[ticker] = price

    except Exception:
        pass

    # Retry failed symbols individually.
    for ticker in tickers:
        if ticker in found:
            continue

        try:
            history = yf.Ticker(ticker).history(
                period="1mo",
                auto_adjust=True
            )

            if history.empty or "Close" not in history:
                errors[ticker] = "No recent price data"
                continue

            series = pd.to_numeric(
                history["Close"],
                errors="coerce"
            ).dropna()

            if series.empty:
                errors[ticker] = "No valid closing prices"
                continue

            price = float(series.iloc[-1])

            if price <= 0:
                errors[ticker] = "Invalid closing price"
                continue

            found[ticker] = price

        except Exception:
            errors[ticker] = "Quote retrieval failed"

    return found, errors


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.subheader("Virtual portfolio performance")

    if sn.empty:
        st.info(
            "No performance snapshots recorded yet. "
            "Go to Daily Decisions to initialize "
            "your $10,000 virtual portfolio."
        )

    else:
        latest = sn.sort_values(
            "run_date"
        ).iloc[-1]

        a, b, c = st.columns(3)

        a.metric(
            "AI Strategy",
            f"${latest.equity:,.2f}",
            f"{(latest.equity / START_CASH - 1) * 100:+.2f}%"
        )

        b.metric(
            "VOO Benchmark",
            f"${latest.benchmark_voo:,.2f}"
        )

        c.metric(
            "QQQ Benchmark",
            f"${latest.benchmark_qqq:,.2f}"
        )

        chart_data = sn.sort_values(
            "run_date"
        ).rename(
            columns={
                "equity": "AI Strategy",
                "benchmark_voo": "VOO",
                "benchmark_qqq": "QQQ"
            }
        )

        fig = px.line(
            chart_data,
            x="run_date",
            y=["AI Strategy", "VOO", "QQQ"],
            title="Growth of $10,000"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        cash, positions = holdings(tr)

        st.subheader("Current virtual holdings")

        if positions:
            position_df = pd.DataFrame(
                [
                    {"Ticker": ticker, "Shares": qty}
                    for ticker, qty in positions.items()
                ]
            )

            st.dataframe(
                position_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No open virtual positions.")

        st.metric("Virtual cash", f"${cash:,.2f}")

    st.caption(
        "Returns are simulated using observed closing prices. "
        "They are not actual executable trading results."
    )

# ==================================================
# DAILY DECISIONS
# ==================================================

elif page == "Daily Decisions":

    st.subheader("Daily investment signals")

    try:
        ranking = signals()

        if ranking.empty:
            st.warning("No valid signals available.")

        else:
            st.dataframe(
                ranking,
                use_container_width=True,
                hide_index=True
            )

        st.warning(
            "This is a deterministic momentum strategy, "
            "not an LLM. Simulated fills use observed closing "
            "prices and may overstate achievable performance."
        )

        if st.button(
            "Run / record today's virtual decisions",
            type="primary"
        ):
            with st.spinner("Running virtual strategy..."):
                result = run(engine)

            st.success("Strategy execution completed.")
            st.write(result)
            st.cache_data.clear()

    except Exception as exc:
        st.error("Strategy execution failed.")
        st.write("Error type:", type(exc).__name__)

    if not de.empty:
        st.subheader("Saved investment decisions")

        st.dataframe(
            de.sort_values(
                "run_date",
                ascending=False
            ).head(50),
            use_container_width=True,
            hide_index=True
        )

# ==================================================
# RESEARCH
# ==================================================

elif page == "Research":

    st.subheader("Investment research")

    ticker = st.selectbox(
        "Select investment",
        WATCH
    )

    try:
        data = market()

        if ticker not in data.columns:
            st.warning("Ticker unavailable.")
        else:
            history = data[ticker].dropna()

            if history.empty:
                st.warning("No price history available.")
            else:
                st.line_chart(history)

                ranking = signals()

                if not ranking.empty:
                    selected = ranking[
                        ranking.ticker == ticker
                    ]

                    if not selected.empty:
                        st.write(
                            selected.iloc[0].reason
                        )

    except Exception as exc:
        st.error("Unable to retrieve research data.")
        st.write("Error type:", type(exc).__name__)

# ==================================================
# TRADE LOG
# ==================================================

elif page == "Trade Log":

    st.subheader("Virtual trade history")

    if tr.empty:
        st.info("No trades recorded yet.")
    else:
        st.dataframe(
            tr.sort_values(
                "id",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    st.download_button(
        "Download trades CSV",
        tr.to_csv(index=False),
        file_name="trades.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download snapshots CSV",
        sn.to_csv(index=False),
        file_name="snapshots.csv",
        mime="text/csv"
    )

# ==================================================
# MY PORTFOLIO COMPARISON
# ==================================================

elif page == "My Portfolio Comparison":

    st.subheader("💼 My Actual Investment Portfolio")

    st.write(
        "Upload your brokerage holdings CSV to calculate "
        "current portfolio value, asset allocation, "
        "and individual holding values."
    )

    st.caption(
        "Required columns: ticker, shares. "
        "Optional column: cost_basis. "
        "Uploaded holdings are processed in memory "
        "and are not saved to Supabase."
    )

    uploaded = st.file_uploader(
        "Upload holdings CSV",
        type=["csv"]
    )

    if uploaded is not None:

        try:
            portfolio = pd.read_csv(uploaded)

            portfolio.columns = (
                portfolio.columns
                .str.strip()
                .str.lower()
            )

            required = {"ticker", "shares"}

            if not required.issubset(portfolio.columns):
                st.error(
                    "CSV must contain ticker and shares columns."
                )
                st.stop()

            portfolio["ticker"] = (
                portfolio["ticker"]
                .astype("string")
                .str.upper()
                .str.strip()
            )

            portfolio["shares"] = pd.to_numeric(
                portfolio["shares"],
                errors="coerce"
            )

            portfolio = portfolio.dropna(
                subset=["ticker", "shares"]
            )

            portfolio = portfolio[
                portfolio["ticker"] != ""
            ]

            portfolio = portfolio[
                portfolio["shares"] >= 0
            ].copy()

            if portfolio.empty:
                st.error("No valid holdings found.")
                st.stop()

            if len(portfolio) > 100:
                st.error(
                    "Maximum 100 holdings per upload."
                )
                st.stop()

            tickers = tuple(
                portfolio["ticker"].unique().tolist()
            )

            with st.spinner("Retrieving market prices..."):
                quote_map, quote_errors = (
                    get_portfolio_prices(tickers)
                )

            portfolio["last_price"] = (
                portfolio["ticker"].map(quote_map)
            )

            portfolio["market_value"] = (
                portfolio["shares"]
                * portfolio["last_price"]
            )

            valid = portfolio[
                portfolio["last_price"].notna()
            ].copy()

            missing = portfolio[
                portfolio["last_price"].isna()
            ].copy()

            # --------------------------------------
            # PORTFOLIO SUMMARY
            # --------------------------------------

            st.subheader("Portfolio overview")

            if valid.empty:
                st.error(
                    "No holdings could be valued. "
                    "Check the ticker symbols or try again later."
                )
            else:
                total_value = float(
                    valid["market_value"].sum()
                )

                a, b, c = st.columns(3)

                a.metric(
                    "Priced Holdings Value",
                    f"${total_value:,.2f}"
                )

                b.metric(
                    "Holdings Priced",
                    len(valid)
                )

                c.metric(
                    "Holdings Missing Prices",
                    len(missing)
                )

                if not missing.empty:
                    st.warning(
                        "The displayed total is incomplete "
                        "because some holdings could not be priced."
                    )

                # ----------------------------------
                # HOLDINGS TABLE
                # ----------------------------------

                st.subheader("Current holdings")

                display = valid.sort_values(
                    "market_value",
                    ascending=False
                )

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "last_price": st.column_config.NumberColumn(
                            "Last Price",
                            format="$%.2f"
                        ),
                        "market_value": st.column_config.NumberColumn(
                            "Market Value",
                            format="$%.2f"
                        )
                    }
                )

                # ----------------------------------
                # ALLOCATION
                # ----------------------------------

                st.subheader("Portfolio allocation")

                allocation = (
                    valid.groupby(
                        "ticker",
                        as_index=False
                    )["market_value"]
                    .sum()
                )

                fig = px.pie(
                    allocation,
                    names="ticker",
                    values="market_value",
                    title="Portfolio Allocation by Holding",
                    hole=0.45
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # ----------------------------------
                # COST BASIS (OPTIONAL)
                # ----------------------------------

                if "cost_basis" in valid.columns:

                    st.subheader("Cost basis information")

                    st.caption(
                        "Cost basis is retained from your CSV. "
                        "It is not used for gain/loss calculations "
                        "because the file may contain per-share "
                        "or total-position cost basis."
                    )

            # --------------------------------------
            # UNPRICED HOLDINGS
            # --------------------------------------

            if not missing.empty:

                st.subheader("⚠️ Holdings needing attention")

                missing["issue"] = (
                    missing["ticker"].map(quote_errors)
                    .fillna("No price available")
                )

                st.dataframe(
                    missing[
                        ["ticker", "shares", "issue"]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    "These holdings were not deleted from "
                    "your uploaded data. They were excluded "
                    "from the priced portfolio total only."
                )

            # --------------------------------------
            # DOWNLOAD RESULTS
            # --------------------------------------

            st.download_button(
                "Download portfolio valuation CSV",
                portfolio.to_csv(index=False),
                file_name="portfolio_valuation.csv",
                mime="text/csv"
            )

            st.caption(
                "Values are based on the latest available "
                "Yahoo Finance closing prices, not guaranteed "
                "real-time quotes. This is a current valuation, "
                "not a historical performance comparison."
            )

        except Exception as exc:
            st.error("Portfolio processing failed.")
            st.write("Error type:", type(exc).__name__)
            st.info(
                "Check your CSV formatting and try again. "
                "If the problem continues, share the error type."
            )

# ==================================================
# METHODOLOGY
# ==================================================

else:

    st.subheader("Methodology and limitations")

    st.markdown("""
### Investment Strategy

- Starting virtual capital: **$10,000**
- Investment universe: stocks, ETFs, bond ETFs and crypto
- Selection: momentum, moving averages and volatility
- Maximum target positions: four
- Initial target allocation: approximately 22.5% per position
- Target cash reserve: 10%
- No leverage or short selling

### Benchmarks

- VOO: S&P 500 ETF
- QQQ: Nasdaq-100 ETF
- Each benchmark starts with a hypothetical $10,000

### Data and execution

- Market data: Yahoo Finance
- Paper trades use observed daily closing prices
- Trading friction: 5 basis points per trade
- Same-close simulation is not a realistically executable strategy
- GitHub Actions can run scheduled trading decisions
- Supabase stores simulated trades and snapshots

### Actual portfolio

- Upload a CSV with `ticker,shares`
- `cost_basis` is optional
- Invalid or unavailable symbols are identified separately
- Current market values are estimated using available quotes
- Uploads are not saved to the database

### Important

This is a rules-based research and paper-trading platform,
not a brokerage trading system or guaranteed profitable AI.
""")
