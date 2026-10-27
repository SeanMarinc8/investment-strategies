
import os
import logging

import streamlit as st
import pandas as pd
import plotly.express as px

from engine import (
    db, init, read, run, prices, score_frame,
    holdings, WATCH, START_CASH
)

st.set_page_config(
    page_title="AI Investment Research Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Investment Research Lab")
st.caption(
    "Research + auditable $10,000 virtual strategy | "
    "Educational only, no real orders"
)

# -------------------------------------------------
# APP SECURITY
# -------------------------------------------------

try:
    database_url = st.secrets.get("DATABASE_URL")
    app_password = st.secrets.get("APP_PASSWORD")
except (FileNotFoundError, KeyError):
    database_url = None
    app_password = None

if database_url:
    os.environ["DATABASE_URL"] = str(database_url)

if app_password:
    pwd = st.text_input(
        "App access password",
        type="password"
    )

    if pwd != app_password:
        st.stop()
else:
    st.warning(
        "No APP_PASSWORD set. Do not publish this app "
        "with private holdings."
    )

# -------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------

if not database_url:
    st.error("DATABASE_URL is missing from Streamlit Secrets.")
    st.info(
        "Go to Manage app → Settings → Secrets and "
        "add your Supabase PostgreSQL connection string."
    )
    st.stop()

try:
    engine = db()
    init(engine)

except Exception as exc:
    st.error("Database connection failed.")
    st.write("Error type:", type(exc).__name__)

    # Log only the exception category here. Full exception
    # strings can contain hostnames or other sensitive details.
    logging.error(
        "Database initialization failed: %s",
        type(exc).__name__
    )

    # Show selected safe diagnostic information.
    error_text = str(exc).lower()

    if "password authentication failed" in error_text:
        st.warning(
            "Supabase rejected the database password. "
            "Check the password in DATABASE_URL."
        )

    elif "could not translate host name" in error_text:
        st.warning(
            "The database hostname could not be resolved. "
            "Copy a fresh connection string from Supabase Connect."
        )

    elif "connection timed out" in error_text or "timeout expired" in error_text:
        st.warning(
            "The database connection timed out. "
            "Check Supabase project status and pooler configuration."
        )

    elif "connection refused" in error_text:
        st.warning(
            "The database server refused the connection. "
            "Check the hostname and port."
        )

    elif "no route to host" in error_text:
        st.warning(
            "The database host is unreachable from Streamlit Cloud."
        )

    elif "ssl" in error_text:
        st.warning(
            "There may be a PostgreSQL SSL configuration issue."
        )

    elif "permission denied" in error_text:
        st.warning(
            "The database user may not have permission "
            "to create or access the required tables."
        )

    else:
        st.info(
            "The database failed to initialize. "
            "Check your Supabase connection settings. "
            "Share the error type shown above for further diagnosis."
        )

    st.stop()

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------

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
        "Quotes are delayed/end-of-day, not guaranteed "
        "real-time. Refresh never automatically trades."
    )

# -------------------------------------------------
# MARKET DATA
# -------------------------------------------------

@st.cache_data(ttl=300)
def market():
    return prices(WATCH)


@st.cache_data(ttl=300)
def signals():
    return score_frame(market())


# -------------------------------------------------
# READ SAVED DATABASE DATA
# -------------------------------------------------

try:
    sn = read(engine, "snapshots")
    tr = read(engine, "trades")
    de = read(engine, "decisions")

except Exception as exc:
    st.error("Unable to read saved investment data.")
    st.write("Error type:", type(exc).__name__)
    st.stop()

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------

if page == "Dashboard":

    st.subheader(
        "Virtual portfolio vs buy-and-hold benchmarks"
    )

    if sn.empty:
        st.info(
            "No trading-day snapshots yet. "
            "Open Daily Decisions and click Run strategy "
            "once to initialize the challenge."
        )

    else:
        latest = sn.sort_values("run_date").iloc[-1]

        a, b, c = st.columns(3)

        a.metric(
            "AI virtual equity",
            f"${latest.equity:,.2f}",
            f"{(latest.equity / START_CASH - 1) * 100:+.2f}%"
        )

        b.metric(
            "VOO hypothetical value",
            f"${latest.benchmark_voo:,.2f}"
        )

        c.metric(
            "QQQ hypothetical value",
            f"${latest.benchmark_qqq:,.2f}"
        )

        plot = sn.sort_values("run_date").rename(
            columns={
                "equity": "AI Strategy",
                "benchmark_voo": "VOO",
                "benchmark_qqq": "QQQ"
            }
        )

        st.plotly_chart(
            px.line(
                plot,
                x="run_date",
                y=["AI Strategy", "VOO", "QQQ"],
                title="Growth of $10,000 since strategy inception"
            ),
            use_container_width=True
        )

        cash, h = holdings(tr)

        st.write(
            "Current virtual positions:",
            h,
            "Cash:",
            round(cash, 2)
        )

    st.info(
        "Performance begins when you first initialize. "
        "No invented history. Trading uses observed daily "
        "closing prices as a simulation proxy."
    )

# -------------------------------------------------
# DAILY DECISIONS
# -------------------------------------------------

elif page == "Daily Decisions":

    st.subheader(
        "Daily candidate ranking and paper decisions"
    )

    try:
        s = signals()

        st.dataframe(
            s,
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "Model: 50/200-day trend + 20-day momentum "
            "minus volatility. Deterministic rules, NOT an LLM. "
            "Trades simulate at the last daily close. "
            "Actual same-close execution is impossible after "
            "observing that close."
        )

        if st.button(
            "Run / record today's virtual decisions",
            type="primary"
        ):
            result = run(engine)
            st.write(result)
            st.cache_data.clear()
            st.rerun()

    except Exception as exc:
        st.error(
            "Market data or strategy execution unavailable."
        )
        st.write("Error type:", type(exc).__name__)

    if not de.empty:
        st.dataframe(
            de.sort_values(
                "run_date",
                ascending=False
            ).head(50),
            use_container_width=True,
            hide_index=True
        )

# -------------------------------------------------
# RESEARCH
# -------------------------------------------------

elif page == "Research":

    st.subheader(
        "Research stocks, bond ETFs and crypto"
    )

    ticker = st.selectbox(
        "Symbol",
        WATCH
    )

    try:
        df = market()[ticker].dropna()

        st.line_chart(df)

        row = signals()
        r = row[row.ticker == ticker]

        if not r.empty:
            st.write(r.iloc[0].reason)

        st.caption(
            "Bonds are represented by ETFs "
            "(TLT, IEF, SHY); crypto by BTC-USD "
            "and ETH-USD. ETF prices are not "
            "individual bond yields."
        )

    except Exception as exc:
        st.error("Unable to retrieve quotes.")
        st.write("Error type:", type(exc).__name__)

# -------------------------------------------------
# TRADE LOG
# -------------------------------------------------

elif page == "Trade Log":

    st.subheader(
        "Paper trade audit log"
    )

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
        "trades.csv",
        "text/csv"
    )

    st.download_button(
        "Download daily snapshots CSV",
        sn.to_csv(index=False),
        "snapshots.csv",
        "text/csv"
    )

# -------------------------------------------------
# PERSONAL PORTFOLIO COMPARISON
# -------------------------------------------------

elif page == "My Portfolio Comparison":

    st.subheader(
        "Compare against your actual holdings"
    )

    st.write(
        "Upload a CSV with columns `ticker,shares` "
        "(optional `cost_basis`). The uploaded file "
        "is processed in memory and is not saved "
        "to the database."
    )

    f = st.file_uploader(
        "Holdings CSV",
        type="csv"
    )

    if f:
        try:
            x = pd.read_csv(f)

            assert {
                "ticker",
                "shares"
            }.issubset(x.columns)

            x["ticker"] = (
                x.ticker.astype(str)
                .str.upper()
                .str.strip()
            )

            x["shares"] = pd.to_numeric(
                x.shares,
                errors="raise"
            )

            assert (x.shares >= 0).all()
            assert len(x) < 101

            current = prices(
                x.ticker.unique().tolist(),
                period="1mo"
            )

            x["last_price"] = x.ticker.map({
                t: float(
                    current[t].dropna().iloc[-1]
                )
                for t in x.ticker.unique()
            })

            x["value"] = (
                x.shares * x.last_price
            )

            st.metric(
                "Current uploaded holdings market value",
                f"${x.value.sum():,.2f}"
            )

            st.dataframe(
                x,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "This is a point-in-time valuation, "
                "not a valid historical performance "
                "comparison. Historical comparison "
                "requires dated valuations and "
                "adjustments for deposits and withdrawals."
            )

        except Exception:
            st.error(
                "Check CSV columns, tickers, "
                "and share quantities."
            )

# -------------------------------------------------
# METHODOLOGY
# -------------------------------------------------

else:

    st.subheader(
        "Methodology and limitations"
    )

    st.markdown("""
- **Data:** Yahoo Finance via yfinance; may be delayed, revised or unavailable.
- **Universe:** Fixed list of stocks, bond ETFs and cryptocurrencies in `engine.py`.
- **Selection:** 200-day trend, 50-day trend, 20-day momentum and realized volatility.
- **Model:** Rules-based strategy, not a generative AI model.
- **Execution:** Simulated fills at observed closing prices, with 5 basis points of friction. This is not realistically executable after observing the same close.
- **Portfolio:** Up to four candidates, approximately 22.5% initial target per holding and a 10% cash buffer.
- **Risk:** No leverage or short selling.
- **Benchmarks:** Separate hypothetical $10,000 investments in VOO and QQQ from strategy inception.
- **Persistence:** PostgreSQL database hosted on Supabase.
- **Automation:** GitHub Actions can execute the strategy on a schedule, subject to data availability.
- **Limitations:** No taxes, reliable intraday execution, market impact or guaranteed profitability.
- **Purpose:** Educational research and forward paper trading, not autonomous brokerage trading.
""")
