
"""Scheduled daily paper trading and personal portfolio valuation.

Runs the six original strategies after the US market closes.
The AI Intraday Trader runs separately through run_intraday().
Existing Supabase positions and history are preserved.
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text

from engine import db, init
import strategy_lab as lab


ET = ZoneInfo("America/New_York")


def ensure_personal_tables(engine):
    """Create missing tables without deleting existing data."""

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


def run_daily_strategies(engine, today):
    """Run the six daily strategies, excluding the intraday bot."""

    snapshots = lab.load(engine, "lab_snapshots")

    completed = set()

    if not snapshots.empty:
        dates = pd.to_datetime(
            snapshots["run_date"],
            errors="coerce"
        ).dt.date

        completed = set(
            snapshots.loc[
                dates == today,
                "strategy"
            ]
        )

    results = []
    failures = []

    for name in lab.MODELS:

        # Intraday trading has its own execution workflow.
        if name == lab.DAY_BOT:
            continue

        if name in completed:
            print(f"SKIP: {name} already completed today")
            continue

        try:
            result = lab.run_strategy(engine, name)

            results.append(result)

            print(
                f"SUCCESS: {name} | "
                f"Equity: ${result.get('equity', 0):,.2f} | "
                f"Trades: {result.get('trades', 0)} | "
                f"Reason: {result.get('reason', '')}"
            )

        except Exception as exc:
            message = (
                f"{name}: "
                f"{type(exc).__name__}: {exc}"
            )

            failures.append(message)

            print(f"FAILED: {message}")

    print()
    print("DAILY STRATEGY SUMMARY")
    print("----------------------")
    print(f"Strategies processed: {len(results)}")
    print(f"Strategies already completed: {len(completed & (set(lab.MODELS) - {lab.DAY_BOT}))}")
    print(f"New paper trades: {sum(r.get('trades', 0) for r in results)}")
    print(f"Strategy failures: {len(failures)}")

    return failures


def value_personal_holdings(engine, today):
    """Record the personal portfolio's daily market value."""

    holdings = pd.read_sql_query(
        text("""
            SELECT ticker, shares
            FROM personal_holdings
            WHERE shares > 0
        """),
        engine
    )

    if holdings.empty:
        print(
            "PERSONAL PORTFOLIO: "
            "No positive-share holdings saved."
        )
        return

    tickers = holdings["ticker"].tolist()

    history = lab.fetch_history(
        tickers,
        "1mo"
    )

    if history.empty:
        print(
            "PERSONAL PORTFOLIO: "
            "Market data unavailable; no snapshot saved."
        )
        return

    quotes = {}

    for ticker in tickers:
        if ticker not in history.columns:
            continue

        series = history[ticker].dropna()

        if series.empty:
            continue

        price = float(series.iloc[-1])

        if price > 0:
            quotes[ticker] = price

    missing = set(tickers) - set(quotes)

    if missing:
        print(
            "PERSONAL PORTFOLIO: "
            "Incomplete pricing; snapshot skipped.",
            sorted(missing)
        )
        return

    value = sum(
        float(row.shares) * quotes[row.ticker]
        for row in holdings.itertuples()
    )

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO personal_snapshots
            (
                snapshot_date,
                market_value,
                priced_count,
                missing_count
            )
            VALUES (:d, :v, :n, 0)

            ON CONFLICT (snapshot_date)
            DO UPDATE SET
                market_value = EXCLUDED.market_value,
                priced_count = EXCLUDED.priced_count,
                missing_count = 0
        """), {
            "d": today,
            "v": value,
            "n": len(holdings)
        })

    print(
        f"PERSONAL PORTFOLIO: "
        f"${value:,.2f} | "
        f"{len(holdings)} priced holdings"
    )


def main():

    now = datetime.now(ET)
    today = now.date()

    print("=" * 60)
    print("AI INVESTMENT RESEARCH LAB")
    print("DAILY AUTOMATED STRATEGY UPDATE")
    print("=" * 60)

    print("Eastern Time:", now.isoformat())

    if now.weekday() >= 5:
        print("Weekend: daily trading skipped.")
        return

    if (now.hour, now.minute) < (16, 20):
        print(
            "Market-close processing starts "
            "at 4:20 PM Eastern."
        )
        return

    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError(
            "DATABASE_URL GitHub Actions secret is missing."
        )

    engine = db()

    init(engine)
    lab.initialize(engine)
    ensure_personal_tables(engine)

    failures = run_daily_strategies(
        engine,
        today
    )

    value_personal_holdings(
        engine,
        today
    )

    if failures:
        raise RuntimeError(
            f"{len(failures)} daily strategies failed: "
            + " | ".join(failures)
        )

    print()
    print("Daily update finished successfully.")


if __name__ == "__main__":
    main()
