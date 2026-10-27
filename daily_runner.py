
"""Scheduled paper-trading and real-portfolio valuation job."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text

from engine import db, init
import strategy_lab as lab


def main():
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5 or (now.hour, now.minute) < (16, 20):
        print("Not an eligible after-close weekday; skipping.")
        return

    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL GitHub Actions secret is missing.")

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

    snapshots = lab.load(engine, "lab_snapshots")
    completed = set()
    if not snapshots.empty:
        completed = set(snapshots.loc[
            pd.to_datetime(snapshots["run_date"]).dt.date == now.date(),
            "strategy"
        ])

    for name in lab.MODELS:
        if name in completed:
            print(name, "already completed today")
            continue
        try:
            print(lab.run_strategy(engine, name))
        except Exception as exc:
            print(name, "FAILED:", type(exc).__name__, str(exc)[:180])

    holdings = pd.read_sql_query(
        text("SELECT ticker, shares FROM personal_holdings"),
        engine
    )

    if holdings.empty:
        print("No personal holdings saved; skipping valuation.")
        return

    history = lab.fetch_history(holdings["ticker"].tolist(), "1mo")
    quotes = {}

    for ticker in holdings["ticker"]:
        if ticker in history.columns:
            s = history[ticker].dropna()
            if not s.empty and float(s.iloc[-1]) > 0:
                quotes[ticker] = float(s.iloc[-1])

    if len(quotes) != len(holdings):
        print(
            "Missing prices; skipping incomplete personal snapshot:",
            set(holdings["ticker"]) - set(quotes)
        )
        return

    value = sum(
        float(row.shares) * quotes[row.ticker]
        for row in holdings.itertuples()
    )

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO personal_snapshots
            (snapshot_date, market_value, priced_count, missing_count)
            VALUES (:d, :v, :n, 0)
            ON CONFLICT (snapshot_date) DO UPDATE SET
                market_value = EXCLUDED.market_value,
                priced_count = EXCLUDED.priced_count,
                missing_count = 0
        """), {
            "d": now.date(),
            "v": value,
            "n": len(holdings)
        })

    print("Personal holdings valuation:", round(value, 2))


if __name__ == "__main__":
    main()
