"""Record timestamped intraday valuations for existing paper portfolios; NEVER trade.

Uses the latest daily strategy positions and recent Yahoo Finance 5-minute bars.
Runs unattended in GitHub Actions; raises on missing required inputs or zero saves.
"""
import json
import os
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from sqlalchemy import text

from engine import db
import strategy_lab as lab

ET = ZoneInfo("America/New_York")
MAX_QUOTE_AGE_MINUTES = 45


def get_quotes(tickers, now):
    quotes = {}
    for ticker in sorted(set(tickers)):
        try:
            bars = yf.download(
                ticker, period="5d", interval="5m", auto_adjust=True,
                progress=False, threads=False, prepost=False,
            )
            if bars.empty:
                print(f"QUOTE MISSING {ticker}: empty response", flush=True)
                continue
            close = bars["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna()
            close = close[close > 0]
            if close.empty:
                print(f"QUOTE MISSING {ticker}: no valid closing prices", flush=True)
                continue
            stamp = pd.Timestamp(close.index[-1])
            if stamp.tzinfo is None:
                print(f"QUOTE MISSING {ticker}: timestamp has no timezone", flush=True)
                continue
            stamp = stamp.tz_convert("UTC").to_pydatetime()
            age = (now - stamp).total_seconds() / 60
            if stamp.astimezone(ET).date() != now.astimezone(ET).date():
                print(f"QUOTE STALE {ticker}: last bar from {stamp.isoformat()}", flush=True)
                continue
            if not -10 <= age <= MAX_QUOTE_AGE_MINUTES:
                print(f"QUOTE STALE {ticker}: age {age:.1f} minutes", flush=True)
                continue
            quotes[ticker] = (float(close.iloc[-1]), stamp)
            print(f"QUOTE OK {ticker}: ${quotes[ticker][0]:.4f} | {stamp.isoformat()} | age {age:.1f}m", flush=True)
        except Exception as exc:
            print(f"QUOTE ERROR {ticker}: {type(exc).__name__}: {exc}", flush=True)
    return quotes


def main():
    now = datetime.now(timezone.utc)
    local = now.astimezone(ET)
    print(f"Collector started: {now.isoformat()} UTC | {local.isoformat()} ET", flush=True)
    if local.weekday() >= 5 or not time(9, 35) <= local.time() <= time(16, 5):
        print("SKIP: Outside regular weekday US market hours.", flush=True)
        return
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL GitHub Actions secret is missing")

    engine = db()
    lab.initialize(engine)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS lab_intraday_snapshots (
                strategy TEXT NOT NULL,
                observed_at TIMESTAMPTZ NOT NULL,
                equity DOUBLE PRECISION NOT NULL,
                cash DOUBLE PRECISION NOT NULL,
                voo_price DOUBLE PRECISION,
                qqq_price DOUBLE PRECISION,
                price_asof TIMESTAMPTZ NOT NULL,
                missing_tickers TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (strategy, observed_at)
            )
        """))

    snapshots = lab.load(engine, "lab_snapshots")
    print(f"Daily strategy snapshots found: {len(snapshots)}", flush=True)
    if snapshots.empty:
        raise RuntimeError("No daily strategy snapshots exist. Run the existing Daily Strategy Lab Update workflow first.")

    latest = snapshots.sort_values("run_date").groupby("strategy", sort=False).tail(1)
    print(f"Strategies with saved positions: {len(latest)} / {len(lab.MODELS)}", flush=True)
    holdings_by_strategy = {}
    symbols = {"VOO", "QQQ"}
    for row in latest.itertuples():
        try:
            raw = json.loads(row.positions_json)
            if not isinstance(raw, dict):
                raise ValueError("positions_json is not an object")
            holdings = {str(t): float(q) for t, q in raw.items() if float(q) > 0}
            holdings_by_strategy[row.strategy] = holdings
            symbols.update(holdings)
            print(f"HOLDINGS {row.strategy}: {len(holdings)} tickers; cash ${float(row.cash):,.2f}; snapshot {row.run_date}", flush=True)
        except (ValueError, TypeError, AttributeError) as exc:
            print(f"INVALID HOLDINGS {row.strategy}: {exc}", flush=True)

    if not holdings_by_strategy:
        raise RuntimeError("No usable strategy holdings could be parsed from daily snapshots.")

    print(f"Fetching {len(symbols)} distinct ticker quotes...", flush=True)
    quotes = get_quotes(symbols, now)
    print(f"Usable quotes: {len(quotes)} / {len(symbols)}", flush=True)
    if not quotes:
        raise RuntimeError("No fresh Yahoo Finance intraday quotes. Check market holiday, provider access or data delay.")

    observed_at = now.replace(second=0, microsecond=0)
    saved = 0
    for row in latest.itertuples():
        holdings = holdings_by_strategy.get(row.strategy)
        if holdings is None:
            continue
        missing = sorted(set(holdings) - set(quotes))
        if missing:
            print(f"SKIP {row.strategy}: missing fresh quotes: {', '.join(missing)}", flush=True)
            continue
        cash = float(row.cash)
        equity = cash + sum(shares * quotes[t][0] for t, shares in holdings.items())
        if not 0 < equity < float("inf"):
            print(f"SKIP {row.strategy}: invalid equity {equity}", flush=True)
            continue
        quote_times = [quotes[t][1] for t in holdings]
        quote_times += [quotes[t][1] for t in ("VOO", "QQQ") if t in quotes]
        price_asof = min(quote_times) if quote_times else now
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO lab_intraday_snapshots
                    (strategy, observed_at, equity, cash, voo_price, qqq_price, price_asof, missing_tickers)
                VALUES (:strategy, :observed_at, :equity, :cash, :voo, :qqq, :price_asof, '')
                ON CONFLICT (strategy, observed_at) DO UPDATE SET
                    equity=EXCLUDED.equity,
                    cash=EXCLUDED.cash,
                    voo_price=EXCLUDED.voo_price,
                    qqq_price=EXCLUDED.qqq_price,
                    price_asof=EXCLUDED.price_asof
            """), {
                "strategy": row.strategy, "observed_at": observed_at,
                "equity": equity, "cash": cash,
                "voo": quotes.get("VOO", (None, None))[0],
                "qqq": quotes.get("QQQ", (None, None))[0],
                "price_asof": price_asof,
            })
        saved += 1
        print(f"SAVED: {row.strategy} | equity ${equity:,.2f} | {observed_at.isoformat()}", flush=True)

    print(f"RESULT: saved {saved} / {len(latest)} available strategy valuations", flush=True)
    if saved == 0:
        raise RuntimeError("ZERO valuations saved; inspect QUOTE STALE / QUOTE MISSING / SKIP lines above.")
    if saved < len(latest):
        print("WARNING: Some strategies skipped due to missing quotes or invalid holdings.", flush=True)


if __name__ == "__main__":
    main()
