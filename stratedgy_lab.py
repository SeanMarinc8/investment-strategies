
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import text

STARTING_CASH = 10000.0

MODELS = {
    "AI Goldman Sachs": {
        "description": "Quality and defensive-factor approximation, not a Goldman Sachs proprietary strategy.",
        "tickers": ["MSFT", "AAPL", "V", "COST", "BRK-B", "JNJ", "PG", "GOOGL"],
        "method": "Select liquid, established companies using recent trend, momentum and volatility."
    },
    "AI Pelosi Disclosure-Inspired": {
        "description": "Technology-oriented basket, not verified live Pelosi trades or holdings.",
        "tickers": ["NVDA", "GOOGL", "MSFT", "AMZN", "AAPL", "AVGO", "CRM", "PANW"],
        "method": "Equal-weight a technology-heavy watchlist. No assumption of actual congressional transactions."
    },
    "AI JPMorgan Chase": {
        "description": "Diversified multi-asset allocation proxy, not JPMorgan proprietary holdings.",
        "tickers": ["VOO", "QQQ", "BND", "IEF", "GLD", "SCHD", "VEA", "VWO"],
        "method": "Balance stocks, bonds, international equities and gold with fixed allocation targets."
    },
    "AI Momentum": {
        "description": "Trend-following model seeking persistent relative strength.",
        "tickers": ["NVDA", "PLTR", "META", "AMZN", "AVGO", "QQQ", "VRT", "MSFT"],
        "method": "Rank positive-trend securities by 63-day return, selecting up to four."
    },
    "AI Defensive Income": {
        "description": "Defensive allocation emphasizing bonds, dividend ETFs and gold.",
        "tickers": ["SCHD", "VIG", "BND", "IEF", "SGOV", "GLD"],
        "method": "Maintain diversified defensive positions with approximately equal allocations."
    },
    "AI Equal-Weight Growth": {
        "description": "Growth-company exposure without market-cap concentration.",
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA"],
        "method": "Equal-weight a predefined growth basket."
    }
}


def initialize(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS lab_trades (
                id BIGSERIAL PRIMARY KEY,
                strategy TEXT NOT NULL,
                trade_date DATE NOT NULL,
                executed_at TIMESTAMPTZ NOT NULL,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                shares DOUBLE PRECISION NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                fee DOUBLE PRECISION NOT NULL DEFAULT 0,
                reason TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS lab_snapshots (
                strategy TEXT NOT NULL,
                run_date DATE NOT NULL,
                equity DOUBLE PRECISION NOT NULL,
                cash DOUBLE PRECISION NOT NULL,
                positions_json TEXT NOT NULL,
                PRIMARY KEY (strategy, run_date)
            )
        """))


def load(engine, table):
    if table not in ("lab_trades", "lab_snapshots"):
        raise ValueError("Invalid table")
    return pd.read_sql_query(
        text(f"SELECT * FROM {table}"),
        engine
    )


def fetch_history(tickers, period="1y"):
    symbols = list(dict.fromkeys(tickers))
    if not symbols:
        return pd.DataFrame()

    raw = yf.download(
        symbols,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = [symbols[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])

    return close.apply(
        pd.to_numeric, errors="coerce"
    ).sort_index().ffill()


def select_weights(name, history):
    config = MODELS[name]
    symbols = [
        t for t in config["tickers"]
        if t in history.columns
        and history[t].dropna().shape[0] >= 65
    ]

    if not symbols:
        return {}, "Insufficient market history."

    if name in ("AI Momentum", "AI Goldman Sachs"):
        candidates = []

        for ticker in symbols:
            series = history[ticker].dropna()
            if len(series) < 65:
                continue

            momentum = float(series.iloc[-1] / series.iloc[-64] - 1)
            moving_average = float(series.tail(50).mean())
            volatility = float(series.pct_change().tail(63).std())

            if series.iloc[-1] <= moving_average:
                continue

            if name == "AI Goldman Sachs":
                score = momentum - 2.0 * volatility
            else:
                score = momentum

            candidates.append((ticker, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        chosen = [item[0] for item in candidates[:4]]

        if not chosen:
            return {}, "No securities passed the positive-trend filter."

        weight = 0.90 / len(chosen)
        return (
            {ticker: weight for ticker in chosen},
            "Selected positive-trend assets using momentum and volatility."
        )

    weight = 0.90 / len(symbols)
    return (
        {ticker: weight for ticker in symbols},
        "Equal-weight allocation with a 10% target cash reserve."
    )


def latest_state(engine, strategy):
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT cash, positions_json
            FROM lab_snapshots
            WHERE strategy = :strategy
            ORDER BY run_date DESC
            LIMIT 1
        """), {"strategy": strategy}).mappings().first()

    if row is None:
        return STARTING_CASH, {}

    return (
        float(row["cash"]),
        json.loads(row["positions_json"])
    )


def run_strategy(engine, name):
    if name not in MODELS:
        raise ValueError("Unknown strategy")

    today = date.today()
    now = datetime.now(timezone.utc)

    history = fetch_history(MODELS[name]["tickers"])
    if history.empty:
        raise RuntimeError("Market data unavailable.")

    latest = {}
    for ticker in history.columns:
        values = history[ticker].dropna()
        if not values.empty:
            latest[ticker] = float(values.iloc[-1])

    cash, positions = latest_state(engine, name)

    equity = cash + sum(
        float(qty) * latest.get(ticker, 0)
        for ticker, qty in positions.items()
    )

    if equity <= 0:
        raise RuntimeError("Invalid portfolio equity.")

    weights, rationale = select_weights(name, history)
    actions = []

    # Rebalance only on the first recorded run and on Mondays.
    # On other days, record a valuation snapshot without trades.
    trades = load(engine, "lab_trades")
    prior_trades = (
        trades[trades["strategy"] == name]
        if not trades.empty else pd.DataFrame()
    )

    should_rebalance = prior_trades.empty or today.weekday() == 0

    if should_rebalance:
        # Sell existing positions first.
        for ticker, qty in list(positions.items()):
            price = latest.get(ticker)
            if price is None or qty <= 0:
                continue

            target_value = equity * weights.get(ticker, 0)
            current_value = qty * price
            amount_to_sell = max(0, current_value - target_value)

            if amount_to_sell < 10:
                continue

            shares = min(qty, amount_to_sell / price)
            fee = shares * price * 0.0005

            cash += shares * price - fee
            positions[ticker] = qty - shares

            actions.append((
                ticker, "SELL", shares, price, fee,
                f"Rebalance: reduce exposure. {rationale}"
            ))

        # Buy toward target allocations.
        for ticker, weight in weights.items():
            price = latest.get(ticker)
            if price is None or price <= 0:
                continue

            current_value = positions.get(ticker, 0) * price
            desired_value = equity * weight
            spend = min(
                max(0, desired_value - current_value),
                cash / 1.0005
            )

            if spend < 10:
                continue

            shares = spend / (price * 1.0005)
            fee = shares * price * 0.0005

            cash -= shares * price + fee
            positions[ticker] = positions.get(ticker, 0) + shares

            actions.append((
                ticker, "BUY", shares, price, fee,
                f"Target weight {weight:.1%}. {rationale}"
            ))

    positions = {
        ticker: float(qty)
        for ticker, qty in positions.items()
        if qty > 0.00000001
    }

    equity = cash + sum(
        qty * latest.get(ticker, 0)
        for ticker, qty in positions.items()
    )

    with engine.begin() as conn:
        for ticker, side, shares, price, fee, reason in actions:
            conn.execute(text("""
                INSERT INTO lab_trades
                (strategy, trade_date, executed_at, ticker,
                 side, shares, price, fee, reason)
                VALUES
                (:strategy, :trade_date, :executed_at, :ticker,
                 :side, :shares, :price, :fee, :reason)
            """), {
                "strategy": name,
                "trade_date": today,
                "executed_at": now,
                "ticker": ticker,
                "side": side,
                "shares": shares,
                "price": price,
                "fee": fee,
                "reason": reason
            })

        conn.execute(text("""
            INSERT INTO lab_snapshots
            (strategy, run_date, equity, cash, positions_json)
            VALUES
            (:strategy, :run_date, :equity, :cash, :positions_json)
            ON CONFLICT (strategy, run_date)
            DO UPDATE SET
                equity = EXCLUDED.equity,
                cash = EXCLUDED.cash,
                positions_json = EXCLUDED.positions_json
        """), {
            "strategy": name,
            "run_date": today,
            "equity": equity,
            "cash": cash,
            "positions_json": json.dumps(positions)
        })

    return {
        "strategy": name,
        "equity": round(equity, 2),
        "trades": len(actions),
        "reason": rationale
    }


def run_all(engine):
    results = []
    for name in MODELS:
        try:
            results.append(run_strategy(engine, name))
        except Exception as exc:
            results.append({
                "strategy": name,
                "error": f"{type(exc).__name__}: {exc}"
            })
    return results
