
"""Seven educational paper-trading strategies.

Existing lab_snapshots and lab_trades are preserved.
Daily run_all() processes six original models.
run_intraday() is a separately scheduled paper-trading bot.
"""

import json
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import text

STARTING_CASH = 10000.0
ET = ZoneInfo("America/New_York")
FEE_RATE = 0.0005
MIN_ORDER = 25.0
DAY_BOT = "AI Intraday Trader"

MODELS = {
    "AI Goldman Sachs": {
        "description": "Quality and trend-factor proxy; not a Goldman Sachs strategy.",
        "tickers": ["MSFT", "AAPL", "V", "COST", "BRK-B", "JNJ", "PG", "GOOGL"],
        "method": "Daily factor ranking; up to four positive-trend stocks, 10% cash reserve.",
    },
    "AI Pelosi Disclosure-Inspired": {
        "description": "Technology basket; not verified congressional transactions.",
        "tickers": ["NVDA", "GOOGL", "MSFT", "AMZN", "AAPL", "AVGO", "CRM", "PANW"],
        "method": "Weekly rotation among strongest technology trends; no disclosure feed.",
    },
    "AI JPMorgan Chase": {
        "description": "Multi-asset proxy; not a JPMorgan proprietary strategy.",
        "tickers": ["VOO", "QQQ", "BND", "IEF", "GLD", "SCHD", "VEA", "VWO"],
        "method": "Weekly risk-sensitive asset allocation.",
    },
    "AI Momentum": {
        "description": "Daily-evaluated relative-strength swing strategy.",
        "tickers": ["NVDA", "PLTR", "META", "AMZN", "AVGO", "QQQ", "VRT", "MSFT"],
        "method": "Select top positive-trend names; exit when signals weaken.",
    },
    "AI Defensive Income": {
        "description": "Defensive allocation to bonds, dividends and gold.",
        "tickers": ["SCHD", "VIG", "BND", "IEF", "SGOV", "GLD"],
        "method": "Weekly defensive rebalancing with a cash reserve.",
    },
    "AI Equal-Weight Growth": {
        "description": "Growth basket with periodic weight restoration.",
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA"],
        "method": "Weekly equal-weight growth allocation.",
    },
    DAY_BOT: {
        "description": "Five-minute-bar intraday paper trading; intended to avoid overnight positions.",
        "tickers": ["QQQ"],
        "method": "Five-minute trend confirmation; one long position, exit before close.",
    },
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

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS lab_decisions (
                id BIGSERIAL PRIMARY KEY,
                strategy TEXT NOT NULL,
                decided_at TIMESTAMPTZ NOT NULL,
                decision_date DATE NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                equity DOUBLE PRECISION,
                UNIQUE (strategy, decided_at, action)
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS lab_intraday_executions (
                strategy TEXT NOT NULL,
                bar_time TIMESTAMPTZ NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY (strategy, bar_time)
            )
        """))


def load(engine, table):
    allowed = ("lab_trades", "lab_snapshots", "lab_decisions")
    if table not in allowed:
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
        close = raw[["Close"]].copy()
        close.columns = [symbols[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])

    return (
        close.apply(pd.to_numeric, errors="coerce")
        .sort_index()
        .ffill()
    )


def _valid_symbols(name, history):
    return [
        symbol
        for symbol in MODELS[name]["tickers"]
        if symbol in history.columns
        and history[symbol].dropna().shape[0] >= 65
    ]


def select_weights(name, history):
    """Calculate target portfolio weights without leverage."""

    if name == DAY_BOT:
        return {}, "Intraday bot uses run_intraday()."

    symbols = _valid_symbols(name, history)

    if not symbols:
        return {}, "Insufficient historical data; retain existing positions."

    trend_models = (
        "AI Momentum",
        "AI Goldman Sachs",
        "AI Pelosi Disclosure-Inspired"
    )

    if name in trend_models:
        ranked = []

        for ticker in symbols:
            series = history[ticker].dropna()

            price = float(series.iloc[-1])
            ma20 = float(series.tail(20).mean())
            ma50 = float(series.tail(50).mean())

            ret21 = price / float(series.iloc[-22]) - 1
            ret63 = price / float(series.iloc[-64]) - 1

            volatility = float(
                series.pct_change().tail(63).std()
            )

            if price <= ma50 or price <= ma20 or ret21 <= 0:
                continue

            if name == "AI Goldman Sachs":
                score = ret63 + 0.5 * ret21 - 2 * volatility
            else:
                score = ret63 + 0.5 * ret21

            ranked.append((ticker, score))

        ranked.sort(
            key=lambda item: item[1],
            reverse=True
        )

        count = 3 if name == "AI Momentum" else 4
        chosen = [ticker for ticker, _ in ranked[:count]]

        if not chosen:
            return {}, "No securities passed trend filters; target cash."

        return (
            {ticker: 0.90 / len(chosen) for ticker in chosen},
            "Positive 20/50-day trend and momentum ranking; 10% cash target."
        )

    if name == "AI JPMorgan Chase":
        base = {
            "VOO": 0.28,
            "QQQ": 0.12,
            "BND": 0.15,
            "IEF": 0.12,
            "GLD": 0.12,
            "SCHD": 0.10,
            "VEA": 0.07,
            "VWO": 0.04
        }

        available = {
            ticker: weight
            for ticker, weight in base.items()
            if ticker in symbols
        }

        total = sum(available.values())

        return (
            {
                ticker: 0.90 * weight / total
                for ticker, weight in available.items()
            },
            "Strategic multi-asset weights, 10% cash; weekly review."
        )

    weight = 0.90 / len(symbols)

    return (
        {ticker: weight for ticker in symbols},
        "Equal-weight target, 10% cash; weekly review."
    )


def latest_state(engine, strategy):
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT cash, positions_json
            FROM lab_snapshots
            WHERE strategy = :strategy
            ORDER BY run_date DESC
            LIMIT 1
        """), {
            "strategy": strategy
        }).mappings().first()

    if row is None:
        return STARTING_CASH, {}

    positions = json.loads(row["positions_json"])

    return (
        float(row["cash"]),
        {
            str(ticker): float(shares)
            for ticker, shares in positions.items()
        }
    )


def _last_snapshot_date(engine, name):
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT MAX(run_date)
            FROM lab_snapshots
            WHERE strategy = :name
        """), {"name": name}).scalar()


def _last_trade_date(engine, name):
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT MAX(trade_date)
            FROM lab_trades
            WHERE strategy = :name
        """), {"name": name}).scalar()


def _record_decision(conn, name, now, action, reason, equity):
    conn.execute(text("""
        INSERT INTO lab_decisions
        (
            strategy, decided_at, decision_date,
            action, reason, equity
        )
        VALUES (:s, :ts, :d, :a, :r, :e)
        ON CONFLICT DO NOTHING
    """), {
        "s": name,
        "ts": now,
        "d": now.astimezone(ET).date(),
        "a": action,
        "r": reason,
        "e": equity
    })


def _save_snapshot(conn, name, day, cash, positions, prices):
    missing = [
        ticker
        for ticker, shares in positions.items()
        if shares > 1e-8
        and (
            ticker not in prices
            or not np.isfinite(prices[ticker])
        )
    ]

    if missing:
        raise RuntimeError(
            f"Missing quotes for held positions: {missing}"
        )

    equity = cash + sum(
        shares * prices[ticker]
        for ticker, shares in positions.items()
    )

    conn.execute(text("""
        INSERT INTO lab_snapshots
        (
            strategy, run_date, equity,
            cash, positions_json
        )
        VALUES (:s, :d, :e, :c, :p)
        ON CONFLICT (strategy, run_date)
        DO UPDATE SET
            equity = EXCLUDED.equity,
            cash = EXCLUDED.cash,
            positions_json = EXCLUDED.positions_json
    """), {
        "s": name,
        "d": day,
        "e": equity,
        "c": cash,
        "p": json.dumps(positions)
    })

    return equity


def _rebalance(cash, positions, prices, weights, equity, reason):
    """Sell before buying. Apply 0.05% fees and $25 minimum."""

    positions = dict(positions)
    actions = []

    for ticker, qty in list(positions.items()):
        if qty <= 0:
            continue

        price = prices[ticker]

        excess = max(
            0.0,
            qty * price - equity * weights.get(ticker, 0.0)
        )

        if excess < MIN_ORDER:
            continue

        shares = min(qty, excess / price)
        fee = shares * price * FEE_RATE

        cash += shares * price - fee
        positions[ticker] = max(0.0, qty - shares)

        actions.append((
            ticker, "SELL", shares, price, fee, reason
        ))

    for ticker, weight in weights.items():
        price = prices.get(ticker)

        if price is None or price <= 0:
            continue

        desired = max(
            0.0,
            equity * weight -
            positions.get(ticker, 0) * price
        )

        spend = min(
            desired,
            cash / (1 + FEE_RATE)
        )

        if spend < MIN_ORDER:
            continue

        shares = spend / price
        fee = spend * FEE_RATE

        cash -= spend + fee

        positions[ticker] = (
            positions.get(ticker, 0) + shares
        )

        actions.append((
            ticker, "BUY", shares, price, fee, reason
        ))

    positions = {
        ticker: float(qty)
        for ticker, qty in positions.items()
        if qty > 1e-8
    }

    return cash, positions, actions


def _insert_trades(conn, name, now, actions):
    for ticker, side, shares, price, fee, reason in actions:
        conn.execute(text("""
            INSERT INTO lab_trades
            (
                strategy, trade_date, executed_at,
                ticker, side, shares, price, fee, reason
            )
            VALUES (
                :s, :d, :ts,
                :t, :side, :q, :p, :f, :r
            )
        """), {
            "s": name,
            "d": now.astimezone(ET).date(),
            "ts": now,
            "t": ticker,
            "side": side,
            "q": shares,
            "p": price,
            "f": fee,
            "r": reason
        })


def run_strategy(engine, name):
    """Evaluate one original strategy once per US trading day."""

    if name not in MODELS:
        raise ValueError("Unknown strategy")

    if name == DAY_BOT:
        raise ValueError(
            "Use run_intraday(engine) for the intraday bot"
        )

    initialize(engine)

    now = datetime.now(timezone.utc)
    market_now = now.astimezone(ET)
    day = market_now.date()

    if market_now.weekday() >= 5:
        return {
            "strategy": name,
            "trades": 0,
            "reason": "Market weekend: skipped"
        }

    if _last_snapshot_date(engine, name) == day:
        return {
            "strategy": name,
            "trades": 0,
            "reason": "Already processed today; no duplicate orders"
        }

    cash, positions = latest_state(engine, name)

    symbols = list(dict.fromkeys(
        MODELS[name]["tickers"] + list(positions)
    ))

    history = fetch_history(symbols)

    if history.empty:
        raise RuntimeError("Market data unavailable")

    prices = {}

    for ticker in history.columns:
        values = history[ticker].dropna()

        if not values.empty:
            price = float(values.iloc[-1])

            if np.isfinite(price):
                prices[ticker] = price

    missing = [
        ticker
        for ticker, shares in positions.items()
        if shares > 1e-8 and ticker not in prices
    ]

    if missing:
        raise RuntimeError(
            f"Refusing to value holdings without quotes: {missing}"
        )

    equity = cash + sum(
        shares * prices[ticker]
        for ticker, shares in positions.items()
    )

    if equity <= 0:
        raise RuntimeError("Invalid portfolio equity")

    weights, rationale = select_weights(name, history)

    last_trade = _last_trade_date(engine, name)

    is_swing = name in (
        "AI Momentum",
        "AI Goldman Sachs"
    )

    is_weekly = not is_swing

    should_trade = (
        last_trade is None
        or is_swing
        or (
            is_weekly
            and market_now.weekday() == 0
        )
    )

    actions = []

    if should_trade:
        if (
            not weights
            and rationale.startswith("Insufficient")
        ):
            should_trade = False
        else:
            cash, positions, actions = _rebalance(
                cash,
                positions,
                prices,
                weights,
                equity,
                rationale
            )

    with engine.begin() as conn:
        _insert_trades(conn, name, now, actions)

        equity = _save_snapshot(
            conn, name, day, cash, positions, prices
        )

        _record_decision(
            conn,
            name,
            now,
            "TRADE" if actions else "HOLD",
            rationale if should_trade
            else "Scheduled HOLD: " + rationale,
            equity
        )

    return {
        "strategy": name,
        "equity": round(equity, 2),
        "trades": len(actions),
        "reason": rationale if actions
        else "HOLD: " + rationale
    }


def run_all(engine):
    """Run the six original strategies, excluding the intraday bot."""

    results = []

    for name in MODELS:
        if name == DAY_BOT:
            continue

        try:
            results.append(run_strategy(engine, name))
        except Exception as exc:
            results.append({
                "strategy": name,
                "error": f"{type(exc).__name__}: {exc}"
            })

    return results


def run_intraday(engine):
    """Execute one paper signal per completed QQQ five-minute bar."""

    initialize(engine)

    now = datetime.now(timezone.utc)
    local = now.astimezone(ET)

    if (
        local.weekday() >= 5
        or not (
            time(9, 40) <= local.time() <= time(16, 0)
        )
    ):
        return {
            "strategy": DAY_BOT,
            "trades": 0,
            "reason": "Outside market session"
        }

    bars = yf.download(
        "QQQ",
        period="5d",
        interval="5m",
        auto_adjust=True,
        progress=False,
        threads=False,
        prepost=False
    )

    if bars.empty:
        raise RuntimeError("No intraday bars returned")

    if isinstance(bars.columns, pd.MultiIndex):
        closes = bars["Close"].iloc[:, 0]
    else:
        closes = bars["Close"]

    closes = pd.to_numeric(
        closes,
        errors="coerce"
    ).dropna()

    if closes.index.tz is None:
        raise RuntimeError(
            "Intraday bar timestamps have no timezone"
        )

    closes.index = closes.index.tz_convert(ET)

    # Yahoo Finance timestamps label the START of each bar.
    # Only use completed bars.
    current_timestamp = pd.Timestamp(now).tz_convert(ET)

    closes = closes[
        closes.index + pd.Timedelta(minutes=5)
        <= current_timestamp
    ]

    closes = closes[
        closes.index.date == local.date()
    ]

    if len(closes) < 8:
        return {
            "strategy": DAY_BOT,
            "trades": 0,
            "reason": "Waiting for eight completed bars"
        }

    bar = closes.index[-1].to_pydatetime()

    if local - bar > timedelta(minutes=20):
        raise RuntimeError(
            "Intraday quote is stale; no trades"
        )

    price = float(closes.iloc[-1])

    if not np.isfinite(price) or price <= 0:
        raise RuntimeError("Invalid intraday price")

    with engine.connect() as conn:
        already = conn.execute(text("""
            SELECT 1
            FROM lab_intraday_executions
            WHERE strategy = :s
              AND bar_time = :b
        """), {
            "s": DAY_BOT,
            "b": bar
        }).first()

    if already:
        return {
            "strategy": DAY_BOT,
            "trades": 0,
            "reason": "Bar already processed"
        }

    cash, positions = latest_state(
        engine, DAY_BOT
    )

    held = float(positions.get("QQQ", 0.0))

    fast = float(closes.tail(3).mean())
    slow = float(closes.tail(8).mean())

    enter = (
        fast > slow * 1.001
        and price > float(closes.iloc[-4])
    )

    exit_signal = fast < slow * 0.999

    flatten = local.time() >= time(15, 50)

    actions = []
    reason = "HOLD: intraday signal unchanged"

    if held > 1e-8 and (
        exit_signal or flatten
    ):
        fee = held * price * FEE_RATE

        cash += held * price - fee

        actions.append((
            "QQQ",
            "SELL",
            held,
            price,
            fee,
            "End-of-day exit" if flatten
            else "Five-minute trend reversal"
        ))

        positions = {}
        reason = actions[-1][-1]

    elif (
        held <= 1e-8
        and enter
        and local.time() < time(15, 30)
    ):
        spend = min(
            cash / (1 + FEE_RATE),
            cash * 0.50
        )

        if spend >= MIN_ORDER:
            shares = spend / price
            fee = spend * FEE_RATE

            cash -= spend + fee

            positions = {"QQQ": shares}

            actions.append((
                "QQQ",
                "BUY",
                shares,
                price,
                fee,
                "Three-bar trend exceeds eight-bar trend by 0.1%"
            ))

            reason = actions[-1][-1]

    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO lab_intraday_executions
            (strategy, bar_time, action, reason)
            VALUES (:s, :b, :a, :r)
            ON CONFLICT DO NOTHING
            RETURNING strategy
        """), {
            "s": DAY_BOT,
            "b": bar,
            "a": "TRADE" if actions else "HOLD",
            "r": reason
        }).first()

        if result is None:
            return {
                "strategy": DAY_BOT,
                "trades": 0,
                "reason": "Bar already processed"
            }

        _insert_trades(
            conn, DAY_BOT, now, actions
        )

        equity = _save_snapshot(
            conn,
            DAY_BOT,
            local.date(),
            cash,
            positions,
            {"QQQ": price}
        )

        _record_decision(
            conn,
            DAY_BOT,
            now,
            "TRADE" if actions else "HOLD",
            reason,
            equity
        )

    return {
        "strategy": DAY_BOT,
        "equity": round(equity, 2),
        "trades": len(actions),
        "reason": reason
    }
