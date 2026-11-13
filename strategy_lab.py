
"""Seven independent educational paper-trading strategies.

Existing lab_snapshots and lab_trades are retained. Daily run_all() processes
all six original models; run_intraday() is a separate, explicitly scheduled job.
No broker connectivity. No guarantees of returns or trading activity.
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
        "description": "Multi-stock five-minute intraday paper trading; no overnight positions intended.",
        "tickers": [
            "NVDA", "MSFT", "GOOGL", "AAPL", "PFE", "AMZN", "META",
            "AVGO", "TSLA", "AMD", "ORCL", "CRM", "JPM", "BAC",
            "V", "MA", "WMT", "COST", "HD", "UNH", "JNJ", "ABBV",
            "XOM", "CAT", "GE", "DIS", "KO", "PG", "NFLX", "INTC"
        ],
        "method": "Scan 30 large-cap companies; rank 5-minute momentum and hold at most three positions.",
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
        # Separate event log: no changes to historical trade or snapshot rows.
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
    if table not in ("lab_trades", "lab_snapshots", "lab_decisions"):
        raise ValueError("Invalid table")
    return pd.read_sql_query(text(f"SELECT * FROM {table}"), engine)


def fetch_history(tickers, period="1y"):
    symbols = list(dict.fromkeys(tickers))
    if not symbols:
        return pd.DataFrame()
    raw = yf.download(symbols, period=period, auto_adjust=True,
                      progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()
        close.columns = [symbols[0]]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    return close.apply(pd.to_numeric, errors="coerce").sort_index().ffill()


def _valid_symbols(name, history):
    return [s for s in MODELS[name]["tickers"]
            if s in history.columns and history[s].dropna().shape[0] >= 65]


def select_weights(name, history):
    """Returns target weights and a plain-language rationale; never leverages."""
    if name == DAY_BOT:
        return {}, "Intraday strategy uses run_intraday(), not daily weights."
    symbols = _valid_symbols(name, history)
    if not symbols:
        return {}, "Insufficient historical data; retain existing positions."

    if name in ("AI Momentum", "AI Goldman Sachs",
                "AI Pelosi Disclosure-Inspired"):
        ranked = []
        for ticker in symbols:
            series = history[ticker].dropna()
            p = float(series.iloc[-1])
            ma20 = float(series.tail(20).mean())
            ma50 = float(series.tail(50).mean())
            ret21 = p / float(series.iloc[-22]) - 1
            ret63 = p / float(series.iloc[-64]) - 1
            vol = float(series.pct_change().tail(63).std())
            if p <= ma50 or p <= ma20 or ret21 <= 0:
                continue
            if name == "AI Goldman Sachs":
                score = ret63 + 0.5 * ret21 - 2 * vol
            else:
                score = ret63 + 0.5 * ret21
            ranked.append((ticker, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        count = 3 if name == "AI Momentum" else 4
        chosen = [ticker for ticker, _ in ranked[:count]]
        if not chosen:
            return {}, "No securities passed trend filters; target cash."
        return ({t: 0.90 / len(chosen) for t in chosen},
                "Positive 20/50-day trend and momentum ranking; 10% cash target.")

    if name == "AI JPMorgan Chase":
        # Distinct from equal-weight: strategic multi-asset risk mix.
        base = {"VOO": .28, "QQQ": .12, "BND": .15, "IEF": .12,
                "GLD": .12, "SCHD": .10, "VEA": .07, "VWO": .04}
        available = {t: w for t, w in base.items() if t in symbols}
        total = sum(available.values())
        return ({t: .90 * w / total for t, w in available.items()},
                "Strategic multi-asset weights, 10% cash; weekly review.")

    weight = .90 / len(symbols)
    return ({t: weight for t in symbols},
            "Equal-weight target, 10% cash; weekly review.")


def latest_state(engine, strategy):
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT cash, positions_json FROM lab_snapshots
            WHERE strategy = :strategy ORDER BY run_date DESC LIMIT 1
        """), {"strategy": strategy}).mappings().first()
    if row is None:
        return STARTING_CASH, {}
    return float(row["cash"]), {str(k): float(v)
                                for k, v in json.loads(row["positions_json"]).items()}


def _last_snapshot_date(engine, name):
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT MAX(run_date) FROM lab_snapshots WHERE strategy = :name
        """), {"name": name}).scalar()


def _last_trade_date(engine, name):
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT MAX(trade_date) FROM lab_trades WHERE strategy = :name
        """), {"name": name}).scalar()


def _record_decision(conn, name, now, action, reason, equity):
    conn.execute(text("""
        INSERT INTO lab_decisions
        (strategy, decided_at, decision_date, action, reason, equity)
        VALUES (:s, :ts, :d, :a, :r, :e)
        ON CONFLICT DO NOTHING
    """), {"s": name, "ts": now, "d": now.astimezone(ET).date(),
           "a": action, "r": reason, "e": equity})


def _save_snapshot(conn, name, day, cash, positions, prices):
    missing = [t for t, q in positions.items()
               if q > 1e-8 and (t not in prices or not np.isfinite(prices[t]))]
    if missing:
        raise RuntimeError(f"Missing quotes for held positions: {missing}")
    equity = cash + sum(q * prices[t] for t, q in positions.items())
    conn.execute(text("""
        INSERT INTO lab_snapshots
        (strategy, run_date, equity, cash, positions_json)
        VALUES (:s, :d, :e, :c, :p)
        ON CONFLICT (strategy, run_date) DO UPDATE SET
            equity=EXCLUDED.equity, cash=EXCLUDED.cash,
            positions_json=EXCLUDED.positions_json
    """), {"s": name, "d": day, "e": equity, "c": cash,
           "p": json.dumps(positions)})
    return equity


def _rebalance(cash, positions, prices, weights, equity, reason):
    """Sells before buys; applies 5 bps fee and $25 minimum order."""
    positions = dict(positions)
    actions = []
    for ticker, qty in list(positions.items()):
        if qty <= 0:
            continue
        price = prices[ticker]
        excess = max(0.0, qty * price - equity * weights.get(ticker, 0.0))
        if excess < MIN_ORDER:
            continue
        shares = min(qty, excess / price)
        fee = shares * price * FEE_RATE
        cash += shares * price - fee
        positions[ticker] = max(0.0, qty - shares)
        actions.append((ticker, "SELL", shares, price, fee, reason))

    for ticker, weight in weights.items():
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        desired = max(0.0, equity * weight - positions.get(ticker, 0) * price)
        spend = min(desired, cash / (1 + FEE_RATE))
        if spend < MIN_ORDER:
            continue
        shares = spend / price
        fee = spend * FEE_RATE
        cash -= spend + fee
        positions[ticker] = positions.get(ticker, 0) + shares
        actions.append((ticker, "BUY", shares, price, fee, reason))
    return cash, {t: float(q) for t, q in positions.items() if q > 1e-8}, actions


def _insert_trades(conn, name, now, actions):
    for ticker, side, shares, price, fee, reason in actions:
        conn.execute(text("""
            INSERT INTO lab_trades
            (strategy, trade_date, executed_at, ticker, side, shares, price, fee, reason)
            VALUES (:s, :d, :ts, :t, :side, :q, :p, :f, :r)
        """), {"s": name, "d": now.astimezone(ET).date(), "ts": now,
               "t": ticker, "side": side, "q": shares, "p": price,
               "f": fee, "r": reason})


def run_strategy(engine, name):
    """Run a single original strategy once per US trading day."""
    if name not in MODELS:
        raise ValueError("Unknown strategy")
    if name == DAY_BOT:
        raise ValueError("Use run_intraday(engine) for the intraday bot")
    initialize(engine)
    now = datetime.now(timezone.utc)
    market_now = now.astimezone(ET)
    day = market_now.date()
    if market_now.weekday() >= 5:
        return {"strategy": name, "trades": 0, "reason": "Market weekend: skipped"}

    # Prevent duplicate daily orders if the workflow is retried.
    if _last_snapshot_date(engine, name) == day:
        return {"strategy": name, "trades": 0,
                "reason": "Already processed this trading date; no duplicate orders"}

    cash, positions = latest_state(engine, name)
    symbols = list(dict.fromkeys(MODELS[name]["tickers"] + list(positions)))
    history = fetch_history(symbols)
    if history.empty:
        raise RuntimeError("Market data unavailable")
    prices = {t: float(history[t].dropna().iloc[-1]) for t in history.columns
              if not history[t].dropna().empty and
              np.isfinite(float(history[t].dropna().iloc[-1]))}
    missing = [t for t, q in positions.items() if q > 1e-8 and t not in prices]
    if missing:
        raise RuntimeError(f"Refusing to value holdings without quotes: {missing}")
    equity = cash + sum(q * prices[t] for t, q in positions.items())
    if equity <= 0:
        raise RuntimeError("Invalid portfolio equity")

    weights, rationale = select_weights(name, history)
    last_trade = _last_trade_date(engine, name)
    is_swing = name in ("AI Momentum", "AI Goldman Sachs")
    is_weekly = not is_swing
    should_trade = (last_trade is None or is_swing or
                    (is_weekly and market_now.weekday() == 0))
    actions = []
    if should_trade:
        # If initial history is insufficient, do not liquidate existing assets.
        if not weights and rationale.startswith("Insufficient"):
            should_trade = False
        else:
            cash, positions, actions = _rebalance(
                cash, positions, prices, weights, equity, rationale)

    with engine.begin() as conn:
        _insert_trades(conn, name, now, actions)
        equity = _save_snapshot(conn, name, day, cash, positions, prices)
        _record_decision(conn, name, now, "TRADE" if actions else "HOLD",
                         rationale if should_trade else "Scheduled HOLD: " + rationale,
                         equity)
    return {"strategy": name, "equity": round(equity, 2),
            "trades": len(actions), "reason": rationale if actions else "HOLD: " + rationale}


def run_all(engine):
    """Existing daily workflow compatibility; does not execute intraday bot."""
    results = []
    for name in MODELS:
        if name == DAY_BOT:
            continue
        try:
            results.append(run_strategy(engine, name))
        except Exception as exc:
            results.append({"strategy": name,
                            "error": f"{type(exc).__name__}: {exc}"})
    return results


def _intraday_close_frame(tickers):
    """Download one batch of five-minute closes; tolerate missing symbols."""
    raw = yf.download(
        tickers, period="5d", interval="5m", auto_adjust=True,
        progress=False, threads=False, prepost=False, group_by="column"
    )
    if raw.empty:
        raise RuntimeError("No intraday stock data returned")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    if close.index.tz is None:
        raise RuntimeError("Market bars have no timezone")
    close.index = close.index.tz_convert(ET)
    return close.apply(pd.to_numeric, errors="coerce").sort_index()


def run_intraday(engine):
    """Trade up to three liquid large-cap stocks using completed five-minute bars.

    All positions are paper-only. The workflow must execute near the close to
    flatten positions; scheduled GitHub runs are not guaranteed.
    """
    initialize(engine)
    now = datetime.now(timezone.utc)
    local = now.astimezone(ET)
    if local.weekday() >= 5 or not (time(9, 40) <= local.time() <= time(16, 0)):
        return {"strategy": DAY_BOT, "trades": 0, "reason": "Outside session"}

    cash, positions = latest_state(engine, DAY_BOT)
    universe = list(dict.fromkeys(MODELS[DAY_BOT]["tickers"] + list(positions)))
    close = _intraday_close_frame(universe)
    close = close[
        (close.index.date == local.date()) &
        (close.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now).tz_convert(ET))
    ]
    if close.empty:
        return {"strategy": DAY_BOT, "trades": 0, "reason": "No completed bars yet"}

    # A common bar key makes duplicate protection independent of the stock chosen.
    bar = close.index[-1].to_pydatetime()
    if local - bar > timedelta(minutes=20):
        raise RuntimeError("Market data is stale; refusing to trade")

    with engine.connect() as conn:
        existing = conn.execute(text("""
            SELECT 1 FROM lab_intraday_executions
            WHERE strategy=:s AND bar_time=:b
        """), {"s": DAY_BOT, "b": bar}).first()
    if existing:
        return {"strategy": DAY_BOT, "trades": 0, "reason": "Bar already processed"}

    quotes = {}
    candidates = []
    for ticker in universe:
        if ticker not in close.columns:
            continue
        series = close[ticker].dropna()
        if series.empty:
            continue
        ticker_bar = series.index[-1].to_pydatetime()
        if bar - ticker_bar > timedelta(minutes=5):
            continue
        price = float(series.iloc[-1])
        if not np.isfinite(price) or price <= 0:
            continue
        quotes[ticker] = price
        if len(series) < 8:
            continue
        fast = float(series.tail(3).mean())
        slow = float(series.tail(8).mean())
        recent = float(series.iloc[-4])
        strength = fast / slow - 1
        if fast > slow * 1.0005 and price > recent:
            candidates.append((ticker, strength))

    missing_held = [ticker for ticker, shares in positions.items()
                    if shares > 1e-8 and ticker not in quotes]
    if missing_held:
        raise RuntimeError(f"No fresh quotes for held stocks: {missing_held}")

    # Existing holdings are always considered for exits, including symbols
    # removed from the new universe in a future version.
    candidates.sort(key=lambda item: item[1], reverse=True)
    top = [ticker for ticker, _ in candidates[:3]]
    flatten = local.time() >= time(15, 50)
    can_enter = time(9, 45) <= local.time() < time(15, 30)
    actions = []
    reason = f"HOLD: scanned {len(quotes)} stocks; {len(candidates)} positive signals"

    for ticker, shares in list(positions.items()):
        if shares <= 1e-8:
            continue
        if flatten or (ticker not in top and can_enter):
            price = quotes[ticker]
            fee = shares * price * FEE_RATE
            cash += shares * price - fee
            actions.append((ticker, "SELL", shares, price, fee,
                            "End-of-day exit" if flatten else "Lost top-three momentum rank"))
            del positions[ticker]

    if can_enter and not flatten:
        for ticker in top:
            if ticker in positions or len(positions) >= 3:
                continue
            price = quotes[ticker]
            # Cap each new holding at 30% of current marked-to-market equity.
            equity = cash + sum(qty * quotes[t] for t, qty in positions.items())
            spend = min(cash / (1 + FEE_RATE), equity * .30)
            if spend < MIN_ORDER:
                continue
            shares = spend / price
            fee = spend * FEE_RATE
            cash -= spend + fee
            positions[ticker] = shares
            actions.append((ticker, "BUY", shares, price, fee,
                            "Top-three 5-minute momentum; 30% position cap"))

    if actions:
        reason = "; ".join(f"{side} {ticker}" for ticker, side, *_ in actions)

    with engine.begin() as conn:
        locked = conn.execute(text("""
            INSERT INTO lab_intraday_executions (strategy, bar_time, action, reason)
            VALUES (:s, :b, :a, :r)
            ON CONFLICT DO NOTHING RETURNING strategy
        """), {"s": DAY_BOT, "b": bar,
               "a": "TRADE" if actions else "HOLD", "r": reason}).first()
        if locked is None:
            return {"strategy": DAY_BOT, "trades": 0, "reason": "Bar already processed"}
        _insert_trades(conn, DAY_BOT, now, actions)
        equity = _save_snapshot(conn, DAY_BOT, local.date(), cash, positions, quotes)
        _record_decision(conn, DAY_BOT, now,
                         "TRADE" if actions else "HOLD", reason, equity)

    return {"strategy": DAY_BOT, "equity": round(equity, 2),
            "trades": len(actions), "reason": reason,
            "scanned": len(quotes), "qualified": len(candidates),
            "holdings": list(positions)}
