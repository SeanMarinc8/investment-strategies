"""Save intraday values for existing paper portfolios; never place trades."""
import json
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from sqlalchemy import text
from engine import db
import strategy_lab as lab

ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')


def get_quotes(tickers):
    quotes = {}
    for ticker in sorted(set(tickers)):
        try:
            bars = yf.download(ticker, period='1d', interval='5m',
                               auto_adjust=True, progress=False,
                               threads=False, prepost=False)
            if bars.empty:
                print('Missing quote:', ticker)
                continue
            close = bars['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors='coerce').dropna()
            close = close[close > 0]
            if close.empty:
                continue
            stamp = pd.Timestamp(close.index[-1])
            if stamp.tzinfo is None:
                print('Missing quote timezone:', ticker)
                continue
            quotes[ticker] = (float(close.iloc[-1]), stamp.tz_convert('UTC').to_pydatetime())
        except Exception as exc:
            print('Quote error:', ticker, type(exc).__name__, str(exc)[:120])
    return quotes


def main():
    now = datetime.now(UTC)
    local = now.astimezone(ET)
    if local.weekday() > 4 or not time(9, 35) <= local.time() <= time(16, 5):
        print('Outside weekday market hours; skipped.')
        return
    if not os.environ.get('DATABASE_URL'):
        raise RuntimeError('DATABASE_URL secret is missing')
    engine = db()
    lab.initialize(engine)
    with engine.begin() as conn:
        conn.execute(text('''
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
        '''))
    snapshots = lab.load(engine, 'lab_snapshots')
    if snapshots.empty:
        print('No strategy positions yet; run the daily strategy job first.')
        return
    latest = snapshots.sort_values('run_date').groupby('strategy').tail(1)
    positions = {}
    symbols = {'VOO', 'QQQ'}
    for row in latest.itertuples():
        try:
            holdings = json.loads(row.positions_json)
            holdings = {str(t): float(q) for t, q in holdings.items() if float(q) > 0}
            positions[row.strategy] = holdings
            symbols.update(holdings)
        except (TypeError, ValueError) as exc:
            print('Invalid holdings:', row.strategy, exc)
    quotes = get_quotes(symbols)
    quotes = {t: (price, stamp) for t, (price, stamp) in quotes.items()
              if stamp.astimezone(ET).date() == local.date()
              and -600 <= (now - stamp).total_seconds() <= 2700}
    if not quotes:
        print('No recent market quotes available; skipped.')
        return
    observed_at = now.replace(second=0, microsecond=0)
    saved = 0
    for row in latest.itertuples():
        holdings = positions.get(row.strategy)
        if holdings is None:
            continue
        missing = sorted(set(holdings) - set(quotes))
        if missing:
            print('Skipped incomplete prices:', row.strategy, ', '.join(missing))
            continue
        equity = float(row.cash) + sum(qty * quotes[t][0] for t, qty in holdings.items())
        if equity <= 0:
            continue
        stamps = [quotes[t][1] for t in holdings]
        stamps.extend(quotes[t][1] for t in ('VOO', 'QQQ') if t in quotes)
        asof = min(stamps) if stamps else now
        with engine.begin() as conn:
            conn.execute(text('''
                INSERT INTO lab_intraday_snapshots
                (strategy, observed_at, equity, cash, voo_price, qqq_price, price_asof, missing_tickers)
                VALUES (:strategy, :observed_at, :equity, :cash, :voo, :qqq, :asof, '')
                ON CONFLICT (strategy, observed_at) DO UPDATE SET
                    equity=EXCLUDED.equity, cash=EXCLUDED.cash,
                    voo_price=EXCLUDED.voo_price, qqq_price=EXCLUDED.qqq_price,
                    price_asof=EXCLUDED.price_asof
            '''), {'strategy': row.strategy, 'observed_at': observed_at,
                   'equity': equity, 'cash': float(row.cash),
                   'voo': quotes.get('VOO', (None, None))[0],
                   'qqq': quotes.get('QQQ', (None, None))[0], 'asof': asof})
        print('Saved', row.strategy, round(equity, 2), observed_at.isoformat())
        saved += 1
    print('Saved strategy valuations:', saved, '/', len(latest))


if __name__ == '__main__':
    main()
