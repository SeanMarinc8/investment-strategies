"""Scheduled intraday paper-trading execution job.

Runs the existing AI Intraday Trader strategy without modifying its rules.
GitHub Actions scheduling is best-effort, not a guaranteed timer.
"""

import os
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

from engine import db, init
import strategy_lab as lab

ET = ZoneInfo('America/New_York')
START = time(9, 35)
END = time(15, 55)


def main():
    now = datetime.now(ET)
    print('=' * 60, flush=True)
    print('AI INVESTMENT RESEARCH LAB - INTRADAY PAPER TRADING', flush=True)
    print('Eastern time:', now.isoformat(), flush=True)
    print('Strategy:', lab.DAY_BOT, flush=True)

    if now.weekday() >= 5:
        print('SKIP: Weekend.', flush=True)
        return
    if not (START <= now.time().replace(tzinfo=None) < END):
        print(f'SKIP: Outside execution window ({START:%H:%M}-{END:%H:%M} ET).', flush=True)
        return

    # The strategy's own market-data and exchange-calendar safeguards must
    # still prevent trades on holidays, halted symbols and stale candles.
    if not os.environ.get('DATABASE_URL'):
        raise RuntimeError('DATABASE_URL GitHub Actions secret is missing.')

    engine = db()
    init(engine)
    lab.initialize(engine)
    result = lab.run_intraday(engine)

    if not isinstance(result, dict):
        raise RuntimeError('Intraday strategy returned an invalid result.')
    if 'error' in result:
        raise RuntimeError(str(result['error']))

    print('INTRADAY EXECUTION RESULT', flush=True)
    print('Strategy:', result.get('strategy'), flush=True)
    print('Trades:', result.get('trades', 0), flush=True)
    print('Reason:', result.get('reason', ''), flush=True)
    if 'equity' in result:
        print('Portfolio equity:', f"${float(result['equity']):,.2f}", flush=True)
    print('Execution completed successfully.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('INTRADAY STRATEGY FAILED:', type(exc).__name__, str(exc), flush=True)
        sys.exit(1)
