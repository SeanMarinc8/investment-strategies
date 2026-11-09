
"""Scheduled intraday paper-trading execution job.

Runs the seventh AI Intraday Trader strategy.
Does not modify the original six strategies.
Uses the existing Supabase database.
"""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from engine import db, init
import strategy_lab as lab


ET = ZoneInfo("America/New_York")


def main():
    now = datetime.now(ET)

    print("=" * 60)
    print("AI INVESTMENT RESEARCH LAB")
    print("INTRADAY PAPER TRADING")
    print("=" * 60)
    print("Eastern time:", now.isoformat())
    print("Strategy:", lab.DAY_BOT)

    if now.weekday() >= 5:
        print("SKIP: Weekend.")
        return

    if not (9 <= now.hour <= 16):
        print("SKIP: Outside trading hours.")
        return

    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError(
            "DATABASE_URL GitHub Actions secret is missing."
        )

    engine = db()

    init(engine)
    lab.initialize(engine)

    try:
        result = lab.run_intraday(engine)
    except Exception as exc:
        print(
            "INTRADAY STRATEGY FAILED:",
            type(exc).__name__,
            str(exc)
        )
        raise

    if not isinstance(result, dict):
        raise RuntimeError(
            "Intraday strategy returned an invalid result."
        )

    if "error" in result:
        raise RuntimeError(str(result["error"]))

    print()
    print("INTRADAY EXECUTION RESULT")
    print("-" * 60)

    print("Strategy:", result.get("strategy"))
    print("Trades:", result.get("trades", 0))
    print("Reason:", result.get("reason", ""))

    if "equity" in result:
        print(
            "Portfolio equity:",
            f"${float(result['equity']):,.2f}"
        )

    print()
    print("Execution completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
