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
