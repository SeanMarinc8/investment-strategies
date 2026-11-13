import os
import hmac
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sqlalchemy import text

import strategy_lab as lab
from engine import db, init

st.set_page_config(page_title='Investment Research Terminal', page_icon='📈', layout='wide', initial_sidebar_state='expanded')
ET = ZoneInfo('America/New_York')
START = 10000.0
NAV = ['Overview', 'Strategy Lab', 'Intraday Trader', 'My Portfolio']
COLORS = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#e11d48', '#db2777', '#0891b2']

st.markdown('''<style>
.stApp {background:#ffffff;color:#172334}
[data-testid="stSidebar"] {background:#f7f9fc;border-right:1px solid #e1e7ef}
[data-testid="stMetric"] {background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px}
[data-testid="stMetricLabel"] {color:#526176}
.stTabs [data-baseweb="tab-list"] {gap:8px}
.stTabs [data-baseweb="tab"] {background:#f1f5f9;border-radius:8px;color:#334155;padding:8px 15px}
.stTabs [aria-selected="true"] {background:#dbeafe;color:#1d4ed8}
[data-testid="stDataFrame"] {border:1px solid #e2e8f0;border-radius:8px}
h1,h2,h3 {letter-spacing:-.025em;color:#172334}
.block-container {padding-top:1.6rem;max-width:1550px}
small {color:#64748b}
</style>''', unsafe_allow_html=True)

try:
    password = str(st.secrets['APP_PASSWORD'])
    database_url = str(st.secrets['DATABASE_URL'])
except Exception:
    st.error('Add APP_PASSWORD and DATABASE_URL to Streamlit Secrets.')
    st.stop()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if not st.session_state.authenticated:
    st.title('Investment Research Terminal')
    attempt = st.text_input('App password', type='password')
    if st.button('Sign in', type='primary'):
        if hmac.compare_digest(attempt, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error('Incorrect password.')
    st.stop()
os.environ['DATABASE_URL'] = database_url

@st.cache_resource
def connection():
    engine = db()
    init(engine)
    lab.initialize(engine)
    with engine.begin() as conn:
        conn.execute(text('''CREATE TABLE IF NOT EXISTS personal_holdings (
            ticker TEXT PRIMARY KEY, shares DOUBLE PRECISION NOT NULL,
            cost_basis DOUBLE PRECISION)'''))
        conn.execute(text('''CREATE TABLE IF NOT EXISTS personal_snapshots (
            snapshot_date DATE PRIMARY KEY, market_value DOUBLE PRECISION NOT NULL,
            priced_count INTEGER NOT NULL, missing_count INTEGER NOT NULL)'''))
    return engine

try:
    engine = connection()
except Exception as exc:
    st.error(f'Database connection failed: {type(exc).__name__}: {exc}')
    st.stop()

@st.cache_data(ttl=120, show_spinner=False)
def table(name):
    allowed = {'lab_snapshots', 'lab_trades', 'lab_decisions', 'lab_intraday_executions',
               'personal_holdings', 'personal_snapshots'}
    if name not in allowed:
        raise ValueError('Table not allowed')
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(text(f'SELECT * FROM {name}'), conn)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def benchmark_prices(symbols, period='3mo', interval='1d'):
    if not symbols:
        return pd.DataFrame()
    raw = yf.download(list(symbols), period=period, interval=interval,
                      auto_adjust=True, progress=False, threads=False)
    if raw.empty:
