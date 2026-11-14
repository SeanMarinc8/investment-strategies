import os
import hmac
import json
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os
import hmac
import json
from datetime import datetime, date
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
        conn.execute(text('''CREATE TABLE IF NOT EXISTS personal_weight_events (
            id BIGSERIAL PRIMARY KEY, effective_date DATE NOT NULL,
            weights_json TEXT NOT NULL, recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )'''))
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
               'personal_holdings', 'personal_snapshots', 'personal_weight_events'}
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
        return pd.DataFrame()
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    elif not isinstance(raw.columns, pd.MultiIndex):
        close.columns = [symbols[0]]
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_convert(ET).tz_localize(None)
    close.index = close.index.normalize()
    return close.apply(pd.to_numeric, errors='coerce').sort_index().ffill()



def portfolio_weights(holdings_frame, price_frame):
    """Current market-value weights, excluding symbols without a valid quote."""
    amounts = {}
    for row in holdings_frame.itertuples(index=False):
        ticker = str(row.ticker).upper()
        if ticker not in price_frame.columns:
            continue
        observations = price_frame[ticker].dropna()
        if observations.empty:
            continue
        last = float(observations.iloc[-1])
        if np.isfinite(last) and last > 0 and float(row.shares) > 0:
            amounts[ticker] = last * float(row.shares)
    total = sum(amounts.values())
    return {t: v / total for t, v in amounts.items()} if total > 0 else {}


def save_weight_event(conn, holdings_frame, prices, effective_date):
    weights = portfolio_weights(holdings_frame, prices)
    if not weights and not holdings_frame.empty:
        raise ValueError('Cannot save allocation history: market quotes unavailable.')
    conn.execute(text("INSERT INTO personal_weight_events (effective_date, weights_json) VALUES (:d, :w)"),
        {'d': effective_date, 'w': json.dumps(weights)})


def historical_weighted_portfolio(prices, initial_weights, events, start):
    """Hypothetical buy-and-hold until a saved allocation event, then rebalance.

    Allocation changes are applied at the prior trading close to the return
    for their effective trading date. No retroactive changes to earlier returns.
    """
    prices = prices.loc[prices.index >= start].sort_index()
    if prices.empty or not initial_weights:
        return pd.Series(dtype=float)
    weights = dict(initial_weights)
    events = sorted(events, key=lambda x: (x[0], x[1]))
    event_index = 0
    first_date = prices.index[0]
    while event_index < len(events) and events[event_index][0] <= first_date:
        weights = dict(events[event_index][2])
        event_index += 1
    if not weights:
        return pd.Series(dtype=float)
    value = 10000.0
    output = {first_date: value}
    prior = prices.iloc[0]
    for current_date, current in prices.iloc[1:].iterrows():
        while event_index < len(events) and events[event_index][0] <= current_date:
            weights = dict(events[event_index][2])
            event_index += 1
        if weights:
            ratios = []
            for ticker, weight in weights.items():
                a, b = prior.get(ticker, np.nan), current.get(ticker, np.nan)
                if pd.notna(a) and pd.notna(b) and a > 0 and b > 0:
                    ratios.append(weight * (b / a))
                else:
                    ratios.append(weight)  # Missing quote: hold previous valuation
            value *= sum(ratios)
        output[current_date] = value
        prior = current
    return pd.Series(output, dtype=float)


def record_portfolio_change(engine, price_frame, effective_date):
    """Save a forward-effective allocation only; historical chart stays frozen."""
    current = pd.read_sql_query(text('SELECT ticker, shares, cost_basis FROM personal_holdings'), engine)
    with engine.begin() as conn:
        save_weight_event(conn, current, price_frame, effective_date)


def snapshots():
    df = table('lab_snapshots').copy()
    if df.empty:
        return df
    df['run_date'] = pd.to_datetime(df['run_date'], errors='coerce').dt.normalize()
    df['equity'] = pd.to_numeric(df['equity'], errors='coerce')
    df['cash'] = pd.to_numeric(df['cash'], errors='coerce')
    return df.dropna(subset=['run_date', 'equity']).sort_values('run_date')


def trades():
    df = table('lab_trades').copy()
    if not df.empty:
        df['executed_at'] = pd.to_datetime(df['executed_at'], errors='coerce', utc=True)
        df = df.sort_values('executed_at', ascending=False)
    return df


def decisions():
    df = table('lab_decisions').copy()
    if not df.empty:
        df['decided_at'] = pd.to_datetime(df['decided_at'], errors='coerce', utc=True)
        df = df.sort_values('decided_at', ascending=False)
    return df


def money(value):
    return f'${value:,.2f}' if pd.notna(value) else '—'


def percent(value):
    return f'{value:+.2f}%' if pd.notna(value) else '—'


def normalize(series):
    series = series.dropna().astype(float)
    if series.empty or series.iloc[0] <= 0:
        return series
    return series / series.iloc[0] * START


def chart(lines, title, height=470, x_start=None):
    fig = go.Figure()
    for i, (name, series) in enumerate(lines.items()):
        s = series.dropna().sort_index()
        if s.empty:
            continue
        color = '#334155' if name == 'VOO' else '#f59e0b' if name == 'QQQ' else COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode='lines',
                                 line=dict(color=color, width=2.5 if name in ('VOO', 'QQQ') else 2),
                                 hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.2f}<extra>%{fullData.name}</extra>'))
    fig.update_layout(title=title, height=height, template='plotly_white',
                      paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                      font=dict(color='#334155'), margin=dict(l=12, r=12, t=52, b=10),
                      hovermode='x unified', legend=dict(orientation='h', y=-0.18),
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#e2e8f0', tickprefix='$'))
    if x_start is not None:
        fig.update_xaxes(range=[pd.Timestamp(x_start), max(pd.Timestamp.now().normalize(), max((v.dropna().index.max() for v in lines.values() if not v.dropna().empty), default=pd.Timestamp.now().normalize()))])
    st.plotly_chart(fig, use_container_width=True)


def strategy_series(df, names):
    result = {}
    for name in names:
        rows = df[df['strategy'] == name].drop_duplicates('run_date', keep='last')
        if not rows.empty:
            result[name] = pd.Series(rows['equity'].values, index=rows['run_date'])
    return result


def comparison(df, names, include_benchmarks=True, height=490):
    lines = strategy_series(df, names)
    if not lines:
        st.info('No strategy snapshots have been saved yet.')
        return
    start = min(s.index.min() for s in lines.values())
    end = max(s.index.max() for s in lines.values())
    if include_benchmarks:
        try:
            prices = benchmark_prices(('VOO', 'QQQ'), period='1y')
            for symbol in ('VOO', 'QQQ'):
                if symbol in prices:
                    s = prices.loc[(prices.index >= start) & (prices.index <= end), symbol].dropna()
                    if not s.empty:
                        lines[symbol] = s
        except Exception as exc:
            st.caption(f'Benchmark data unavailable: {exc}')
    # All lines start at $10k on their own first observed date; no fabricated earlier points.
    chart({name: normalize(s) for name, s in lines.items()},
          'Growth of $10,000 · first recorded date per series', height)
    st.caption('Each line is rebased to $10,000 at its first available observation. Different start dates and daily vs. intraday sampling limit direct comparisons.')


def performance_table(df):
    if df.empty:
        return pd.DataFrame()
    rows = []
    for name, group in df.groupby('strategy'):
        g = group.sort_values('run_date')
        first, last = g.iloc[0], g.iloc[-1]
        prev = g.iloc[-2]['equity'] if len(g) > 1 else np.nan
        rows.append({'Strategy': name, 'Equity': last['equity'],
                     'Since first record (%)': (last['equity'] / first['equity'] - 1) * 100 if first['equity'] > 0 else np.nan,
                     'Last snapshot change (%)': (last['equity'] / prev - 1) * 100 if pd.notna(prev) and prev > 0 else np.nan,
                     'Cash': last['cash'], 'Snapshots': len(g),
                     'Last update': last['run_date'].date()})
    return pd.DataFrame(rows).sort_values('Equity', ascending=False)


def holdings_from_snapshot(df, name):
    rows = df[df['strategy'] == name].sort_values('run_date')
    if rows.empty:
        return pd.DataFrame(columns=['Ticker', 'Shares'])
    try:
        pos = json.loads(rows.iloc[-1]['positions_json'])
        return pd.DataFrame([{'Ticker': t, 'Shares': float(q)} for t, q in pos.items() if float(q) > 0])
    except Exception:
        return pd.DataFrame(columns=['Ticker', 'Shares'])


@st.cache_data(ttl=300, show_spinner=False)
def scanner(symbols):
    raw = yf.download(list(symbols), period='5d', interval='5m',
                      auto_adjust=True, progress=False, threads=False, prepost=False)
    if raw.empty:
        return pd.DataFrame(), None
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    elif not isinstance(raw.columns, pd.MultiIndex):
        close.columns = [symbols[0]]
    if close.index.tz is None:
        return pd.DataFrame(), None
    close.index = close.index.tz_convert(ET)
    now = pd.Timestamp.now(tz=ET)
    close = close[(close.index.date == now.date()) &
                  (close.index + pd.Timedelta(minutes=5) <= now)]
    if close.empty:
        return pd.DataFrame(), None
    common = close.index[-1]
    result = []
    for symbol in symbols:
        if symbol not in close:
            result.append({'Ticker': symbol, 'Signal': 'Unavailable'})
            continue
        s = pd.to_numeric(close[symbol], errors='coerce').dropna()
        if len(s) < 8 or common - s.index[-1] > pd.Timedelta(minutes=5):
            result.append({'Ticker': symbol, 'Signal': 'Insufficient / stale'})
            continue
        price = float(s.iloc[-1]); fast = float(s.tail(3).mean()); slow = float(s.tail(8).mean())
        if slow <= 0 or price <= 0:
            continue
        score = (fast / slow - 1) * 100
        qualified = fast > slow * 1.0005 and price > float(s.iloc[-4])
        result.append({'Ticker': symbol, 'Price': price, 'Momentum %': score,
                       'Last 5m %': (price / float(s.iloc[-2]) - 1) * 100 if s.iloc[-2] > 0 else np.nan,
                       'Signal': 'Qualified' if qualified else 'Hold'})
    out = pd.DataFrame(result)
    if not out.empty:
        out = out.sort_values('Momentum %', ascending=False, na_position='last')
        qualified = out[out['Signal'] == 'Qualified'].head(3)['Ticker'].tolist()
        out['Top 3'] = out['Ticker'].isin(qualified)
    return out, common


def show_table(frame, height=360):
    # pandas Styler has .data but no .empty attribute.
    data = frame if isinstance(frame, pd.DataFrame) else getattr(frame, "data", None)
    if data is None or data.empty:
        st.info('No saved records yet.')
    else:
        st.dataframe(frame, hide_index=True, use_container_width=True, height=height)


st.sidebar.markdown('## ◈ RESEARCH TERMINAL')
st.sidebar.caption('Seven paper-trading strategies · US equities')
page = st.sidebar.radio('Workspace', NAV, label_visibility='collapsed')
st.sidebar.divider()
if st.sidebar.button('↻ Refresh saved data', use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption('Data may be delayed · Paper trades only')
st.sidebar.caption('Eastern time: ' + datetime.now(ET).strftime('%b %d · %I:%M %p'))

st.title(page)
st.caption('AI INVESTMENT RESEARCH LAB  /  ' + page.upper())
snaps = snapshots()
all_trades = trades()
all_decisions = decisions()
strategies = list(lab.MODELS.keys())

if page == 'Overview':
    perf = performance_table(snaps)
    if not perf.empty:
        daily = perf[perf['Strategy'] != lab.DAY_BOT]
        best = daily.iloc[0] if not daily.empty else perf.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Tracked strategies', f'{len(perf)} / {len(strategies)}')
        c2.metric('Top strategy equity', money(best['Equity']))
        c3.metric('Recorded paper trades', len(all_trades))
        c4.metric('Last snapshot', str(perf['Last update'].max()))
    else:
        st.info('No strategy snapshots yet. Your GitHub workflows populate this dashboard.')
    st.subheader('All strategies vs. the market')
    selected = st.multiselect('Show on chart', strategies, default=strategies,
                              help='Select any combination of the seven strategies.')
    benchmarks = st.checkbox('Include VOO and QQQ benchmarks', value=True)
    comparison(snaps, selected, benchmarks)
    st.subheader('Performance leaderboard')
    if not perf.empty:
        show_table(perf.style.format({'Equity': '${:,.2f}', 'Cash': '${:,.2f}',
                                      'Since first record (%)': '{:+.2f}',
                                      'Last snapshot change (%)': '{:+.2f}'}, na_rep='—'), 335)
    st.subheader('Latest activity')
    a, b = st.columns([1.3, 1])
    with a:
        st.markdown('**Recent transactions**')
        show_table(all_trades[['executed_at', 'strategy', 'ticker', 'side', 'shares', 'price', 'reason']].head(12)
                   if not all_trades.empty else pd.DataFrame(), 335)
    with b:
        st.markdown('**Recent strategy decisions**')
        show_table(all_decisions[['decided_at', 'strategy', 'action', 'reason']].head(12)
                   if not all_decisions.empty else pd.DataFrame(), 335)

elif page == 'Strategy Lab':
    st.caption('Analyze the six daily models and the seventh intraday model in one consistent workspace.')
    tabs = st.tabs(['Compare all', 'Strategy details', 'Transactions & decisions', 'Methodology'])
    with tabs[0]:
        names = st.multiselect('Strategies to compare', strategies, default=strategies, key='lab_compare')
        comparison(snaps, names)
        show_table(performance_table(snaps), 350)
    with tabs[1]:
        chosen = st.selectbox('Choose strategy', strategies)
        model = lab.MODELS[chosen]
        st.caption(model.get('description', ''))
        group = snaps[snaps['strategy'] == chosen] if not snaps.empty else pd.DataFrame()
        if not group.empty:
            group = group.sort_values('run_date')
            latest = group.iloc[-1]
            a, b, c, d = st.columns(4)
            a.metric('Equity', money(latest['equity']))
            b.metric('Cash', money(latest['cash']))
            c.metric('Return since first record', percent((latest['equity'] / group.iloc[0]['equity'] - 1) * 100))
            d.metric('Snapshots', len(group))
            comparison(snaps, [chosen])
            st.markdown('**Current recorded holdings**')
            show_table(holdings_from_snapshot(snaps, chosen), 230)
        else:
            st.info('This strategy has no saved equity history yet.')
        st.markdown('**Recent trades**')
        t = all_trades[all_trades['strategy'] == chosen] if not all_trades.empty else pd.DataFrame()
        show_table(t.head(50), 300)
        st.markdown('**Why the bot acted**')
        d = all_decisions[all_decisions['strategy'] == chosen] if not all_decisions.empty else pd.DataFrame()
        show_table(d.head(50), 300)
    with tabs[2]:
        filter_name = st.selectbox('Filter activity', ['All strategies'] + strategies)
        t = all_trades if filter_name == 'All strategies' or all_trades.empty else all_trades[all_trades['strategy'] == filter_name]
        d = all_decisions if filter_name == 'All strategies' or all_decisions.empty else all_decisions[all_decisions['strategy'] == filter_name]
        st.markdown('**Trade history**')
        show_table(t, 420)
        if not t.empty:
            st.download_button('Download trades CSV', t.to_csv(index=False), 'strategy_trades.csv', 'text/csv')
        st.markdown('**Decision history**')
        show_table(d, 360)
    with tabs[3]:
        for name, spec in lab.MODELS.items():
            with st.expander(name):
                st.write(spec.get('description', ''))
                st.write('**Method:**', spec.get('method', ''))
                st.write('**Watchlist:**', ', '.join(spec.get('tickers', [])))
        st.caption('Strategy names describe educational proxies, not affiliations with the named institutions or individuals.')
        st.caption('Paper executions use model prices and simulated fees. Snapshot returns are not audited live investment returns.')

elif page == 'Intraday Trader':
    st.caption('30-stock momentum scanner · five-minute bars · independent $10,000 paper account')
    bot_rows = snaps[snaps['strategy'] == lab.DAY_BOT] if not snaps.empty else pd.DataFrame()
    bot_trades = all_trades[all_trades['strategy'] == lab.DAY_BOT] if not all_trades.empty else pd.DataFrame()
    bot_decisions = all_decisions[all_decisions['strategy'] == lab.DAY_BOT] if not all_decisions.empty else pd.DataFrame()
    if not bot_rows.empty:
        last = bot_rows.sort_values('run_date').iloc[-1]
        a, b, c, d = st.columns(4)
        a.metric('Paper equity', money(last['equity']))
        b.metric('Cash available', money(last['cash']))
        c.metric('Executed trades', len(bot_trades))
        d.metric('Last recorded date', str(last['run_date'].date()))
    if not bot_rows.empty:
        chart(strategy_series(snaps, [lab.DAY_BOT]), 'Intraday bot paper equity · daily closing snapshots', 340)
    else:
        st.info('No intraday equity snapshots have been recorded yet.')
    st.subheader('Bot portfolio & trading history')
    a, b = st.columns([1, 1.3])
    with a:
        st.markdown('**Current positions**')
        show_table(holdings_from_snapshot(snaps, lab.DAY_BOT), 230)
    with b:
        st.markdown('**Recent executions**')
        show_table(bot_trades.head(40), 280)
    st.subheader('Live momentum scan')
    watchlist = tuple(lab.MODELS.get(lab.DAY_BOT, {}).get('tickers', []))
    if st.button('Refresh five-minute scan'):
        scanner.clear()
    if watchlist:
        try:
            with st.spinner('Reading five-minute market bars...'):
                scan, bar = scanner(watchlist)
            if scan.empty:
                st.info('No completed intraday bars available for today. The scanner is most useful during US market hours.')
            else:
                qualified = scan[scan['Signal'] == 'Qualified']
                a, b, c = st.columns(3)
                a.metric('Watchlist', len(watchlist))
                b.metric('Qualified signals', len(qualified))
                c.metric('Top 3 candidates', ', '.join(qualified.head(3)['Ticker']) or 'None')
                st.caption('Latest completed bar: ' + str(bar) + ' · Momentum = (3-bar mean / 8-bar mean − 1) × 100')
                show_table(scan, 550)
        except Exception as exc:
            st.warning(f'Scanner unavailable: {type(exc).__name__}: {exc}')
    else:
        st.warning('No intraday watchlist found in strategy_lab.py.')
    with st.expander('Decision log and operational limitations'):
        show_table(bot_decisions.head(80), 330)
        st.caption('The scanner is indicative, not an execution feed. GitHub scheduled jobs can be delayed or skipped; closing positions before market close is not guaranteed. The database stores one equity snapshot per day, not a full intraday equity curve.')

elif page == 'My Portfolio':
    st.caption('Your entered real holdings · market-value estimates · comparison against simulated strategies')
    holdings = table('personal_holdings')
    if holdings.empty:
        st.info('No personal holdings saved yet. Use the CSV importer below.')
    else:
        symbols = holdings['ticker'].astype(str).tolist()
        try:
            px = benchmark_prices(tuple(symbols), period='1mo')
            priced = []
            for row in holdings.itertuples(index=False):
                s = px[row.ticker].dropna() if row.ticker in px else pd.Series(dtype=float)
                price = float(s.iloc[-1]) if not s.empty else np.nan
                priced.append({'Ticker': row.ticker, 'Shares': row.shares, 'Price': price,
                               'Market value': float(row.shares) * price if pd.notna(price) else np.nan,
                               'Cost basis': row.cost_basis})
            priced = pd.DataFrame(priced)
            total = priced['Market value'].sum(min_count=1)
            a, b, c = st.columns(3)
            a.metric('Priced market value', money(total))
            b.metric('Positions', len(priced))
            c.metric('Missing prices', int(priced['Price'].isna().sum()))
        except Exception as exc:
            st.warning(f'Holdings valuation unavailable: {exc}')
    st.subheader('Personal portfolio history vs. paper strategies')
    st.caption('Your portfolio is modeled from June 1, 2026 using its initial saved weights. Future edits take effect on the next trading day. Each AI strategy starts on its first real snapshot date.')
    custom_ticker = st.text_input(
        'Add a stock or ETF to this comparison',
        placeholder='NVDA, AAPL, PFE, RUM, SPY',
        help='Chart only: this does not change your holdings.'
    ).strip().upper()
    chosen = st.multiselect('Compare with paper strategies', strategies,
                            default=strategies, key='personal_strategy_compare')
    cutoff = pd.Timestamp('2026-06-01')
    lines = {}
    # Each strategy enters the graph on its actual first saved date, not a common start.
    for name, series in strategy_series(snaps, chosen).items():
        series = series.loc[series.index >= cutoff].dropna().sort_index()
        if not series.empty:
            lines[name] = normalize(series)

    # Historical period uses the current portfolio allocation as a hypothetical
    # June 1 baseline. Future changes use saved weight events and never rewrite
    # earlier dates. This is a model, not a record of actual historical trades.
    weight_events = table('personal_weight_events')
    saved_events = []
    if not weight_events.empty:
        for record in weight_events.itertuples(index=False):
            try:
                saved_events.append((pd.Timestamp(record.effective_date).normalize(),
                                     int(record.id), json.loads(record.weights_json)))
            except (ValueError, TypeError, AttributeError):
                continue
    historical_base = None
    if saved_events:
        # The first event's weights are the frozen hypothetical initial mix.
        historical_base = sorted(saved_events, key=lambda x: (x[0], x[1]))[0][2]
    elif not holdings.empty:
        try:
            history_quotes = benchmark_prices(tuple(holdings['ticker'].astype(str).tolist()), period='5y')
            historical_base = portfolio_weights(holdings, history_quotes)
            if historical_base:
                # Persist initial weights so later edits cannot change the backtest.
                with engine.begin() as conn:
                    save_weight_event(conn, holdings, history_quotes,
                                      pd.Timestamp.now(tz=ET).date())
                st.cache_data.clear()
                saved_events = [(pd.Timestamp.now(tz=ET).normalize(), 0, historical_base)]
        except Exception as exc:
            st.warning(f'Portfolio history setup unavailable: {exc}')
    if historical_base:
        try:
            universe = sorted(set(historical_base).union(*[set(e[2]) for e in saved_events]))
            historical_quotes = benchmark_prices(tuple(universe), period='5y')
            # Exclude the initial baseline event from forward rebalance events.
            future_events = sorted(saved_events, key=lambda x: (x[0], x[1]))[1:]
            modeled = historical_weighted_portfolio(historical_quotes, historical_base,
                                                    future_events, cutoff)
            if not modeled.empty:
                lines['My Portfolio'] = modeled
        except Exception as exc:
            st.warning(f'Portfolio history calculation unavailable: {exc}')
    else:
        st.caption('Add holdings to create a hypothetical portfolio history from June 1.')

    symbols_to_compare = ['VOO', 'QQQ']
    if custom_ticker:
        if len(custom_ticker) > 16 or not all(c.isalnum() or c in '.-' for c in custom_ticker):
            st.warning('Invalid ticker. Try NVDA, BRK-B, or RUM.')
        elif custom_ticker not in symbols_to_compare:
            symbols_to_compare.append(custom_ticker)
    try:
        prices = benchmark_prices(tuple(symbols_to_compare), period='5y')
        for ticker in symbols_to_compare:
            if ticker not in prices.columns:
                if ticker == custom_ticker:
                    st.warning(f'No historical prices found for {ticker}.')
                continue
            series = prices[ticker].dropna().sort_index()
            series = series.loc[series.index >= cutoff]
            if not series.empty:
                lines[ticker] = normalize(series)
    except Exception as exc:
        st.warning(f'Could not retrieve comparison prices: {exc}')

    if lines:
        chart(lines, 'Growth of $10,000 · each series begins at its own first observation',
              520, x_start=cutoff)
        st.caption('My Portfolio: hypothetical $10,000 buy-and-hold using the initial saved allocation from June 1; later edits apply to future trading days only. Paper strategies start on their own first recorded date. This is not actual brokerage performance.')
    else:
        st.info('No historical data available for the comparison yet.')
    st.caption('Adding a ticker only changes this comparison chart. It does not place trades or change Supabase holdings.')
    if 'priced' in locals() and 'total' in locals():
        st.subheader('Holdings and allocation')
        priced['Weight %'] = priced['Market value'] / total * 100 if pd.notna(total) and total > 0 else np.nan
        show_table(priced.sort_values('Market value', ascending=False), 390)
        if pd.notna(total) and total > 0:
            fig = go.Figure(go.Pie(labels=priced['Ticker'], values=priced['Market value'].fillna(0), hole=.6))
            fig.update_layout(template='plotly_white', paper_bgcolor='#ffffff', height=330,
                              margin=dict(l=5,r=5,t=5,b=5), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader('Manage my holdings')
    st.caption('Update individual tickers without replacing your portfolio CSV. All changes affect your saved holdings, not your simulated strategies.')
    existing_tickers = sorted(holdings['ticker'].astype(str).str.upper().tolist()) if not holdings.empty else []
    with st.form('edit_single_holding', clear_on_submit=False):
        left, middle, right = st.columns([1.3, 1, 1])
        with left:
            edit_ticker = st.text_input('Ticker', placeholder='RUM').strip().upper()
        with middle:
            edit_action = st.selectbox('Action', ['Buy / add shares', 'Sell / subtract shares', 'Set exact shares', 'Remove holding'])
        with right:
            edit_qty = st.number_input('Shares', min_value=0.0, value=0.0, step=1.0, format='%.6f',
                                       disabled=edit_action == 'Remove holding')
        st.caption('Example: sell 100 RUM shares → ticker RUM, Sell / subtract shares, Shares 100. Your other holdings stay unchanged.')
        submitted = st.form_submit_button('Save holding change', type='primary')
    if submitted:
        if not edit_ticker or len(edit_ticker) > 16 or not all(ch.isalnum() or ch in '.-' for ch in edit_ticker):
            st.error('Enter a valid ticker symbol.')
        elif edit_action != 'Remove holding' and edit_qty <= 0 and edit_action != 'Set exact shares':
            st.error('Enter a positive number of shares.')
        else:
            try:
                with engine.begin() as conn:
                    # Lock the existing row so simultaneous edits cannot overwrite one another.
                    existing = conn.execute(text('SELECT shares, cost_basis FROM personal_holdings WHERE ticker=:t FOR UPDATE'),
                                            {'t': edit_ticker}).mappings().first()
                    old_qty = float(existing['shares']) if existing else 0.0
                    if edit_action == 'Buy / add shares':
                        new_qty = old_qty + float(edit_qty)
                    elif edit_action == 'Sell / subtract shares':
                        new_qty = old_qty - float(edit_qty)
                    elif edit_action == 'Set exact shares':
                        new_qty = float(edit_qty)
                    else:
                        new_qty = 0.0
                    if new_qty < -1e-8:
                        raise ValueError(f'You only have {old_qty:g} shares of {edit_ticker}; cannot sell {edit_qty:g}.')
                    if new_qty <= 1e-8:
                        conn.execute(text('DELETE FROM personal_holdings WHERE ticker=:t'), {'t': edit_ticker})
                    else:
                        conn.execute(text("""INSERT INTO personal_holdings (ticker,shares,cost_basis)
                                             VALUES (:t,:s,:c)
                                             ON CONFLICT (ticker) DO UPDATE SET shares=EXCLUDED.shares"""),
                                     {'t': edit_ticker, 's': new_qty,
                                      'c': float(existing['cost_basis']) if existing and existing['cost_basis'] is not None else None})
                record_portfolio_change(engine, benchmark_prices(tuple(sorted(set(existing_tickers + [edit_ticker]))), period='1mo'),
                                        pd.Timestamp.now(tz=ET).date() + pd.Timedelta(days=1))
                st.cache_data.clear()
                st.success(f'{edit_ticker}: {old_qty:g} → {max(new_qty, 0):g} shares saved.')
                st.rerun()
            except Exception as exc:
                st.error(f'Could not save change: {exc}')

    if existing_tickers:
        with st.expander('Current saved share counts'):
            show_table(holdings[['ticker', 'shares', 'cost_basis']].sort_values('ticker'), 260)

    with st.expander('Import or replace my holdings (CSV)', expanded=holdings.empty):
        st.caption('Upload a CSV with ticker and shares columns; cost_basis is optional. Uploading replaces the existing holdings table, but not strategy histories.')
        upload = st.file_uploader('Holdings CSV', type=['csv'])
        if upload is not None:
            try:
                incoming = pd.read_csv(upload)
                incoming.columns = incoming.columns.str.strip().str.lower()
                if not {'ticker', 'shares'}.issubset(incoming.columns):
                    st.error('CSV must contain ticker and shares columns.')
                else:
                    incoming['ticker'] = incoming['ticker'].astype(str).str.upper().str.strip()
                    incoming['shares'] = pd.to_numeric(incoming['shares'], errors='coerce')
                    if 'cost_basis' not in incoming:
                        incoming['cost_basis'] = np.nan
                    incoming['cost_basis'] = pd.to_numeric(incoming['cost_basis'], errors='coerce')
                    incoming = incoming.dropna(subset=['ticker', 'shares'])
                    incoming = incoming[(incoming['shares'] >= 0) & (incoming['ticker'] != '') & (incoming['ticker'] != 'NAN')]
                    incoming = incoming.groupby('ticker', as_index=False).agg({'shares': 'sum', 'cost_basis': 'first'})
                    st.dataframe(incoming, hide_index=True)
                    if st.button('Confirm: replace my holdings', type='primary'):
                        with engine.begin() as conn:
                            conn.execute(text('DELETE FROM personal_holdings'))
                            for r in incoming.itertuples(index=False):
                                basis = float(r.cost_basis) if pd.notna(r.cost_basis) else None
                                conn.execute(text('INSERT INTO personal_holdings (ticker,shares,cost_basis) VALUES (:t,:s,:c)'),
                                             {'t': r.ticker, 's': float(r.shares), 'c': basis})
                        updated_tickers = tuple(incoming['ticker'].astype(str).tolist())
                        record_portfolio_change(engine, benchmark_prices(updated_tickers, period='1mo'),
                                                pd.Timestamp.now(tz=ET).date() + pd.Timedelta(days=1))
                        st.cache_data.clear()
                        st.rerun()
            except Exception as exc:
                st.error(f'Could not import CSV: {exc}')

st.divider()
st.caption('Educational paper trading only · Quotes may be delayed · Historical results do not predict future performance')

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
        return pd.DataFrame()
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    elif not isinstance(raw.columns, pd.MultiIndex):
        close.columns = [symbols[0]]
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_convert(ET).tz_localize(None)
    close.index = close.index.normalize()
    return close.apply(pd.to_numeric, errors='coerce').sort_index().ffill()


def snapshots():
    df = table('lab_snapshots').copy()
    if df.empty:
        return df
    df['run_date'] = pd.to_datetime(df['run_date'], errors='coerce').dt.normalize()
    df['equity'] = pd.to_numeric(df['equity'], errors='coerce')
    df['cash'] = pd.to_numeric(df['cash'], errors='coerce')
    return df.dropna(subset=['run_date', 'equity']).sort_values('run_date')


def trades():
    df = table('lab_trades').copy()
    if not df.empty:
        df['executed_at'] = pd.to_datetime(df['executed_at'], errors='coerce', utc=True)
        df = df.sort_values('executed_at', ascending=False)
    return df


def decisions():
    df = table('lab_decisions').copy()
    if not df.empty:
        df['decided_at'] = pd.to_datetime(df['decided_at'], errors='coerce', utc=True)
        df = df.sort_values('decided_at', ascending=False)
    return df


def money(value):
    return f'${value:,.2f}' if pd.notna(value) else '—'


def percent(value):
    return f'{value:+.2f}%' if pd.notna(value) else '—'


def normalize(series):
    series = series.dropna().astype(float)
    if series.empty or series.iloc[0] <= 0:
        return series
    return series / series.iloc[0] * START


def chart(lines, title, height=470, x_start=None):
    fig = go.Figure()
    for i, (name, series) in enumerate(lines.items()):
        s = series.dropna().sort_index()
        if s.empty:
            continue
        color = '#334155' if name == 'VOO' else '#f59e0b' if name == 'QQQ' else COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode='lines',
                                 line=dict(color=color, width=2.5 if name in ('VOO', 'QQQ') else 2),
                                 hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.2f}<extra>%{fullData.name}</extra>'))
    fig.update_layout(title=title, height=height, template='plotly_white',
                      paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                      font=dict(color='#334155'), margin=dict(l=12, r=12, t=52, b=10),
                      hovermode='x unified', legend=dict(orientation='h', y=-0.18),
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#e2e8f0', tickprefix='$'))
    if x_start is not None:
        fig.update_xaxes(range=[pd.Timestamp(x_start), max(pd.Timestamp.now().normalize(), max((v.dropna().index.max() for v in lines.values() if not v.dropna().empty), default=pd.Timestamp.now().normalize()))])
    st.plotly_chart(fig, use_container_width=True)


def strategy_series(df, names):
    result = {}
    for name in names:
        rows = df[df['strategy'] == name].drop_duplicates('run_date', keep='last')
        if not rows.empty:
            result[name] = pd.Series(rows['equity'].values, index=rows['run_date'])
    return result


def comparison(df, names, include_benchmarks=True, height=490):
    lines = strategy_series(df, names)
    if not lines:
        st.info('No strategy snapshots have been saved yet.')
        return
    start = min(s.index.min() for s in lines.values())
    end = max(s.index.max() for s in lines.values())
    if include_benchmarks:
        try:
            prices = benchmark_prices(('VOO', 'QQQ'), period='1y')
            for symbol in ('VOO', 'QQQ'):
                if symbol in prices:
                    s = prices.loc[(prices.index >= start) & (prices.index <= end), symbol].dropna()
                    if not s.empty:
                        lines[symbol] = s
        except Exception as exc:
            st.caption(f'Benchmark data unavailable: {exc}')
    # All lines start at $10k on their own first observed date; no fabricated earlier points.
    chart({name: normalize(s) for name, s in lines.items()},
          'Growth of $10,000 · first recorded date per series', height)
    st.caption('Each line is rebased to $10,000 at its first available observation. Different start dates and daily vs. intraday sampling limit direct comparisons.')


def performance_table(df):
    if df.empty:
        return pd.DataFrame()
    rows = []
    for name, group in df.groupby('strategy'):
        g = group.sort_values('run_date')
        first, last = g.iloc[0], g.iloc[-1]
        prev = g.iloc[-2]['equity'] if len(g) > 1 else np.nan
        rows.append({'Strategy': name, 'Equity': last['equity'],
                     'Since first record (%)': (last['equity'] / first['equity'] - 1) * 100 if first['equity'] > 0 else np.nan,
                     'Last snapshot change (%)': (last['equity'] / prev - 1) * 100 if pd.notna(prev) and prev > 0 else np.nan,
                     'Cash': last['cash'], 'Snapshots': len(g),
                     'Last update': last['run_date'].date()})
    return pd.DataFrame(rows).sort_values('Equity', ascending=False)


def holdings_from_snapshot(df, name):
    rows = df[df['strategy'] == name].sort_values('run_date')
    if rows.empty:
        return pd.DataFrame(columns=['Ticker', 'Shares'])
    try:
        pos = json.loads(rows.iloc[-1]['positions_json'])
        return pd.DataFrame([{'Ticker': t, 'Shares': float(q)} for t, q in pos.items() if float(q) > 0])
    except Exception:
        return pd.DataFrame(columns=['Ticker', 'Shares'])


@st.cache_data(ttl=300, show_spinner=False)
def scanner(symbols):
    raw = yf.download(list(symbols), period='5d', interval='5m',
                      auto_adjust=True, progress=False, threads=False, prepost=False)
    if raw.empty:
        return pd.DataFrame(), None
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    elif not isinstance(raw.columns, pd.MultiIndex):
        close.columns = [symbols[0]]
    if close.index.tz is None:
        return pd.DataFrame(), None
    close.index = close.index.tz_convert(ET)
    now = pd.Timestamp.now(tz=ET)
    close = close[(close.index.date == now.date()) &
                  (close.index + pd.Timedelta(minutes=5) <= now)]
    if close.empty:
        return pd.DataFrame(), None
    common = close.index[-1]
    result = []
    for symbol in symbols:
        if symbol not in close:
            result.append({'Ticker': symbol, 'Signal': 'Unavailable'})
            continue
        s = pd.to_numeric(close[symbol], errors='coerce').dropna()
        if len(s) < 8 or common - s.index[-1] > pd.Timedelta(minutes=5):
            result.append({'Ticker': symbol, 'Signal': 'Insufficient / stale'})
            continue
        price = float(s.iloc[-1]); fast = float(s.tail(3).mean()); slow = float(s.tail(8).mean())
        if slow <= 0 or price <= 0:
            continue
        score = (fast / slow - 1) * 100
        qualified = fast > slow * 1.0005 and price > float(s.iloc[-4])
        result.append({'Ticker': symbol, 'Price': price, 'Momentum %': score,
                       'Last 5m %': (price / float(s.iloc[-2]) - 1) * 100 if s.iloc[-2] > 0 else np.nan,
                       'Signal': 'Qualified' if qualified else 'Hold'})
    out = pd.DataFrame(result)
    if not out.empty:
        out = out.sort_values('Momentum %', ascending=False, na_position='last')
        qualified = out[out['Signal'] == 'Qualified'].head(3)['Ticker'].tolist()
        out['Top 3'] = out['Ticker'].isin(qualified)
    return out, common


def show_table(frame, height=360):
    # pandas Styler has .data but no .empty attribute.
    data = frame if isinstance(frame, pd.DataFrame) else getattr(frame, "data", None)
    if data is None or data.empty:
        st.info('No saved records yet.')
    else:
        st.dataframe(frame, hide_index=True, use_container_width=True, height=height)


st.sidebar.markdown('## ◈ RESEARCH TERMINAL')
st.sidebar.caption('Seven paper-trading strategies · US equities')
page = st.sidebar.radio('Workspace', NAV, label_visibility='collapsed')
st.sidebar.divider()
if st.sidebar.button('↻ Refresh saved data', use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption('Data may be delayed · Paper trades only')
st.sidebar.caption('Eastern time: ' + datetime.now(ET).strftime('%b %d · %I:%M %p'))

st.title(page)
st.caption('AI INVESTMENT RESEARCH LAB  /  ' + page.upper())
snaps = snapshots()
all_trades = trades()
all_decisions = decisions()
strategies = list(lab.MODELS.keys())

if page == 'Overview':
    perf = performance_table(snaps)
    if not perf.empty:
        daily = perf[perf['Strategy'] != lab.DAY_BOT]
        best = daily.iloc[0] if not daily.empty else perf.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Tracked strategies', f'{len(perf)} / {len(strategies)}')
        c2.metric('Top strategy equity', money(best['Equity']))
        c3.metric('Recorded paper trades', len(all_trades))
        c4.metric('Last snapshot', str(perf['Last update'].max()))
    else:
        st.info('No strategy snapshots yet. Your GitHub workflows populate this dashboard.')
    st.subheader('All strategies vs. the market')
    selected = st.multiselect('Show on chart', strategies, default=strategies,
                              help='Select any combination of the seven strategies.')
    benchmarks = st.checkbox('Include VOO and QQQ benchmarks', value=True)
    comparison(snaps, selected, benchmarks)
    st.subheader('Performance leaderboard')
    if not perf.empty:
        show_table(perf.style.format({'Equity': '${:,.2f}', 'Cash': '${:,.2f}',
                                      'Since first record (%)': '{:+.2f}',
                                      'Last snapshot change (%)': '{:+.2f}'}, na_rep='—'), 335)
    st.subheader('Latest activity')
    a, b = st.columns([1.3, 1])
    with a:
        st.markdown('**Recent transactions**')
        show_table(all_trades[['executed_at', 'strategy', 'ticker', 'side', 'shares', 'price', 'reason']].head(12)
                   if not all_trades.empty else pd.DataFrame(), 335)
    with b:
        st.markdown('**Recent strategy decisions**')
        show_table(all_decisions[['decided_at', 'strategy', 'action', 'reason']].head(12)
                   if not all_decisions.empty else pd.DataFrame(), 335)

elif page == 'Strategy Lab':
    st.caption('Analyze the six daily models and the seventh intraday model in one consistent workspace.')
    tabs = st.tabs(['Compare all', 'Strategy details', 'Transactions & decisions', 'Methodology'])
    with tabs[0]:
        names = st.multiselect('Strategies to compare', strategies, default=strategies, key='lab_compare')
        comparison(snaps, names)
        show_table(performance_table(snaps), 350)
    with tabs[1]:
        chosen = st.selectbox('Choose strategy', strategies)
        model = lab.MODELS[chosen]
        st.caption(model.get('description', ''))
        group = snaps[snaps['strategy'] == chosen] if not snaps.empty else pd.DataFrame()
        if not group.empty:
            group = group.sort_values('run_date')
            latest = group.iloc[-1]
            a, b, c, d = st.columns(4)
            a.metric('Equity', money(latest['equity']))
            b.metric('Cash', money(latest['cash']))
            c.metric('Return since first record', percent((latest['equity'] / group.iloc[0]['equity'] - 1) * 100))
            d.metric('Snapshots', len(group))
            comparison(snaps, [chosen])
            st.markdown('**Current recorded holdings**')
            show_table(holdings_from_snapshot(snaps, chosen), 230)
        else:
            st.info('This strategy has no saved equity history yet.')
        st.markdown('**Recent trades**')
        t = all_trades[all_trades['strategy'] == chosen] if not all_trades.empty else pd.DataFrame()
        show_table(t.head(50), 300)
        st.markdown('**Why the bot acted**')
        d = all_decisions[all_decisions['strategy'] == chosen] if not all_decisions.empty else pd.DataFrame()
        show_table(d.head(50), 300)
    with tabs[2]:
        filter_name = st.selectbox('Filter activity', ['All strategies'] + strategies)
        t = all_trades if filter_name == 'All strategies' or all_trades.empty else all_trades[all_trades['strategy'] == filter_name]
        d = all_decisions if filter_name == 'All strategies' or all_decisions.empty else all_decisions[all_decisions['strategy'] == filter_name]
        st.markdown('**Trade history**')
        show_table(t, 420)
        if not t.empty:
            st.download_button('Download trades CSV', t.to_csv(index=False), 'strategy_trades.csv', 'text/csv')
        st.markdown('**Decision history**')
        show_table(d, 360)
    with tabs[3]:
        for name, spec in lab.MODELS.items():
            with st.expander(name):
                st.write(spec.get('description', ''))
                st.write('**Method:**', spec.get('method', ''))
                st.write('**Watchlist:**', ', '.join(spec.get('tickers', [])))
        st.caption('Strategy names describe educational proxies, not affiliations with the named institutions or individuals.')
        st.caption('Paper executions use model prices and simulated fees. Snapshot returns are not audited live investment returns.')

elif page == 'Intraday Trader':
    st.caption('30-stock momentum scanner · five-minute bars · independent $10,000 paper account')
    bot_rows = snaps[snaps['strategy'] == lab.DAY_BOT] if not snaps.empty else pd.DataFrame()
    bot_trades = all_trades[all_trades['strategy'] == lab.DAY_BOT] if not all_trades.empty else pd.DataFrame()
    bot_decisions = all_decisions[all_decisions['strategy'] == lab.DAY_BOT] if not all_decisions.empty else pd.DataFrame()
    if not bot_rows.empty:
        last = bot_rows.sort_values('run_date').iloc[-1]
        a, b, c, d = st.columns(4)
        a.metric('Paper equity', money(last['equity']))
        b.metric('Cash available', money(last['cash']))
        c.metric('Executed trades', len(bot_trades))
        d.metric('Last recorded date', str(last['run_date'].date()))
    if not bot_rows.empty:
        chart(strategy_series(snaps, [lab.DAY_BOT]), 'Intraday bot paper equity · daily closing snapshots', 340)
    else:
        st.info('No intraday equity snapshots have been recorded yet.')
    st.subheader('Bot portfolio & trading history')
    a, b = st.columns([1, 1.3])
    with a:
        st.markdown('**Current positions**')
        show_table(holdings_from_snapshot(snaps, lab.DAY_BOT), 230)
    with b:
        st.markdown('**Recent executions**')
        show_table(bot_trades.head(40), 280)
    st.subheader('Live momentum scan')
    watchlist = tuple(lab.MODELS.get(lab.DAY_BOT, {}).get('tickers', []))
    if st.button('Refresh five-minute scan'):
        scanner.clear()
    if watchlist:
        try:
            with st.spinner('Reading five-minute market bars...'):
                scan, bar = scanner(watchlist)
            if scan.empty:
                st.info('No completed intraday bars available for today. The scanner is most useful during US market hours.')
            else:
                qualified = scan[scan['Signal'] == 'Qualified']
                a, b, c = st.columns(3)
                a.metric('Watchlist', len(watchlist))
                b.metric('Qualified signals', len(qualified))
                c.metric('Top 3 candidates', ', '.join(qualified.head(3)['Ticker']) or 'None')
                st.caption('Latest completed bar: ' + str(bar) + ' · Momentum = (3-bar mean / 8-bar mean − 1) × 100')
                show_table(scan, 550)
        except Exception as exc:
            st.warning(f'Scanner unavailable: {type(exc).__name__}: {exc}')
    else:
        st.warning('No intraday watchlist found in strategy_lab.py.')
    with st.expander('Decision log and operational limitations'):
        show_table(bot_decisions.head(80), 330)
        st.caption('The scanner is indicative, not an execution feed. GitHub scheduled jobs can be delayed or skipped; closing positions before market close is not guaranteed. The database stores one equity snapshot per day, not a full intraday equity curve.')

elif page == 'My Portfolio':
    st.caption('Your entered real holdings · market-value estimates · comparison against simulated strategies')
    holdings = table('personal_holdings')
    if holdings.empty:
        st.info('No personal holdings saved yet. Use the CSV importer below.')
    else:
        symbols = holdings['ticker'].astype(str).tolist()
        try:
            px = benchmark_prices(tuple(symbols), period='1mo')
            priced = []
            for row in holdings.itertuples(index=False):
                s = px[row.ticker].dropna() if row.ticker in px else pd.Series(dtype=float)
                price = float(s.iloc[-1]) if not s.empty else np.nan
                priced.append({'Ticker': row.ticker, 'Shares': row.shares, 'Price': price,
                               'Market value': float(row.shares) * price if pd.notna(price) else np.nan,
                               'Cost basis': row.cost_basis})
            priced = pd.DataFrame(priced)
            total = priced['Market value'].sum(min_count=1)
            a, b, c = st.columns(3)
            a.metric('Priced market value', money(total))
            b.metric('Positions', len(priced))
            c.metric('Missing prices', int(priced['Price'].isna().sum()))
        except Exception as exc:
            st.warning(f'Holdings valuation unavailable: {exc}')
    st.subheader('Personal portfolio history vs. paper strategies')
    st.caption('Chart timeline begins June 1, 2026. Each paper strategy starts on its actual first recorded snapshot date; no earlier performance is invented.')
    custom_ticker = st.text_input(
        'Add a stock or ETF to this comparison',
        placeholder='NVDA, AAPL, PFE, RUM, SPY',
        help='Chart only: this does not change your holdings.'
    ).strip().upper()
    chosen = st.multiselect('Compare with paper strategies', strategies,
                            default=strategies, key='personal_strategy_compare')
    cutoff = pd.Timestamp('2026-06-01')
    lines = {}
    # Each strategy enters the graph on its actual first saved date, not a common start.
    for name, series in strategy_series(snaps, chosen).items():
        series = series.loc[series.index >= cutoff].dropna().sort_index()
        if not series.empty:
            lines[name] = normalize(series)

    personal = table('personal_snapshots')
    if not personal.empty and {'snapshot_date', 'market_value'}.issubset(personal.columns):
        personal['snapshot_date'] = pd.to_datetime(personal['snapshot_date'], errors='coerce').dt.normalize()
        personal['market_value'] = pd.to_numeric(personal['market_value'], errors='coerce')
        personal = personal.dropna(subset=['snapshot_date', 'market_value']).sort_values('snapshot_date')
        personal = personal.loc[personal['snapshot_date'] >= cutoff]
        if not personal.empty:
            series = pd.Series(personal['market_value'].values, index=personal['snapshot_date'])
            lines['My Portfolio'] = normalize(series)
    if 'My Portfolio' not in lines:
        st.caption('Your portfolio line starts when the first personal snapshot was actually saved. Earlier holdings history cannot be reconstructed accurately from current shares alone.')

    symbols_to_compare = ['VOO', 'QQQ']
    if custom_ticker:
        if len(custom_ticker) > 16 or not all(c.isalnum() or c in '.-' for c in custom_ticker):
            st.warning('Invalid ticker. Try NVDA, BRK-B, or RUM.')
        elif custom_ticker not in symbols_to_compare:
            symbols_to_compare.append(custom_ticker)
    try:
        prices = benchmark_prices(tuple(symbols_to_compare), period='5y')
        for ticker in symbols_to_compare:
            if ticker not in prices.columns:
                if ticker == custom_ticker:
                    st.warning(f'No historical prices found for {ticker}.')
                continue
            series = prices[ticker].dropna().sort_index()
            series = series.loc[series.index >= cutoff]
            if not series.empty:
                lines[ticker] = normalize(series)
    except Exception as exc:
        st.warning(f'Could not retrieve comparison prices: {exc}')

    if lines:
        chart(lines, 'Growth of $10,000 · each series begins at its own first observation',
              520, x_start=cutoff)
        st.caption('All lines are independently rebased to $10,000 at their first available observation on or after June 1. Personal portfolio snapshots may reflect deposits, withdrawals and changes in holdings; this is not a cash-flow-adjusted investment return.')
    else:
        st.info('No historical data available for the comparison yet.')
    st.caption('Adding a ticker only changes this comparison chart. It does not place trades or change Supabase holdings.')
    if 'priced' in locals() and 'total' in locals():
        st.subheader('Holdings and allocation')
        priced['Weight %'] = priced['Market value'] / total * 100 if pd.notna(total) and total > 0 else np.nan
        show_table(priced.sort_values('Market value', ascending=False), 390)
        if pd.notna(total) and total > 0:
            fig = go.Figure(go.Pie(labels=priced['Ticker'], values=priced['Market value'].fillna(0), hole=.6))
            fig.update_layout(template='plotly_white', paper_bgcolor='#ffffff', height=330,
                              margin=dict(l=5,r=5,t=5,b=5), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader('Manage my holdings')
    st.caption('Update individual tickers without replacing your portfolio CSV. All changes affect your saved holdings, not your simulated strategies.')
    existing_tickers = sorted(holdings['ticker'].astype(str).str.upper().tolist()) if not holdings.empty else []
    with st.form('edit_single_holding', clear_on_submit=False):
        left, middle, right = st.columns([1.3, 1, 1])
        with left:
            edit_ticker = st.text_input('Ticker', placeholder='RUM').strip().upper()
        with middle:
            edit_action = st.selectbox('Action', ['Buy / add shares', 'Sell / subtract shares', 'Set exact shares', 'Remove holding'])
        with right:
            edit_qty = st.number_input('Shares', min_value=0.0, value=0.0, step=1.0, format='%.6f',
                                       disabled=edit_action == 'Remove holding')
        st.caption('Example: sell 100 RUM shares → ticker RUM, Sell / subtract shares, Shares 100. Your other holdings stay unchanged.')
        submitted = st.form_submit_button('Save holding change', type='primary')
    if submitted:
        if not edit_ticker or len(edit_ticker) > 16 or not all(ch.isalnum() or ch in '.-' for ch in edit_ticker):
            st.error('Enter a valid ticker symbol.')
        elif edit_action != 'Remove holding' and edit_qty <= 0 and edit_action != 'Set exact shares':
            st.error('Enter a positive number of shares.')
        else:
            try:
                with engine.begin() as conn:
                    # Lock the existing row so simultaneous edits cannot overwrite one another.
                    existing = conn.execute(text('SELECT shares, cost_basis FROM personal_holdings WHERE ticker=:t FOR UPDATE'),
                                            {'t': edit_ticker}).mappings().first()
                    old_qty = float(existing['shares']) if existing else 0.0
                    if edit_action == 'Buy / add shares':
                        new_qty = old_qty + float(edit_qty)
                    elif edit_action == 'Sell / subtract shares':
                        new_qty = old_qty - float(edit_qty)
                    elif edit_action == 'Set exact shares':
                        new_qty = float(edit_qty)
                    else:
                        new_qty = 0.0
                    if new_qty < -1e-8:
                        raise ValueError(f'You only have {old_qty:g} shares of {edit_ticker}; cannot sell {edit_qty:g}.')
                    if new_qty <= 1e-8:
                        conn.execute(text('DELETE FROM personal_holdings WHERE ticker=:t'), {'t': edit_ticker})
                    else:
                        conn.execute(text("""INSERT INTO personal_holdings (ticker,shares,cost_basis)
                                             VALUES (:t,:s,:c)
                                             ON CONFLICT (ticker) DO UPDATE SET shares=EXCLUDED.shares"""),
                                     {'t': edit_ticker, 's': new_qty,
                                      'c': float(existing['cost_basis']) if existing and existing['cost_basis'] is not None else None})
                st.cache_data.clear()
                st.success(f'{edit_ticker}: {old_qty:g} → {max(new_qty, 0):g} shares saved.')
                st.rerun()
            except Exception as exc:
                st.error(f'Could not save change: {exc}')

    if existing_tickers:
        with st.expander('Current saved share counts'):
            show_table(holdings[['ticker', 'shares', 'cost_basis']].sort_values('ticker'), 260)

    with st.expander('Import or replace my holdings (CSV)', expanded=holdings.empty):
        st.caption('Upload a CSV with ticker and shares columns; cost_basis is optional. Uploading replaces the existing holdings table, but not strategy histories.')
        upload = st.file_uploader('Holdings CSV', type=['csv'])
        if upload is not None:
            try:
                incoming = pd.read_csv(upload)
                incoming.columns = incoming.columns.str.strip().str.lower()
                if not {'ticker', 'shares'}.issubset(incoming.columns):
                    st.error('CSV must contain ticker and shares columns.')
                else:
                    incoming['ticker'] = incoming['ticker'].astype(str).str.upper().str.strip()
                    incoming['shares'] = pd.to_numeric(incoming['shares'], errors='coerce')
                    if 'cost_basis' not in incoming:
                        incoming['cost_basis'] = np.nan
                    incoming['cost_basis'] = pd.to_numeric(incoming['cost_basis'], errors='coerce')
                    incoming = incoming.dropna(subset=['ticker', 'shares'])
                    incoming = incoming[(incoming['shares'] >= 0) & (incoming['ticker'] != '') & (incoming['ticker'] != 'NAN')]
                    incoming = incoming.groupby('ticker', as_index=False).agg({'shares': 'sum', 'cost_basis': 'first'})
                    st.dataframe(incoming, hide_index=True)
                    if st.button('Confirm: replace my holdings', type='primary'):
                        with engine.begin() as conn:
                            conn.execute(text('DELETE FROM personal_holdings'))
                            for r in incoming.itertuples(index=False):
                                basis = float(r.cost_basis) if pd.notna(r.cost_basis) else None
                                conn.execute(text('INSERT INTO personal_holdings (ticker,shares,cost_basis) VALUES (:t,:s,:c)'),
                                             {'t': r.ticker, 's': float(r.shares), 'c': basis})
                        st.cache_data.clear()
                        st.rerun()
            except Exception as exc:
                st.error(f'Could not import CSV: {exc}')

st.divider()
st.caption('Educational paper trading only · Quotes may be delayed · Historical results do not predict future performance')
