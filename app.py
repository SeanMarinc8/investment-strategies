import os
import hmac
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

from engine import db, init, read, run, prices, score_frame, holdings, WATCH, START_CASH

st.set_page_config(page_title='AI Investment Research Lab', page_icon='📈', layout='wide')
st.title('📈 AI Investment Research Lab')
st.caption('Research, explainable signals and a $10,000 forward paper-trading experiment. No real orders.')

try:
    secrets = st.secrets
    app_password = secrets.get('APP_PASSWORD')
    database_url = secrets.get('DATABASE_URL')
except (FileNotFoundError, KeyError):
    app_password = None
    database_url = None

if not app_password:
    st.error('APP_PASSWORD is missing from Streamlit Secrets.')
    st.stop()
password = st.text_input('App access password', type='password')
if not hmac.compare_digest(password, str(app_password)):
    st.stop()
if not database_url:
    st.error('DATABASE_URL is missing from Streamlit Secrets.')
    st.stop()
os.environ['DATABASE_URL'] = str(database_url)

try:
    engine = db()
    init(engine)
    sn = read(engine, 'snapshots')
    tr = read(engine, 'trades')
    de = read(engine, 'decisions')
except Exception as exc:
    st.error('Database unavailable.')
    msg = str(exc).lower()
    if 'password authentication failed' in msg:
        st.warning('Supabase rejected the database password.')
    elif 'tenant or user not found' in msg:
        st.warning('Check the Supabase pooler username and host.')
    elif 'timeout' in msg:
        st.warning('The database connection timed out.')
    else:
        st.write('Error category:', type(exc).__name__)
    logging.error('Database initialization failed: %s', type(exc).__name__)
    st.stop()

with st.sidebar:
    page = st.radio('Pages', ['Dashboard', 'Daily Decisions', 'Research', 'Trade Log', 'My Portfolio Comparison', 'Data & Methodology'])
    if st.button('Refresh market data'):
        st.cache_data.clear()
        st.rerun()
    st.caption('Refresh updates quotes, not trades. Your scheduled GitHub Action is separate.')

@st.cache_data(ttl=300, show_spinner=False)
def get_quotes(symbols, period='1y'):
    symbols = tuple(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
    if not symbols:
        return pd.DataFrame()
    raw = yf.download(list(symbols), period=period, interval='1d', auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close'].copy()
    else:
        close = raw[['Close']].rename(columns={'Close': symbols[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    close.index = pd.to_datetime(close.index, utc=True).tz_localize(None).normalize()
    close = close.sort_index()
    close = close.loc[:, ~close.columns.duplicated()]
    return close.apply(pd.to_numeric, errors='coerce')

@st.cache_data(ttl=300, show_spinner=False)
def strategy_signals():
    return score_frame(prices(WATCH))

def dollar(x):
    return f'${x:,.2f}'

def strategy_series(snapshots):
    if snapshots.empty:
        return pd.Series(dtype=float)
    x = snapshots.copy()
    x['date'] = pd.to_datetime(x['run_date'], errors='coerce').dt.normalize()
    x['equity'] = pd.to_numeric(x['equity'], errors='coerce')
    x = x.dropna(subset=['date', 'equity']).sort_values('date').drop_duplicates('date', keep='last')
    return pd.Series(x['equity'].to_numpy(), index=pd.DatetimeIndex(x['date']), name='AI Strategy')

def price_comparison(symbols, period, include_ai=True):
    """Return a common $10,000 starting-value chart; AI uses saved daily snapshots only."""
    quote = get_quotes(tuple(symbols), period)
    if quote.empty:
        return pd.DataFrame(), [], None
    valid = [s for s in symbols if s in quote.columns and quote[s].notna().any()]
    missing = [s for s in symbols if s not in valid]
    if not valid:
        return pd.DataFrame(), missing, None
    quote = quote[valid]
    start = quote.index.min()
    ai = strategy_series(sn) if include_ai else pd.Series(dtype=float)
    if not ai.empty:
        # Compare over the overlap period, never create a synthetic AI backtest.
        start = max(start, ai.index.min())
    quote = quote.loc[quote.index >= start]
    if quote.empty:
        return pd.DataFrame(), missing, None
    # Anchor all quoted assets to the same first date with complete valid prices.
    complete = quote.dropna(how='any')
    if complete.empty:
        return pd.DataFrame(), missing, None
    common_start = complete.index.min()
    quote = quote.loc[quote.index >= common_start]
    base = quote.loc[common_start]
    indexed = quote.divide(base).multiply(START_CASH)
    if include_ai and not ai.empty:
        # AI snapshots are only shown on recorded dates; no fabricated daily points.
        ai = ai.loc[ai.index >= common_start]
        if not ai.empty:
            ai_base = ai.iloc[0]
            if ai_base > 0:
                indexed['AI Strategy'] = ai / ai_base * START_CASH
    return indexed, missing, common_start

if page == 'Dashboard':
    st.subheader('Virtual strategy vs. buy-and-hold')
    if sn.empty:
        st.info('No strategy snapshots yet. Use Daily Decisions to record the first virtual trading day.')
    else:
        latest = sn.sort_values('run_date').iloc[-1]
        a, b, c = st.columns(3)
        a.metric('AI strategy equity', dollar(float(latest.equity)), f'{(float(latest.equity)/START_CASH-1):+.2%}')
        b.metric('VOO benchmark', dollar(float(latest.benchmark_voo)) if pd.notna(latest.benchmark_voo) else 'N/A')
        c.metric('QQQ benchmark', dollar(float(latest.benchmark_qqq)) if pd.notna(latest.benchmark_qqq) else 'N/A')
        plot = sn.sort_values('run_date').rename(columns={'equity':'AI Strategy','benchmark_voo':'VOO','benchmark_qqq':'QQQ'})
        fig = px.line(plot, x='run_date', y=['AI Strategy','VOO','QQQ'], markers=True, title='Recorded portfolio values ($)')
        fig.update_layout(legend_title_text='Portfolio / benchmark', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        cash, positions = holdings(tr)
        st.metric('Virtual cash balance', dollar(cash))
        if positions:
            st.dataframe(pd.DataFrame([{'Ticker':k,'Shares':v} for k,v in positions.items()]), hide_index=True, use_container_width=True)
    with st.expander('What do these numbers mean?', expanded=True):
        st.markdown('**AI strategy equity** = remaining simulated cash + market value of simulated holdings at the most recent recorded run. **VOO** tracks a hypothetical $10,000 investment in an S&P 500 ETF; **QQQ** tracks a separate $10,000 investment in a Nasdaq-100 ETF. Each begins at the strategy inception date. The graph uses only saved daily snapshots, not a backtest.')
        st.caption('Simulated fills use already-observed closing prices. Actual execution at those same prices is not assured.')

elif page == 'Daily Decisions':
    st.subheader('Explainable daily investment decisions')
    st.markdown('The engine ranks a fixed watchlist using **trend, momentum and volatility**. It is a deterministic scoring model, not an LLM or predictive AI.')
    with st.expander('How does the score work?', expanded=True):
        st.markdown('''**200-day trend:** +2 if the latest price is above its 200-day average, otherwise −2.  
**50-day vs 200-day average:** +1 if the 50-day average is higher, otherwise −1.  
**20-day momentum:** the last 20 trading-day return × 20, limited to −2 through +2.  
**Volatility penalty:** subtract annualized standard deviation of recent daily returns, capped at 1.5.  
**Eligibility:** only assets with score above 0.8 and price above their 200-day average qualify. The four highest-ranked qualifying assets can be bought, subject to cash limits.''')
    try:
        ranked = strategy_signals()
        if ranked.empty:
            st.warning('No signals available. Market data may be missing.')
        else:
            view = ranked.copy()
            view['ret20'] = view['ret20'] * 100
            view['vol'] = view['vol'] * 100
            view = view.rename(columns={'ticker':'Ticker','price':'Last close ($)','score':'Score','trend':'Above 200-day avg','ret20':'20-day return (%)','vol':'Annualized volatility (%)','reason':'Why this score?'})
            st.dataframe(view, hide_index=True, use_container_width=True)
        st.warning('The strategy uses observed closing prices to choose and simulate trades at that same close. This creates optimistic execution assumptions. Treat results as educational paper trading, not an executable trading record.')
        if st.button('Run / record today’s virtual decisions', type='primary'):
            with st.spinner('Evaluating strategy...'):
                result = run(engine)
            st.write(result)
            st.rerun()
    except Exception as exc:
        st.error('Unable to calculate or record strategy decisions.')
        st.write('Error category:', type(exc).__name__)
    if not de.empty:
        st.subheader('Saved decisions')
        st.dataframe(de.sort_values('run_date', ascending=False).head(60), hide_index=True, use_container_width=True)

elif page == 'Research':
    st.subheader('🔬 Multi-asset research & strategy comparison')
    st.write('Select any combination of stocks, ETFs and crypto. Add or remove symbols from the selector; the chart and legend update automatically.')
    options = sorted(set(WATCH + ['DIA','VTI','SCHD','BND','TSLA','AMD','COST','BRK-B','RUM']))
    if 'research_symbols' not in st.session_state:
        st.session_state.research_symbols = ['VOO','QQQ','AAPL','NVDA']
    pending = st.session_state.pop('pending_research_ticker', None)
    if pending and pending not in st.session_state.research_symbols:
        st.session_state.research_symbols = st.session_state.research_symbols + [pending]
    options = sorted(set(options + st.session_state.research_symbols))
    selected = st.multiselect('Assets to compare (type to search or click × to remove)', options=options, key='research_symbols', max_selections=12)
    custom = st.text_input('Add another Yahoo Finance ticker (optional)', placeholder='Example: PLTR or BTC-USD')
    if st.button('Add ticker') and custom.strip():
        symbol = custom.strip().upper()
        if symbol in st.session_state.research_symbols:
            st.info(f'{symbol} is already selected.')
        elif len(st.session_state.research_symbols) >= 12:
            st.warning('Remove a ticker before adding another (12 maximum).')
        else:
            # The multiselect widget already exists on this run.
            # Defer updating its state until the next rerun.
            st.session_state['pending_research_ticker'] = symbol
            st.rerun()
    c1,c2,c3 = st.columns([2,1,1])
    with c1:
        period = st.selectbox('Time period', ['1mo','3mo','6mo','1y','2y','5y'], index=3)
    with c2:
        include_ai = st.checkbox('Show AI Strategy', value=True)
    with c3:
        show_bench = st.checkbox('Add VOO + QQQ', value=True)
    chosen = list(dict.fromkeys(selected + (['VOO','QQQ'] if show_bench else [])))
    if not chosen:
        st.info('Choose at least one ticker, or enable VOO + QQQ.')
    else:
        with st.spinner('Downloading historical prices and building comparison...'):
            try:
                chart, missing, start = price_comparison(chosen, period, include_ai)
            except Exception as exc:
                chart, missing, start = pd.DataFrame(), [], None
                st.error(f'Market data request failed ({type(exc).__name__}). Try Refresh market data.')
        if missing:
            st.warning('No usable price data for: ' + ', '.join(missing))
        if not chart.empty:
            fig = go.Figure()
            for col in chart.columns:
                series = chart[col].dropna()
                if series.empty:
                    continue
                is_ai = col == 'AI Strategy'
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values,
                    mode='lines+markers' if is_ai else 'lines',
                    name='AI Strategy (recorded)' if is_ai else col,
                    line={'width':4 if is_ai else 2, 'dash':'dash' if is_ai else 'solid'},
                    marker={'size':7 if is_ai else 4},
                    connectgaps=False,
                    hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.2f}<extra>%{fullData.name}</extra>'
                ))
            fig.update_layout(
                title='Growth of a hypothetical $10,000 per asset',
                xaxis_title='Date', yaxis_title='Value of $10,000 invested ($)',
                legend={'title':'Click a name to hide/show','orientation':'h','y':-0.25,'x':0},
                hovermode='x unified', height=570,
                margin={'l':20,'r':20,'t':70,'b':100}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption('Legend: each colored line is an asset. The thicker dashed line is the AI Strategy, using only saved snapshots. Click a legend item to hide/show a line; double-click to isolate it.')
            if start is not None:
                st.caption(f'Comparison begins {start.date()} and rebases each displayed series to $10,000. Historical asset prices do not imply the AI Strategy was active in the past.')
            latest = chart.ffill().iloc[-1]
            base = pd.DataFrame({'Series':latest.index, 'Latest normalized value ($)':latest.values})
            base['Return since chart start (%)'] = (base['Latest normalized value ($)']/START_CASH - 1)*100
            st.dataframe(base, hide_index=True, use_container_width=True)
        else:
            st.info('No overlapping usable price data for these selections and period. Try fewer assets or a longer period.')
        if include_ai and sn.empty:
            st.info('AI Strategy has no saved snapshots yet. Open Daily Decisions and record the first run to start its forward performance history.')
        elif include_ai and not sn.empty:
            st.info('AI Strategy points represent recorded virtual equity, not historical price-based backtesting. The line may be short or absent for a selected period.')

    st.subheader('What the research data means')
    with st.expander('Why normalize everything to $10,000?', expanded=True):
        st.markdown('A $600 stock and a $50 ETF cannot be compared fairly on a raw-price chart. Normalization answers: **What would $10,000 invested in each asset at the same starting date be worth at each later date?** It compares percentage growth, not share prices. Each line is a separate hypothetical $10,000 investment; the values are not added together.')
    with st.expander('What does the AI Strategy line mean?', expanded=True):
        st.markdown('It is the **recorded total value of the virtual portfolio**: cash plus simulated holdings. The comparison rebases its first visible snapshot to $10,000, so the line can share the same scale as individual assets. **It is not a historical AI backtest.** The line only exists on dates saved in Supabase, and there may be gaps if daily runs were missed.')
    with st.expander('How does the strategy use these indicators?'):
        st.markdown('The engine checks each watchlist asset against its **200-day moving average** (long-term direction), compares the **50-day average** with the 200-day average (intermediate trend), calculates **20-day return** (recent momentum), and subtracts a **volatility penalty** (how erratically price has moved). It ranks assets by the resulting score and selects up to four qualifying assets. The chart is for analysis; selecting an asset here **does not change the engine’s trading watchlist**.')
    with st.expander('How should I interpret risk and performance?'):
        st.markdown('**Higher return** means the line finishes farther above $10,000. **Drawdown** is a decline from a previous peak; **volatility** measures how much returns fluctuate. A strong recent return does not guarantee future gains. ETFs can overlap substantially (for example, VOO and QQQ hold some of the same companies), and crypto trades on weekends while US stocks generally do not.')
    st.caption('Yahoo Finance adjusted closing prices may be delayed, revised, or unavailable. This is educational research, not personalized investment advice.')

elif page == 'Trade Log':
    st.subheader('Paper trade audit trail')
    if tr.empty:
        st.info('No virtual trades yet.')
    else:
        st.dataframe(tr.sort_values('id', ascending=False), hide_index=True, use_container_width=True)
    st.download_button('Download trades CSV', tr.to_csv(index=False), 'trades.csv', 'text/csv')
    st.download_button('Download snapshots CSV', sn.to_csv(index=False), 'snapshots.csv', 'text/csv')
    with st.expander('How to read the trade log', expanded=True):
        st.markdown('**BUY/SELL** is the simulated direction; **shares × price** approximates trade notional; **fee** is modeled friction of 0.05% per trade; **reason** records the model signal. These are hypothetical transactions, not brokerage executions.')

elif page == 'My Portfolio Comparison':
    st.subheader('💼 Value my actual holdings')
    st.write('Upload a CSV with `ticker,shares` and optionally `cost_basis`. This is a point-in-time valuation, not a historical return calculation. Files are processed in memory, not written to Supabase.')
    uploaded = st.file_uploader('Holdings CSV', type='csv')
    if uploaded is not None:
        try:
            portfolio = pd.read_csv(uploaded)
            portfolio.columns = portfolio.columns.str.strip().str.lower()
            if not {'ticker','shares'}.issubset(portfolio.columns):
                st.error('The CSV needs ticker and shares columns.')
                st.stop()
            portfolio['ticker'] = portfolio['ticker'].astype('string').str.upper().str.strip()
            portfolio['shares'] = pd.to_numeric(portfolio['shares'], errors='coerce')
            invalid = portfolio[portfolio['ticker'].isna() | (portfolio['ticker']=='') | portfolio['shares'].isna() | (portfolio['shares']<0)]
            if not invalid.empty:
                st.warning(f'{len(invalid)} invalid rows were excluded. Check ticker names and share quantities.')
            portfolio = portfolio.drop(invalid.index).copy()
            if portfolio.empty or len(portfolio)>100:
                st.error('Provide 1–100 valid holdings.')
                st.stop()
            tickers = portfolio['ticker'].unique().tolist()
            quote_map = {}
            failures = {}
            with st.spinner('Retrieving quotes...'):
                try:
                    batch = get_quotes(tuple(tickers), '1mo')
                    for t in tickers:
                        if t in batch.columns:
                            s = batch[t].dropna()
                            if not s.empty and float(s.iloc[-1])>0:
                                quote_map[t] = float(s.iloc[-1])
                except Exception:
                    pass
                for t in tickers:
                    if t in quote_map:
                        continue
                    try:
                        hist = yf.Ticker(t).history(period='1mo', auto_adjust=True)
                        if not hist.empty and 'Close' in hist and not hist['Close'].dropna().empty:
                            p = float(hist['Close'].dropna().iloc[-1])
                            if p>0:
                                quote_map[t] = p
                                continue
                        failures[t] = 'No valid recent Yahoo Finance close'
                    except Exception:
                        failures[t] = 'Quote lookup failed'
            portfolio['last_price'] = portfolio['ticker'].map(quote_map)
            portfolio['market_value'] = portfolio['shares']*portfolio['last_price']
            valued = portfolio.dropna(subset=['last_price']).copy()
            missing = portfolio[portfolio['last_price'].isna()].copy()
            a,b,c = st.columns(3)
            a.metric('Priced holdings value', dollar(float(valued['market_value'].sum())))
            b.metric('Holdings priced', len(valued))
            c.metric('Holdings missing quotes', len(missing))
            if not valued.empty:
                st.dataframe(valued.sort_values('market_value', ascending=False), hide_index=True, use_container_width=True)
                alloc = valued.groupby('ticker', as_index=False)['market_value'].sum()
                st.plotly_chart(px.pie(alloc, names='ticker', values='market_value', hole=0.45, title='Allocation by current market value'), use_container_width=True)
            if not missing.empty:
                st.warning('Portfolio value is incomplete because some symbols could not be priced.')
                missing['issue'] = missing['ticker'].map(failures).fillna('Price unavailable')
                st.dataframe(missing[['ticker','shares','issue']], hide_index=True, use_container_width=True)
            st.download_button('Download valuation CSV', portfolio.to_csv(index=False), 'portfolio_valuation.csv', 'text/csv')
            with st.expander('What is this telling me?', expanded=True):
                st.markdown('**Market value** = shares owned × latest available adjusted close. **Allocation** = each position’s market value ÷ total priced holdings value. This can reveal concentration, but excludes unpriced positions and any cash not included in your CSV. **Cost basis** is not used to calculate profit because CSVs differ on whether it means per-share cost or total position cost.')
        except Exception as exc:
            st.error('Portfolio processing failed. Check CSV headers and data.')
            st.write('Error category:', type(exc).__name__)

else:
    st.subheader('How this investment research system works')
    st.markdown('''### 1. Data collection
The program requests adjusted daily closing prices from Yahoo Finance for a fixed universe of stocks, ETFs, bond ETFs and crypto. Adjusted prices may reflect splits and distributions; feeds are not exchange-grade.

### 2. Signal construction
For each eligible asset, it computes the 50-day and 200-day moving averages, a 20-trading-day return, and recent annualized volatility. A transparent formula turns those into a ranking score.

### 3. Paper-trading decision
The model selects up to four positive-trend assets with scores above 0.8. New positions target roughly 22.5% of portfolio equity each, with a 10% cash reserve. It does not short or borrow.

### 4. Simulated execution
Trades use the latest observed closing price and charge 5 basis points (0.05%) in modeled trading friction. Because the model sees the close before it simulates a fill at that close, this can overstate results.

### 5. Permanent record
Supabase stores decisions, virtual trades and daily portfolio snapshots. The AI Strategy line reflects only these saved observations. Missing runs are not automatically backfilled.

### 6. Benchmarks
VOO and QQQ represent separate $10,000 buy-and-hold comparisons. They are not a blended benchmark. The Research chart can also compare any selected tickers on the same normalized dollar scale.

### 7. What this is not
This is a deterministic rules-based research experiment, not a live AI portfolio manager, not a brokerage connection, and not proof that a strategy can outperform the market. Taxes, real execution slippage, spreads and market impact are not fully modeled.
''')
