import os
import streamlit as st
import pandas as pd
import plotly.express as px
from engine import db,init,read,run,prices,score_frame,holdings,WATCH,START_CASH

st.set_page_config(page_title='AI Investment Research Lab',page_icon='📈',layout='wide')
st.title('📈 AI Investment Research Lab')
st.caption('Research + auditable $10,000 virtual strategy | Educational only, no real orders')
if 'DATABASE_URL' in st.secrets: os.environ['DATABASE_URL']=st.secrets['DATABASE_URL']
if 'APP_PASSWORD' in st.secrets:
    pwd=st.text_input('App access password',type='password')
    if pwd!=st.secrets['APP_PASSWORD']:st.stop()
else: st.warning('No APP_PASSWORD set. Do not publish this app with private holdings.')
engine=db()
try: init(engine)
except Exception as exc: st.error('Database unavailable. Check Streamlit Secrets and Supabase connectivity.');st.stop()
page=st.sidebar.radio('Pages',['Dashboard','Daily Decisions','Research','Trade Log','My Portfolio Comparison','Data & Methodology'])
with st.sidebar:
    if st.button('Refresh latest quotes / signals'):
        st.cache_data.clear();st.rerun()
    st.caption('Quotes are delayed/end-of-day, not guaranteed real-time. Refresh never automatically trades.')
@st.cache_data(ttl=300)
def market():return prices(WATCH)
@st.cache_data(ttl=300)
def signals():return score_frame(market())
sn=read(engine,'snapshots');tr=read(engine,'trades');de=read(engine,'decisions')
if page=='Dashboard':
    st.subheader('Virtual portfolio vs buy-and-hold benchmarks')
    if sn.empty:st.info('No trading-day snapshots yet. Open Daily Decisions and click Run strategy once to initialize the challenge.')
    else:
        latest=sn.sort_values('run_date').iloc[-1]
        a,b,c=st.columns(3)
        a.metric('AI virtual equity',f'${latest.equity:,.2f}',f'{(latest.equity/START_CASH-1)*100:+.2f}%')
        b.metric('VOO hypothetical value',f'${latest.benchmark_voo:,.2f}')
        c.metric('QQQ hypothetical value',f'${latest.benchmark_qqq:,.2f}')
        plot=sn.sort_values('run_date').rename(columns={'equity':'AI Strategy','benchmark_voo':'VOO','benchmark_qqq':'QQQ'})
        st.plotly_chart(px.line(plot,x='run_date',y=['AI Strategy','VOO','QQQ'],title='Growth of $10,000 since strategy inception'),use_container_width=True)
        cash,h=holdings(tr)
        st.write('Current virtual positions:',h,'Cash:',round(cash,2))
    st.info('Performance only begins when you first initialize. No invented history. Trading uses observed daily closing prices as a simulation proxy.')
elif page=='Daily Decisions':
    st.subheader('Daily candidate ranking and paper decisions')
    try:
        s=signals();st.dataframe(s,use_container_width=True,hide_index=True)
        st.warning('Model: 50/200-day trend + 20-day momentum − volatility. Deterministic rules, NOT an LLM. Trades simulate at last daily close; actual same-close execution is impossible after observing that close. Results are optimistic and require forward-only testing.')
        if st.button('Run / record today’s virtual decisions',type='primary'):
            st.write(run(engine));st.cache_data.clear();st.rerun()
    except Exception as ex:st.error(f'Market data unavailable: {type(ex).__name__}. Try later.')
    if not de.empty:st.dataframe(de.sort_values('run_date',ascending=False).head(50),use_container_width=True,hide_index=True)
elif page=='Research':
    st.subheader('Research stocks, bond ETFs and crypto')
    ticker=st.selectbox('Symbol',WATCH)
    try:
        df=market()[ticker].dropna();st.line_chart(df)
        row=signals();r=row[row.ticker==ticker]
        if not r.empty:st.write(r.iloc[0].reason)
        st.caption('Bonds are represented by ETFs (TLT, IEF, SHY); crypto by BTC-USD and ETH-USD. ETF prices are not individual bond yields.')
    except Exception:st.error('Unable to retrieve quotes.')
elif page=='Trade Log':
    st.subheader('Immutable-style paper trade audit log')
    st.dataframe(tr.sort_values('id',ascending=False),use_container_width=True,hide_index=True)
    st.download_button('Download trades CSV',tr.to_csv(index=False),'trades.csv','text/csv')
    st.download_button('Download daily snapshots CSV',sn.to_csv(index=False),'snapshots.csv','text/csv')
elif page=='My Portfolio Comparison':
    st.subheader('Compare against your actual holdings — without linking brokerage accounts')
    st.write('Upload a CSV with columns `ticker,shares` (optional `cost_basis`). The uploaded file is processed in memory and is not saved to the database.')
    f=st.file_uploader('Holdings CSV',type='csv')
    if f:
        try:
            x=pd.read_csv(f);assert {'ticker','shares'}.issubset(x.columns)
            x['ticker']=x.ticker.astype(str).str.upper().str.strip();x['shares']=pd.to_numeric(x.shares,errors='raise')
            assert (x.shares>=0).all() and len(x)<101
            current=prices(x.ticker.unique().tolist(),period='1mo')
            x['last_price']=x.ticker.map({t:float(current[t].dropna().iloc[-1]) for t in x.ticker.unique()})
            x['value']=x.shares*x.last_price
            st.metric('Current uploaded holdings market value',f'${x.value.sum():,.2f}')
            st.dataframe(x,use_container_width=True,hide_index=True)
            st.caption('This is a point-in-time valuation, NOT a valid historical performance comparison. To compare your portfolio over 6 months, record dated daily valuations or time-weighted returns and account for deposits/withdrawals.')
        except Exception as ex:st.error('Check CSV columns, tickers, and share quantities.')
else:
    st.subheader('Methodology and limitations')
    st.markdown('''- **Data:** Yahoo Finance via yfinance; not exchange-grade, can be delayed, revised or unavailable. Crypto trades 24/7; stocks/ETFs use market sessions.
- **Universe:** fixed list in `engine.py`. Securities are selected by 200-day trend, 50-day trend, 20-day momentum and realized volatility. This is a rules-based model, not a generative AI.
- **Execution:** paper fills at the latest observed close, with 5 bps simulated friction. Since the signal also uses that close, this is *not* a realistic executable fill and may overstate returns. A production backtest should execute at the next available open with spread/slippage and holiday controls.
- **Portfolio:** max four candidates, ~22.5% initial target each, 10% cash buffer; no leverage or shorts. Sells positions outside the candidate set.
- **Benchmark:** $10,000 allocated to each of VOO and QQQ at the first recorded date; not a blended allocation.
- **Persistence:** set `DATABASE_URL` to a persistent Supabase PostgreSQL URL. SQLite is only for local development; Streamlit Cloud local files can disappear.
- **Automation:** GitHub Actions workflow can run once daily, but GitHub scheduling is best effort and the provider may return stale closes. Same-date reruns are blocked; use backups and database access restrictions.
- **Caveats:** No taxes, dividends as cash flows, borrow fees, real brokerage fills, market-impact modeling, or reliable intraday data. Never interpret a signal as a guaranteed profitable trade.''')
