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
