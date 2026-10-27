# AI Investment Research Lab

Streamlit research dashboard + auditable, rules-based $10,000 paper trading experiment against VOO and QQQ. **No real trading or brokerage credentials.**

## Deploy to Streamlit Community Cloud
1. Create a **private** GitHub repo; upload all files, including `.github/workflows/daily.yml`.
2. Create a Streamlit app using `app.py` and install from `requirements.txt`.
3. Set Streamlit Secrets:
   ```toml
   DATABASE_URL = "postgresql://...your Supabase session pooler URL..."
   APP_PASSWORD = "a-long-unique-app-password"
   ```
4. In GitHub repo **Settings → Secrets and variables → Actions**, add `DATABASE_URL` as a **repository secret**. This is required for scheduled runs. Never commit the value.
5. Open **Daily Decisions** and click **Run / record today’s virtual decisions** to initialize the starting date. Scheduled runs will then record each trading day when market data is available.
6. Use **Actions → Daily paper portfolio snapshot → Run workflow** to test the automation. Check logs for success.
7. Export trade and snapshot CSVs regularly; configure Supabase backups and test restore. Limit database user privileges and restrict access to the Streamlit app.

## Local run
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
This is a **research prototype**, not reliable production trading infrastructure. Yahoo Finance data is delayed/unofficial; the paper model fills at an observed close (look-ahead execution bias). Schedule is best-effort and weekday runs can include market holidays. GitHub Actions scheduling may be delayed. A stronger second version should use next-session open execution, explicit trading calendars, recorded signal timestamps, corporate actions, point-in-time historical data, a dedicated market-data API, automated backups, authentication, tests, and an optional LLM research commentary layer. Current app is rules-based and does not use an AI API.

### Data model
`trades`, `decisions`, `snapshots`, `config` tables in the configured database. The first run anchors benchmark starting prices; the comparison only covers dates after launch. User portfolio CSV is never saved.
