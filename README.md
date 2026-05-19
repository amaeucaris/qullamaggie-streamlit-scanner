# Qullamaggie NASDAQ Scanner

App Streamlit per scansionare ticker NASDAQ daily con dati `yfinance`.

## Avvio

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Consigliato usare un virtualenv pulito: `pandas-ta` moderno usa stack NumPy/Pandas recente.

## Deploy Streamlit Cloud

1. Pubblica questa cartella su GitHub.
2. In GitHub apri **Actions** -> **Update scanner data** -> **Run workflow**.
3. La workflow crea `data/history_prices.parquet` e `data/metadata.json`.
4. Su Streamlit Cloud fai deploy puntando a `app.py`.
5. In **Advanced settings** seleziona Python `3.12`.
6. Nell'app usa modalita **Precomputed** per leggere i dati gia scaricati.

La workflow e schedulata dopo la chiusura USA nei giorni feriali. Questo evita di chiamare `yfinance` a ogni apertura dell'app da smartphone.

Per test rapidi locali:

```bash
python update_data.py --universe NASDAQ --max-tickers 200 --chunk-size 50 --pause-seconds 1
streamlit run app.py
```

## Funzioni

- Universo NASDAQ da Nasdaq Trader, con opzione ETF.
- Universo All US listed opzionale per includere NYSE/AMEX.
- Download daily via `yfinance`, cache con `@st.cache_data` e pulsante refresh.
- Modalita precomputed da `data/history_prices.parquet` per deploy stabile.
- Returns 1M, 3M, 6M e momentum rank top 2%.
- Filtri: ADR% 20 giorni, price sopra SMA10/SMA20, avg volume > 200k, price > 5.
- Mappa estensione ATR-to-SMA50 con zone non extended, extended e hyper extended.
- Tab Stockbee 4% breakout.
- Tab Backtest Qullamaggie: segnale a fine giornata, entrata next open, uscita dopo N sedute.
- Chart Plotly candlestick con SMA10/SMA20.
- Export CSV via download e salvataggio locale nella cartella `exports/`.
