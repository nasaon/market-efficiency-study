# -*- coding: utf-8 -*-
"""Part C: 1시간봉 인트라데이 수집 (yfinance, 최대 ~730일)."""
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "intraday");
SUB = {
    "us": "AAPL MSFT NVDA AMZN META GOOGL TSLA JPM XOM UNH JNJ V WMT HD PG".split(),
    "kr": "005930.KS 000660.KS 035420.KS 005380.KS 051910.KS 000270.KS 068270.KS 005490.KS".split(),
    "crypto": [c + "-USD" for c in "BTC ETH BNB XRP SOL ADA DOGE LINK LTC DOT AVAX MATIC".split()],
}


def main():
    for mk, tickers in SUB.items():
        d = os.path.join(OUT, mk); os.makedirs(d, exist_ok=True)
        data = yf.download(tickers, period="730d", interval="1h", progress=False, auto_adjust=False, group_by="ticker", threads=True)
        ok = 0
        for t in tickers:
            try:
                sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
                df.columns = ["open", "high", "low", "close", "volume"]
                if len(df) < 500: continue
                df.index.name = "dt"
                df.to_csv(os.path.join(d, f"{t.replace('-','_').replace('.','_')}.csv"), encoding="utf-8-sig")
                ok += 1
            except Exception:
                pass
        print(f"[{mk}] {ok}/{len(tickers)} (1h)")
    print("완료.")


if __name__ == "__main__":
    main()
