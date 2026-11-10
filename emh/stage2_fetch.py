# -*- coding: utf-8 -*-
"""
2단계 개별종목 데이터 수집 (yfinance). 2015-2025 일봉.
- 미국: S&P 대형주 ~80
- 한국: KOSPI 대형주 ~40 (.KS)
- 가상자산: 시총상위 코인 ~22 (-USD, BTC는 벤치마크라 제외)
출력: data/stage2/{market}/{ticker}.csv
"""
import os, sys, time
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "stage2")
START, END = "2015-01-01", "2026-01-01"

US = ("AAPL MSFT NVDA AMZN GOOGL META TSLA BRK-B JPM V UNH XOM JNJ WMT MA PG HD CVX LLY ABBV "
      "AVGO KO PEP COST MRK ADBE CSCO ACN TMO ABT DHR LIN NEE TXN NKE PM WFC BMY UNP RTX QCOM "
      "HON UPS LOW INTC IBM CAT GS AMGN BA DE SBUX MMM GE F GM T VZ DIS NFLX CRM ORCL AMD PYPL "
      "INTU AMAT MU LRCX ADI KLAC MDT CB MO GILD CL DUK SO TGT CVS").split()

KR = ("005930.KS 000660.KS 373220.KS 207940.KS 005380.KS 000270.KS 068270.KS 105560.KS 035420.KS "
      "005490.KS 012330.KS 055550.KS 035720.KS 051910.KS 028260.KS 006400.KS 086790.KS 015760.KS "
      "032830.KS 066570.KS 096770.KS 017670.KS 030200.KS 009150.KS 010130.KS 034730.KS 018260.KS "
      "090430.KS 010950.KS 024110.KS 097950.KS 078930.KS 036570.KS 033780.KS 011170.KS 003550.KS "
      "267260.KS 316140.KS 138040.KS 071050.KS").split()

CRYPTO = ("ETH BNB XRP ADA SOL DOGE DOT LTC LINK BCH XLM TRX ETC XMR EOS AAVE ATOM ALGO XTZ FIL VET ICP").split()
CRYPTO = [c + "-USD" for c in CRYPTO]

MARKETS = {"us": US, "kr": KR, "crypto": CRYPTO}


def save_one(raw, ticker, outdir):
    if raw is None or len(raw) == 0:
        return False
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < 250:
        return False
    df.to_csv(os.path.join(outdir, f"{ticker.replace('-','_').replace('.','_')}.csv"), encoding="utf-8-sig")
    return True


def main():
    for mk, tickers in MARKETS.items():
        outdir = os.path.join(OUT, mk); os.makedirs(outdir, exist_ok=True)
        ok = 0
        data = yf.download(tickers, start=START, end=END, progress=False, auto_adjust=False, group_by="ticker", threads=True)
        for t in tickers:
            try:
                sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                if save_one(sub, t, outdir):
                    ok += 1
            except Exception:
                pass
        print(f"[{mk}] {ok}/{len(tickers)} 저장 -> {outdir}")
    print("완료.")


if __name__ == "__main__":
    main()
