# -*- coding: utf-8 -*-
"""
1단계 데이터 수집 — 지수 일봉 OHLCV (2015-01-01 ~ 2025-12-31)
소스: yfinance 통일
  - 한국  : ^KS200  (KOSPI 200)
  - 미국  : ^GSPC   (S&P 500)
  - 가상자산: BTC-USD (Bitcoin)
출력: stage1/data/{market}.csv  (date, open, high, low, close, volume)
"""
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

START = "2015-01-01"
END = "2026-01-01"          # yfinance end는 배타적 → 2025-12-31 포함

MARKETS = {
    "kospi200": "^KS200",
    "sp500": "^GSPC",
    "btc": "BTC-USD",
}


def fetch_one(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"{ticker}: 빈 데이터")
    # 단일 티커도 최근 yfinance는 MultiIndex 컬럼 반환 → 평탄화
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def main():
    summary = []
    for name, ticker in MARKETS.items():
        df = fetch_one(ticker)
        out = os.path.join(DATA, f"{name}.csv")
        df.to_csv(out, encoding="utf-8-sig")
        summary.append({
            "market": name, "ticker": ticker, "rows": len(df),
            "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
        })
        print(f"[OK] {name:9s} {ticker:9s} rows={len(df):5d}  "
              f"{df.index[0].date()} ~ {df.index[-1].date()}  -> {out}")
    pd.DataFrame(summary).to_csv(os.path.join(DATA, "_summary.csv"),
                                 index=False, encoding="utf-8-sig")
    print("\n수집 완료. 요약: stage1/data/_summary.csv")


if __name__ == "__main__":
    main()
