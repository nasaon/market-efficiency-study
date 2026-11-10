# -*- coding: utf-8 -*-
"""S&P500 전체 OHLCV 수집 (volume 포함) → sp500_px/.
거래대금(종가x거래량) 기반 '연도별 대장주' 선정에 쓰임.
기존 px_*.csv 는 volume 이 없어 별도 폴더에 새로 받는다."""
import os, sys, warnings
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
PXDIR = os.path.join(HERE, "sp500_px")
os.makedirs(PXDIR, exist_ok=True)


def get_tickers():
    url = ("https://raw.githubusercontent.com/datasets/"
           "s-and-p-500-companies/master/data/constituents.csv")
    t = pd.read_csv(url)
    syms = [str(s).replace(".", "-").strip() for s in t["Symbol"]]
    return sorted(set(syms))


def safe(tk):
    return tk.replace("-", "_")


def main():
    tks = get_tickers()
    print(f"S&P500 종목 {len(tks)}개 수집 시작 (2015~2026, volume 포함)", flush=True)
    B = 40
    saved = skipped = 0
    for i in range(0, len(tks), B):
        batch = tks[i:i + B]
        try:
            data = yf.download(batch, start="2015-01-01", end="2026-12-31",
                               progress=False, auto_adjust=True,
                               group_by="ticker", threads=True)
        except Exception as e:
            print(f"  [batch {i}] 다운로드 실패: {e}", flush=True)
            continue
        multi = isinstance(data.columns, pd.MultiIndex)
        for tk in batch:
            try:
                sub = data[tk] if multi else data
                sub = sub.dropna(how="all")
                if sub is None or len(sub) < 60:
                    skipped += 1
                    continue
                out = pd.DataFrame({
                    "date": pd.DatetimeIndex(sub.index).strftime("%Y-%m-%d"),
                    "o": sub["Open"].values, "h": sub["High"].values,
                    "l": sub["Low"].values, "c": sub["Close"].values,
                    "v": sub["Volume"].values,
                }).dropna()
                if len(out) < 60:
                    skipped += 1
                    continue
                out.to_csv(os.path.join(PXDIR, f"px_{safe(tk)}.csv"), index=False)
                saved += 1
            except Exception:
                skipped += 1
        print(f"  진행 {min(i + B, len(tks))}/{len(tks)}  저장 {saved}  스킵 {skipped}",
              flush=True)
    print(f"\n완료: {saved}종목 저장, {skipped}스킵 -> {PXDIR}", flush=True)


if __name__ == "__main__":
    main()
