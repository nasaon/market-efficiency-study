# -*- coding: utf-8 -*-
"""
세계적 퀀트 전략 4종을 같은 3시장(개별종목 초과수익)에 검정.
- TSMOM  : 시계열 모멘텀 sign(과거252일수익)        [Moskowitz·Ooi·Pedersen 2012, AQR]
- XSMOM  : 횡단면 모멘텀 12-1, 시장내 중위수 대비    [Jegadeesh & Titman 1993]
- REV    : 단기 반전 -sign(과거5일수익)             [Jegadeesh 1990 / Lehmann 1990]
- MA     : 이동평균 추세 sign(MA50-MA200)            [Paul Tudor Jones / 터틀 / AQR]
모든 신호: +1=롱 콜. ex_N ~ signal + 통제, 이중클러스터. coef>0 = 전략이 초과수익을 예측.
"""
import os, sys, glob
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import signals as S

RES = os.path.join(S.HERE, "results"); os.makedirs(RES, exist_ok=True)
S2 = os.path.join(S.HERE, "data", "stage2")
NS = [5, 10, 20]; CTRL = ["vol20", "logvol"]
MAP = [("us", "미국", "sp500"), ("kr", "한국", "kospi200"), ("crypto", "가상자산", "btc")]
STRATS = ["TSMOM", "XSMOM", "REV", "MA"]


def feat(df):
    df = S.add_basics(df)
    c = df["close"]
    df["ret252"] = c / c.shift(252) - 1
    df["ret_12_1"] = c.shift(21) / c.shift(252) - 1
    df["ret5"] = c / c.shift(5) - 1
    df["ma50"] = c.rolling(50).mean(); df["ma200"] = c.rolling(200).mean()
    df["TSMOM"] = np.sign(df["ret252"])
    df["REV"] = -np.sign(df["ret5"])
    df["MA"] = np.sign(df["ma50"] - df["ma200"])
    return df


def build(folder, idxkey):
    idx = pd.read_csv(os.path.join(S.OUT, f"{idxkey}_panel.csv"), parse_dates=["date"], index_col="date")
    idx = idx[[f"fwd_{N}" for N in NS]].rename(columns={f"fwd_{N}": f"idx_{N}" for N in NS})
    frames = []
    for fp in glob.glob(os.path.join(S2, folder, "*.csv")):
        tic = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        if len(df) < 300: continue
        df = feat(df)
        m = df.join(idx, how="inner")
        for N in NS: m[f"ex_{N}"] = m[f"fwd_{N}"] - m[f"idx_{N}"]
        m["ticker"] = tic
        cols = ["ticker"] + [f"ex_{N}" for N in NS] + CTRL + ["TSMOM", "REV", "MA", "ret_12_1"]
        frames.append(m[cols].reset_index())
    p = pd.concat(frames, ignore_index=True)
    # XSMOM: 시장내 날짜별 12-1 모멘텀 중위수 대비 부호
    p["XSMOM"] = p.groupby("date")["ret_12_1"].transform(lambda s: np.sign(s - s.median()))
    return p


def clreg(p, ycol, xcols):
    d = p[[ycol, "date", "ticker"] + xcols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200: return None
    y = d[ycol].values; X = sm.add_constant(d[xcols].values)
    g = np.column_stack([pd.factorize(d["date"])[0], pd.factorize(d["ticker"])[0]])
    try:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    except Exception:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g[:, 0]})
    return m, d


def main():
    rows = []
    for folder, kor, idxkey in MAP:
        p = build(folder, idxkey)
        print(f"[{folder}] {p['ticker'].nunique()}종목 {len(p):,}행")
        for st in STRATS:
            for N in NS:
                out = clreg(p, f"ex_{N}", [st] + CTRL)
                if out is None: continue
                m, d = out
                rows.append(dict(market=folder, kor=kor, strat=st, N=N,
                                 coef=m.params[1], t=m.tvalues[1], p=m.pvalues[1], nobs=len(d)))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(RES, "quant.csv"), index=False, encoding="utf-8-sig")
    r["star"] = np.where(r["p"] < .05, "*", "")
    print("\n=== 퀀트 전략 (개별종목 초과수익, coef>0=전략 예측력) ===")
    print(r.assign(cf=lambda x:(x.coef*100).round(3))[["kor","strat","N","cf","p","star"]].to_string(index=False))
    print(f"\n유의(p<.05): {int((r['p']<.05).sum())}/{len(r)}")
    print("방향일치(coef>0) & 유의:", int(((r['coef']>0)&(r['p']<.05)).sum()))
    print("\n완료. results/quant.csv")


if __name__ == "__main__":
    main()
