# -*- coding: utf-8 -*-
"""
Part A: 종목 선택 × SMC. "올바른 종목을 고르면 기술적분석이 유의해지나?"
- 유동성 버킷(저/고): 종목별 중앙 달러거래대금 < 시장중앙 → 저유동(소외/비효율 프록시).
- 가치/성장 프록시: 과거 252일 수익. 날짜별 중앙값 미만=가치(소외/역행), 초과=성장/모멘텀.
- 각 버킷서 FVG 방향신호(FVG_s)·추세선(TL_s)의 초과수익 예측 계수 비교. 이중클러스터.
"""
import os, sys, glob
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import signals as S
from combined_run import with_all

RES = os.path.join(S.HERE, "results"); os.makedirs(RES, exist_ok=True)
S2 = os.path.join(S.HERE, "data", "stage2")
NS = [5, 10, 20]; CTRL = ["vol20", "logvol"]
MAP = [("us", "미국", "sp500"), ("kr", "한국", "kospi200"), ("crypto", "가상자산", "btc")]


def build(folder, idxkey):
    idx = pd.read_csv(os.path.join(S.OUT, f"{idxkey}_panel.csv"), parse_dates=["date"], index_col="date")
    idx = idx[[f"fwd_{N}" for N in NS]].rename(columns={f"fwd_{N}": f"idx_{N}" for N in NS})
    frames = []
    for fp in glob.glob(os.path.join(S2, folder, "*.csv")):
        tic = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        if len(df) < 300: continue
        df = with_all(df)
        df["dvol"] = df["close"] * df["volume"]
        df["mom252"] = df["close"] / df["close"].shift(252) - 1
        m = df.join(idx, how="inner")
        for N in NS: m[f"ex_{N}"] = m[f"fwd_{N}"] - m[f"idx_{N}"]
        m["ticker"] = tic
        frames.append(m[["ticker", "dvol", "mom252"] + [f"ex_{N}" for N in NS] + CTRL + ["FVG_s", "TL_s"]].reset_index())
    p = pd.concat(frames, ignore_index=True)
    # 유동성: 종목별 중앙 dvol → 시장중앙 기준 저/고
    med = p.groupby("ticker")["dvol"].median()
    cut = med.median()
    lowliq = set(med[med < cut].index)
    p["liq"] = np.where(p["ticker"].isin(lowliq), "저유동", "고유동")
    # 가치/성장: 날짜별 mom252 중앙값
    p["vg"] = np.where(p["mom252"] < p.groupby("date")["mom252"].transform("median"), "가치", "성장")
    return p


def clreg(p, ycol, xcol):
    d = p[[ycol, "date", "ticker", xcol] + CTRL].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200 or (d[xcol] != 0).sum() < 10: return None
    y = d[ycol].values; X = sm.add_constant(d[[xcol] + CTRL].values)
    g = np.column_stack([pd.factorize(d["date"])[0], pd.factorize(d["ticker"])[0]])
    try:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    except Exception:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g[:, 0]})
    return m


def main():
    rows = []
    for folder, kor, idxkey in MAP:
        p = build(folder, idxkey)
        print(f"[{folder}] {p['ticker'].nunique()}종목 {len(p):,}행")
        for sig in ["FVG_s", "TL_s"]:
            for axis, groups in [("liq", ["저유동", "고유동"]), ("vg", ["가치", "성장"])]:
                for grp in groups:
                    sub = p[p[axis] == grp]
                    for N in NS:
                        m = clreg(sub, f"ex_{N}", sig)
                        if m is None: continue
                        rows.append(dict(market=folder, kor=kor, sig=sig, axis=axis, group=grp, N=N,
                                         coef=m.params[1], t=m.tvalues[1], p=m.pvalues[1]))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(RES, "partA.csv"), index=False, encoding="utf-8-sig")
    r["star"] = np.where(r["p"] < .05, "*", "")
    print("\n=== Part A: 버킷별 FVG 예측계수 (개별종목 초과수익, %p) ===")
    for folder, kor, _ in MAP:
        print(f"\n[{kor}]")
        s = r[(r["market"] == folder) & (r["sig"] == "FVG_s")]
        for axis in ["liq", "vg"]:
            for grp in (["저유동", "고유동"] if axis == "liq" else ["가치", "성장"]):
                ss = s[(s["axis"] == axis) & (s["group"] == grp)].sort_values("N")
                if len(ss) == 0: continue
                cells = " ".join(f"N{int(r2.N)}:{r2.coef*100:+.3f}{r2.star}" for _, r2 in ss.iterrows())
                print(f"  FVG {grp:5s}: {cells}")
    print("\n완료. results/partA.csv")


if __name__ == "__main__":
    main()
