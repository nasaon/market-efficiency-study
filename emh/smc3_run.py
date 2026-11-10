# -*- coding: utf-8 -*-
"""SMC 3종(추세선·채널·페이크아웃) 1단계·2단계 분석. 같은 3시장·프레임."""
import os, sys, glob
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import signals as S
from smc3_signals import trend_channel_fakeout, COLS

RES = os.path.join(S.HERE, "results"); os.makedirs(RES, exist_ok=True)
S2 = os.path.join(S.HERE, "data", "stage2")
NS = [5, 10, 20]; CTRL = ["vol20", "logvol"]
MAP = [("us", "미국", "sp500"), ("kr", "한국", "kospi200"), ("crypto", "가상자산", "btc")]
FAMS = {"TL": ("TL_L", "TL_S"), "CH": ("CH_L", "CH_S"), "FO": ("FO_L", "FO_S")}


def with_signals(df):
    df = S.add_basics(df)
    sh, sl = S.swing_points(df)
    sig = trend_channel_fakeout(df, sh, sl)
    for col, idxs in sig.items():
        v = np.zeros(len(df), int)
        for j in idxs: v[j] = 1
        df[col] = v
    return df


def hac(y, X, N):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": N})


def clreg(panel, ycol, xcols):
    d = panel[[ycol, "date", "ticker"] + xcols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200 or (d[xcols[0]] != 0).sum() < 10:
        return None
    y = d[ycol].values; X = sm.add_constant(d[xcols].values)
    g = np.column_stack([pd.factorize(d["date"])[0], pd.factorize(d["ticker"])[0]])
    try:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    except Exception:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g[:, 0]})
    return m, d


# ── 1단계 (지수) ──
def stage1():
    rows = []
    for _, kor, idxkey in MAP:
        df = with_signals(pd.read_csv(os.path.join(S.DATA, f"{idxkey}.csv"), parse_dates=["date"], index_col="date"))
        for col in COLS:
            for N in NS:
                d = df[[f"fwd_{N}", col] + CTRL].replace([np.inf, -np.inf], np.nan).dropna()
                if d[col].sum() < 10: continue
                m = hac(d[f"fwd_{N}"].values, d[[col] + CTRL].values, N)
                rows.append(dict(market=idxkey, kor=kor, sig=col, N=N, coef=m.params[1], p=m.pvalues[1], nsig=int(d[col].sum())))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(RES, "smc3_stage1.csv"), index=False, encoding="utf-8-sig")
    return r


# ── 2단계 (개별종목) ──
def build_panel(folder, idxkey):
    idx = pd.read_csv(os.path.join(S.OUT, f"{idxkey}_panel.csv"), parse_dates=["date"], index_col="date")
    idx = idx[[f"fwd_{N}" for N in NS]].rename(columns={f"fwd_{N}": f"idx_{N}" for N in NS})
    frames = []
    for fp in glob.glob(os.path.join(S2, folder, "*.csv")):
        tic = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        if len(df) < 250: continue
        df = with_signals(df)
        m = df.join(idx, how="inner")
        for N in NS: m[f"ex_{N}"] = m[f"fwd_{N}"] - m[f"idx_{N}"]
        m["ticker"] = tic
        frames.append(m[["ticker"] + [f"ex_{N}" for N in NS] + CTRL + COLS].reset_index())
    return pd.concat(frames, ignore_index=True)


def stage2():
    h1rows, prows = [], []
    panels = {}
    for folder, kor, idxkey in MAP:
        p = build_panel(folder, idxkey); panels[folder] = (kor, p)
        print(f"[{folder}] {p['ticker'].nunique()}종목 {len(p):,}행")
        for col in COLS:
            for N in NS:
                out = clreg(p, f"ex_{N}", [col] + CTRL)
                if out is None: continue
                m, d = out
                h1rows.append(dict(market=folder, kor=kor, sig=col, N=N, coef=m.params[1], t=m.tvalues[1], p=m.pvalues[1], nsig=int(d[col].sum())))
        for fam, (cl, cs) in FAMS.items():
            p2 = p.copy(); p2["signed"] = p2[cl].astype(int) - p2[cs].astype(int)
            for N in NS:
                out = clreg(p2, f"ex_{N}", ["signed"] + CTRL)
                if out is None: continue
                m, d = out
                prows.append(dict(market=folder, kor=kor, family=fam, N=N, coef=m.params[1], t=m.tvalues[1], p=m.pvalues[1], nsig=int((d["signed"] != 0).sum())))
    h1 = pd.DataFrame(h1rows); h1.to_csv(os.path.join(RES, "smc3_stage2_h1.csv"), index=False, encoding="utf-8-sig")
    pool = pd.DataFrame(prows); pool.to_csv(os.path.join(RES, "smc3_pooled.csv"), index=False, encoding="utf-8-sig")
    return h1, pool


def main():
    print("=== SMC 3종 (추세선·채널·페이크아웃) ===")
    s1 = stage1()
    print(f"[1단계 지수] {len(s1)}회귀, 유의 p<.05: {int((s1['p']<.05).sum())}")
    h1, pool = stage2()
    h1["dir_ok"] = np.where(h1["sig"].str.endswith("_L"), h1["coef"] > 0, h1["coef"] < 0)
    print("\n[2단계 H1] 방향일치 & p<.05 / 전체:")
    for folder, kor, _ in MAP:
        s = h1[h1["market"] == folder]
        print(f"  {kor:7s}: {int((s['dir_ok'] & (s['p']<.05)).sum())}/{len(s)} (유의 {int((s['p']<.05).sum())})")
    print("\n[2단계 방향풀링] (signed, 별표 p<.05)")
    pool["star"] = np.where(pool["p"] < .05, "*", "")
    print(pool.assign(cf=lambda x:(x.coef*100).round(3))[["kor","family","N","cf","p","star","nsig"]].to_string(index=False))
    print(f"  유의: {int((pool['p']<.05).sum())}/{len(pool)}")
    print("\n완료. results/smc3_*.csv")


if __name__ == "__main__":
    main()
