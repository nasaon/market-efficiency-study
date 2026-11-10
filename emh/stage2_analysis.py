# -*- coding: utf-8 -*-
"""
2단계 개별종목 패널 분석.
- 종목별 신호 생성(FVG, OB A/C) + 초과수익(종목 - 시장지수) + 통제.
- 패널 회귀: excess_N ~ 신호더미 + vol20 + logvol, 이중 클러스터(날짜·종목) 표준오차.
- H1(계수≠0·방향), H3(시장 비교), 방향풀링, H2(컨플루언스).
출력: results/stage2_*.csv
"""
import os, sys, glob
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import signals as S

HERE = S.HERE
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
S2 = os.path.join(HERE, "data", "stage2")
NS = [5, 10, 20]
SIGS = ["FVG_L", "FVG_S", "OB_A_L", "OB_A_S", "OB_C_L", "OB_C_S"]
CTRL = ["vol20", "logvol"]
# 시장 -> (종목폴더, 벤치마크 지수 패널)
MAP = [("us", "S&P500", "sp500"), ("kr", "코스피200", "kospi200"), ("crypto", "가상자산", "btc")]


def index_fwd(idxkey):
    df = pd.read_csv(os.path.join(S.OUT, f"{idxkey}_panel.csv"), parse_dates=["date"], index_col="date")
    return df[[f"fwd_{N}" for N in NS]].rename(columns={f"fwd_{N}": f"idx_{N}" for N in NS})


def build_market(folder, idxkey):
    idx = index_fwd(idxkey)
    frames = []
    for fp in glob.glob(os.path.join(S2, folder, "*.csv")):
        tic = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        if len(df) < 250:
            continue
        df = S.add_basics(df)
        sh, sl = S.swing_points(df)
        fvg, _ = S.fvg_signals(df)
        ob, obev = S.ob_signals(df, sh, sl)
        for d, idxs in fvg.items():
            v = np.zeros(len(df), int)
            for j in idxs: v[j] = 1
            df[f"FVG_{d}"] = v
        for (r, d), idxs in ob.items():
            v = np.zeros(len(df), int)
            for j in idxs: v[j] = 1
            df[f"OB_{r}_{d}"] = v
        # OB 컨플루언스(부스터>=1) 진입봉
        confbar = {}
        for e in obev:
            cc = int(e["sweep"]) + int(e["fvg"]) + int(e["gap2x"])
            confbar[(e["rule"], e["dir"], e["entry"])] = max(confbar.get((e["rule"], e["dir"], e["entry"]), 0), cc)
        for r in ["A", "C"]:
            for d in ["L", "S"]:
                hi = np.zeros(len(df), int); ba = np.zeros(len(df), int)
                for (rr, dd, j), cc in confbar.items():
                    if rr == r and dd == d:
                        (hi if cc >= 1 else ba)[j] = 1
                df[f"OB_{r}_{d}_hi"] = hi; df[f"OB_{r}_{d}_base"] = ba
        # 초과수익
        m = df.join(idx, how="inner")
        for N in NS:
            m[f"ex_{N}"] = m[f"fwd_{N}"] - m[f"idx_{N}"]
        m["ticker"] = tic
        keep = ["ticker"] + [f"ex_{N}" for N in NS] + CTRL + \
               [c for c in m.columns if c.startswith(("FVG_", "OB_"))]
        frames.append(m[keep].reset_index())
    panel = pd.concat(frames, ignore_index=True)
    return panel


def clreg(panel, ycol, xcols):
    d = panel[[ycol, "date", "ticker"] + xcols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200:
        return None
    y = d[ycol].values
    X = sm.add_constant(d[xcols].values)
    dcode = pd.factorize(d["date"])[0]; tcode = pd.factorize(d["ticker"])[0]
    groups = np.column_stack([dcode, tcode])
    try:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": dcode})
    return m, d


def main():
    panels = {}
    for folder, kor, idxkey in MAP:
        p = build_market(folder, idxkey)
        panels[folder] = (kor, p)
        print(f"[{folder}] 종목패널 {p['ticker'].nunique()}종목 {len(p):,}행")

    # ── H1 (개별종목 초과수익) ──
    rows = []
    for folder, (kor, p) in panels.items():
        for sig in SIGS:
            for N in NS:
                out = clreg(p, f"ex_{N}", [sig] + CTRL)
                if out is None: continue
                m, d = out
                rows.append(dict(market=folder, kor=kor, sig=sig, N=N,
                                 coef=m.params[1], t=m.tvalues[1], p=m.pvalues[1],
                                 nsig=int(d[sig].sum()), nobs=len(d)))
    h1 = pd.DataFrame(rows); h1.to_csv(os.path.join(RES, "stage2_h1.csv"), index=False, encoding="utf-8-sig")
    h1d = h1.copy()
    h1d["dir_ok"] = np.where(h1d["sig"].str.endswith("_L"), h1d["coef"] > 0, h1d["coef"] < 0)
    print("\n=== 2단계 H1 (초과수익, 이중클러스터) ===")
    print("방향일치 & p<.05 건수 / 전체:")
    for folder, (kor, p) in panels.items():
        s = h1d[h1d["market"] == folder]
        hit = int((s["dir_ok"] & (s["p"] < .05)).sum())
        print(f"  {kor:9s}: {hit}/{len(s)}  (방향무관 유의 {int((s['p']<.05).sum())})")

    # ── 방향 풀링 ──
    prows = []
    fams = {"FVG": ("FVG_L", "FVG_S"), "OB_A": ("OB_A_L", "OB_A_S"), "OB_C": ("OB_C_L", "OB_C_S")}
    for folder, (kor, p) in panels.items():
        for fam, (cl, cs) in fams.items():
            p2 = p.copy(); p2["signed"] = p2[cl].astype(int) - p2[cs].astype(int)
            for N in NS:
                out = clreg(p2, f"ex_{N}", ["signed"] + CTRL)
                if out is None: continue
                m, d = out
                prows.append(dict(market=folder, kor=kor, family=fam, N=N,
                                  coef=m.params[1], t=m.tvalues[1], p=m.pvalues[1],
                                  nsig=int((d["signed"] != 0).sum())))
    pool = pd.DataFrame(prows); pool.to_csv(os.path.join(RES, "stage2_pooled.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 2단계 방향풀링 (signed = +long/-short) ===")
    print(pool.assign(coef_pp=lambda x:(x.coef*100).round(3))[["kor","family","N","coef_pp","t","p","nsig"]].to_string(index=False))
    print(f"유의(p<.05): {int((pool['p']<.05).sum())}/{len(pool)}")

    # ── H2 컨플루언스 (개별종목): hi vs base OB ──
    hrows = []
    for folder, (kor, p) in panels.items():
        for r in ["A", "C"]:
            for d in ["L", "S"]:
                hi, ba = f"OB_{r}_{d}_hi", f"OB_{r}_{d}_base"
                for N in NS:
                    out = clreg(p, f"ex_{N}", [hi, ba] + CTRL)
                    if out is None: continue
                    m, dd = out
                    hrows.append(dict(market=folder, kor=kor, rule=r, dir=d, N=N,
                                      coef_hi=m.params[1], p_hi=m.pvalues[1], n_hi=int(dd[hi].sum()),
                                      coef_base=m.params[2], p_base=m.pvalues[2], n_base=int(dd[ba].sum())))
    h2 = pd.DataFrame(hrows); h2.to_csv(os.path.join(RES, "stage2_h2.csv"), index=False, encoding="utf-8-sig")
    if len(h2):
        h2["hi_dir"] = np.where(h2["dir"] == "L", h2["coef_hi"], -h2["coef_hi"])
        h2["base_dir"] = np.where(h2["dir"] == "L", h2["coef_base"], -h2["coef_base"])
        print("\n=== 2단계 H2 컨플루언스 ===")
        print(f"  컨플루언스 OB 방향조정 평균: {h2['hi_dir'].mean()*100:+.3f}%p (유의 {int((h2['p_hi']<.05).sum())}/{len(h2)})")
        print(f"  base OB 방향조정 평균      : {h2['base_dir'].mean()*100:+.3f}%p (유의 {int((h2['p_base']<.05).sum())}/{len(h2)})")
    print("\n완료. results/stage2_*.csv 저장.")


if __name__ == "__main__":
    main()
