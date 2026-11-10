# -*- coding: utf-8 -*-
"""
1단계 확장 분석: H2(컨플루언스) · 방향풀링 · H5(수익성 gross→net) · 기간분할 · FDR.
출력: results/ 하위 CSV. (지수 시계열, HAC)
"""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import signals as S

HERE = S.HERE
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
MK = [("kospi200", "코스피200"), ("sp500", "S&P500"), ("btc", "비트코인")]
NS = [5, 10, 20]
CTRL = ["vol20", "logvol"]


def load(key):
    df = pd.read_csv(os.path.join(S.DATA, f"{key}.csv"), parse_dates=["date"], index_col="date")
    df = S.add_basics(df)
    sh, sl = S.swing_points(df)
    _, oev = S.ob_signals(df, sh, sl)
    return df, oev


def hac(y, X, N):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": N})


# ── H2: 컨플루언스(부스터) 겹친 OB가 더 예측하는가 ──
def h2():
    rows = []
    for key, kor in MK:
        df, oev = load(key)
        n = len(df)
        for rule in ["A", "C"]:
            for d in ["L", "S"]:
                # 진입봉별 최대 컨플루언스 수
                conf = {}
                for e in oev:
                    if e["rule"] == rule and e["dir"] == d:
                        cc = int(e["sweep"]) + int(e["fvg"]) + int(e["gap2x"])
                        conf[e["entry"]] = max(conf.get(e["entry"], 0), cc)
                if not conf:
                    continue
                hi = np.zeros(n); base = np.zeros(n)
                for j, cc in conf.items():
                    if cc >= 1: hi[j] = 1
                    else: base[j] = 1
                for N in NS:
                    d2 = df[[f"fwd_{N}"] + CTRL].copy()
                    d2["hi"] = hi; d2["base"] = base
                    d2 = d2.replace([np.inf, -np.inf], np.nan).dropna()
                    if d2["hi"].sum() < 8 or d2["base"].sum() < 8:
                        continue
                    m = hac(d2[f"fwd_{N}"].values, d2[["hi", "base"] + CTRL].values, N)
                    rows.append(dict(market=key, kor=kor, rule=rule, dir=d, N=N,
                                     coef_hi=m.params[1], p_hi=m.pvalues[1], n_hi=int(d2["hi"].sum()),
                                     coef_base=m.params[2], p_base=m.pvalues[2], n_base=int(d2["base"].sum())))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(RES, "h2_confluence.csv"), index=False, encoding="utf-8-sig")
    return r


# ── 방향 풀링: signed = +1(long)/-1(short)/0, fwd ~ signed ──
def pooled():
    rows = []
    fams = {"FVG": ("FVG_L", "FVG_S"), "OB_A": ("OB_A_L", "OB_A_S"), "OB_C": ("OB_C_L", "OB_C_S")}
    for key, kor in MK:
        df = pd.read_csv(os.path.join(S.OUT, f"{key}_panel.csv"), parse_dates=["date"], index_col="date")
        for fam, (cl, cs) in fams.items():
            signed = df[cl].astype(int) - df[cs].astype(int)   # +1/-1/0 (둘다면 0)
            for N in NS:
                d2 = df[[f"fwd_{N}"] + CTRL].copy(); d2["signed"] = signed.values
                d2 = d2.replace([np.inf, -np.inf], np.nan).dropna()
                d2 = d2[d2["signed"] != 0].append(d2[d2["signed"] == 0]) if False else d2
                if (d2["signed"] != 0).sum() < 15:
                    continue
                m = hac(d2[f"fwd_{N}"].values, d2[["signed"] + CTRL].values, N)
                rows.append(dict(market=key, kor=kor, family=fam, N=N,
                                 coef=m.params[1], t=m.tvalues[1], p=m.pvalues[1],
                                 n_sig=int((d2["signed"] != 0).sum())))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(RES, "pooled_directional.csv"), index=False, encoding="utf-8-sig")
    return r


# ── H5: 수익성 gross→net (방향조정 평균수익, 거래비용 차감) ──
def profitability():
    rows = []
    fams = {"FVG": ("FVG_L", "FVG_S"), "OB_A": ("OB_A_L", "OB_A_S"), "OB_C": ("OB_C_L", "OB_C_S")}
    costs = [0.0, 0.001, 0.002, 0.005]   # 왕복 거래비용 가정
    for key, kor in MK:
        df = pd.read_csv(os.path.join(S.OUT, f"{key}_panel.csv"), parse_dates=["date"], index_col="date")
        for fam, (cl, cs) in fams.items():
            for N in NS:
                long_ret = df.loc[df[cl] == 1, f"fwd_{N}"].dropna()
                short_ret = -df.loc[df[cs] == 1, f"fwd_{N}"].dropna()   # 숏은 부호반전
                allr = pd.concat([long_ret, short_ret])
                if len(allr) < 15:
                    continue
                gross = allr.mean()
                row = dict(market=key, kor=kor, family=fam, N=N, n=len(allr), gross=gross)
                for c in costs:
                    row[f"net_{int(c*1000)}bp"] = gross - c
                rows.append(row)
    r = pd.DataFrame(rows); r.to_csv(os.path.join(RES, "profitability.csv"), index=False, encoding="utf-8-sig")
    return r


# ── 기간분할 + FDR (H1) ──
def periodsplit_fdr():
    SIGS = ["FVG_L", "FVG_S", "OB_A_L", "OB_A_S", "OB_C_L", "OB_C_S"]
    splits = {"전체": (None, None), "전반(15-21)": (None, "2021-12-31"), "후반(22-25)": ("2022-01-01", None)}
    rows = []
    for key, kor in MK:
        df = pd.read_csv(os.path.join(S.OUT, f"{key}_panel.csv"), parse_dates=["date"], index_col="date")
        for sname, (a, b) in splits.items():
            sub = df.loc[(slice(a, b))]
            for sig in SIGS:
                for N in NS:
                    d2 = sub[[f"fwd_{N}", sig] + CTRL].replace([np.inf, -np.inf], np.nan).dropna()
                    if d2[sig].sum() < 8 or len(d2) < 40:
                        rows.append(dict(market=key, period=sname, sig=sig, N=N, coef=np.nan, p=np.nan, n=int(d2[sig].sum())))
                        continue
                    m = hac(d2[f"fwd_{N}"].values, d2[[sig] + CTRL].values, N)
                    rows.append(dict(market=key, period=sname, sig=sig, N=N, coef=m.params[1], p=m.pvalues[1], n=int(d2[sig].sum())))
    r = pd.DataFrame(rows)
    # FDR (Benjamini-Hochberg) on 전체 기간 p값
    full = r[(r["period"] == "전체") & r["p"].notna()].copy().sort_values("p").reset_index(drop=True)
    mm = len(full)
    full["rank"] = np.arange(1, mm + 1)
    full["bh_crit"] = full["rank"] / mm * 0.05
    full["fdr_sig"] = full["p"] <= full["bh_crit"]
    r.to_csv(os.path.join(RES, "periodsplit.csv"), index=False, encoding="utf-8-sig")
    full.to_csv(os.path.join(RES, "h1_fdr.csv"), index=False, encoding="utf-8-sig")
    return r, full


def main():
    print("── H2 컨플루언스 ──")
    h2r = h2()
    # 요약: conf vs base 평균계수 (방향조정: L은 +, S는 -가 좋음)
    if len(h2r):
        h2r["hi_dir"] = np.where(h2r["dir"] == "L", h2r["coef_hi"], -h2r["coef_hi"])
        h2r["base_dir"] = np.where(h2r["dir"] == "L", h2r["coef_base"], -h2r["coef_base"])
        print(f"  컨플루언스 OB 방향조정 평균계수: {h2r['hi_dir'].mean()*100:+.3f}%p (유의 p<.05: {int((h2r['p_hi']<.05).sum())}/{len(h2r)})")
        print(f"  base OB 방향조정 평균계수      : {h2r['base_dir'].mean()*100:+.3f}%p (유의: {int((h2r['p_base']<.05).sum())}/{len(h2r)})")

    print("── 방향 풀링 ──")
    pr = pooled()
    print(pr.assign(coef=lambda x: (x.coef*100).round(3))[["kor", "family", "N", "coef", "p", "n_sig"]].to_string(index=False))
    print(f"  유의(p<.05): {int((pr['p']<.05).sum())}/{len(pr)}")

    print("── H5 수익성 (방향조정 평균, %) ──")
    pf = profitability()
    show = pf.assign(gross=lambda x:(x.gross*100).round(3), net20=lambda x:(x['net_2bp']*100).round(3))
    print(show[["kor","family","N","n","gross","net20"]].to_string(index=False))

    print("── 기간분할 + FDR ──")
    ps, fdr = periodsplit_fdr()
    print(f"  H1 전체 p값 {len(fdr)}개 중 FDR(BH .05) 통과: {int(fdr['fdr_sig'].sum())}개  (최소 p={fdr['p'].min():.3f})")
    print("\n완료. results/ 에 CSV 저장.")


if __name__ == "__main__":
    main()
