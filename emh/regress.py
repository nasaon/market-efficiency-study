# -*- coding: utf-8 -*-
"""
1단계 회귀 — fwd_N ~ 신호더미 + 통제(vol20, logvol), HAC(Newey-West, lag=N) 표준오차.
시장(코스피·S&P·BTC) × 신호 6종(FVG L/S, OB A/C × L/S) × N{5,10,20}.
- H1: 신호더미 계수 != 0?  (방향: _L은 +, _S는 -가 '신호대로')
- H3: 예측력(유의성·크기) BTC > 한국 > 미국?
- A vs C: 어느 엥걸프 룰이 더 강한가.
강세장 편향 방어: 절편이 시장평균 흡수 → 더미 '계수'로만 해석.
"""
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
PAN = os.path.join(HERE, "panels")
MK = [("kospi200", "코스피200"), ("sp500", "S&P500"), ("btc", "비트코인")]
SIGS = ["FVG_L", "FVG_S", "OB_A_L", "OB_A_S", "OB_C_L", "OB_C_S"]
NS = [5, 10, 20]
CTRL = ["vol20", "logvol"]


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def run():
    rows = []
    for key, kor in MK:
        df = pd.read_csv(os.path.join(PAN, f"{key}_panel.csv"), parse_dates=["date"], index_col="date")
        for sig in SIGS:
            for N in NS:
                cols = [f"fwd_{N}", sig] + CTRL
                d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
                nsig = int(d[sig].sum())
                if nsig < 10 or len(d) < 50:
                    rows.append(dict(market=key, kor=kor, sig=sig, N=N, n=len(d), nsig=nsig,
                                     coef=np.nan, t=np.nan, p=np.nan, note="신호부족"))
                    continue
                y = d[f"fwd_{N}"].values
                X = sm.add_constant(d[[sig] + CTRL].values)
                m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": N})
                # 인덱스1 = 신호더미
                coef, t, p = m.params[1], m.tvalues[1], m.pvalues[1]
                rows.append(dict(market=key, kor=kor, sig=sig, N=N, n=len(d), nsig=nsig,
                                 coef=coef, t=t, p=p, note=""))
    return pd.DataFrame(rows)


def main():
    res = run()
    res.to_csv(os.path.join(HERE, "stage1_regression_results.csv"), index=False, encoding="utf-8-sig")

    # 보기 좋게: 신호×시장 (N=10 대표) 계수(%p)와 유의성
    print("=" * 78)
    print("[1단계 회귀] 신호더미 계수 = 신호 후 N일 수익률에 대한 '추가' 효과 (단위: %p)")
    print("  방향 기대: _L(상승신호)→계수 +,  _S(하락신호)→계수 -   | HAC(lag=N) p값 별표")
    print("=" * 78)
    for N in NS:
        print(f"\n───────────── N = {N}일 보유 ─────────────")
        sub = res[res["N"] == N]
        piv = sub.pivot(index="sig", columns="market", values="coef").reindex(SIGS)
        pp = sub.pivot(index="sig", columns="market", values="p").reindex(SIGS)
        nn = sub.pivot(index="sig", columns="market", values="nsig").reindex(SIGS)
        print(f"{'신호':10s} | {'코스피200':>16s} | {'S&P500':>16s} | {'비트코인':>16s}")
        for sig in SIGS:
            cells = []
            for key, _ in MK:
                c = piv.loc[sig, key]; p = pp.loc[sig, key]; ns = int(nn.loc[sig, key])
                if pd.isna(c):
                    cells.append(f"{'(부족)':>16s}")
                else:
                    cells.append(f"{c*100:+7.3f}%p{stars(p):<3s}(n{ns:<4d})")
            print(f"{sig:10s} | " + " | ".join(cells))

    # H1 요약: 방향 맞고 유의한(p<.05) 건수
    ok = res.dropna(subset=["coef"]).copy()
    ok["dir_ok"] = np.where(ok["sig"].str.endswith("_L"), ok["coef"] > 0, ok["coef"] < 0)
    ok["sig_05"] = ok["p"] < 0.05
    print("\n" + "=" * 78)
    print("[H1 요약] 방향 일치 & p<.05 (= 신호대로 유의하게 예측) 건수 / 전체")
    for key, kor in MK:
        s = ok[ok["market"] == key]
        hit = int((s["dir_ok"] & s["sig_05"]).sum())
        print(f"  {kor:9s}: {hit:2d} / {len(s):2d}   (방향무관 유의 {int(s['sig_05'].sum())}건)")
    print("\n→ H3 점검: 위 '유의 건수'가 비트코인 > 한국 > 미국 순이면 효율성 기울기 지지.")
    print("→ 전체 결과 CSV: stage1_regression_results.csv")


if __name__ == "__main__":
    main()
