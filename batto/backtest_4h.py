# -*- coding: utf-8 -*-
"""4시간봉 정제 전략(추세선A ∪ OB+FVG겹침) — 표본 확보. 타점 수 + 보유봉수별 성과 + MFE/MAE.
1h(yfinance 730d) → 4h 리샘플."""
import os, sys, warnings
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import golden_strategy as G
import backtest_refined as BR

SYMS = [("BTC","BTC-USD"), ("S&P","^GSPC"), ("코스피","^KS11"), ("삼성","005930.KS"), ("하이닉스","000660.KS")]


def load_4h(tk):
    fp = os.path.join(G.HERE, f"h4_{tk.replace('.','_').replace('^','x')}.csv")
    if not os.path.exists(fp):
        d = yf.download(tk, period="730d", interval="1h", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d = d[["Open","High","Low","Close"]].dropna(); d.columns = ["o","h","l","c"]
        try: r = d.resample("4h").agg({"o":"first","h":"max","l":"min","c":"last"}).dropna()
        except Exception: r = d.resample("4H").agg({"o":"first","h":"max","l":"min","c":"last"}).dropna()
        r.index.name = "date"; r.to_csv(fp, encoding="utf-8-sig")
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")


def stat(rs):
    a = np.array(rs); n = len(a)
    if n < 3: return n, np.nan, np.nan, np.nan
    return n, a.mean(), (a > 0).mean()*100, a.mean()/(a.std(ddof=1)/np.sqrt(n))


def main():
    allpts = []
    print("종목별 4시간봉:")
    for kor, tk in SYMS:
        df = load_4h(tk); c = df["c"].values; h = df["h"].values; l = df["l"].values
        sigs = BR.sigs_refined(df)
        ref = [s for s in sigs if s["trig"] == "추세선" or s["grade"] == "OB+FVG"]
        print(f"  {kor:8s} 4h봉 {len(df):6d}  전체타점 {len(sigs):4d}  정제 {len(ref):3d}")
        for s in ref: allpts.append((c, h, l, s["j"], s["dir"]))
    print(f"\n4h 정제 타점(추세선A ∪ 겹침) 총 {len(allpts)}개\n")

    print("보유봉수별 (4h봉, 6봉=하루(코인)):")
    print(f"  {'봉수':>5s}{'≈일':>6s}{'타점':>6s}{'평균%':>8s}{'승률%':>7s}{'t':>6s}")
    for N in [6, 12, 18, 30, 60, 120]:
        rs = []
        for c, h, l, j, d in allpts:
            if j+N >= len(c): continue
            rs.append((1 if d == "L" else -1)*(c[j+N]/c[j]-1)*100)
        n, m, wr, t = stat(rs)
        f = lambda v, p: f"{v:{p}}" if v == v else "  -  "
        print(f"  {N:5d}{N/6:6.1f}{n:6d}{f(m,'+8.2f')}{f(wr,'7.0f')}{f(t,'+6.2f')}")

    print("\n타점 잠재력 (진입 후 90봉, 방향 고가/저가):")
    mfe = []; mae = []
    for c, h, l, j, d in allpts:
        H = min(j+90, len(c)-1)
        if H <= j: continue
        if d == "L":
            fav = (h[j+1:H+1].max()/c[j]-1)*100; adv = (l[j+1:H+1].min()/c[j]-1)*100
        else:
            fav = (1-l[j+1:H+1].min()/c[j])*100; adv = (1-h[j+1:H+1].max()/c[j])*100
        mfe.append(fav); mae.append(adv)
    print(f"  평균 MFE {np.mean(mfe):+.1f}%   평균 MAE {np.mean(mae):+.1f}%   비 {np.mean(mfe)/abs(np.mean(mae)):.2f}")


if __name__ == "__main__":
    main()
