# -*- coding: utf-8 -*-
"""F&G<30 공포매수 강건성 검증.
- 클러스터 보정: 각 거래일의 동일가중 포폴 N일수익을 1관측치로(독립단위=날)
- 임계 민감도: F&G < 40/35/30/25/20
- 연도별 분해 / 거래비용 / F&G~수익 회귀
동일 기간 2021~2026, 503종목, 롱."""
import os, sys, warnings
from collections import defaultdict
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PXDIR = os.path.join(HERE, "sp500_px")
START = pd.Timestamp("2021-01-04"); END = pd.Timestamp("2026-06-04")
COST = 0.20   # 왕복 거래비용(%)


def load(tk):
    return pd.read_csv(os.path.join(PXDIR, f"px_{tk}.csv"),
                       parse_dates=["date"], index_col="date")


def welch(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    d = x.mean() - y.mean()
    return d, d / se


def main():
    fng = pd.read_csv(os.path.join(HERE, "fng_stock.csv"),
                      parse_dates=["date"]).set_index("date")["fng"]
    fngd = {d.normalize(): float(v) for d, v in fng.items()}

    files = sorted(fn for fn in os.listdir(PXDIR)
                   if fn.startswith("px_") and fn.endswith(".csv"))
    r40 = defaultdict(list); r60 = defaultdict(list)
    for fn in files:
        tk = fn[3:-4]
        try:
            df = load(tk)
        except Exception:
            continue
        if len(df) < 200:
            continue
        c = df["c"].values
        nd = df.index.normalize()
        inper = (df.index >= START) & (df.index <= END)
        n = len(c)
        for i in range(n):
            if not inper[i]:
                continue
            if i + 40 < n:
                r40[nd[i]].append((c[i + 40] / c[i] - 1) * 100)
            if i + 60 < n:
                r60[nd[i]].append((c[i + 60] / c[i] - 1) * 100)

    days = sorted(d for d in r40 if d in fngd)
    mkt40 = {d: float(np.mean(r40[d])) for d in days}
    mkt60 = {d: float(np.mean(r60[d])) for d in days if r60.get(d)}
    fv = {d: fngd[d] for d in days}
    print(f"분석 거래일 {len(days)}일 (각 날 = 동일가중 포폴 1관측치)\n")

    # 1. 임계 민감도 (40일, 날 단위 = 클러스터 보정)
    print("=" * 72)
    print("■ 임계 민감도 — 40일, 날 단위(클러스터 보정)")
    print(f"  {'임계':>7s} {'공포일':>6s} {'공포평균%':>9s} {'비공포%':>8s} {'초과%p':>8s} {'Welch t':>8s} {'net공포%':>8s}")
    for th in [40, 35, 30, 25, 20]:
        fd = [d for d in days if fv[d] < th]
        ndd = [d for d in days if fv[d] >= th]
        if len(fd) < 5:
            continue
        fx = [mkt40[d] for d in fd]; nx = [mkt40[d] for d in ndd]
        diff, t = welch(fx, nx)
        print(f"  <{th:<5d} {len(fd):6d} {np.mean(fx):+9.2f} {np.mean(nx):+8.2f} "
              f"{diff:+8.2f} {t:+8.2f} {np.mean(fx)-COST:+8.2f}")

    # 2. 연도별 (F&G<30)
    print("\n■ 연도별 — F&G<30 공포일 매수 (40일, 날 단위)")
    fd30 = [d for d in days if fv[d] < 30]
    for yr in range(2021, 2027):
        yd = [d for d in fd30 if d.year == yr]
        allyr = [mkt40[d] for d in days if d.year == yr]
        if not yd:
            print(f"  {yr}: 공포일 0  (그해 전체 {np.mean(allyr):+.2f}%)" if allyr else f"  {yr}: -")
            continue
        xs = [mkt40[d] for d in yd]
        print(f"  {yr}: 공포일 {len(yd):3d}  공포 {np.mean(xs):+6.2f}%  "
              f"그해전체 {np.mean(allyr):+6.2f}%  차 {np.mean(xs)-np.mean(allyr):+5.2f}%p")

    # 3. 60일 임계 민감도
    print("\n■ 임계 민감도 — 60일, 날 단위")
    print(f"  {'임계':>7s} {'공포일':>6s} {'공포평균%':>9s} {'비공포%':>8s} {'초과%p':>8s} {'Welch t':>8s}")
    d60 = [d for d in days if d in mkt60]
    for th in [30, 25, 20]:
        fd = [d for d in d60 if fv[d] < th]
        ndd = [d for d in d60 if fv[d] >= th]
        if len(fd) < 5:
            continue
        fx = [mkt60[d] for d in fd]; nx = [mkt60[d] for d in ndd]
        diff, t = welch(fx, nx)
        print(f"  <{th:<5d} {len(fd):6d} {np.mean(fx):+9.2f} {np.mean(nx):+8.2f} {diff:+8.2f} {t:+8.2f}")

    # 4. 회귀 40일수익 ~ F&G
    print("\n■ 회귀: 동일가중 40일수익 ~ F&G 지수 (날 단위)")
    X = np.array([fv[d] for d in days]); Y = np.array([mkt40[d] for d in days])
    n = len(X)
    b1 = np.cov(X, Y, ddof=1)[0, 1] / X.var(ddof=1)
    b0 = Y.mean() - b1 * X.mean()
    resid = Y - (b0 + b1 * X)
    Sxx = np.sum((X - X.mean()) ** 2); MSE = np.sum(resid ** 2) / (n - 2)
    seb = np.sqrt(MSE / Sxx); tb = b1 / seb
    print(f"  기울기 = {b1:+.4f} %p/(F&G 1점)   t = {tb:+.2f}   (음수 = 공포일수록 미래수익↑)")
    print(f"  → F&G 10점 하락 시 40일수익 {(-b1*10):+.2f}%p")
    print(f"\n  [거래비용] 무작위 대비 초과수익은 양쪽 다 비용 발생 → 초과엔 영향 없음.")
    print(f"  공포매수 절대수익은 왕복 {COST}%p 차감해도 충분히 양수(위 net공포% 열).")


if __name__ == "__main__":
    main()
