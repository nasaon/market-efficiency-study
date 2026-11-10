# -*- coding: utf-8 -*-
"""진입 시점 벤치마크 비교 — Batto 신호 vs 무작위 vs F&G<30(공포 매수).
동일 기간(2021-01-04~2026-06-04)·동일 503종목·롱 한정·고정 40/60일 청산.
질문: Batto 신호 진입이 '아무 날 진입(무작위)'보다 유의하게 나은가? = 매도/진입 알파 존재?"""
import os, sys, json, random, warnings
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_strategy as G          # noqa: F401
import batto_v1 as BV

PXDIR = os.path.join(HERE, "sp500_px")
random.seed(7); np.random.seed(7)
START = pd.Timestamp("2021-01-04")
END = pd.Timestamp("2026-06-04")
RAND_PER_STOCK = 100   # 종목당 무작위 진입 표본


def load(tk):
    return pd.read_csv(os.path.join(PXDIR, f"px_{tk}.csv"),
                       parse_dates=["date"], index_col="date")


def ret(c, j, N):
    if j + N >= len(c):
        return None
    return (c[j + N] / c[j] - 1) * 100   # 롱 방향수익(%)


def stat(rs):
    a = np.array([r for r in rs if r is not None]); n = len(a)
    if n < 5:
        return n, np.nan, np.nan, np.nan
    m = a.mean(); t = m / (a.std(ddof=1) / np.sqrt(n))
    return n, m, (a > 0).mean() * 100, t


def welch(x, y):
    x = np.array([v for v in x if v is not None])
    y = np.array([v for v in y if v is not None])
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    return x.mean() - y.mean(), (x.mean() - y.mean()) / se


def main():
    fng = pd.read_csv(os.path.join(HERE, "fng_stock.csv"),
                      parse_dates=["date"]).set_index("date")["fng"]
    fear = set(fng[fng < 30].index.normalize())

    files = sorted(fn for fn in os.listdir(PXDIR)
                   if fn.startswith("px_") and fn.endswith(".csv"))
    print(f"종목 {len(files)}개, 기간 {START.date()}~{END.date()}, 롱only, 공포일 {len(fear)}일",
          flush=True)

    A = {"r40": [], "r60": []}   # Batto 신호
    B = {"r40": [], "r60": []}   # 무작위
    C = {"r40": [], "r60": []}   # F&G<30
    nstock = done = 0
    for fn in files:
        tk = fn[3:-4]
        try:
            df = load(tk)
        except Exception:
            continue
        if len(df) < 200:
            continue
        c = df["c"].values; dates = df.index
        inper = (dates >= START) & (dates <= END)
        # A — Batto 롱 신호
        try:
            sigs = BV.build_batto(df, 42)
        except Exception:
            sigs = []
        for s in sigs:
            if s["dir"] != "L":
                continue
            j = s["j"]
            if not inper[j]:
                continue
            A["r40"].append(ret(c, j, 40)); A["r60"].append(ret(c, j, 60))
        # 기간내 + 60일 확보된 유효 진입 인덱스
        valid = [i for i in range(len(c)) if inper[i] and i + 61 < len(c)]
        # B — 무작위
        if valid:
            for i in random.sample(valid, min(RAND_PER_STOCK, len(valid))):
                B["r40"].append(ret(c, i, 40)); B["r60"].append(ret(c, i, 60))
        # C — F&G<30 공포일
        norm = dates.normalize()
        for i in valid:
            if norm[i] in fear:
                C["r40"].append(ret(c, i, 40)); C["r60"].append(ret(c, i, 60))
        nstock += 1
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(files)}  A={len(A['r40'])} B={len(B['r40'])} C={len(C['r40'])}",
                  flush=True)

    print(f"\n처리 {nstock}종목\n")
    print("=" * 74)
    print("■ 진입 시점 벤치마크 — 롱 고정 40/60일 (2021~2026, 503종목)")
    print(f"  {'전략':16s} {'타점':>7s} {'평균40%':>8s} {'승률40':>7s} {'t40':>7s} {'평균60%':>8s} {'t60':>7s}")
    out = {}
    f = lambda v, p: f"{v:{p}}" if v == v else "  -  "
    for name, D in [("Batto 신호", A), ("무작위 진입", B), ("F&G<30 진입", C)]:
        n40, m40, w40, t40 = stat(D["r40"])
        n60, m60, w60, t60 = stat(D["r60"])
        print(f"  {name:16s} {n40:7d} {f(m40,'+8.2f')} {f(w40,'6.0f')}% {f(t40,'+7.2f')} "
              f"{f(m60,'+8.2f')} {f(t60,'+7.2f')}")
        out[name] = dict(n=n40, m40=m40, w40=w40, t40=t40, m60=m60, t60=t60)

    print("\n■ 무작위 대비 초과수익 (Welch t — 진입 알파 검정)")
    for name, D in [("Batto 신호", A), ("F&G<30", C)]:
        d40, t40 = welch(D["r40"], B["r40"])
        d60, t60 = welch(D["r60"], B["r60"])
        print(f"  {name:12s} vs 무작위:  40일 {d40:+.2f}%p (t={t40:+.2f})   "
              f"60일 {d60:+.2f}%p (t={t60:+.2f})")

    with open(os.path.join(HERE, "benchmark_results.json"), "w", encoding="utf-8") as fp:
        json.dump({k: {kk: (vv if vv == vv else None) for kk, vv in v.items()}
                   for k, v in out.items()}, fp, ensure_ascii=False, indent=2)
    print("\n저장: benchmark_results.json")


if __name__ == "__main__":
    main()
