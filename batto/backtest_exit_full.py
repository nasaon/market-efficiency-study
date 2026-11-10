# -*- coding: utf-8 -*-
"""상위 3가지 매도 방식 — S&P500 503종목 전체 통합 검증.
표본: 503종목 × 2016~2026 전 타점 (대장주/무작위 구분 없음).
방식: 고정 40일 / ATR 손익비 1:3 / 반익반본 ATR×3.
신호: Batto v1 (strict 컨플루언스). 청산 함수는 backtest_exit 재사용.
주: 같은 종목 다수 타점 → 단순 t는 종목 클러스터로 다소 과대평가(상대비교 목적)."""
import os, sys, json, warnings
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_strategy as G          # noqa: F401  (build_batto 내부 의존)
import batto_v1 as BV
import backtest_exit as BE           # exit_fixed / exit_atr_rr / exit_half_profit / stat

PXDIR = os.path.join(HERE, "sp500_px")
Y0, Y1 = 2016, 2026


def load(tk):
    return pd.read_csv(os.path.join(PXDIR, f"px_{tk}.csv"),
                       parse_dates=["date"], index_col="date")


def fnum(v, p):
    return f"{v:{p}}" if v == v else "   -  "


def main():
    files = sorted(fn for fn in os.listdir(PXDIR)
                   if fn.startswith("px_") and fn.endswith(".csv"))
    print(f"종목 {len(files)}개 처리 시작...", flush=True)
    pts = []
    done = skipped = 0
    for fn in files:
        tk = fn[3:-4]
        try:
            df = load(tk)
        except Exception:
            skipped += 1; continue
        if len(df) < 200:
            skipped += 1; continue
        o, h, l, c = (df[x].values for x in ["o", "h", "l", "c"])
        try:
            sigs = BV.build_batto(df, 42)
        except Exception:
            skipped += 1; continue
        atr = BV.atr14(h, l, c)
        yrs = df.index.year.values
        for s in sigs:
            j, d = s["j"], s["dir"]
            if not (Y0 <= yrs[j] <= Y1) or j + 61 >= len(c):
                continue
            pts.append(dict(c=c, h=h, l=l, j=j, d=d, atr=atr[j],
                            tk=tk, yr=int(yrs[j])))
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(files)}  누적타점 {len(pts)}", flush=True)

    nL = sum(p["d"] == "L" for p in pts); nS = sum(p["d"] == "S" for p in pts)
    print(f"\n처리 {done}종목 (스킵 {skipped})  총 타점 {len(pts)}개  (롱 {nL} / 숏 {nS})\n", flush=True)

    # ── 상위 3가지 매도 방식 ──
    methods = [
        ("고정 40일",      lambda pt: BE.exit_fixed(pt, 40)),
        ("ATR 손익비 1:3", lambda pt: BE.exit_atr_rr(pt, 3.0)),
        ("반익반본 ATR×3", lambda pt: BE.exit_half_profit(pt, 3.0)),
    ]
    print("=" * 60)
    print("■ 상위 3가지 매도 방식 — 503종목 전체 통합")
    print(f"  {'방식':18s} {'타점':>6s} {'평균%':>7s} {'승률':>6s} {'t':>7s}")
    results = []
    for name, fn in methods:
        rs = [r for pt in pts if (r := fn(pt)) is not None]
        n, m, wr, t = BE.stat(rs)
        mark = " ◀유의" if (t == t and t >= 2.0) else ""
        print(f"  {name:18s} {n:6d} {fnum(m,'+7.2f')} {fnum(wr,'5.0f')}% {fnum(t,'+7.2f')}{mark}")
        results.append((name, n, m, wr, t))

    # ── 고정 N일 그리드 (원래 질문: 왜 20일?) ──
    print("\n■ 고정 N일 그리드 (보유기간별 — 큰 표본 재확인)")
    print(f"  {'N':>4s} {'타점':>6s} {'평균%':>7s} {'승률':>6s} {'t':>7s}")
    grid = []
    for N in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
        rs = [r for pt in pts if (r := BE.exit_fixed(pt, N)) is not None]
        n, m, wr, t = BE.stat(rs)
        mark = " ◀" if (t == t and t >= 2.0) else ""
        print(f"  {N:4d} {n:6d} {fnum(m,'+7.2f')} {fnum(wr,'5.0f')}% {fnum(t,'+7.2f')}{mark}")
        grid.append((N, n, m, wr, t))

    # ── 연도별 (고정 40일) ──
    print("\n■ 연도별 (고정 40일)")
    for yr in range(Y0, Y1 + 1):
        rs = [r for pt in pts if pt["yr"] == yr and (r := BE.exit_fixed(pt, 40)) is not None]
        n, m, wr, t = BE.stat(rs)
        print(f"  {yr}: {n:4d}타점  {fnum(m,'+6.2f')}%  승률{fnum(wr,'4.0f')}%  t={fnum(t,'+5.2f')}")

    # ── 방향별 (고정 40일) ──
    print("\n■ 방향별 (고정 40일)")
    for d, lb in [("L", "롱"), ("S", "숏")]:
        rs = [r for pt in pts if pt["d"] == d and (r := BE.exit_fixed(pt, 40)) is not None]
        n, m, wr, t = BE.stat(rs)
        print(f"  {lb}: {n:4d}타점  {fnum(m,'+6.2f')}%  승률{fnum(wr,'4.0f')}%  t={fnum(t,'+5.2f')}")

    out = {
        "n_points": len(pts), "n_long": nL, "n_short": nS, "n_stocks": done,
        "methods": [(nm, n, (m if m == m else None), (wr if wr == wr else None),
                     (t if t == t else None)) for (nm, n, m, wr, t) in results],
        "fixed_grid": [(N, n, (m if m == m else None), (wr if wr == wr else None),
                        (t if t == t else None)) for (N, n, m, wr, t) in grid],
    }
    with open(os.path.join(HERE, "exit_full_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: exit_full_results.json")


if __name__ == "__main__":
    main()
