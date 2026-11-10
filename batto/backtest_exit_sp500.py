# -*- coding: utf-8 -*-
"""상위 3가지 매도 방식 — S&P500 전체 풀 확장 검증.
표본: 2016~2026 각 연도 [전년 거래대금(c*v) 1위 1 + 무작위 3].
방식: 고정 40일 / ATR 손익비 1:3 / 반익반본 ATR×3.
신호: Batto v1 (strict 컨플루언스). 청산 함수는 backtest_exit 재사용."""
import os, sys, json, random, warnings
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
random.seed(7); np.random.seed(7)
YEARS = list(range(2016, 2027))


def load(tk):
    fp = os.path.join(PXDIR, f"px_{tk.replace('-', '_')}.csv")
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")


def list_tickers():
    out = []
    for fn in os.listdir(PXDIR):
        if fn.startswith("px_") and fn.endswith(".csv"):
            out.append(fn[3:-4].replace("_", "-"))
    return sorted(out)


def build_sample(loaded):
    """연도별 [대장주1 + 무작위3]. 대장주 = 전년 거래대금(c*v) 최대 (룩어헤드 없음)."""
    sample = {}
    for yr in YEARS:
        prev = yr - 1
        dv = {}
        for tk, df in loaded.items():
            sp = df[df.index.year == prev]
            sc = df[df.index.year == yr]
            if len(sp) < 150 or len(sc) < 120:   # 전년·당해 거래일 충분해야
                continue
            dv[tk] = float((sp["c"] * sp["v"]).sum())
        if len(dv) < 4:
            continue
        top = max(dv, key=dv.get)
        pool = [t for t in dv if t != top]
        rand3 = random.sample(pool, 3)
        sample[yr] = [("대장주", top)] + [("무작위", t) for t in rand3]
    return sample


def collect(sample, loaded):
    pts, meta = [], []
    cache = {}
    for yr, items in sample.items():
        for role, tk in items:
            df = loaded[tk]
            o, h, l, c = (df[x].values for x in ["o", "h", "l", "c"])
            if tk not in cache:
                atr = BV.atr14(h, l, c)
                cache[tk] = (BV.build_batto(df, 42), atr,
                             BV.detect_ob_filtered(o, h, l, c, atr),
                             BV.detect_fvg_filtered(o, h, l, c, atr))
            sigs, atr, obs, fvs = cache[tk]
            ym = df.index.year == yr
            a = int(np.argmax(ym)); b = a + int(ym.sum()) - 1
            cnt = 0
            for s in sigs:
                j, d = s["j"], s["dir"]
                if not (a <= j <= b) or j + 61 >= len(c):
                    continue
                pts.append(dict(c=c, h=h, l=l, j=j, d=d, atr=atr[j],
                                obs=obs, fvs=fvs, tk=tk, yr=yr))
                cnt += 1
            meta.append((yr, role, tk, cnt))
    return pts, meta


def fnum(v, p):
    return f"{v:{p}}" if v == v else "   -  "


def main():
    tks = list_tickers()
    print(f"수집된 종목 {len(tks)}개 로드 중...", flush=True)
    loaded = {}
    for tk in tks:
        try:
            df = load(tk)
            if len(df) >= 150:
                loaded[tk] = df
        except Exception:
            pass
    print(f"유효 {len(loaded)}종목\n", flush=True)

    sample = build_sample(loaded)
    pts, meta = collect(sample, loaded)

    print("=" * 66)
    print("■ 표본 구성 (연도별 대장주1 + 무작위3, 대장주=전년 거래대금 1위)")
    print(f"  {'연도':>4s}  {'구분':4s} {'종목':9s} {'타점':>4s}")
    for yr in sorted(sample):
        for (myr, role, tk, cnt) in meta:
            if myr == yr:
                print(f"  {yr:>4d}  {role:4s} {tk:9s} {cnt:>4d}")
    nL = sum(p["d"] == "L" for p in pts); nS = sum(p["d"] == "S" for p in pts)
    print(f"\n총 타점 {len(pts)}개  (롱 {nL} / 숏 {nS})\n")

    methods = [
        ("고정 40일",      lambda pt: BE.exit_fixed(pt, 40)),
        ("ATR 손익비 1:3", lambda pt: BE.exit_atr_rr(pt, 3.0)),
        ("반익반본 ATR×3", lambda pt: BE.exit_half_profit(pt, 3.0)),
    ]
    print("=" * 66)
    print("■ 상위 3가지 매도 방식 — 확장 표본 결과")
    print(f"  {'방식':18s} {'타점':>5s} {'평균%':>7s} {'승률':>6s} {'t':>6s}")
    results = []
    for name, fn in methods:
        rs = [r for pt in pts if (r := fn(pt)) is not None]
        n, m, wr, t = BE.stat(rs)
        mark = " ◀유의" if (t == t and t >= 2.0) else ""
        print(f"  {name:18s} {n:5d} {fnum(m,'+7.2f')} {fnum(wr,'5.0f')}% {fnum(t,'+6.2f')}{mark}")
        results.append((name, n, m, wr, t))

    print("\n■ 연도별 성과 (고정 40일)")
    for yr in sorted(set(p["yr"] for p in pts)):
        rs = [r for pt in pts if pt["yr"] == yr and (r := BE.exit_fixed(pt, 40)) is not None]
        n, m, wr, t = BE.stat(rs)
        print(f"  {yr}: {n:3d}타점  {fnum(m,'+6.2f')}%  승률{fnum(wr,'4.0f')}%  t={fnum(t,'+5.2f')}")

    print("\n■ 방향별 (고정 40일)")
    for d, lb in [("L", "롱"), ("S", "숏")]:
        rs = [r for pt in pts if pt["d"] == d and (r := BE.exit_fixed(pt, 40)) is not None]
        n, m, wr, t = BE.stat(rs)
        print(f"  {lb}: {n:3d}타점  {fnum(m,'+6.2f')}%  승률{fnum(wr,'4.0f')}%  t={fnum(t,'+5.2f')}")

    out = {
        "sample": {str(yr): [[role, tk, cnt] for (myr, role, tk, cnt) in meta if myr == yr]
                   for yr in sorted(sample)},
        "n_points": len(pts), "n_long": nL, "n_short": nS,
        "methods": [(name, n, (m if m == m else None), (wr if wr == wr else None),
                     (t if t == t else None)) for (name, n, m, wr, t) in results],
    }
    with open(os.path.join(HERE, "exit_sp500_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: exit_sp500_results.json")


if __name__ == "__main__":
    main()
