# -*- coding: utf-8 -*-
"""매도 전략 전수 비교 — 4가지 방식 × 그리드.
S&P500 50종목 × 2021~2025 연도별 10개 × Batto v1(개선 파라미터)."""
import os, sys, random, warnings
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import golden_strategy as G
import batto_v1 as BV

random.seed(7); np.random.seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))
SNP = open(os.path.join(HERE, "snp50.txt")).read().strip().split()
YEARS = [2021, 2022, 2023, 2024, 2025]


def build_sample():
    sample = {}
    for yr in YEARS:
        pool = [tk for tk in SNP if (BV.load(tk).index.year == yr).sum() >= 200]
        sample[yr] = random.sample(pool, min(10, len(pool)))
    return sample


def stat(rs):
    a = np.array(rs); n = len(a)
    if n < 5: return n, np.nan, np.nan, np.nan
    m = a.mean(); t = m / (a.std(ddof=1) / np.sqrt(n))
    return n, m, (a > 0).mean() * 100, t


# ── 타점 수집 (전체 데이터 미리) ──────────────────────
def collect_signals(sample):
    pts = []
    for yr, tks in sample.items():
        for tk in tks:
            df = BV.load(tk)
            c = df["c"].values; h = df["h"].values; l = df["l"].values
            sigs = BV.build_batto(df, 42)
            atr = BV.atr14(h, l, c)
            obs = BV.detect_ob_filtered(df["o"].values, h, l, c, atr)
            fvs = BV.detect_fvg_filtered(df["o"].values, h, l, c, atr)
            ym = df.index.year == yr; a = int(np.argmax(ym)); b = a + int(ym.sum()) - 1
            for s in sigs:
                j = s["j"]; d = s["dir"]
                if not (a <= j <= b) or j + 61 >= len(c): continue
                pts.append(dict(c=c, h=h, l=l, j=j, d=d,
                                atr=atr[j], obs=obs, fvs=fvs, tk=tk, yr=yr))
    return pts


# ── 1. 고정 N일 ──────────────────────────────────────
def exit_fixed(pt, N):
    c, j, d = pt["c"], pt["j"], pt["d"]
    if j + N >= len(c): return None
    sgn = 1 if d == "L" else -1
    return sgn * (c[j + N] / c[j] - 1) * 100


# ── 2. ATR 손익비 (Stop=ATR×1, Target=ATR×RR) ───────
def exit_atr_rr(pt, RR):
    c, h, l, j, d, atr = pt["c"], pt["h"], pt["l"], pt["j"], pt["d"], pt["atr"]
    if atr == 0: return None
    sgn = 1 if d == "L" else -1
    stop_dist = atr * 1.0
    tgt_dist = atr * RR
    stop = c[j] - sgn * stop_dist
    tgt  = c[j] + sgn * tgt_dist
    for k in range(j + 1, min(j + 61, len(c))):
        if d == "L":
            if l[k] <= stop: return -(stop_dist / c[j]) * 100
            if h[k] >= tgt:  return  (tgt_dist  / c[j]) * 100
        else:
            if h[k] >= stop: return -(stop_dist / c[j]) * 100
            if l[k] <= tgt:  return  (tgt_dist  / c[j]) * 100
    return (1 if d == "L" else -1) * (c[min(j+60, len(c)-1)] / c[j] - 1) * 100


# ── 3. 반익반본 (1차 ATR×1.5 절반청산, 나머지 최대40일) ──
def exit_half_profit(pt, first_mult=1.5, max_days=40):
    c, h, l, j, d, atr = pt["c"], pt["h"], pt["l"], pt["j"], pt["d"], pt["atr"]
    if atr == 0: return None
    sgn = 1 if d == "L" else -1
    tgt1 = c[j] + sgn * atr * first_mult
    stop0 = c[j] - sgn * atr * 1.0
    stop = stop0; half_done = False; half_ret = 0.0
    for k in range(j + 1, min(j + max_days + 1, len(c))):
        if not half_done:
            hit_stop = l[k] <= stop if d == "L" else h[k] >= stop
            hit_tgt1 = h[k] >= tgt1 if d == "L" else l[k] <= tgt1
            if hit_stop:
                return sgn * (stop / c[j] - 1) * 100
            if hit_tgt1:
                half_ret = (atr * first_mult / c[j]) * 100
                stop = c[j]  # 손절 진입가로 이동(본전)
                half_done = True
        else:
            hit_be = l[k] <= stop if d == "L" else h[k] >= stop
            if hit_be:
                return (half_ret + 0.0) / 2  # 절반 익절 + 절반 본전
    # 시간 만료
    final = sgn * (c[min(j + max_days, len(c)-1)] / c[j] - 1) * 100
    if half_done:
        return (half_ret + final) / 2
    return final


# ── 4. 트레일링 스탑 (고점/저점 기준 ATR×mult 이탈) ─
def exit_trailing(pt, trail_mult):
    c, h, l, j, d, atr = pt["c"], pt["h"], pt["l"], pt["j"], pt["d"], pt["atr"]
    if atr == 0: return None
    sgn = 1 if d == "L" else -1
    best = c[j]
    for k in range(j + 1, min(j + 61, len(c))):
        cur_h = h[k]; cur_l = l[k]; cur_c = c[k]
        if d == "L":
            best = max(best, cur_h)
            trail = best - atr * trail_mult
            if cur_l <= trail: return (trail / c[j] - 1) * 100
        else:
            best = min(best, cur_l)
            trail = best + atr * trail_mult
            if cur_h >= trail: return -(trail / c[j] - 1) * 100
    return sgn * (c[min(j + 60, len(c)-1)] / c[j] - 1) * 100


# ── 5. 구조 기반 (반대 OB/FVG 출현 시 청산, 최대 N일) ─
def exit_structure(pt, max_days=40):
    c, h, l, j, d = pt["c"], pt["h"], pt["l"], pt["j"], pt["d"]
    obs, fvs = pt["obs"], pt["fvs"]
    opp = "S" if d == "L" else "L"
    stop = c[j] - (1 if d == "L" else -1) * pt["atr"] * 1.0
    for k in range(j + 1, min(j + max_days + 1, len(c))):
        hit_stop = l[k] <= stop if d == "L" else h[k] >= stop
        if hit_stop:
            return (1 if d == "L" else -1) * (stop / c[j] - 1) * 100
        # 반대 OB/FVG 등장 체크
        opp_ob = any(e["dir"] == opp and e["form"] == k for e in obs)
        opp_fv = any(e["dir"] == opp and e["form"] == k for e in fvs)
        if opp_ob or opp_fv:
            return (1 if d == "L" else -1) * (c[k] / c[j] - 1) * 100
    return (1 if d == "L" else -1) * (c[min(j + max_days, len(c)-1)] / c[j] - 1) * 100


def fmt(n, m, wr, t):
    f = lambda v, p: f"{v:{p}}" if v == v else "  -  "
    return f"{n:5d} {f(m,'+6.2f')} {f(wr,'5.0f')} {f(t,'+5.2f')}"


def main():
    print("샘플 구성 중...")
    sample = build_sample()
    print("신호 수집 중...")
    pts = collect_signals(sample)
    print(f"총 {len(pts)}개 타점\n")

    results = []  # (방식, 파라미터, n, m, wr, t)

    # ── 1. 고정 N일 ──
    print("="*60)
    print("■ 1. 고정 N일 청산")
    print(f"  {'N':>4s} {'타점':>5s} {'평균%':>7s} {'승률':>5s} {'t':>6s}")
    for N in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
        rs = [r for pt in pts if (r := exit_fixed(pt, N)) is not None]
        n, m, wr, t = stat(rs)
        mark = " ◀" if (t == t and t >= 2.0) else ""
        print(f"  {N:4d} {fmt(n,m,wr,t)}{mark}")
        results.append(("고정N일", f"{N}일", n, m, wr, t))

    # ── 2. ATR 손익비 ──
    print("\n■ 2. ATR 손익비 (손절=ATR×1)")
    print(f"  {'손익비':>6s} {'타점':>5s} {'평균%':>7s} {'승률':>5s} {'t':>6s}")
    for RR in [1.0, 1.5, 2.0, 2.5, 3.0]:
        rs = [r for pt in pts if (r := exit_atr_rr(pt, RR)) is not None]
        n, m, wr, t = stat(rs)
        mark = " ◀" if (t == t and t >= 2.0) else ""
        print(f"  1:{RR:<4.1f} {fmt(n,m,wr,t)}{mark}")
        results.append(("ATR손익비", f"1:{RR}", n, m, wr, t))

    # ── 3. 반익반본 ──
    print("\n■ 3. 반익반본 (1차목표 ATR×mult, 나머지 최대40일)")
    print(f"  {'1차목표':>8s} {'타점':>5s} {'평균%':>7s} {'승률':>5s} {'t':>6s}")
    for mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
        rs = [r for pt in pts if (r := exit_half_profit(pt, mult)) is not None]
        n, m, wr, t = stat(rs)
        mark = " ◀" if (t == t and t >= 2.0) else ""
        print(f"  ATR×{mult:<3.1f}  {fmt(n,m,wr,t)}{mark}")
        results.append(("반익반본", f"ATR×{mult}", n, m, wr, t))

    # ── 4. 트레일링 스탑 ──
    print("\n■ 4. 트레일링 스탑 (고점/저점 기준)")
    print(f"  {'트레일':>8s} {'타점':>5s} {'평균%':>7s} {'승률':>5s} {'t':>6s}")
    for mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        rs = [r for pt in pts if (r := exit_trailing(pt, mult)) is not None]
        n, m, wr, t = stat(rs)
        mark = " ◀" if (t == t and t >= 2.0) else ""
        print(f"  ATR×{mult:<3.1f}  {fmt(n,m,wr,t)}{mark}")
        results.append(("트레일링", f"ATR×{mult}", n, m, wr, t))

    # ── 5. 구조 기반 ──
    print("\n■ 5. 구조 기반 (반대 OB/FVG 출현 청산, 최대 N일)")
    print(f"  {'최대보유':>8s} {'타점':>5s} {'평균%':>7s} {'승률':>5s} {'t':>6s}")
    for N in [20, 30, 40, 60]:
        rs = [r for pt in pts if (r := exit_structure(pt, N)) is not None]
        n, m, wr, t = stat(rs)
        mark = " ◀" if (t == t and t >= 2.0) else ""
        print(f"  최대{N}일  {fmt(n,m,wr,t)}{mark}")
        results.append(("구조기반", f"최대{N}일", n, m, wr, t))

    # ── 종합 상위 10 ──
    print("\n" + "="*60)
    print("■ 전체 방식 상위 10 (t값 기준)")
    print(f"  {'방식':10s} {'파라미터':10s} {'타점':>5s} {'평균%':>7s} {'승률':>5s} {'t':>6s}")
    valid = [(w, p, n, m, wr, t) for w, p, n, m, wr, t in results if t == t and n >= 5]
    for w, p, n, m, wr, t in sorted(valid, key=lambda x: x[5], reverse=True)[:10]:
        f2 = lambda v, p2: f"{v:{p2}}" if v == v else "  -  "
        print(f"  {w:10s} {p:10s} {n:5d} {f2(m,'+6.2f')} {f2(wr,'5.0f')} {f2(t,'+5.2f')}")
    import json
    with open(os.path.join(HERE, "exit_results.json"), "w") as f3:
        json.dump([(w,p,n,float(m) if m==m else None,float(wr) if wr==wr else None,float(t) if t==t else None)
                   for w,p,n,m,wr,t in results], f3, ensure_ascii=False)
    print("\n결과 저장: exit_results.json")


if __name__ == "__main__":
    main()
