# -*- coding: utf-8 -*-
"""BATTO 일봉장기 — 피보나치 되돌림 DCA 백테스트 (forward EV).

피보 100%(고점) 정의 두 모드 비교:
 - first   : 골든크로스 후 '첫 swing high'(K=5 pivot) = 고점 (구버전)
 - runhigh : 골든크로스 후 '제일 높은 고점'을 동적(running-high)으로 갱신하며 되돌림 매수.
             각 시점까지 보이는 최고점만 사용(look-ahead 없음). 새 고점이 나오면 피보 리셋.

공통:
 1) 일봉200MA & 4h200MA → 4h200MA가 일봉200MA 상향돌파 = 골든크로스
 2) 최저점(0%) = 고점 시점 기준 과거 252일(1년) / 126일(6개월)
 3) 하락 중 0.382/0.5/0.618 도달(intraday low ≤ 레벨) 시 진입(레벨가, 갭하락시 시가), 비중 1:1:2
 4) 사이클: 다음 골든크로스 또는 0% 종가 이탈 시 종료(레벨당 1회)
 5) forward EV: 진입 후 [10,20,40,60]일 종가수익률. baseline=같은종목 유효구간 평균 → alpha.

유니버스: fib_universe.json union (미국 70종목)
"""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import golden_strategy as G
import backtest_4h as B4

HERE = G.HERE
K = 5
RATIOS = [0.382, 0.5, 0.618]
WEIGHT = {0.382: 1, 0.5: 1, 0.618: 2}
FWD = [10, 20, 40, 60]
LOOKBACKS = {"1y": 252, "6m": 126}


def daily_4h_ma(h4, win=200):
    """4h 종가 200SMA → 일별 마지막값 시계열(naive index).
    4h CSV는 혼합 tz오프셋(-04:00/-05:00 등)이라 object Index로 파싱됨 → UTC 통일."""
    s = h4["c"].rolling(win).mean()
    idx = pd.to_datetime(s.index, utc=True)
    s.index = idx.tz_convert(None)
    s = s.resample("1D").last().dropna()
    s.index = s.index.normalize()
    return s


def _fwd(c, j, entry, n):
    return {N: (c[j+N]/entry - 1.0)*100 for N in FWD if j+N < n}


def analyze(sym, peak_mode="first"):
    df = G.load_price(sym)
    try:
        h4 = B4.load_4h(sym)
    except Exception:
        return [], {}
    if len(h4) < 220 or len(df) < 260:
        return [], {}
    o, h, l, c = [df[x].values.astype(float) for x in ["o", "h", "l", "c"]]
    idx = df.index; n = len(c)
    dma = pd.Series(c).rolling(200).mean().values
    h4d = daily_4h_ma(h4); h4_al = h4d.reindex(idx, method="ffill").values
    gc = np.zeros(n, bool)
    for i in range(1, n):
        a0, a1, b0, b1 = dma[i-1], dma[i], h4_al[i-1], h4_al[i]
        if np.isnan(a0) or np.isnan(a1) or np.isnan(b0) or np.isnan(b1):
            continue
        if b0 <= a0 and b1 > a1:
            gc[i] = True
    sh, sl = G.swings(h, l)
    gci = np.where(gc)[0]
    recs = []
    for ci, g in enumerate(gci):
        nextg = gci[ci+1] if ci+1 < len(gci) else n
        if peak_mode == "first":
            peak = None
            for i in range(g+1, nextg):
                if sh[i]:
                    peak = i; break
            if peak is None:
                continue
            confirm = peak + K
            if confirm >= nextg:
                continue
            high = h[peak]
            for lbname, lbd in LOOKBACKS.items():
                low = l[max(0, peak-lbd):peak+1].min()
                if low >= high:
                    continue
                levels = {r: high - r*(high-low) for r in RATIOS}
                filled = set(); j = confirm + 1
                while j < nextg:
                    if c[j] < low:
                        break
                    for r in RATIOS:
                        if r in filled:
                            continue
                        if l[j] <= levels[r]:
                            entry = min(o[j], levels[r]); filled.add(r)
                            recs.append(dict(sym=sym, lb=lbname, r=r, w=WEIGHT[r], j=j,
                                             entry=entry, fwd=_fwd(c, j, entry, n)))
                    if len(filled) == 3:
                        break
                    j += 1
        else:  # runhigh — 동적 최고점, look-ahead 없음
            for lbname, lbd in LOOKBACKS.items():
                run_high = -1e18; levels = None; low_j = None; filled = set()
                for j in range(g+1, nextg):
                    if h[j] > run_high:
                        run_high = h[j]
                        low_j = l[max(0, j-lbd):j+1].min()
                        if low_j < run_high:
                            levels = {r: run_high - r*(run_high-low_j) for r in RATIOS}
                            filled = set()
                        else:
                            levels = None
                    elif levels is not None:
                        if c[j] < low_j:
                            levels = None      # 0% 이탈 → 새 고점 대기
                            continue
                        for r in RATIOS:
                            if r in filled:
                                continue
                            if l[j] <= levels[r]:
                                entry = min(o[j], levels[r]); filled.add(r)
                                recs.append(dict(sym=sym, lb=lbname, r=r, w=WEIGHT[r], j=j,
                                                 entry=entry, fwd=_fwd(c, j, entry, n)))
    valid = ~(np.isnan(dma) | np.isnan(h4_al))
    base = {}
    for N in FWD:
        vs = [(c[i+N]/c[i]-1.0)*100 for i in range(n) if valid[i] and i+N < n]
        base[N] = float(np.mean(vs)) if len(vs) >= 20 else float("nan")
    return recs, base


def stat(vals):
    a = np.array(vals, float); nn = len(a)
    if nn < 3:
        return nn, np.nan, np.nan, np.nan
    m = a.mean(); wr = (a > 0).mean()*100
    t = m / (a.std(ddof=1)/np.sqrt(nn)) if a.std(ddof=1) > 0 else np.nan
    return nn, m, wr, t


def wmean(pairs):
    if not pairs:
        return np.nan, 0
    rs = np.array([p[0] for p in pairs]); ws = np.array([p[1] for p in pairs])
    return float((rs*ws).sum()/ws.sum()), len(rs)


def report(recs, key="fwd", label="forward EV"):
    lines = []
    for lb in LOOKBACKS:
        sub = [x for x in recs if x["lb"] == lb]
        lines.append(f"\n  --- 룩백 {lb} (최저점 과거 {LOOKBACKS[lb]}일) · {label} ---")
        lines.append(f"  {'레벨':>6s}{'타점':>6s} " + "".join(f"{key.upper()+'@'+str(N):>10s}" for N in FWD)
                     + f"{'승률@20':>9s}{'t@20':>7s}")
        for r in RATIOS:
            rr = [x for x in sub if x["r"] == r]
            row = f"  {r:>6.3f}{len(rr):>6d} "
            for N in FWD:
                vals = [x[key][N] for x in rr if N in x.get(key, {})]
                _, m, _, _ = stat(vals)
                row += f"{m:>10.2f}" if m == m else f"{'-':>10s}"
            v20 = [x[key][20] for x in rr if 20 in x.get(key, {})]
            _, _, wr, t = stat(v20)
            row += f"{wr:>9.1f}" if wr == wr else f"{'-':>9s}"
            row += f"{t:>7.2f}" if t == t else f"{'-':>7s}"
            lines.append(row)
        for tag, weighted in [("전체(동일)", False), ("전체(1:1:2)", True)]:
            row = f"  {tag:>9s}{len(sub):>3d} "
            for N in FWD:
                if weighted:
                    pairs = [(x[key][N], x["w"]) for x in sub if N in x.get(key, {})]
                    m, _ = wmean(pairs)
                else:
                    vals = [x[key][N] for x in sub if N in x.get(key, {})]
                    _, m, _, _ = stat(vals)
                row += f"{m:>10.2f}" if m == m else f"{'-':>10s}"
            v20 = [x[key][20] for x in sub if 20 in x.get(key, {})]
            _, _, wr, t = stat(v20)
            row += f"{wr:>9.1f}" if wr == wr else f"{'-':>9s}"
            row += f"{t:>7.2f}" if t == t else f"{'-':>7s}"
            lines.append(row)
    return "\n".join(lines)


def main():
    uni = json.load(open(os.path.join(HERE, "fib_universe.json"), encoding="utf-8"))["union"]
    out_blocks = []
    for mode in ["first", "runhigh"]:
        allrecs = []; nsym = 0; base_avg = {N: [] for N in FWD}
        for t in uni:
            try:
                recs, base = analyze(t, mode)
            except Exception as e:
                print(f"  {t} ERR {e}", flush=True); continue
            if recs:
                nsym += 1
            for N in FWD:
                if base.get(N) == base.get(N):
                    base_avg[N].append(base[N])
            for x in recs:
                x["alpha"] = {N: x["fwd"][N] - base[N] for N in x["fwd"]
                              if N in base and base[N] == base[N]}
            allrecs.extend(recs)
        title = "첫 고점" if mode == "first" else "골든크로스후 제일 높은 고점(동적 running-high)"
        head = f"\n\n==================== 고점정의 = {mode} ({title}) ===================="
        head += f"\n분석종목 {nsym}/{len(uni)}   총 진입(레벨×룩백) {len(allrecs)}"
        head += "\n시장 baseline: " + "  ".join(f"@{N}={np.mean(v):+.2f}%" for N, v in base_avg.items() if v)
        block = head + "\n#### 절대 forward EV ####" + report(allrecs, "fwd", "절대EV")
        block += "\n\n#### ALPHA (초과수익) ####" + report(allrecs, "alpha", "ALPHA")
        print(block); out_blocks.append(block)
    with open(os.path.join(HERE, "FIB_DCA_RESULT.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_blocks))
    print("\n저장: FIB_DCA_RESULT.txt")


if __name__ == "__main__":
    main()
