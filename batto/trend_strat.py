# -*- coding: utf-8 -*-
"""전략B: 추세추종 모멘텀 (공포매수 無) — 코인·한국 맞춤.
규칙: 지수>MA(위험ON)일 때만 → 모멘텀 상위K 동일비중 보유, 월 리밸. 위험OFF=현금.
히스테리시스 레짐(진입선>청산선) + 거래비용. 시장 무관(US/KR/CRYPTO)."""
import os, sys, json, time
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import rb_sim as rb, mkt_sim as ms

HERE = os.path.dirname(os.path.abspath(__file__))


def regime_arr(D, me, mx, conf):
    T = D["T"]; gspc = D["gspc"]; ma_e = D["spx_ma"][me]; ma_x = D["spx_ma"][mx]
    reg = np.zeros(T, bool); state = False; below = 0
    for t in range(1, T):
        pv = gspc[t - 1]
        if not state:
            if not np.isnan(ma_e[t - 1]) and pv > ma_e[t - 1]:
                state = True; below = 0
        else:
            if not np.isnan(ma_x[t - 1]) and pv < ma_x[t - 1]:
                below += 1
                if below > conf:
                    state = False
            else:
                below = 0
        reg[t] = state
    return reg


def run_trend(cfg, D):
    ca = cfg["core_asset"]; key = int(ca[3:]) if ca[3:].isdigit() else ca[3:]
    mom_ret = D["mom"][key]; mturn = D["momturn"][key]
    me = cfg["mkt_ma"]; mx = cfg.get("mkt_exit_ma", me); conf = cfg.get("mkt_confirm", 0)
    reg = regime_arr(D, me, mx, conf)
    alloc = cfg.get("alloc", 1.0)
    w = reg.astype(float) * alloc
    # 변동성 타겟팅: 직전 실현변동성이 목표보다 높으면 비중 축소
    tv = cfg.get("target_vol", 0)
    if tv > 0:
        ppy = D["ppy"]
        vw = cfg.get("vol_win", 30)   # 재의심: 30 하드코딩이었음
        rv = pd.Series(mom_ret).rolling(vw).std().values * np.sqrt(ppy)
        scale = np.ones(len(w))
        for t in range(len(w)):
            if t >= 1 and not np.isnan(rv[t - 1]) and rv[t - 1] > 0:
                scale[t] = min(1.0, tv / rv[t - 1])
        w = w * scale
    w_prev = np.concatenate([[0.0], w[:-1]])
    port = w_prev * mom_ret
    turn = np.abs(np.diff(w, prepend=0)) + w * mturn
    port = port - turn * (cfg.get("cost_bps", 5) / 1e4)
    invested = w_prev > 1e-6
    return port, invested, D["yrs"]


def evalc(cfg, D):
    p, i, y = run_trend(cfg, D); ppy = D["ppy"]
    mask = (y >= D["y0"]) & (y <= D["y1"]); p, i, y = p[mask], i[mask], y[mask]
    m = rb.metrics(p, i, y, ppy=ppy); s = rb.score(m)
    mid = (D["y0"] + D["y1"]) // 2
    h1 = rb.score(rb.metrics(p[y <= mid], i[y <= mid], y[y <= mid], ppy=ppy))["total"]
    h2 = rb.score(rb.metrics(p[y > mid], i[y > mid], y[y > mid], ppy=ppy))["total"]
    return m, s, h1, h2, p, y


def sweep(mkt):
    D = ms.load_market(mkt)
    Ks = [3, 5, 8] if D["n"] <= 25 else [5, 8, 10, 15]
    print(f"\n{'='*86}\n■ [{mkt}] 추세추종 모멘텀 스윕 (지수레짐 + 모멘텀top-K, 하락장 현금)")
    print(f"  {'코어':>9s} {'룩백':>4s} {'레짐MA':>9s} {'alloc':>5s} | {'full':>5s} {'강건':>5s} {'샤프':>5s} {'MDD':>7s} {'CAGR':>6s} {'소르':>5s} {'노출':>5s}")
    best = None
    for K in Ks:
        for lb in ["", "_126"]:
            for me in [150, 200, 220]:
                for mx in [100, 150]:
                    if mx > me:
                        continue
                    for alloc in [0.6, 0.8, 1.0]:
                        ca = f"mom{K}{lb}"
                        if (int(ca[3:]) if ca[3:].isdigit() else ca[3:]) not in D["mom"]:
                            continue
                        cfg = dict(core_asset=ca, mkt_ma=me, mkt_exit_ma=mx, mkt_confirm=0,
                                   alloc=alloc, cost_bps=5)
                        m, s, h1, h2, p, y = evalc(cfg, D)
                        mh = min(h1, h2)
                        row = (s["total"], mh, m["sharpe"], m["mdd"], m["cagr"], m["sortino"], m["expo"], cfg)
                        if best is None or (mh, s["total"]) > (best[1], best[0]):
                            best = row
                        if mh >= (45 if mkt == "CRYPTO" else 60) or s["total"] >= 80:
                            print(f"  {ca:>9s} {lb or '252':>4s} {me:>4d}/{mx:<3d}  {alloc:>5.1f} | {s['total']:>5.1f} {mh:>5.1f} {m['sharpe']:>5.2f} {m['mdd']:>+6.1f}% {m['cagr']:>+5.1f}% {m['sortino']:>5.2f} {m['expo']:>4.0f}%")
    print(f"  ★강건최고: full {best[0]:.1f} 강건 {best[1]:.1f} 샤프 {best[2]:.2f} MDD {best[3]:+.1f}% CAGR {best[4]:+.1f}% | {best[7]}")
    # 최고 연도별
    m, s, h1, h2, p, y = evalc(best[7], D)
    yr = {int(yy): (np.prod(1 + p[y == yy]) - 1) * 100 for yy in np.unique(y)}
    print(f"  연도별: " + " ".join(f"{k}:{v:+.0f}%" for k, v in yr.items()))
    return best


if __name__ == "__main__":
    for mkt in (sys.argv[1:] or ["CRYPTO", "KR"]):
        sweep(mkt)
