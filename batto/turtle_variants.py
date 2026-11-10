# -*- coding: utf-8 -*-
"""터틀 미검증 축 — 피라미딩·듀얼시스템(S1+S2). 청산=채택안 고정, 진입/포지션 구축만 변형.
질문: 원조 터틀의 (a)피라미딩(승자에 add-up) (b)듀얼시스템(20/55 병행)이 단일 대비 도움되나?
평가: 바스켓9·다국면 중앙값Calmar 평균(Train 4국면)/Test/종목승(vs 단일). 무레버리지(총 ≤100%)."""
import os, sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import turtle_exit_evolve as EV
import turtle_exit_robust as RB
import turtle_entry_filters as F1

SLN, TRAILN = 1.5, 2.0


def pos_series(D, mode):
    """진입 변형별 포지션(0~1). 청산=채택안(저점이탈 절반익절+트레일2·1.5N손절) 공통.
    mode: single / dual(20+55) / pyr(50%돌파+50%at+1N)."""
    h, l, c, a = D["h"], D["l"], D["c"], D["a"]; n = D["n"]; sma = D["sma"]
    eN = D["eN"]; xN = 10
    hh = pd.Series(h).rolling(eN).max().shift(1).values          # 기본 진입 고점
    hh2 = pd.Series(h).rolling(55).max().shift(1).values          # S2 고점(듀얼)
    ll = pd.Series(l).rolling(xN).min().shift(1).values
    ll2 = pd.Series(l).rolling(20).min().shift(1).values
    start = max(eN, D["smaN"], 55) + 1
    pos = 0.0; stop = peak = ent = 0.0; half = False; addDone = False; arr = np.zeros(n)
    for i in range(start, n):
        smaok = (not np.isnan(sma[i])) and c[i] > sma[i]
        if pos == 0:
            brk = (not np.isnan(hh[i]) and c[i] > hh[i])
            brk2 = (not np.isnan(hh2[i]) and c[i] > hh2[i])
            if smaok and (brk or (mode == "dual" and brk2)):
                ent = c[i]; stop = c[i] - SLN * a[i]; peak = h[i]; half = False; addDone = False
                if mode == "single": pos = 1.0
                elif mode == "pyr": pos = 0.5            # 절반 진입 후 add-up
                elif mode == "dual":                      # 20돌파 절반·55동시돌파면 전량
                    pos = 1.0 if (brk and brk2) else 0.5
        else:
            peak = max(peak, h[i]); trail = peak - TRAILN * a[i]
            # 포지션 구축(add-up) — 무레버리지 총 ≤1.0
            if mode == "pyr" and not addDone and pos < 1.0 and c[i] >= ent + 1.0 * a[i]:
                pos = 1.0; addDone = True
            if mode == "dual" and pos < 1.0 and not np.isnan(hh2[i]) and c[i] > hh2[i]:
                pos = 1.0                                  # S2(55) 추가 돌파 시 전량
            # 청산(공통·채택안)
            if not half:
                if l[i] <= stop: pos = 0.0
                elif not np.isnan(ll[i]) and c[i] < ll[i]: pos = pos * 0.5; half = True
            else:
                if l[i] <= max(stop, trail): pos = 0.0
        arr[i] = pos
    return pd.Series(arr, index=D["idx"])


def robust(DATA, mode, wins):
    vs = []
    for w in wins:
        lo, hi = pd.Timestamp(w[1]), pd.Timestamp(w[2]); cs = []
        for D in DATA:
            cal = F1.metrics_win(D, pos_series(D, mode), lo, hi)
            if cal == cal: cs.append(cal)
        if cs: vs.append(np.median(cs))
    return float(np.mean(vs)) if vs else np.nan


def per_asset(DATA, mode, base="single"):
    win = tot = 0; deltas = []
    for D in DATA:
        lo, hi = D["idx"][0], D["idx"][-1] + pd.Timedelta(days=1)
        c1 = F1.metrics_win(D, pos_series(D, mode), lo, hi)
        c0 = F1.metrics_win(D, pos_series(D, base), lo, hi)
        if c1 == c1 and c0 == c0: tot += 1; deltas.append((D["tk"], c1 - c0)); win += (c1 > c0)
    return win, tot, deltas


def main():
    print("데이터 로드...")
    DATA = []
    for tk, kind, eN, smaN, fee in EV.BASKET:
        try:
            D = EV.load_asset(tk, kind, eN, smaN, fee); DATA.append(D)
        except Exception as e: print(f"  {tk} 스킵 {e}")
    L = ["터틀 미검증 축 — 피라미딩·듀얼시스템 (청산=채택안 고정)",
         f"바스켓 {len(DATA)}(코인3+미국강추세6)·Train 4국면 중앙값Calmar 평균/Test/종목승(vs 단일)·무레버리지\n",
         f"  {'변형':28s}{'Train':>7s}{'Test':>7s}{'종목승':>8s}"]
    base_tr = robust(DATA, "single", F1.TRAIN_W); base_te = robust(DATA, "single", F1.WINDOWS[4:])
    L.append(f"  {'단일(현재·100% 돌파진입)':28s}{base_tr:>7.2f}{base_te:>7.2f}{'-':>8s}")
    for label, m in [("피라미딩(50%돌파+50%at+1N)", "pyr"), ("듀얼시스템(20돌파+55추가)", "dual")]:
        tr = robust(DATA, m, F1.TRAIN_W); te = robust(DATA, m, F1.WINDOWS[4:])
        win, tot, d = per_asset(DATA, m)
        mark = "  ★도움" if (tr > base_tr + 0.02 and win > tot*0.55 and te >= base_te) else "  무효/역효과"
        L.append(f"  {label:28s}{tr:>7.2f}{te:>7.2f}{win:>5d}/{tot:<2d}{mark}")
    L.append(f"\n  기준(단일) Train {base_tr:.2f}·Test {base_te:.2f}")
    L.append("  ※ 무레버리지라 피라미딩은 '늦게 100% 채움'(초기노출↓)일 뿐 — 레버리지 없으면 증폭 제한.")
    L.append("  ★도움=Train↑&종목과반&Test우위 동시. 슬리피지/세금·생존편향 미반영.")
    txt = "\n".join(L); print(txt)
    open(os.path.join(EV.HERE, "TURTLE_VARIANTS_LOG.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
