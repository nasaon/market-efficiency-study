# -*- coding: utf-8 -*-
"""터틀 청산 config들의 '다기간 robust' 검증 — 한 OOS 슬라이스 운빨 배제.
turtle_exit_evolve의 엔진 재사용. 핵심 config를 국면별 윈도우마다 바스켓 중앙값 Calmar로 비교.
어떤 청산이 '모든 국면서 고르게' 좋은가? (한 윈도우만 좋으면 국면운)."""
import os, sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import turtle_exit_evolve as EV

WINDOWS = [("18-20", "2018-01-01", "2021-01-01"), ("21불장", "2021-01-01", "2022-01-01"),
           ("22약세", "2022-01-01", "2023-01-01"), ("23회복", "2023-01-01", "2024-01-01"),
           ("24불장", "2024-01-01", "2025-01-01"), ("25-26", "2025-01-01", "2027-01-01")]

CONFIGS = [
    ("원전(저점10전량)", [{"frac": 1.0, "type": "donch", "param": 10}]),
    ("현재채택(저점10½+트레일2½)", [{"frac": 0.5, "type": "donch", "param": 10}, {"frac": 0.5, "type": "trail", "param": 2.0}]),
    ("타이트1+와이드3", [{"frac": 0.5, "type": "trail", "param": 1.0}, {"frac": 0.5, "type": "trail", "param": 3.0}]),
    ("3단(1/2/4)", [{"frac": 0.34, "type": "trail", "param": 1.0}, {"frac": 0.33, "type": "trail", "param": 2.0}, {"frac": 0.33, "type": "trail", "param": 4.0}]),
    ("RSI85+익절타깃8+트레일4", [{"frac": 0.34, "type": "rsi", "param": 85}, {"frac": 0.33, "type": "ptarget", "param": 8}, {"frac": 0.33, "type": "trail", "param": 4.0}]),
    ("반납15%+트레일3", [{"frac": 0.5, "type": "giveback", "param": 0.15}, {"frac": 0.5, "type": "trail", "param": 3.0}]),
    ("진화챔프(저점7+트레일1.4+저점6)", [{"frac": 0.36, "type": "donch", "param": 7}, {"frac": 0.39, "type": "trail", "param": 1.41}, {"frac": 0.25, "type": "donch", "param": 6}]),
]


def win_median_calmar(DATA, genome, lo, hi):
    lo = pd.Timestamp(lo); hi = pd.Timestamp(hi)
    cals = []
    for D in DATA:
        eq = EV.simulate(D, genome)
        m = EV.metrics_slice(eq, lo, hi)
        if m and m[3] == m[3]: cals.append(m[3])
    return float(np.median(cals)) if cals else np.nan


def main():
    print("데이터 로드...")
    DATA = []
    for tk, kind, eN, smaN, fee in EV.BASKET:
        try: DATA.append(EV.load_asset(tk, kind, eN, smaN, fee))
        except Exception: pass
    L = ["터틀 청산 다기간 robust 검증 — 국면별 바스켓 중앙값 Calmar (높을수록↑)",
         f"바스켓 {len(DATA)}종목(코인3+미국강추세6)\n",
         "  " + f"{'청산 config':30s}" + "".join(f"{w[0]:>8s}" for w in WINDOWS) + f"{'평균':>7s}{'최악':>7s}"]
    rows = []
    for name, tr in CONFIGS:
        g = EV.norm({"tranches": [dict(t) for t in tr]})
        vals = [win_median_calmar(DATA, g, w[1], w[2]) for w in WINDOWS]
        valid = [v for v in vals if v == v]
        avg = np.mean(valid) if valid else np.nan
        worst = np.min(valid) if valid else np.nan
        rows.append((name, vals, avg, worst))
        cells = "".join((f"{v:>8.2f}" if v == v else f"{'-':>8s}") for v in vals)
        L.append(f"  {name:30s}{cells}{avg:>7.2f}{worst:>7.2f}")
    L.append("")
    # 최고 평균 / 최고 최악(robust)
    best_avg = max(rows, key=lambda r: (r[2] if r[2] == r[2] else -9))
    best_worst = max(rows, key=lambda r: (r[3] if r[3] == r[3] else -9))
    L.append(f"  ★평균 최고: {best_avg[0]} (평균 {best_avg[2]:.2f})")
    L.append(f"  ★최악국면 최고(robust): {best_worst[0]} (최악 {best_worst[3]:.2f})")
    L.append("  ※ '고점익절'(RSI/익절타깃)이 모든 국면 robust하면 채택가치. 한 윈도우만 좋으면 국면운.")
    L.append("  ※ 22약세는 음수 가능(추세추종 하락장 약점). 생존편향·슬리피지/세금 미반영.")
    txt = "\n".join(L); print(txt)
    open(os.path.join(EV.HERE, "TURTLE_EXIT_ROBUST_LOG.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
