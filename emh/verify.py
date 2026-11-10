# -*- coding: utf-8 -*-
"""
신호 검증 — (1) 정의·look-ahead 전수 감사, (2) 실제 예시 캔들 차트 렌더.
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import signals as S

OUTDIR = os.path.join(S.HERE, "verify")
os.makedirs(OUTDIR, exist_ok=True)


def load(name):
    df = pd.read_csv(os.path.join(S.DATA, f"{name}.csv"), parse_dates=["date"], index_col="date")
    df = S.add_basics(df)
    is_sh, is_sl = S.swing_points(df)
    _, fvg_ev = S.fvg_signals(df)
    _, ob_ev = S.ob_signals(df, is_sh, is_sl)
    return df, is_sh, is_sl, fvg_ev, ob_ev


# ── 전수 감사 ──────────────────────────────────────────
def audit_fvg(df, events):
    h, l, o, c = (df["high"].values, df["low"].values,
                  df["open"].values, df["close"].values)
    body, atr = df["body"].values, df["atr"].values
    n = len(df); bad = 0; checked = 0
    for e in events:
        b, j, d = e["form"], e["entry"], e["dir"]
        a = b - 2
        checked += 1
        ok = True
        # 1) 갭 정의
        if d == "L":
            ok &= (h[a] < l[b]) and abs(e["g_lo"] - h[a]) < 1e-6 and abs(e["g_hi"] - l[b]) < 1e-6
        else:
            ok &= (l[a] > h[b]) and abs(e["g_lo"] - h[b]) < 1e-6 and abs(e["g_hi"] - l[a]) < 1e-6
        # 2) look-ahead: 진입은 형성 이후, 유효창 이내
        ok &= (b < j <= b + S.WINDOW_K)
        # 3) 크기필터
        mid = b - 1
        if e["size"] == "x2":
            nb = max(body[mid - 1], body[mid + 1])
            ok &= body[mid] >= 2.0 * nb - 1e-9
        elif e["size"] == "atr15":
            ok &= body[mid] >= 1.5 * atr[mid] - 1e-9
        if not ok:
            bad += 1
    return checked, bad


def audit_ob(df, events):
    h, l, o, c = (df["high"].values, df["low"].values,
                  df["open"].values, df["close"].values)
    bad = 0; checked = 0
    for e in events:
        s, m, p, j, d = e["swing"], e["bos"], e["ob"], e["entry"], e["dir"]
        checked += 1
        ok = True
        # 스윙 확정(+k) 후 BOS
        ok &= (s + S.SWING_K) <= m
        if d == "L":
            ok &= c[m] > h[s] and c[m - 1] <= h[s]   # 풀바디 종가 돌파(교차)
            ok &= c[p] < o[p]                          # OB=마지막 음봉
            ok &= l[j] <= e["z_hi"]                     # 되돌림(하락) 터치
        else:
            ok &= c[m] < l[s] and c[m - 1] >= l[s]
            ok &= c[p] > o[p]                          # OB=마지막 양봉
            ok &= h[j] >= e["z_lo"]                     # 되돌림(상승) 터치
        # look-ahead: OB는 BOS 이전, 진입은 BOS 이후·유효창 이내
        ok &= (p < m < j <= m + S.WINDOW_K)
        if not ok:
            bad += 1
    return checked, bad


# ── 캔들 차트 ──────────────────────────────────────────
def draw_candles(ax, df, i0, i1):
    sub = df.iloc[i0:i1 + 1]
    for k, (_, row) in enumerate(sub.iterrows()):
        up = row["close"] >= row["open"]
        col = "#d62728" if up else "#1f77b4"   # 한국식: 상승=빨강
        ax.plot([k, k], [row["low"], row["high"]], color=col, lw=0.8, zorder=2)
        lo = min(row["open"], row["close"]); hi = max(row["open"], row["close"])
        ax.add_patch(Rectangle((k - 0.3, lo), 0.6, max(hi - lo, 1e-9),
                               facecolor=col, edgecolor=col, zorder=3))
    return sub


def chart_fvg(df, ev, name, tag):
    b, j, d = ev["form"], ev["entry"], ev["dir"]
    i0 = max(b - 5, 0); i1 = min(j + 8, len(df) - 1)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sub = draw_candles(ax, df, i0, i1)
    # 갭 음영 (형성봉~진입봉)
    gx0 = b - i0; gx1 = j - i0
    ax.add_patch(Rectangle((gx0 - 0.5, ev["g_lo"]), (gx1 - gx0) + 1.0,
                           ev["g_hi"] - ev["g_lo"], facecolor="gold", alpha=0.35, zorder=1))
    ax.axvline(b - i0, color="green", ls="--", lw=1, label=f"형성봉(3번) idx={b}")
    ax.axvline(j - i0, color="purple", ls="-", lw=1.5, label=f"진입봉(되돌림 {ev['fill']}) idx={j}")
    ax.set_title(f"[{name}] FVG {d} size={ev['size']} fill={ev['fill']}  "
                 f"갭=[{ev['g_lo']:.2f}, {ev['g_hi']:.2f}]  "
                 f"형성→진입 {j-b}봉", fontsize=11)
    xt = range(0, len(sub), max(1, len(sub) // 8))
    ax.set_xticks(list(xt)); ax.set_xticklabels([sub.index[t].strftime("%y-%m-%d") for t in xt], fontsize=8)
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.2)
    p = os.path.join(OUTDIR, f"fvg_{name}_{tag}.png")
    plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
    return p


def chart_ob(df, ev, name, tag):
    s, m, p_, j, d = ev["swing"], ev["bos"], ev["ob"], ev["entry"], ev["dir"]
    i0 = max(min(s, p_) - 3, 0); i1 = min(j + 6, len(df) - 1)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sub = draw_candles(ax, df, i0, i1)
    # OB 존 음영 (OB봉~진입봉)
    zx0 = p_ - i0; zx1 = j - i0
    ax.add_patch(Rectangle((zx0 - 0.5, ev["z_lo"]), (zx1 - zx0) + 1.0,
                           ev["z_hi"] - ev["z_lo"], facecolor="orange", alpha=0.30, zorder=1))
    swing_price = df["high"].values[s] if d == "L" else df["low"].values[s]
    ax.axhline(swing_price, color="gray", ls=":", lw=1, label=f"스윙{'고' if d=='L' else '저'}점 idx={s}")
    ax.axvline(p_ - i0, color="green", ls="--", lw=1, label=f"OB봉(마지막 {'음' if d=='L' else '양'}봉) idx={p_}")
    ax.axvline(m - i0, color="red", ls="-", lw=1, label=f"BOS(풀바디 돌파) idx={m}")
    ax.axvline(j - i0, color="purple", ls="-", lw=1.5, label=f"진입봉(되돌림 터치) idx={j}")
    ax.set_title(f"[{name}] OB {d} zone={ev['zone']} fvgreq={ev['fvgreq']}  "
                 f"존=[{ev['z_lo']:.2f}, {ev['z_hi']:.2f}]", fontsize=11)
    xt = range(0, len(sub), max(1, len(sub) // 8))
    ax.set_xticks(list(xt)); ax.set_xticklabels([sub.index[t].strftime("%y-%m-%d") for t in xt], fontsize=8)
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.2)
    pth = os.path.join(OUTDIR, f"ob_{name}_{tag}.png")
    plt.tight_layout(); plt.savefig(pth, dpi=110); plt.close()
    return pth


def pick(events, **crit):
    cand = [e for e in events if all(e[k] == v for k, v in crit.items())]
    # 중간 위치, 형성→진입 간격이 적당한 것
    cand = [e for e in cand if 200 < e["entry"] < len(_GLOBAL_DF) - 50]
    return cand


def trace_fvg(df, e):
    a, b, j = e["form"] - 2, e["form"], e["entry"]
    h, l = df["high"].values, df["low"].values
    print(f"  형성봉 idx={b} ({df.index[b].date()}): 1번캔들 high={h[a]:.2f}, 3번캔들 low={l[b]:.2f}"
          f"  → {'high<low ✓ 상승갭' if h[a]<l[b] else '갭X'}")
    print(f"  갭구간=[{e['g_lo']:.2f}, {e['g_hi']:.2f}], 중간={0.5*(e['g_lo']+e['g_hi']):.2f}")
    print(f"  진입봉 idx={j} ({df.index[j].date()}): low={l[j]:.2f} → '{e['fill']}' 트리거 충족, 형성+{j-b}봉")


def trace_ob(df, e):
    s, m, p, j = e["swing"], e["bos"], e["ob"], e["entry"]
    h, l, o, c = df["high"].values, df["low"].values, df["open"].values, df["close"].values
    print(f"  스윙고점 idx={s} ({df.index[s].date()}) high={h[s]:.2f}")
    print(f"  BOS idx={m} ({df.index[m].date()}): close={c[m]:.2f} > 스윙고 {h[s]:.2f}? "
          f"{'✓ 풀바디 돌파' if c[m]>h[s] else 'X'}  (직전봉 close={c[m-1]:.2f} ≤ {h[s]:.2f})")
    print(f"  OB봉 idx={p} ({df.index[p].date()}): open={o[p]:.2f} close={c[p]:.2f} "
          f"→ {'음봉 ✓' if c[p]<o[p] else 'X'}  존=[{e['z_lo']:.2f},{e['z_hi']:.2f}]")
    print(f"  진입봉 idx={j} ({df.index[j].date()}): low={l[j]:.2f} ≤ 존상단 {e['z_hi']:.2f}? "
          f"{'✓ 되돌림 터치' if l[j]<=e['z_hi'] else 'X'}  BOS+{j-m}봉")


_GLOBAL_DF = None


def main():
    global _GLOBAL_DF
    print("=" * 64)
    print("1) 전수 감사 (정의·look-ahead 위반 건수)")
    print("=" * 64)
    store = {}
    for name in ["kospi200", "sp500", "btc"]:
        df, is_sh, is_sl, fvg_ev, ob_ev = load(name)
        store[name] = (df, fvg_ev, ob_ev)
        cf, bf = audit_fvg(df, fvg_ev)
        co, bo = audit_ob(df, ob_ev)
        print(f"[{name:9s}] FVG 이벤트 {cf:5d}건 위반 {bf} | OB 이벤트 {co:5d}건 위반 {bo}")

    print("\n" + "=" * 64)
    print("2) 실제 예시 캔들 차트 렌더")
    print("=" * 64)
    saved = []

    # FVG 예시: BTC, x2/touch/Long
    df, fvg_ev, ob_ev = store["btc"]
    _GLOBAL_DF = df
    cand = pick(fvg_ev, size="x2", fill="touch", dir="L")
    if cand:
        e = sorted(cand, key=lambda x: x["g_hi"] - x["g_lo"], reverse=True)[0]  # 갭 큰 것
        print("\n[FVG 예시 — BTC, 상승 x2/touch]")
        trace_fvg(df, e)
        saved.append(chart_fvg(df, e, "btc", "x2_touch_L"))

    # OB 예시: S&P500, full/any/Long
    df, fvg_ev, ob_ev = store["sp500"]
    _GLOBAL_DF = df
    cand = [e for e in ob_ev if e["zone"] == "full" and e["fvgreq"] == "any"
            and e["dir"] == "L" and 200 < e["entry"] < len(df) - 50]
    if cand:
        e = sorted(cand, key=lambda x: x["z_hi"] - x["z_lo"], reverse=True)[0]
        print("\n[OB 예시 — S&P500, 상승 full/any]")
        trace_ob(df, e)
        saved.append(chart_ob(df, e, "sp500", "full_any_L"))

    print("\n저장된 차트:")
    for p in saved:
        print("  ", p)


if __name__ == "__main__":
    main()
