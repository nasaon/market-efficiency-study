# -*- coding: utf-8 -*-
"""
1단계 신호 생성 — 위키(쉽알남 1·2장) 충실 정의. (2026-06-01 전면 재정의)

FVG (3캔들):
  - 상승: high[1] < low[3], 2번캔들=강한 양봉, 2번 몸통 >= 2x max(1번,3번 몸통). 갭=[high1, low3].
  - 하락: low[1] > high[3], 2번캔들=강한 음봉, 동일 크기조건.     갭=[high3, low1].
  - 진입 = 갭 구간 '터치'(되돌림). 유효창 K봉.

OB (2캔들 엥걸프, 존=몸통):
  - 상승: [음봉 c1, 양봉 c2]
      A 몸통엥걸프: c2.open <= c1.close  AND  c2.close >= c1.open
      C 고가돌파  : c2.open <= c1.close  AND  c2.close >  c1.high
      OB존 = 음봉 몸통 [c1.close, c1.open]
  - 하락: [양봉 c1, 음봉 c2] (거울)
      A: c2.open >= c1.close AND c2.close <= c1.open
      C: c2.open >= c1.close AND c2.close <  c1.low
      OB존 = 양봉 몸통 [c1.open, c1.close]
  - 진입 = 가격이 OB 몸통존 '터치'(상승=하락 되돌림 low<=z_hi / 하락=상승 되돌림 high>=z_lo). 유효창 K봉.
  - 컨플루언스 플래그(부스터, H2용): has_sweep(직전 저/고점 스윕후 반전) · has_fvg(직후 FVG) · gap2x(c2몸통>=2x c1몸통).

폐기: BOS, 채움 50/100, FVG none, 3캔들 OB.
look-ahead: OB는 c2 마감 시 확정, FVG는 3번캔들 마감 시 확정. 진입봉 t는 그 이후. fwd return만 미래.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "panels")
os.makedirs(OUT, exist_ok=True)

SWING_K = 5
WINDOW_K = 20
ATR_PERIOD = 14
N_LIST = [5, 10, 20]
FVG_MIN_MULT = 2.0          # 2번 캔들 몸통 >= 2배 (위키 "최소 2~3배")
ENGULF_RULES = ["A", "C"]   # A=몸통엥걸프, C=고가돌파


def add_basics(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    df["body"] = (c - o).abs()
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    df["ret1"] = c.pct_change()
    df["vol20"] = df["ret1"].rolling(20, min_periods=20).std()
    df["logvol"] = np.log(df["volume"].replace(0, np.nan))
    for N in N_LIST:
        df[f"fwd_{N}"] = c.shift(-N) / c - 1.0
    return df


def swing_points(df, k=SWING_K):
    h, l = df["high"].values, df["low"].values
    n = len(df)
    is_sh = np.zeros(n, dtype=bool); is_sl = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if h[i] == h[i - k:i + k + 1].max() and h[i - k:i + k + 1].argmax() == k:
            is_sh[i] = True
        if l[i] == l[i - k:i + k + 1].min() and l[i - k:i + k + 1].argmin() == k:
            is_sl[i] = True
    return is_sh, is_sl


# ── FVG (위키) ──────────────────────────────────────────
def fvg_signals(df):
    o, h, l, c = (df["open"].values, df["high"].values, df["low"].values, df["close"].values)
    body = df["body"].values
    n = len(df)
    out = {d: set() for d in ("L", "S")}
    events = []
    for mid in range(1, n - 1):
        a, b = mid - 1, mid + 1            # 1번, 3번 캔들. 형성봉 = b
        big = body[mid] >= FVG_MIN_MULT * max(body[a], body[b])
        if not big:
            continue
        bull = (h[a] < l[b]) and (c[mid] > o[mid])   # 강한 양봉
        bear = (l[a] > h[b]) and (c[mid] < o[mid])   # 강한 음봉
        if not (bull or bear):
            continue
        if bull:
            g_lo, g_hi, d = h[a], l[b], "L"
        else:
            g_lo, g_hi, d = h[b], l[a], "S"
        form = b
        j_hit = None
        for j in range(form + 1, min(form + WINDOW_K, n - 1) + 1):
            if d == "L" and l[j] <= g_hi:      # 위에서 갭으로 하락 진입
                j_hit = j; break
            if d == "S" and h[j] >= g_lo:      # 아래서 갭으로 상승 진입
                j_hit = j; break
        if j_hit is not None:
            out[d].add(j_hit)
            events.append({"dir": d, "form": form, "entry": j_hit, "g_lo": g_lo, "g_hi": g_hi})
    return out, events


# ── OB (위키 2캔들 엥걸프) ──────────────────────────────
def _prior_swing(flags, arr, idx):
    for s in range(idx - 1, -1, -1):
        if flags[s] and (s + SWING_K) <= idx:
            return s, arr[s]
    return None, None


def ob_signals(df, is_sh, is_sl):
    o, h, l, c = (df["open"].values, df["high"].values, df["low"].values, df["close"].values)
    body = df["body"].values
    n = len(df)
    out = {(r, d): set() for r in ENGULF_RULES for d in ("L", "S")}
    events = []

    for p in range(1, n - 1):
        c1, c2 = p, p + 1                     # c1=OB캔들, c2=엥걸프캔들. 형성=c2 마감
        # ── 상승 OB: 음봉 c1 → 양봉 c2 ──
        if c[c1] < o[c1] and c[c2] > o[c2]:
            ruleA = (o[c2] <= c[c1]) and (c[c2] >= o[c1])
            ruleC = (o[c2] <= c[c1]) and (c[c2] > h[c1])
            if ruleA or ruleC:
                z_lo, z_hi = c[c1], o[c1]      # 음봉 몸통
                j_hit = None
                for j in range(c2 + 1, min(c2 + WINDOW_K, n - 1) + 1):
                    if l[j] <= z_hi:
                        j_hit = j; break
                if j_hit is not None:
                    _, psl = _prior_swing(is_sl, l, c1)
                    sweep = (psl is not None) and (min(l[c1], l[c2]) < psl) and (c[c2] > psl)
                    has_fvg = (c2 + 1 < n) and (h[c1] < l[c2 + 1])
                    gap2x = body[c2] >= 2.0 * body[c1] if body[c1] > 0 else False
                    for r in (["A"] if ruleA else []) + (["C"] if ruleC else []):
                        out[(r, "L")].add(j_hit)
                        events.append({"rule": r, "dir": "L", "c1": c1, "c2": c2, "entry": j_hit,
                                       "z_lo": z_lo, "z_hi": z_hi,
                                       "sweep": bool(sweep), "fvg": bool(has_fvg), "gap2x": bool(gap2x)})
        # ── 하락 OB: 양봉 c1 → 음봉 c2 ──
        if c[c1] > o[c1] and c[c2] < o[c2]:
            ruleA = (o[c2] >= c[c1]) and (c[c2] <= o[c1])
            ruleC = (o[c2] >= c[c1]) and (c[c2] < l[c1])
            if ruleA or ruleC:
                z_lo, z_hi = o[c1], c[c1]      # 양봉 몸통 [open, close]
                j_hit = None
                for j in range(c2 + 1, min(c2 + WINDOW_K, n - 1) + 1):
                    if h[j] >= z_lo:
                        j_hit = j; break
                if j_hit is not None:
                    _, psh = _prior_swing(is_sh, h, c1)
                    sweep = (psh is not None) and (max(h[c1], h[c2]) > psh) and (c[c2] < psh)
                    has_fvg = (c2 + 1 < n) and (l[c1] > h[c2 + 1])
                    gap2x = body[c2] >= 2.0 * body[c1] if body[c1] > 0 else False
                    for r in (["A"] if ruleA else []) + (["C"] if ruleC else []):
                        out[(r, "S")].add(j_hit)
                        events.append({"rule": r, "dir": "S", "c1": c1, "c2": c2, "entry": j_hit,
                                       "z_lo": z_lo, "z_hi": z_hi,
                                       "sweep": bool(sweep), "fvg": bool(has_fvg), "gap2x": bool(gap2x)})
    return out, events


def build_panel(name):
    df = pd.read_csv(os.path.join(DATA, f"{name}.csv"), parse_dates=["date"], index_col="date")
    df = add_basics(df)
    is_sh, is_sl = swing_points(df)
    fvg, _ = fvg_signals(df)
    for d, idxs in fvg.items():
        v = np.zeros(len(df), dtype=int)
        for j in idxs: v[j] = 1
        df[f"FVG_{d}"] = v
    ob, _ = ob_signals(df, is_sh, is_sl)
    for (r, d), idxs in ob.items():
        v = np.zeros(len(df), dtype=int)
        for j in idxs: v[j] = 1
        df[f"OB_{r}_{d}"] = v
    df.to_csv(os.path.join(OUT, f"{name}_panel.csv"), encoding="utf-8-sig")
    return df


def main():
    rows = []
    for name in ["kospi200", "sp500", "btc"]:
        df = build_panel(name)
        sig = [col for col in df.columns if col.startswith(("FVG_", "OB_"))]
        for col in sig:
            rows.append({"market": name, "signal": col, "count": int(df[col].sum())})
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(OUT, "_signal_counts.csv"), index=False, encoding="utf-8-sig")
    piv = rep.pivot(index="signal", columns="market", values="count").fillna(0).astype(int)
    print("=== 신호 개수 (시장 x 명세) - 위키 충실 정의 ===")
    print(piv[["kospi200", "sp500", "btc"]].to_string())


if __name__ == "__main__":
    main()
