# -*- coding: utf-8 -*-
"""터틀 진입 필터 실험 — RSI·스토캐스틱·MACD·일목균형표 16조합(2^4) + 기타 추세지표.
================================================================================
질문: 터틀 진입(30/20일 고점돌파 + SMA)에 보조 추세지표를 AND로 더하면 도움되나?
방법: 청산=채택안(저점이탈 절반익절+트레일2ATR·1.5N손절) 고정, ★진입 필터만 변경해 격리.
평가: robust — 바스켓9(코인3+미국강추세6)·다국면 중앙값 Calmar 평균 + 종목별 승률(vs 무필터).
정직성: 필터는 진입을 '더 엄격'하게 함(거래수↓). 휩쏘 줄이면 이득, 추세 놓치면 손해. 그 균형을 측정.
       돌파 시점엔 대개 RSI/MACD가 이미 불리시라 필터가 안 걸릴 수 있음(무효 가능성).
================================================================================
"""
import os, sys, itertools
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import turtle_exit_evolve as EV
import turtle_exit_robust as RB

BASKET = EV.BASKET
WINDOWS = RB.WINDOWS
TRAIN_W = WINDOWS[:4]


def emav(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def rollmax(x, n): return pd.Series(x).rolling(n).max().values
def rollmin(x, n): return pd.Series(x).rolling(n).min().values
def rollmean(x, n): return pd.Series(x).rolling(n).mean().values


def indicators(df):
    """각 지표의 '상승추세 확인' 불리언 배열 반환(진입 필터용). 미래참조 없음."""
    h, l, c = df["h"].values, df["l"].values, df["c"].values
    out = {}
    # RSI(14) > 50
    out["RSI"] = EV.S.rsi(c, 14) > 50
    # Stochastic %K(14) > 50
    hh14 = rollmax(h, 14); ll14 = rollmin(l, 14)
    k = 100 * (c - ll14) / np.where(hh14 - ll14 == 0, np.nan, hh14 - ll14)
    out["Stoch"] = k > 50
    # MACD(12,26,9): macd > signal
    macd = emav(c, 12) - emav(c, 26); sig = emav(macd, 9)
    out["MACD"] = macd > sig
    # 일목균형표: 종가 > 구름대 상단(선행스팬 A/B, 26 시프트 = 미래참조 없음)
    tenkan = (rollmax(h, 9) + rollmin(l, 9)) / 2
    kijun = (rollmax(h, 26) + rollmin(l, 26)) / 2
    spanA = pd.Series((tenkan + kijun) / 2).shift(26).values
    spanB = pd.Series((rollmax(h, 52) + rollmin(l, 52)) / 2).shift(26).values
    cloud_top = np.fmax(spanA, spanB)
    out["Ichi"] = ~np.isnan(cloud_top) & (c > cloud_top)
    return out


def base_entry(D):
    """터틀 기본 진입: eN일 고점 돌파 + SMA 위 (필터 이전)."""
    h, c = D["h"], D["c"]; eN = D["eN"]; sma = D["sma"]
    hh = pd.Series(h).rolling(eN).max().shift(1).values
    return (~np.isnan(hh)) & (c > hh) & (~np.isnan(sma)) & (c > sma)


def turtle_pos(D, entry_ok, xN=10, slN=1.5, trailN=2.0):
    """진입=entry_ok(불리언), 청산=채택안(저점이탈 절반익절+트레일2·1.5N손절). pos 0/0.5/1."""
    h, l, c, a = D["h"], D["l"], D["c"], D["a"]; n = D["n"]
    ll = pd.Series(l).rolling(xN).min().shift(1).values
    start = max(D["eN"], D["smaN"], 52, 26) + 1
    pos = 0.0; stop = peak = 0.0; half = False; arr = np.zeros(n)
    for i in range(start, n):
        if pos == 0:
            if entry_ok[i]:
                pos = 1.0; stop = c[i] - slN * a[i]; peak = h[i]; half = False
        else:
            peak = max(peak, h[i]); trail = peak - trailN * a[i]
            if not half:
                if l[i] <= stop: pos = 0.0
                elif not np.isnan(ll[i]) and c[i] < ll[i]: pos = 0.5; half = True
            else:
                if l[i] <= max(stop, trail): pos = 0.0
        arr[i] = pos
    return pd.Series(arr, index=D["idx"])


def metrics_win(D, pos, lo, hi):
    idx = D["idx"]; c = D["c"]; fee = D["fee"]
    cret = np.concatenate([[0], c[1:] / c[:-1] - 1])
    p = pos.values
    net = np.concatenate([[0], p[:-1]]) * cret - np.abs(np.diff(p, prepend=0)) * fee
    eq = pd.Series(np.cumprod(1 + np.nan_to_num(net)), index=idx)
    e = eq[(eq.index >= lo) & (eq.index < hi)]
    if len(e) < 60 or e.iloc[0] <= 0: return None
    e = e / e.iloc[0]; pk = np.maximum.accumulate(e.values)
    mdd = (1 - e.values / pk).max() * 100; yrs = len(e) / 252.0
    cagr = e.iloc[-1] ** (1 / yrs) - 1 if e.iloc[-1] > 0 else -1
    return cagr / (mdd / 100) if mdd > 0 else np.nan


def robust_score(DATA, entries, wins):
    """다국면 중앙값 Calmar의 평균(entries: {tk:entry_ok})."""
    vs = []
    for w in wins:
        lo, hi = pd.Timestamp(w[1]), pd.Timestamp(w[2])
        cs = []
        for D in DATA:
            cal = metrics_win(D, turtle_pos(D, entries[D["tk"]]), lo, hi)
            if cal == cal: cs.append(cal)
        if cs: vs.append(np.median(cs))
    return float(np.mean(vs)) if vs else np.nan


def per_asset_win(DATA, entries, base_entries):
    win = 0; tot = 0; deltas = []
    for D in DATA:
        lo, hi = D["idx"][0], D["idx"][-1] + pd.Timedelta(days=1)
        c1 = metrics_win(D, turtle_pos(D, entries[D["tk"]]), lo, hi)
        c0 = metrics_win(D, turtle_pos(D, base_entries[D["tk"]]), lo, hi)
        if c1 == c1 and c0 == c0:
            tot += 1; deltas.append((D["tk"], c1 - c0)); win += (c1 > c0)
    return win, tot, deltas


def trades_count(D, entry_ok):
    p = turtle_pos(D, entry_ok).values
    return int(np.sum((p[1:] > 0) & (p[:-1] == 0)))


def main():
    print("데이터 로드...")
    DATA = []
    for tk, kind, eN, smaN, fee in BASKET:
        try:
            D = EV.load_asset(tk, kind, eN, smaN, fee)
            D["ind"] = indicators(pd.DataFrame({"h": D["h"], "l": D["l"], "c": D["c"]}, index=D["idx"]))
            D["base"] = base_entry(D)
            DATA.append(D)
        except Exception as e:
            print(f"  {tk} 스킵 {e}")
    FILTERS = ["RSI", "Stoch", "MACD", "Ichi"]

    def make_entries(combo):
        ent = {}
        for D in DATA:
            m = D["base"].copy()
            for f in combo: m = m & D["ind"][f]
            ent[D["tk"]] = m
        return ent

    base_entries = make_entries([])
    base_train = robust_score(DATA, base_entries, TRAIN_W)
    base_test = robust_score(DATA, base_entries, WINDOWS[4:])

    L = ["터틀 진입 필터 실험 — RSI·스토캐스틱·MACD·일목 16조합 (청산=채택안 고정)",
         f"바스켓 {len(DATA)}(코인3+미국강추세6)·robust: Train 4국면 중앙값Calmar 평균 / Test 2국면 / 종목승률\n",
         f"  {'조합(추가 필터)':30s}{'Train':>7s}{'Test':>7s}{'종목승':>7s}{'평균거래수':>9s}"]
    rows = []
    for r in range(5):
        for combo in itertools.combinations(FILTERS, r):
            ent = make_entries(list(combo))
            tr = robust_score(DATA, ent, TRAIN_W); te = robust_score(DATA, ent, WINDOWS[4:])
            win, tot, _ = per_asset_win(DATA, ent, base_entries)
            avg_tr = np.mean([trades_count(D, ent[D["tk"]]) for D in DATA])
            name = "무필터(기준=현재)" if not combo else "+".join(combo)
            rows.append((name, tr, te, win, tot, avg_tr, len(combo)))
    # 무필터 먼저, 나머지는 Train 내림차순
    rows.sort(key=lambda x: (x[6] > 0, -x[1]))
    for name, tr, te, win, tot, avgt, nc in rows:
        mark = "  (기준)" if nc == 0 else ("  ★Train↑" if tr > base_train + 0.02 else "")
        L.append(f"  {name:30s}{tr:>7.2f}{te:>7.2f}{win:>5d}/{tot:<2d}{avgt:>9.1f}{mark}")
    L.append(f"\n  기준(무필터): Train {base_train:.2f} · Test {base_test:.2f}")
    best = max([r for r in rows if r[6] > 0], key=lambda x: x[1])
    L.append(f"  Train 최고 조합: {best[0]} (Train {best[1]:.2f}·Test {best[2]:.2f}·종목 {best[3]}/{best[4]})")
    L.append("  ※ 필터 켤수록 거래수↓(엄격). 종목승 5/9 안팎=동전던지기(robust 엣지 아님).")
    txt = "\n".join(L); print(txt)
    open(os.path.join(EV.HERE, "TURTLE_ENTRY_FILTERS_LOG.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
