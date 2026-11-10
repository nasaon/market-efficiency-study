# -*- coding: utf-8 -*-
"""터틀 진입 필터 2부 — 기타 추세지표(ADX/DMI·수퍼트렌드·Aroon·Vortex·돈치안중심선·CCI·PSAR).
1부(RSI/Stoch/MACD/일목)에 이어, 다른 추세확인 지표를 진입 AND필터로 개별 검증. 청산=채택안 고정.
평가 동일: 바스켓9·다국면 중앙값Calmar 평균(Train) / Test / 종목별 승률(vs 무필터)."""
import os, sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import turtle_exit_evolve as EV
import turtle_exit_robust as RB
import turtle_entry_filters as F1


def wild(x, n): return pd.Series(x).ewm(alpha=1/n, adjust=False).mean().values
def rmax(x, n): return pd.Series(x).rolling(n).max().values
def rmin(x, n): return pd.Series(x).rolling(n).min().values


def true_range(h, l, c):
    pc = np.roll(c, 1); pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def adx_bull(h, l, c, n=14, adx_th=20):
    up = np.concatenate([[0], h[1:] - h[:-1]]); dn = np.concatenate([[0], l[:-1] - l[1:]])
    pDM = np.where((up > dn) & (up > 0), up, 0.0); mDM = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = wild(true_range(h, l, c), n)
    pDI = 100 * wild(pDM, n) / np.where(atr == 0, np.nan, atr)
    mDI = 100 * wild(mDM, n) / np.where(atr == 0, np.nan, atr)
    dx = 100 * np.abs(pDI - mDI) / np.where(pDI + mDI == 0, np.nan, pDI + mDI)
    adx = wild(dx, n)
    return (pDI > mDI) & (adx > adx_th)


def supertrend_bull(h, l, c, n=10, mult=3.0):
    atr = wild(true_range(h, l, c), n); hl2 = (h + l) / 2
    up = hl2 + mult * atr; dn = hl2 - mult * atr
    N = len(c); fu = up.copy(); fl = dn.copy(); trend = np.ones(N)
    for i in range(1, N):
        fu[i] = min(up[i], fu[i-1]) if c[i-1] <= fu[i-1] else up[i]
        fl[i] = max(dn[i], fl[i-1]) if c[i-1] >= fl[i-1] else dn[i]
        if c[i] > fu[i-1]: trend[i] = 1
        elif c[i] < fl[i-1]: trend[i] = -1
        else: trend[i] = trend[i-1]
    return trend > 0


def aroon_bull(h, l, n=25):
    sh = pd.Series(h).rolling(n).apply(lambda x: n - 1 - np.argmax(x), raw=True).values
    sl = pd.Series(l).rolling(n).apply(lambda x: n - 1 - np.argmin(x), raw=True).values
    aup = 100 * (n - sh) / n; adn = 100 * (n - sl) / n
    return aup > adn


def vortex_bull(h, l, c, n=14):
    pc = np.roll(c, 1); pc[0] = c[0]; pl = np.roll(l, 1); pl[0] = l[0]; ph = np.roll(h, 1); ph[0] = h[0]
    vmp = np.abs(h - pl); vmm = np.abs(l - ph); tr = true_range(h, l, c)
    vip = pd.Series(vmp).rolling(n).sum().values / np.where(pd.Series(tr).rolling(n).sum().values == 0, np.nan, pd.Series(tr).rolling(n).sum().values)
    vim = pd.Series(vmm).rolling(n).sum().values / np.where(pd.Series(tr).rolling(n).sum().values == 0, np.nan, pd.Series(tr).rolling(n).sum().values)
    return vip > vim


def donchmid_bull(h, l, c, n=20):
    mid = (rmax(h, n) + rmin(l, n)) / 2
    return c > mid


def cci_bull(h, l, c, n=20):
    tp = (h + l + c) / 3; sma = pd.Series(tp).rolling(n).mean().values
    md = pd.Series(tp).rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).values
    cci = (tp - sma) / np.where(md == 0, np.nan, 0.015 * md)
    return cci > 0


def psar_bull(h, l, c, af0=0.02, afmax=0.2):
    N = len(c); sar = np.zeros(N); bull = True; af = af0; ep = h[0]; sar[0] = l[0]
    for i in range(1, N):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if bull:
            if l[i] < sar[i]:
                bull = False; sar[i] = ep; ep = l[i]; af = af0
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af0, afmax)
        else:
            if h[i] > sar[i]:
                bull = True; sar[i] = ep; ep = h[i]; af = af0
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af0, afmax)
        # 표시용 bull 상태 저장
    # 다시 계산해 bull 상태 배열 반환
    out = np.zeros(N, dtype=bool); b = True; af = af0; ep = h[0]; s = l[0]
    for i in range(1, N):
        s = s + af * (ep - s)
        if b:
            if l[i] < s: b = False; s = ep; ep = l[i]; af = af0
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af0, afmax)
        else:
            if h[i] > s: b = True; s = ep; ep = h[i]; af = af0
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af0, afmax)
        out[i] = b
    return out & (c > 0)


OTHERS = {
    "ADX/DMI": lambda D: adx_bull(D["h"], D["l"], D["c"]),
    "수퍼트렌드": lambda D: supertrend_bull(D["h"], D["l"], D["c"]),
    "Aroon": lambda D: aroon_bull(D["h"], D["l"]),
    "Vortex": lambda D: vortex_bull(D["h"], D["l"], D["c"]),
    "돈치안중심선": lambda D: donchmid_bull(D["h"], D["l"], D["c"]),
    "CCI": lambda D: cci_bull(D["h"], D["l"], D["c"]),
    "PSAR": lambda D: psar_bull(D["h"], D["l"], D["c"]),
}


def main():
    print("데이터 로드...")
    DATA = []
    for tk, kind, eN, smaN, fee in F1.BASKET:
        try:
            D = EV.load_asset(tk, kind, eN, smaN, fee); D["base"] = F1.base_entry(D); DATA.append(D)
        except Exception as e:
            print(f"  {tk} 스킵 {e}")
    base_ent = {D["tk"]: D["base"] for D in DATA}
    base_tr = F1.robust_score(DATA, base_ent, F1.TRAIN_W); base_te = F1.robust_score(DATA, base_ent, F1.WINDOWS[4:])

    L = ["터틀 진입 필터 2부 — 기타 추세지표 개별 검증 (청산=채택안 고정)",
         f"바스켓 {len(DATA)}(코인3+미국강추세6)·Train 4국면 중앙값Calmar 평균 / Test / 종목승(vs무필터)\n",
         f"  {'추가 필터':16s}{'Train':>7s}{'Test':>7s}{'종목승':>7s}{'평균거래':>8s}"]
    L.append(f"  {'무필터(기준)':16s}{base_tr:>7.2f}{base_te:>7.2f}{'-':>7s}{np.mean([F1.trades_count(D, D['base']) for D in DATA]):>8.1f}")
    results = []
    for name, fn in OTHERS.items():
        ent = {}
        for D in DATA:
            try: m = D["base"] & fn(D)
            except Exception as e: m = D["base"]; print(f"  {name}/{D['tk']} err {e}")
            ent[D["tk"]] = m
        tr = F1.robust_score(DATA, ent, F1.TRAIN_W); te = F1.robust_score(DATA, ent, F1.WINDOWS[4:])
        win, tot, _ = F1.per_asset_win(DATA, ent, base_ent)
        avgt = np.mean([F1.trades_count(D, ent[D["tk"]]) for D in DATA])
        results.append((name, tr, te, win, tot, avgt))
    results.sort(key=lambda x: -x[1])
    for name, tr, te, win, tot, avgt in results:
        mark = "  ★Train↑" if tr > base_tr + 0.02 else ("  ~동등" if abs(tr - base_tr) <= 0.02 else "")
        L.append(f"  {name:16s}{tr:>7.2f}{te:>7.2f}{win:>5d}/{tot:<2d}{avgt:>8.1f}{mark}")
    best = max(results, key=lambda x: x[1])
    L.append(f"\n  기준(무필터) Train {base_tr:.2f}·Test {base_te:.2f}")
    L.append(f"  최고: {best[0]} Train {best[1]:.2f}·Test {best[2]:.2f}·종목 {best[3]}/{best[4]}")
    L.append("  ※ Train>기준+0.02 & 종목 과반 & Test 우위라야 진짜 도움. 그 외엔 무효/역효과.")
    txt = "\n".join(L); print(txt)
    open(os.path.join(EV.HERE, "TURTLE_ENTRY_FILTERS2_LOG.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
