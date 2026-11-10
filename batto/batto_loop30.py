# -*- coding: utf-8 -*-
"""BATTO 30+ 라운드 전략 루프 — 다약세장(2018Q4·2020 COVID) 포함 2018~2026 검증.

년도 윈도우: 19~26 (전년6/1~당해5/31, 각 5월말 보유). 200봉 이후만 진입.
각 라운드 = 다른 전략. 측정: 타점수 / 지수(^GSPC)초과 / baseline알파(같은종목 아무날) / 년도일관성.
핵심: 다약세장 넣어도 baseline 알파가 유지되는 전략이 '진짜'. (강세장 한정 결과 걸러내기)
생존편향 주의: 유니버스=2026 시총상위 70종목 → 2018시점엔 미래우량주 정보 포함. 정직히 명시.
"""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import golden_strategy as G
import batto_vix_gc_grid as BG

HERE = G.HERE
MIN_GAP = 5
YEARS = [(f"{y%100:02d}", f"{y-1}-06-01", f"{y}-05-31") for y in range(2019, 2027)]
BEAR_YEARS = {"19", "20", "22", "23"}   # 약세장 국면 포함 년도(2018Q4/COVID/2022)


def sma(c, n): return pd.Series(c).rolling(n).mean().values
def ema(c, n): return pd.Series(c).ewm(span=n, adjust=False).mean().values


def prep(sym, vixser, gser):
    df = G.load_price(sym); idx = df.index
    o = df["o"].values.astype(float); h = df["h"].values.astype(float)
    l = df["l"].values.astype(float); c = df["c"].values.astype(float)
    s = pd.Series(c)
    std20 = s.rolling(20).std().values; m20 = sma(c, 20)
    macd = ema(c, 12) - ema(c, 26); macds = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    def mom(n):
        r = np.full(len(c), np.nan)
        r[n:] = c[n:]/c[:-n] - 1
        return r
    return dict(idx=idx, o=o, h=h, l=l, c=c, n=len(c),
                rsi14=BG.rsi_wilder(c, 14), rsi5=BG.rsi_wilder(c, 5),
                rsi3=BG.rsi_wilder(c, 3), rsi2=BG.rsi_wilder(c, 2),
                s20=m20, s50=sma(c, 50), s100=sma(c, 100), s200=sma(c, 200),
                bbl2=m20-2*std20, bbl25=m20-2.5*std20,
                macd=macd, macds=macds,
                mom20=mom(20), mom60=mom(60), mom120=mom(120),
                hi250=s.rolling(250).max().values,
                vix=vixser.reindex(idx, method="ffill").values,
                gspc=gser.reindex(idx, method="ffill").values)


def ybaseline(d):
    bl = {}
    for ylab, ys, ye in YEARS:
        idx = d["idx"]; c = d["c"]
        me = idx <= pd.Timestamp(ye)
        if not me.any():
            bl[ylab] = np.nan; continue
        ei = np.where(me)[0][-1]; endc = c[ei]
        if np.isnan(endc):
            bl[ylab] = np.nan; continue
        win = np.where((idx >= pd.Timestamp(ys)) & (idx <= pd.Timestamp(ye)))[0]
        rr = [(endc/c[i]-1)*100 for i in win if i >= 200 and c[i] > 0]
        bl[ylab] = np.mean(rr) if rr else np.nan
    return bl


# ── 전략 술어 헬퍼 ──
def bear(d, i, v, r): return d["vix"][i] >= v and d["rsi14"][i] <= r
def conn(d, i, r2):   return d["c"][i] > d["s200"][i] and d["rsi2"][i] <= r2
def qs(d, i):         return d["c"][i] > d["s200"][i] and d["c"][i] < d["s20"][i] and d["rsi5"][i] <= 45
def bbdn(d, i):       return d["c"][i] > d["s200"][i] and d["c"][i] < d["bbl2"][i]

# ── 30+ 전략 (라운드) ──
STRATS = {
 "R01 VIX20&RSI36":            lambda d, i: bear(d, i, 20, 36),
 "R02 VIX25&RSI36":            lambda d, i: bear(d, i, 25, 36),
 "R03 VIX28&RSI36":            lambda d, i: bear(d, i, 28, 36),
 "R04 VIX30&RSI40":            lambda d, i: bear(d, i, 30, 40),
 "R05 VIX25&RSI30":            lambda d, i: bear(d, i, 25, 30),
 "R06 >200&RSI2<=3":           lambda d, i: conn(d, i, 3),
 "R07 >200&RSI2<=5":           lambda d, i: conn(d, i, 5),
 "R08 >200&RSI2<=10":          lambda d, i: conn(d, i, 10),
 "R09 >200&RSI2<=15":          lambda d, i: conn(d, i, 15),
 "R10 QS(>200&<20&RSI5<=45)":  lambda d, i: qs(d, i),
 "R11 >200&RSI14<=40":         lambda d, i: d["c"][i] > d["s200"][i] and d["rsi14"][i] <= 40,
 "R12 >50&RSI2<=5":            lambda d, i: d["c"][i] > d["s50"][i] and d["rsi2"][i] <= 5,
 "R13 >200&>50&RSI2<=10":      lambda d, i: d["c"][i] > d["s200"][i] and d["c"][i] > d["s50"][i] and d["rsi2"][i] <= 10,
 "R14 s50>s200&RSI2<=10":      lambda d, i: d["s50"][i] > d["s200"][i] and d["rsi2"][i] <= 10,
 "R15 >200&close<BB2":         lambda d, i: bbdn(d, i),
 "R16 >200&low<BB2&close>BB2": lambda d, i: d["c"][i] > d["s200"][i] and d["l"][i] < d["bbl2"][i] and d["c"][i] > d["bbl2"][i],
 "R17 close<BB2.5":            lambda d, i: d["c"][i] < d["bbl25"][i],
 "R18 >200&mom20>0&RSI2<=10":  lambda d, i: d["c"][i] > d["s200"][i] and d["mom20"][i] > 0 and d["rsi2"][i] <= 10,
 "R19 신고가풀백 >0.95hi&RSI2<=10": lambda d, i: d["c"][i] > d["s200"][i] and d["c"][i] > 0.95*d["hi250"][i] and d["rsi2"][i] <= 10,
 "R20 mom60>10%&RSI14<=40":    lambda d, i: d["mom60"][i] > 0.10 and d["rsi14"][i] <= 40,
 "R21 >200&MACD>sig&RSI2<=10": lambda d, i: d["c"][i] > d["s200"][i] and d["macd"][i] > d["macds"][i] and d["rsi2"][i] <= 10,
 "R22 >200&RSI3<=10":          lambda d, i: d["c"][i] > d["s200"][i] and d["rsi3"][i] <= 10,
 "R23 BEAR|Connors5 [기존최적]": lambda d, i: bear(d, i, 25, 36) or conn(d, i, 5),
 "R24 BEAR|QS":                lambda d, i: bear(d, i, 25, 36) or qs(d, i),
 "R25 BEAR|BB하단":            lambda d, i: bear(d, i, 25, 36) or bbdn(d, i),
 "R26 BEAR|(>200&RSI2<=5&MACD)": lambda d, i: bear(d, i, 25, 36) or (d["c"][i] > d["s200"][i] and d["rsi2"][i] <= 5 and d["macd"][i] > d["macds"][i]),
 "R27 BEAR|Connors5|mom풀백":  lambda d, i: bear(d, i, 25, 36) or conn(d, i, 5) or (d["mom60"][i] > 0.10 and d["rsi14"][i] <= 40),
 "R28 BEAR|Connors5|BB하단":   lambda d, i: bear(d, i, 25, 36) or conn(d, i, 5) or bbdn(d, i),
 "R29 VIX30&RSI36|Connors3":   lambda d, i: bear(d, i, 30, 36) or conn(d, i, 3),
 "R30 >200&RSI2<=5&mom120>0":  lambda d, i: d["c"][i] > d["s200"][i] and d["rsi2"][i] <= 5 and d["mom120"][i] > 0,
 "R31 >100&RSI2<=5":           lambda d, i: d["c"][i] > d["s100"][i] and d["rsi2"][i] <= 5,
 "R32 BEAR(28,34)|Connors5":   lambda d, i: bear(d, i, 28, 34) or conn(d, i, 5),
 "R33 BEAR|Connors5|BB(3way)": lambda d, i: bear(d, i, 25, 36) or conn(d, i, 5) or bbdn(d, i),
 "R34 >200&RSI2<=5&close>open": lambda d, i: d["c"][i] > d["s200"][i] and d["rsi2"][i] <= 5 and d["c"][i] > d["o"][i],
 "R35 BEAR(20,40)|Connors10":  lambda d, i: bear(d, i, 20, 40) or conn(d, i, 10),
}


def run_strategy(data, bls, uni, fn):
    """년도별 (타점, 종목수익평균, 지수수익평균, 알파평균) + 전체."""
    per = {}
    tot_s, tot_g, tot_a = [], [], []
    for ylab, ys, ye in YEARS:
        ss, gg, aa = [], [], []
        s_ts = pd.Timestamp(ys); e_ts = pd.Timestamp(ye)
        for sym in uni:
            d = data[sym]; idx = d["idx"]; c = d["c"]; gspc = d["gspc"]
            me = idx <= e_ts
            if not me.any():
                continue
            ei = np.where(me)[0][-1]; endc = c[ei]; endg = gspc[ei]
            if np.isnan(endc):
                continue
            blsy = bls[sym][ylab]
            win = np.where((idx >= s_ts) & (idx <= e_ts))[0]
            last = -10**9
            for i in win:
                if i < 200 or c[i] <= 0:
                    continue
                if not fn(d, i):
                    continue
                if i - last < MIN_GAP:
                    continue
                last = i
                sret = (endc/c[i]-1)*100
                ss.append(sret)
                if gspc[i] == gspc[i] and gspc[i] > 0 and endg == endg:
                    gg.append((endg/gspc[i]-1)*100)
                if blsy == blsy:
                    aa.append(sret - blsy)
        per[ylab] = (len(ss), np.mean(ss) if ss else np.nan,
                     np.mean(gg) if gg else np.nan, np.mean(aa) if aa else np.nan)
        tot_s += ss; tot_g += gg; tot_a += aa
    tot = (len(tot_s), np.mean(tot_s) if tot_s else np.nan,
           np.mean(tot_g) if tot_g else np.nan, np.mean(tot_a) if tot_a else np.nan)
    return per, tot


def main():
    uni = json.load(open(os.path.join(HERE, "fib_universe.json"), encoding="utf-8"))["union"]
    vixser = G.load_vix()["vix"]
    gdf = G.load_price("^GSPC"); gser = pd.Series(gdf["c"].values.astype(float), index=gdf.index)
    data = {s: prep(s, vixser, gser) for s in uni}
    bls = {s: ybaseline(data[s]) for s in uni}

    L = ["BATTO 30+ 라운드 전략 루프 — 2018~2026 (다약세장 19=2018Q4, 20=COVID, 22 포함)",
         "초과=종목−지수동기 / 알파=종목−baseline(같은종목 아무날). 단위 %p. 단위년도 5월말 보유.",
         "생존편향: 유니버스=2026 시총상위 70종목(미래우량주 정보 포함). 절대수치보다 '전략 간 상대비교'로 해석.\n"]
    rows = []
    print("실행 중...", flush=True)
    for name, fn in STRATS.items():
        per, tot = run_strategy(data, bls, uni, fn)
        k, ms, mg, ma = tot
        exc = ms - mg if (ms == ms and mg == mg) else np.nan
        # 약세장 년도 알파 평균(다약세장 견고성)
        bear_alpha = [per[y][3] for y in BEAR_YEARS if y in per and per[y][3] == per[y][3]]
        bear_a = np.mean(bear_alpha) if bear_alpha else np.nan
        # 알파 양수 년도수
        ha = sum(1 for y, _, _ in YEARS if per[y][0] > 0)
        pa = sum(1 for y, _, _ in YEARS if per[y][3] == per[y][3] and per[y][3] > 0)
        rows.append((name, k, exc, ma, bear_a, pa, ha))
        print(f"  {name:<30s} 타점{k:>6d} 초과{exc:>+6.1f} 알파{ma:>+6.1f} 약세장알파{bear_a:>+6.1f} 알파년{pa}/{ha}", flush=True)

    L.append(f"  {'라운드':<30s}{'타점':>7s}{'초과%p':>7s}{'알파%p':>7s}{'약세장알파':>9s}{'알파년':>7s}")
    for r in rows:
        L.append(f"  {r[0]:<30s}{r[1]:>7d}{r[2]:>+7.1f}{r[3]:>+7.1f}{r[4]:>+9.1f}{str(r[5])+'/'+str(r[6]):>7s}")

    L.append("\n[정렬1] 약세장 알파(다약세장 견고성) 상위 10:")
    for r in sorted(rows, key=lambda x: (x[4] if x[4] == x[4] else -99), reverse=True)[:10]:
        L.append(f"  {r[0]:<30s} 약세장알파{r[4]:>+6.1f}  전체알파{r[3]:>+6.1f}  초과{r[2]:>+6.1f}  타점{r[1]}  알파년{r[5]}/{r[6]}")

    L.append("\n[정렬2] 전체 알파(baseline 대비 타이밍실력) 상위 10:")
    for r in sorted(rows, key=lambda x: (x[3] if x[3] == x[3] else -99), reverse=True)[:10]:
        L.append(f"  {r[0]:<30s} 전체알파{r[3]:>+6.1f}  약세장알파{r[4]:>+6.1f}  초과{r[2]:>+6.1f}  타점{r[1]}  알파년{r[5]}/{r[6]}")

    txt = "\n".join(L)
    print("\n" + txt)
    with open(os.path.join(HERE, "BATTO_LOOP30_LOG.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n저장: BATTO_LOOP30_LOG.md")


if __name__ == "__main__":
    main()
