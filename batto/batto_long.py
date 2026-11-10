# -*- coding: utf-8 -*-
"""BATTO 장기 재검증 — 일봉 2007~2026 (2008 금융위기 포함, 약세장 6~7회).
핵심 결론 재확인: VIX공포=진짜 엣지인가, Connors풀백=가짜인가, 우량주/낙폭과대는?

년도 윈도우: 08~26 (전년6/1~당해5/31, 5월말 보유). 200봉 이후만 진입.
측정: 타점 / 절대수익 / 지수(^GSPC)초과 / baseline알파(같은종목 아무날). 약세장견고성.
※ 유니버스=2026 시총상위 70종목 → 2007시점 생존편향 큼. 절대/초과는 부풀려짐.
   baseline알파(종목내 비교)가 그나마 강건. 신생종목(META12·TSLA10·ABBV13·CRWD19 등)은
   상장 전 년도엔 자동 제외(run이 데이터 있는 윈도우만 집계).
"""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import batto_vix_gc_grid as BG

HERE = os.path.dirname(os.path.abspath(__file__))
YEARS = [(f"{y%100:02d}", f"{y-1}-06-01", f"{y}-05-31") for y in range(2008, 2027)]
BEAR = {"08", "09", "11", "16", "18", "20", "22"}   # 약세장 포함 년도


def load_long(tk):
    fp = os.path.join(HERE, f"px_long_{tk.replace('.', '_').replace('^', 'x')}.csv")
    if not os.path.exists(fp):
        r = yf.download(tk, start="2007-01-01", auto_adjust=True, progress=False)
        if isinstance(r.columns, pd.MultiIndex):
            r.columns = r.columns.get_level_values(0)
        r = r[["Open", "High", "Low", "Close"]].dropna(); r.columns = ["o", "h", "l", "c"]
        r.index.name = "date"; r.to_csv(fp, encoding="utf-8-sig")
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")


def load_vix_long():
    fp = os.path.join(HERE, "vix_long.csv")
    if not os.path.exists(fp):
        v = yf.download("^VIX", start="2007-01-01", auto_adjust=True, progress=False)
        if isinstance(v.columns, pd.MultiIndex):
            v.columns = v.columns.get_level_values(0)
        v = v[["Close"]].dropna(); v.columns = ["vix"]; v.index.name = "date"
        v.to_csv(fp, encoding="utf-8-sig")
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")


def sma(c, n): return pd.Series(c).rolling(n).mean().values


def prep(sym, vixser, gser):
    df = load_long(sym); idx = df.index
    c = df["c"].values.astype(float); n = len(c)
    s200 = sma(c, 200)
    up504 = np.zeros(n, bool)
    if n > 504:
        up504[504:] = c[504:] > c[:-504]
    ret = np.zeros(n); ret[1:] = c[1:]/c[:-1] - 1
    return dict(idx=idx, c=c, n=n, rsi14=BG.rsi_wilder(c, 14), rsi2=BG.rsi_wilder(c, 2),
                s200=s200, up504=up504,
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
        win = np.where((idx >= pd.Timestamp(ys)) & (idx <= pd.Timestamp(ye)))[0]
        rr = [(endc/c[i]-1)*100 for i in win if i >= 200 and c[i] > 0]
        bl[ylab] = np.mean(rr) if rr else np.nan
    return bl


def run(data, bls, uni, fn):
    per = {}; ts, tg, ta = [], [], []
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
            last = -10**9
            for i in np.where((idx >= s_ts) & (idx <= e_ts))[0]:
                if i < 200 or c[i] <= 0 or not fn(d, i):
                    continue
                if i - last < 5:
                    continue
                last = i
                sret = (endc/c[i]-1)*100; ss.append(sret)
                if gspc[i] == gspc[i] and gspc[i] > 0 and endg == endg:
                    gg.append((endg/gspc[i]-1)*100)
                if blsy == blsy:
                    aa.append(sret - blsy)
        per[ylab] = (len(ss), np.mean(ss) if ss else np.nan,
                     np.mean(gg) if gg else np.nan, np.mean(aa) if aa else np.nan)
        ts += ss; tg += gg; ta += aa
    tot = (len(ts), np.mean(ts) if ts else np.nan, np.mean(tg) if tg else np.nan, np.mean(ta) if ta else np.nan)
    return per, tot


def vf(d, i, v, r): return d["vix"][i] >= v and d["rsi14"][i] <= r
def conn(d, i, r2): return d["c"][i] > d["s200"][i] and d["rsi2"][i] <= r2
def below(d, i): return d["c"][i] < d["s200"][i]
def up(d, i): return bool(d["up504"][i])

STRATS = {
 "VIX25&RSI36":              lambda d, i: vf(d, i, 25, 36),
 "VIX28&RSI36":              lambda d, i: vf(d, i, 28, 36),
 "VIX30&RSI40":              lambda d, i: vf(d, i, 30, 40),
 "VIX25&낙폭과대":            lambda d, i: vf(d, i, 25, 36) and below(d, i),
 "VIX25&낙폭과대&장기하락":    lambda d, i: vf(d, i, 25, 36) and below(d, i) and (not up(d, i)),
 "VIX25&낙폭과대&장기우상향":  lambda d, i: vf(d, i, 25, 36) and below(d, i) and up(d, i),
 "Connors RSI2<=5":          lambda d, i: conn(d, i, 5),
 "Connors RSI2<=10":         lambda d, i: conn(d, i, 10),
 "VIX25|Connors5(전천후)":    lambda d, i: vf(d, i, 25, 36) or conn(d, i, 5),
 "VIX&장기우상향(게이트)":     lambda d, i: vf(d, i, 25, 36) and up(d, i),
}


def main():
    uni = json.load(open(os.path.join(HERE, "fib_universe.json"), encoding="utf-8"))["union"]
    print("데이터 로딩(2007~, 첫 실행시 다운로드)...", flush=True)
    vixser = load_vix_long()["vix"]
    gser = pd.Series(load_long("^GSPC")["c"].values.astype(float), index=load_long("^GSPC").index)
    data = {}
    for i, s in enumerate(uni, 1):
        try:
            data[s] = prep(s, vixser, gser)
            print(f"  {i}/{len(uni)} {s} {data[s]['idx'][0].date()}~ ({data[s]['n']}봉)", flush=True)
        except Exception as e:
            print(f"  {s} ERR {e}", flush=True)
    uni = [s for s in uni if s in data]
    bls = {s: ybaseline(data[s]) for s in data}

    L = ["BATTO 장기 재검증 — 일봉 2007~2026 (08~26, 약세장 08·09·11·16·18·20·22 포함).",
         "절대=종목 5월말보유 / 초과=종목−지수동기 / 알파=종목−baseline(같은종목 아무날). 약세장알파=BEAR년 평균.",
         "※ 70종목=2026 시총상위 생존자 → 절대/초과 부풀려짐. baseline알파가 강건 지표.\n"]
    L.append(f"  {'전략':<24s}{'타점':>7s}{'절대%':>7s}{'초과%p':>7s}{'알파%p':>7s}{'약세장알파':>9s}{'알파년':>7s}")
    rows = []
    for name, fn in STRATS.items():
        per, tot = run(data, bls, uni, fn)
        k, ms, mg, ma = tot
        exc = ms - mg if (ms == ms and mg == mg) else float("nan")
        ba = [per[y][3] for y in BEAR if per[y][3] == per[y][3]]
        bear_a = np.mean(ba) if ba else float("nan")
        pa = sum(1 for y, _, _ in YEARS if per[y][3] == per[y][3] and per[y][3] > 0)
        ha = sum(1 for y, _, _ in YEARS if per[y][0] > 0)
        rows.append((name, k, ms, exc, ma, bear_a, pa, ha))
        L.append(f"  {name:<24s}{k:>7d}{ms:>+7.1f}{exc:>+7.1f}{ma:>+7.1f}{bear_a:>+9.1f}{str(pa)+'/'+str(ha):>7s}")
        print(f"  ▶ {name:<24s} 타점{k:>5d} 알파{ma:>+6.1f} 약세장{bear_a:>+6.1f} 알파년{pa}/{ha}", flush=True)

    # 금융위기 년도(09=2008.6~2009.5) 상세
    L.append("\n[2008 금융위기 구간 09년도(2008.6~2009.5) 상세 — 절대수익%(타점)]")
    for name, fn in STRATS.items():
        per, _ = run(data, bls, uni, fn)
        k, ms, mg, ma = per["09"]
        L.append(f"  {name:<24s} 절대{ms:>+7.1f}  알파{ma:>+7.1f}  타점{k}")

    txt = "\n".join(L)
    print("\n" + txt)
    with open(os.path.join(HERE, "BATTO_LONG_LOG.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n저장: BATTO_LONG_LOG.txt")


if __name__ == "__main__":
    main()
