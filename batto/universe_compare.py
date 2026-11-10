# -*- coding: utf-8 -*-
"""유니버스 비교 — S&P500 / 나스닥100 / 러셀(S&P600 소형) / 통합. 폭이 넓을수록 선별 알파 오르나.
선별=직전252일 변동성+모멘텀 z 상위8. 상대알파=중앙(선별)-중앙(전체). 종가만(z·수익 다 종가).
배치 다운로드 캐시(_univ_close.pkl). 워크포워드 2017~2026. ★생존편향(현 구성종목·상폐없음) 여전히 존재."""
import os, sys, time
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
MIN_HIST = 504; TOPN = 8; YEARS = list(range(2017, 2027))
CACHE = os.path.join(HERE, "_univ_close.pkl")


def get_lists():
    sp500 = open(os.path.join(HERE, "_sp500.txt")).read().split(",")
    nq100 = open(os.path.join(HERE, "_nq100.txt")).read().split(",")
    sp600 = open(os.path.join(HERE, "_sp600.txt")).read().split(",")
    return {"S&P500": sp500, "나스닥100": nq100, "러셀(S&P600소형)": sp600}


def download_all(univ):
    if os.path.exists(CACHE):
        px = pd.read_pickle(CACHE)
        print(f"캐시 로드: {px.shape[1]}종목")
        return px
    import yfinance as yf
    allt = sorted(set(sum(univ.values(), [])))
    print(f"배치 다운로드 {len(allt)}종목...")
    frames = []
    for i in range(0, len(allt), 150):
        chunk = allt[i:i+150]
        try:
            d = yf.download(chunk, start="2014-01-01", progress=False, auto_adjust=True)["Close"]
            frames.append(d); print(f"  {i+len(chunk)}/{len(allt)}")
        except Exception as e:
            print(f"  청크 실패 {e}")
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()].dropna(how="all")
    px.to_pickle(CACHE); print(f"저장: {px.shape}")
    return px


def sel_alpha(px, names):
    """유니버스별 워크포워드 상대알파(중앙) + 평균 유효종목수."""
    names = [n for n in names if n in px.columns]
    alphas = []; nsel = []
    for yr in YEARS:
        asof = pd.Timestamp(f"{yr}-01-01")
        prev = px[px.index < asof]; cur = px[(px.index >= asof) & (px.index <= pd.Timestamp(f"{yr}-12-31"))]
        scores = {}
        for s in names:
            p = prev[s].dropna()
            if len(p) < MIN_HIST: continue
            ret = p.pct_change().dropna()
            vol = ret.tail(252).std() * np.sqrt(252)
            mom = p.iloc[-1] / p.iloc[-252] - 1
            if mom == mom and vol == vol: scores[s] = (vol, mom)
        if len(scores) < TOPN: continue
        vols = np.array([scores[s][0] for s in scores]); moms = np.array([scores[s][1] for s in scores])
        zv = (vols - vols.mean()) / (vols.std() or 1); zm = (moms - moms.mean()) / (moms.std() or 1)
        z = {s: zv[i] + zm[i] for i, s in enumerate(scores)}
        top = sorted(z, key=lambda s: -z[s])[:TOPN]
        def fwd(s):
            a = cur[s].dropna()
            return (a.iloc[-1] / a.iloc[0] - 1) * 100 if len(a) > 30 else np.nan
        sel = [fwd(s) for s in top]; sel = [x for x in sel if x == x]
        allf = [fwd(s) for s in scores]; allf = [x for x in allf if x == x]
        if sel and allf:
            alphas.append((yr, np.median(sel) - np.median(allf), np.median(sel), np.median(allf)))
            nsel.append(len(scores))
    return alphas, (np.mean(nsel) if nsel else 0)


def main():
    univ = get_lists()
    px = download_all(univ)
    univ["통합(3지수)"] = sorted(set(univ["S&P500"] + univ["나스닥100"] + univ["러셀(S&P600소형)"]))
    L = ["유니버스 비교 — 지수별 스크리너 선별 상대알파 (워크포워드 2017~2026·종가)",
         "폭 넓을수록 선별력↑ 가설 검증. ★현 구성종목이라 생존편향 여전(절대수익 낙관).\n",
         f"  {'유니버스':18s}{'유효종목수':>9s}{'상대알파중앙':>11s}{'OOS(24~26)':>16s}"]
    for nm in ["나스닥100", "러셀(S&P600소형)", "S&P500", "통합(3지수)"]:
        al, navg = sel_alpha(px, univ[nm])
        med = np.median([a[1] for a in al]) if al else np.nan
        d = dict((a[0], a[1]) for a in al)
        oos = " ".join(f"{d.get(y,0):+.0f}" for y in (2024, 2025, 2026))
        L.append(f"  {nm:18s}{navg:>9.0f}{med:>+10.0f}%p{oos:>16s}")
    L.append("\n  ★폭↑(종목수↑)수록 상대알파↑면 '유니버스 확대=선별력향상' 확증(편향아닌 breadth).")
    L.append("  소형(러셀)만은 모멘텀 지속 약하고 생존편향 최악(상폐多)일 수 있음 — 절대 맹신 금지.")
    L.append("  ※ 실행은 여전히 TOP8(소액)·풀만 넓힘=관리부담 안늘고 선별품질↑. 생존편향·수수료 미반영.")
    txt = "\n".join(L); print(txt)
    open(os.path.join(HERE, "UNIVERSE_COMPARE_LOG.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
