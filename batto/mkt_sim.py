# -*- coding: utf-8 -*-
"""다중시장 일반화 로더 — 미국/한국/코인. rb_sim의 run_fast·metrics·score 재사용.
시장별: 종목유니버스 · 추세오버레이 지수 · 공포지수(방향) · 연율화기준(ppy)."""
import os, sys
import numpy as np, pandas as pd
import rb_sim as rb

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-12

MARKETS = {
    "US":    dict(pxdir="sp500_px", idx=("gspc_full.csv", "c"), fear=("vix_full.csv", "vix"),
                  fear_dir="high", y0=2015, y1=2025, ppy=252),
    "KR":    dict(pxdir="korea_px", idx=("korea_px/_KS11.csv", "c"), fear=("vix_full.csv", "vix"),
                  fear_dir="high", y0=2016, y1=2025, ppy=252),
    "CRYPTO": dict(pxdir="crypto_px", idx=("crypto_px/px_BTC.csv", "c"), fear=("fng_crypto.csv", "fng"),
                   fear_dir="low", y0=2019, y1=2025, ppy=365),
}


def _mom(prices, pret, cal, T, nn, month, imamat=None):
    """모멘텀 시리즈 사전계산.
    specs: (win, skip, 주기, K들, suffix). skip=21→12-1. W/Q=주간/분기 리밸.
    tfw>0 → 개별종목 추세필터: top-K 중 자기 MA(tfw) 위인 것만 보유, 나머지 슬롯 현금
            (K분모 고정 — 폭망코인(LUNA류)이 MA 깨지면 다음날 자동 이탈)."""
    week = cal.to_period("W"); quarter = cal.to_period("Q")
    specs = [
        (126, 0, "M", (3, 5, 8, 10, 15), "_126", 0),
        (252, 0, "M", (3, 5, 8, 10, 15), None, 0),
        (63,  0, "M", (3, 5, 8, 10), "_63", 0),
        (252, 21, "M", (3, 5, 8), "_121", 0),
        (126, 0, "W", (3, 5, 8), "_126w", 0),
        (126, 0, "M", (3, 5, 8, 10), "_126tf50", 50),
        (126, 0, "M", (3, 5, 8, 10), "_126tf100", 100),
        (63,  0, "W", (3, 5, 8), "_63w", 0),
        (252, 0, "W", (3, 5, 8), "_252w", 0),
        (126, 0, "M", (3, 5, 8, 10), "_126tf150", 150),
        (126, 0, "M", (3, 5, 8, 10), "_126tf200", 200),
        (126, 0, "W", (3, 5, 8), "_126wtf100", 100),
        # vadj=True → 랭킹=수익률/σ(63d, 전일까지) — 리스크조정 모멘텀(재의심 4R)
        (126, 0, "M", (3, 5, 8, 10), "_126v", 0, True),
        (252, 0, "M", (3, 5, 8), "_v", 0, True),
        (126, 0, "W", (3, 5, 8), "_126wv", 0, True),
        (126, 0, "W", (3, 5, 8), "_126wvtf100", 100, True),
        (126, 0, "M", (3, 5, 8, 10), "_126vtf100", 100, True),
    ]
    mom = {}; momturn = {}
    sdm = pd.DataFrame(pret).rolling(63).std().shift(1).values   # t-1까지 σ(룩어헤드 없음)
    for spec in specs:
        win, skip, freq, Ks, suf, tfw = spec[:6]
        vadj = spec[6] if len(spec) > 6 else False
        per = month if freq == "M" else week if freq == "W" else quarter
        trailw = np.full((T, nn), np.nan)
        for t in range(win + 1, T):
            base = prices[t - win - 1]
            trailw[t] = np.where(base > 0, prices[t - 1 - skip] / base - 1, np.nan)
        if vadj:
            trailw = trailw / np.maximum(sdm, 3e-3)   # σ하한 — 초저변동 폭주 방지
        ifma = imamat.get(tfw) if (tfw and imamat) else None
        for K in Ks:
            sel = np.zeros((T, nn), bool); cur = None; pm = None; turn = np.zeros(T); prev = None
            for t in range(T):
                if t >= win + 1 and per[t] != pm:
                    tr = trailw[t]; idx = np.where(~np.isnan(tr))[0]
                    if len(idx) >= K:
                        cur = idx[np.argsort(tr[idx])[::-1][:K]]
                        if prev is not None:
                            turn[t] = len(set(cur.tolist()) - set(prev.tolist())) / K
                        prev = cur
                    pm = per[t]
                if cur is not None:
                    if ifma is None:
                        sel[t, cur] = True
                    else:   # 개별 추세필터: 전일 종가>전일 MA만 (룩어헤드 없음)
                        for c in cur:
                            v = ifma[t - 1, c]
                            sel[t, c] = (not np.isnan(v)) and prices[t - 1, c] > v
            if ifma is None:
                cnt = sel.sum(1)
                series = np.where(cnt > 0, (pret * sel).sum(1) / np.maximum(cnt, 1), 0.0)
            else:   # K분모 고정(미통과 슬롯=현금) + 일별 이탈/재진입 회전 반영
                series = (pret * sel).sum(1) / K
                dsel = np.abs(np.diff(sel.astype(float), axis=0, prepend=sel[:1] * 0)).sum(1) / K
                turn = dsel   # 리밸교체+필터 이탈/재진입 일별 전체 반영
            key = K if suf is None else f"{K}{suf}"
            mom[key] = series; momturn[key] = turn

    # ── 모멘텀 앙상블(재의심 7R): 126+252 cross-sectional rank 합으로 랭킹 ──
    # 단일 룩백 과적합 의심 → 두 룩백 백분위 순위를 더해 강건화. 룩어헤드 없음(t-1 정보).
    def _trail(win):
        tw = np.full((T, nn), np.nan)
        for t in range(win + 1, T):
            base = prices[t - win - 1]
            tw[t] = np.where(base > 0, prices[t - 1] / base - 1, np.nan)
        return tw

    def _xrank(arr):
        out = np.full((T, nn), np.nan)
        for t in range(T):
            row = arr[t]; mm = ~np.isnan(row); k = int(mm.sum())
            if k == 0:
                continue
            rr = np.empty(k); rr[row[mm].argsort()] = np.arange(k)
            out[t, mm] = rr / max(k - 1, 1)
        return out

    ens_score = _xrank(_trail(126)) + _xrank(_trail(252))   # 2룩백 합
    ens3_score = _xrank(_trail(63)) + _xrank(_trail(126)) + _xrank(_trail(252))  # 3룩백 합(8R)
    for freq, fsuf in (("M", "_ens"), ("W", "_ensw"), ("W", "_ens3w"), ("M", "_ens3")):
        per = month if freq == "M" else week
        escore = ens3_score if "ens3" in fsuf else ens_score
        for tfw in (0, 100):
            ifma = imamat.get(tfw) if (tfw and imamat) else None
            tsuf = fsuf + (f"tf{tfw}" if tfw else "")
            for K in (3, 5, 8):
                sel = np.zeros((T, nn), bool); cur = None; pm = None
                for t in range(T):
                    if t >= 253 and per[t] != pm:
                        sc_ = escore[t]; idx = np.where(~np.isnan(sc_))[0]
                        if len(idx) >= K:
                            cur = idx[np.argsort(sc_[idx])[::-1][:K]]
                        pm = per[t]
                    if cur is not None:
                        if ifma is None:
                            sel[t, cur] = True
                        else:
                            for c in cur:
                                v = ifma[t - 1, c]
                                sel[t, c] = (not np.isnan(v)) and prices[t - 1, c] > v
                if ifma is None:
                    cnt = sel.sum(1)
                    series = np.where(cnt > 0, (pret * sel).sum(1) / np.maximum(cnt, 1), 0.0)
                else:
                    series = (pret * sel).sum(1) / K
                turn = np.abs(np.diff(sel.astype(float), axis=0, prepend=sel[:1] * 0)).sum(1) / K
                mom[f"{K}{tsuf}"] = series; momturn[f"{K}{tsuf}"] = turn
    return mom, momturn


DEAD = {   # 생존편향 보정용 주입 종목(상폐·폭망)
    "CRYPTO": ["LUNA1", "UST", "FTT", "CEL", "LUNC", "WAVES", "HNT", "GALA", "RUNE"],
    "US": ["XRX", "M", "NWL", "COTY", "WU", "VFC", "ZION", "DXC", "MHK", "AAL", "CCL"],
    "KR": ["215600", "042660", "068760", "003490"],
}


def load_market(name, exclude=None):
    m = MARKETS[name]
    exclude = set(exclude or [])
    pxdir = os.path.join(HERE, m["pxdir"])
    cols = {}
    for fn in sorted(os.listdir(pxdir)):
        if not (fn.startswith("px_") and fn.endswith(".csv")):
            continue
        if fn[3:-4] in exclude:
            continue
        try:
            df = pd.read_csv(os.path.join(pxdir, fn), parse_dates=["date"], index_col="date")
        except Exception:
            continue
        if len(df) < 300:
            continue
        df = df[(df.index.year >= m["y0"] - 1) & (df.index.year <= m["y1"])]   # -1=워밍업
        if len(df) < 250:
            continue
        cols[fn[3:-4]] = df["c"]
    px = pd.DataFrame(cols).sort_index().ffill()
    # 분석기간 시작 워밍업 확보 위해 전체 유지, 평가시 연도필터는 run에서 yrs로
    cal = px.index; pret = px.pct_change().fillna(0.0).values
    yrs = cal.year.values; T = len(cal); nn = px.shape[1]
    # 지수(추세 오버레이)
    ipath, icol = m["idx"]
    g = pd.read_csv(os.path.join(HERE, ipath), parse_dates=["date"]).set_index("date")[icol]
    g = g.reindex(cal).ffill(); gspc = g.values
    # 공포지수
    fpath, fcol = m["fear"]
    fs = pd.read_csv(os.path.join(HERE, fpath), parse_dates=["date"]).set_index("date")[fcol]
    fd = {d.normalize(): float(v) for d, v in fs.items()}
    vixarr = np.array([fd.get(d.normalize(), np.nan) for d in cal])
    # 연말/지표 행렬
    lastday = np.zeros(T, bool)
    for y in np.unique(yrs):
        idx = np.where(yrs == y)[0]; lastday[idx[-1]] = True
    cl = px
    dl = cl.diff()
    Rmats = {}
    for p_ in (7, 10, 14, 21):
        upp = dl.clip(lower=0).ewm(alpha=1/p_, adjust=False).mean()
        dnp = (-dl.clip(upper=0)).ewm(alpha=1/p_, adjust=False).mean()
        Rmats[p_] = (100 - 100/(1 + upp/(dnp + EPS))).values
    Rmat = Rmats[14]
    closemat = cl.values
    mamat = {w: cl.rolling(w).mean().values for w in (50, 100, 150, 200)}
    gser = pd.Series(gspc); spx_ma = {}; mktok = {}
    for w in (50, 100, 110, 120, 130, 140, 150, 160, 175, 190, 200, 210, 220, 235, 250):
        gm = gser.rolling(w).mean().values; spx_ma[w] = gm
        ok = np.zeros(T, bool); ok[1:] = (~np.isnan(gm[:-1])) & (gspc[:-1] > gm[:-1]); mktok[w] = ok
    # 레짐 대안(재의심): 듀얼모멘텀 — 전일 기준 지수 N일 수익>0이면 ON (MA 불필요)
    for w, key in ((126, "dm126"), (252, "dm252")):
        ok = np.zeros(T, bool)
        ok[w + 1:] = gspc[w:-1] / np.maximum(gspc[: T - w - 1], EPS) > 1.0
        mktok[key] = ok; spx_ma[key] = np.full(T, np.nan)
    gret = gser.pct_change().fillna(0.0).values
    s50 = gser.rolling(50).mean().values; s200 = gser.rolling(200).mean().values
    deep = np.zeros(T, bool)
    deep[1:] = (~np.isnan(s200[:-1])) & (gspc[:-1] < s200[:-1]) & (gspc[:-1] < s50[:-1])
    ewall = px.pct_change().mean(axis=1).fillna(0.0).values
    month = cal.to_period("M")
    mom, momturn = _mom(px.values, pret, cal, T, nn, month, imamat=mamat)  # tf용 개별MA=mamat 재사용
    # 자체 공포지수 대안(KR용: 미국VIX 차용 의심) — 지수 실현변동성/고점대비낙폭
    rv20 = pd.Series(gret).rolling(20).std().values * np.sqrt(m["ppy"]) * 100
    ddown = (1 - gspc / np.maximum.accumulate(np.where(np.isnan(gspc), -np.inf, gspc))) * 100
    return dict(px=px, cal=cal, pret=pret, yrs=yrs, lastday=lastday, vixarr=vixarr,
                fear_dir=m["fear_dir"], ppy=m["ppy"], y0=m["y0"], y1=m["y1"],
                T=T, n=nn, cols=list(px.columns), gspc=gspc, ewall_ret=ewall, m7_ret=ewall,
                mom=mom, momturn=momturn, Rmat=Rmat, Rmats=Rmats, closemat=closemat,
                mamat=mamat, mktok=mktok, spx_ma=spx_ma, gret=gret, deep=deep,
                fear_alt=dict(rv20=rv20, ddown=ddown))


def run_eval(cfg, D):
    """평가기간(y0~y1)만 잘라 metrics+score. ppy 반영."""
    p, i, y = rb.run_fast(cfg, D)
    mask = (y >= D["y0"]) & (y <= D["y1"])
    ppy = D["ppy"]
    m = rb.metrics(p[mask], i[mask], y[mask], ppy=ppy); s = rb.score(m)
    return p[mask], i[mask], y[mask], m, s


if __name__ == "__main__":
    for name in ["US", "KR", "CRYPTO"]:
        D = load_market(name)
        print(f"{name}: 종목 {D['n']} · 거래일 {D['T']} ({D['cal'][0].date()}~{D['cal'][-1].date()}) "
              f"· 공포 {D['fear_dir']} · ppy {D['ppy']}")
