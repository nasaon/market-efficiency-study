# -*- coding: utf-8 -*-
"""레드/블루팀 전략 연구 시뮬레이터 — 손절·추적손절·분할진입·포지션수 조절.
일별 포트폴리오, 2015~2025, 미국 502종목. 객관적 100점 루브릭 채점.

config 키:
  vix, rsi, ma            진입 임계
  slots                   완전투자 목표 포지션수(=1포지션당 1/slots NAV, 총합 1 상한)
  maxtranche, gap         분할진입 횟수 / 추가 최소간격(일)
  stop                    고정손절 %(평단대비, 0=off)
  trail                   추적손절 %(고점대비, 0=off)
  tp_rsi                  익절 RSI(0=off→연말까지)
  yearend                 연말청산(bool)
  reentry                 청산 후 재진입 허용(bool)
"""
import os, sys, json, warnings, copy
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PXDIR = os.path.join(HERE, "sp500_px")
EPS = 1e-12
EW_BH_SHARPE = 0.92   # 동일비중 매수보유 벤치마크 샤프(앞서 산출)


def rsi(c, p=14):
    s = pd.Series(c); d = s.diff()
    ru = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    rd = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
    return (100 - 100 / (1 + ru / (rd + EPS))).values


# ── 데이터 1회 로드/사전계산 ──
def load_all(exclude=None):
    exclude = set(exclude or [])
    vix = pd.read_csv(os.path.join(HERE, "vix_full.csv"), parse_dates=["date"]).set_index("date")["vix"]
    vixd = {d.normalize(): float(x) for d, x in vix.items()}
    files = sorted(fn for fn in os.listdir(PXDIR) if fn.startswith("px_") and fn.endswith(".csv"))
    cols = {}
    for fn in files:
        if fn[3:-4] in exclude:
            continue
        try:
            df = pd.read_csv(os.path.join(PXDIR, fn), parse_dates=["date"], index_col="date")
        except Exception:
            continue
        if len(df) < 260:
            continue
        df = df[(df.index.year >= 2015) & (df.index.year <= 2025)]
        if len(df) < 200:
            continue
        cols[fn[3:-4]] = df["c"]
    px = pd.DataFrame(cols).sort_index()
    cal = px.index
    pret = px.pct_change().fillna(0.0).values
    # SPX 추세 오버레이용
    g = pd.read_csv(os.path.join(HERE, "gspc_full.csv"), parse_dates=["date"]).set_index("date")["c"]
    g = g.reindex(cal).ffill()
    gspc = g.values
    # M7 동일비중 바스켓 수익(후견편향 — ETF 대용, 참고용)
    M7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    m7cols = [tk for tk in M7 if tk in px.columns]
    m7_ret = px[m7cols].pct_change().mean(axis=1).fillna(0.0).values
    yrs = cal.year.values
    T = len(cal)
    # 동일비중 전종목(무편향 코어)
    ewall_ret = pd.DataFrame(cols).sort_index().pct_change().mean(axis=1).fillna(0.0).values
    # 모멘텀 코어(선행참조 없음): 매월초 직전 252일 수익 상위K 동일비중, 다음 리밸까지 보유
    prices = px.values
    retm = pret
    nn = px.shape[1]
    month = cal.to_period("M")
    week = cal.to_period("W"); quarter = cal.to_period("Q")
    mom = {}; momturn = {}
    # (win, skip, 리밸주기, K목록, 키suffix). 252/0/M은 int 키(호환).
    # skip=21 → 12-1 모멘텀(최근 1개월 제외, 학계표준). W=주간, Q=분기 리밸.
    specs = [
        (126, 0, "M", (5, 8, 10, 15, 20, 30, 40, 50), "_126"),
        (189, 0, "M", (5, 8, 10, 15, 20, 30, 40, 50), "_189"),
        (252, 0, "M", (5, 8, 10, 15, 20, 30, 40, 50), None),
        (63,  0, "M", (5, 8, 10, 15), "_63"),
        (252, 21, "M", (5, 8, 10, 15), "_121"),
        (126, 0, "W", (5, 8, 10), "_126w"),
        (126, 0, "Q", (5, 8, 10), "_126q"),
        (63,  0, "W", (5, 8, 10), "_63w"),
        (252, 0, "W", (5, 8, 10), "_252w"),
        # vadj=True → 랭킹을 수익률/σ(63d, 전일까지)로 — 리스크조정 모멘텀(재의심 4R)
        (252, 0, "M", (5, 8, 10), "_v", True),
        (126, 0, "M", (5, 8, 10), "_126v", True),
        (252, 0, "W", (5, 8), "_252wv", True),
    ]
    sdm = pd.DataFrame(retm).rolling(63).std().shift(1).values   # sdm[t]=t-1까지 σ(룩어헤드 없음)
    for spec in specs:
        win, skip, freq, Ks, suf = spec[:5]
        vadj = spec[5] if len(spec) > 5 else False
        per = month if freq == "M" else week if freq == "W" else quarter
        trailw = np.full((T, nn), np.nan)
        for t in range(win + 1, T):
            base = prices[t - win - 1]
            trailw[t] = np.where(base > 0, prices[t - 1 - skip] / base - 1, np.nan)
        if vadj:
            trailw = trailw / np.maximum(sdm, 3e-3)   # σ하한(연5%상당) — 초저변동 폭주 방지
        for K in Ks:
            sel = np.zeros((T, nn), bool); cur = None; pm = None
            turn = np.zeros(T); prev = None
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
                    sel[t, cur] = True
            cnt = sel.sum(1)
            series = np.where(cnt > 0, (retm * sel).sum(1) / np.maximum(cnt, 1), 0.0)
            key = K if suf is None else f"{K}{suf}"
            mom[key] = series; momturn[key] = turn

    # ── 모멘텀 앙상블(재의심 7/10R): 126+252 cross-sectional rank 합 랭킹 ──
    def _trail_u(win):
        tw = np.full((T, nn), np.nan)
        for t in range(win + 1, T):
            base = prices[t - win - 1]
            tw[t] = np.where(base > 0, prices[t - 1] / base - 1, np.nan)
        return tw

    def _xrank_u(arr):
        out = np.full((T, nn), np.nan)
        for t in range(T):
            row = arr[t]; mm = ~np.isnan(row); k = int(mm.sum())
            if k == 0:
                continue
            rr = np.empty(k); rr[row[mm].argsort()] = np.arange(k)
            out[t, mm] = rr / max(k - 1, 1)
        return out

    ens_u = _xrank_u(_trail_u(126)) + _xrank_u(_trail_u(252))
    for freq, fsuf in (("M", "_ens"), ("W", "_ensw")):
        per = month if freq == "M" else week
        for K in (5, 8, 10):
            sel = np.zeros((T, nn), bool); cur = None; pm = None
            for t in range(T):
                if t >= 253 and per[t] != pm:
                    sc_ = ens_u[t]; idx = np.where(~np.isnan(sc_))[0]
                    if len(idx) >= K:
                        cur = idx[np.argsort(sc_[idx])[::-1][:K]]
                    pm = per[t]
                if cur is not None:
                    sel[t, cur] = True
            cnt = sel.sum(1)
            series = np.where(cnt > 0, (retm * sel).sum(1) / np.maximum(cnt, 1), 0.0)
            turn = np.abs(np.diff(sel.astype(float), axis=0, prepend=sel[:1] * 0)).sum(1) / np.maximum(K, 1)
            mom[f"{K}{fsuf}"] = series; momturn[f"{K}{fsuf}"] = turn
    lastday = np.zeros(T, bool)
    for y in np.unique(yrs):
        idx = np.where(yrs == y)[0]
        lastday[idx[-1]] = True
    vixarr = np.array([vixd.get(d.normalize(), np.nan) for d in cal])
    stocks = {}
    for ci, tk in enumerate(px.columns):
        c = px[tk].ffill().values
        R = rsi(c); ma = pd.Series(c).rolling(200).mean().values  # ma200 base; ma_win override below
        stocks[tk] = dict(ci=ci, c=c, R=R)
    # ma는 cfg.ma 가변 → 따로 계산 위해 close 보관
    closes = {tk: px[tk].ffill().values for tk in px.columns}
    # ── 고속 벡터화용 행렬 사전계산 ──
    cl = px.ffill()
    dl = cl.diff()
    Rmats = {}
    for p_ in (7, 10, 14, 21):   # RSI 기간 가변(기존 14 고정 의심)
        upp = dl.clip(lower=0).ewm(alpha=1/p_, adjust=False).mean()
        dnp = (-dl.clip(upper=0)).ewm(alpha=1/p_, adjust=False).mean()
        Rmats[p_] = (100 - 100/(1 + upp/(dnp + EPS))).values
    Rmat = Rmats[14]
    closemat = cl.values
    mamat = {w: cl.rolling(w).mean().values for w in (50, 100, 150, 200)}
    gser = pd.Series(gspc)
    mktok = {}
    spx_ma = {}
    for w in (50, 100, 120, 150, 175, 200, 220, 250):
        gm = gser.rolling(w).mean().values
        spx_ma[w] = gm
        ok = np.zeros(T, bool)
        ok[1:] = (~np.isnan(gm[:-1])) & (gspc[:-1] > gm[:-1])
        mktok[w] = ok
    # 레짐 대안(재의심): 듀얼모멘텀 — 전일 기준 지수 N일 수익>0이면 ON
    for w, key in ((126, "dm126"), (252, "dm252")):
        ok = np.zeros(T, bool)
        ok[w + 1:] = gspc[w:-1] / np.maximum(gspc[: T - w - 1], EPS) > 1.0
        mktok[key] = ok; spx_ma[key] = np.full(T, np.nan)
    gret = gser.pct_change().fillna(0.0).values
    s50 = gser.rolling(50).mean().values; s200 = gser.rolling(200).mean().values
    deep = np.zeros(T, bool)
    deep[1:] = (~np.isnan(s200[:-1])) & (gspc[:-1] < s200[:-1]) & (gspc[:-1] < s50[:-1])
    return dict(px=px, cal=cal, pret=pret, yrs=yrs, lastday=lastday, vixarr=vixarr,
                stocks=stocks, closes=closes, T=T, n=px.shape[1], cols=list(px.columns),
                gspc=gspc, m7_ret=m7_ret, ewall_ret=ewall_ret, mom=mom, momturn=momturn,
                Rmat=Rmat, Rmats=Rmats, closemat=closemat, mamat=mamat, mktok=mktok,
                spx_ma=spx_ma, gret=gret, deep=deep)


def run_fast(cfg, D):
    """벡터화 고속버전 — mt=1·무손절·무익절(tp_rsi/tp_pct=0) 생산영역 전용.
    포지션 보유여부를 행렬 accumulate로 계산. run()과 동일 결과(해당 영역)."""
    T, n = D["T"], D["n"]; pret = D["pret"]; yrs = D["yrs"]; lastday = D["lastday"]
    vixarr = D["vixarr"]; closemat = D["closemat"]
    rsi_p = cfg.get("rsi_p", 14)
    Rmat = D["Rmats"][rsi_p] if "Rmats" in D else D["Rmat"]
    rsi_th = cfg["rsi"]; vix_th = cfg["vix"]
    maw = cfg["ma"]   # 0 = 종목 MA 필터 없음
    slots = cfg["slots"]; vixmax = cfg.get("vixmax", 0)
    # 시장추세 레짐 (히스테리시스: 진입선 > 청산선, 청산 확인일수)
    me = cfg.get("mkt_ma", 200); mx = cfg.get("mkt_exit_ma", me); conf = cfg.get("mkt_confirm", 0)
    if mx == me and conf == 0:
        mkt_ok = D["mktok"][me]                      # 단순(기존)과 동일
    else:
        gspc0 = D["gspc"]; ma_e = D["spx_ma"][me]; ma_x = D["spx_ma"][mx]
        mkt_ok = np.zeros(T, bool); state = False; below = 0
        for t in range(1, T):
            pv = gspc0[t - 1]
            if not state:
                if not np.isnan(ma_e[t - 1]) and pv > ma_e[t - 1]:
                    state = True; below = 0
            else:
                if not np.isnan(ma_x[t - 1]) and pv < ma_x[t - 1]:
                    below += 1
                    if below > conf:
                        state = False
                else:
                    below = 0
            mkt_ok[t] = state

    # 진입조건 (전일 t-1 확정): RSI<=th & close>ma & 공포조건, 상승장(mkt_ok)에서만
    if maw == 0:
        cond = (Rmat <= rsi_th)
    else:
        ma = D["mamat"][maw]
        cond = (Rmat <= rsi_th) & (closemat > ma) & (~np.isnan(ma))
    fdir = D.get("fear_dir", "high")   # high=VIX(높을수록공포), low=F&G(낮을수록공포)
    if fdir == "low":
        vok = vixarr <= vix_th
        if vixmax > 0:
            vok = vok & (vixarr >= vixmax)   # 너무 극단(예:0)은 거름
    else:
        vok = vixarr >= vix_th
        if vixmax > 0:
            vok = vok & (vixarr <= vixmax)
    condv = cond & vok[:, None]
    entry = np.zeros((T, n), bool)
    entry[1:] = condv[:-1] & mkt_ok[1:, None]      # t-1 신호 → t 진입, 당일 상승장
    # 시장 전체 청산일: 연말(yearend=True일 때) or 추세이탈
    exitday = lastday.copy() if cfg.get("yearend", True) else np.zeros(T, bool)
    if cfg.get("mkt_exit", True):
        exitday = exitday | (~mkt_ok)
    ar = np.arange(T)
    eidx = np.where(entry, ar[:, None], -1)
    last_entry = np.maximum.accumulate(eidx, axis=0)
    xidx = np.where(exitday, ar, -1)
    last_exit = np.maximum.accumulate(xidx)
    held = (last_entry > last_exit[:, None]) & (last_entry >= 0)
    mh = cfg.get("maxhold", 0)   # 재의심: 보유기간 상한(0=무제한이 기존값)
    if mh > 0:
        held = held & ((ar[:, None] - last_entry) <= mh)

    if cfg.get("rsi_weight", False):
        # 재의심: 동일비중 대신 진입신호 RSI 깊이 비례 가중(총노출 동일)
        ei = np.maximum(last_entry - 1, 0)
        sig_rsi = np.take_along_axis(Rmat, ei, axis=0)
        depth = np.where(held, np.clip(rsi_th - sig_rsi, 0, None) + 1.0, 0.0)
        rowsum_d = depth.sum(1)
        nheld = held.sum(1)
        target = np.minimum(nheld / slots, 1.0)
        dollar = depth * np.where(rowsum_d > 0, target / np.maximum(rowsum_d, EPS), 0.0)[:, None]
    else:
        dollar = held.astype(float) / slots
        rowsum = dollar.sum(axis=1)
        scale = np.where(rowsum > 1.0, 1.0 / np.maximum(rowsum, EPS), 1.0)
        dollar = dollar * scale[:, None]
    held_prev = np.vstack([np.zeros((1, n)), dollar[:-1]])
    port = (held_prev * pret).sum(axis=1)
    invested = held_prev.sum(axis=1) > 1e-6
    # 위성 회전율(매수/매도 비중변화 합)
    turn = np.abs(np.diff(dollar, axis=0, prepend=dollar[:1] * 0)).sum(axis=1)

    gret = D["gret"]
    wcore = np.zeros(T)
    if cfg.get("cash_in_spx", False):
        ca = cfg.get("core_asset", "spx")
        mkey = None
        if ca.startswith("mom"):
            k = ca[3:]; mkey = int(k) if k.isdigit() else k
        core_ret = (D["m7_ret"] if ca == "m7" else D["ewall_ret"] if ca == "ewall"
                    else D["mom"][mkey] if mkey is not None else gret)
        frac = cfg.get("cash_spx_frac", 1.0)
        # 공포 단계별 노출 증액(재의심): VIX≥vix_deep이면 frac을 frac_deep로(전일 VIX 기준, 룩어헤드 없음)
        vd = cfg.get("vix_deep", 0); fd = cfg.get("frac_deep", frac)
        if vd > 0:
            vprev = np.concatenate([[np.nan], vixarr[:-1]])
            frac_t = np.where(vprev >= vd, fd, frac)
        else:
            frac_t = frac
        resid = np.clip(1.0 - dollar.sum(axis=1), 0, 1) * mkt_ok.astype(float) * frac_t
        resid_prev = np.concatenate([[0.0], resid[:-1]])
        port = port + resid_prev * core_ret
        invested = invested | (resid_prev > 1e-6)
        wcore = resid
        # 코어 회전: 진입/이탈(|Δwcore|) + 모멘텀 바스켓 내부 리밸런싱
        turn = turn + np.abs(np.diff(wcore, prepend=0))
        if mkey is not None:
            turn = turn + wcore * D["momturn"][mkey]
    hf = cfg.get("hedge_frac", 0); hl = cfg.get("hedge_lev", 1)
    if hf > 0:
        on = D["deep"] if cfg.get("hedge_mode", "inverse") == "deep" else (~mkt_ok)
        conf = cfg.get("hedge_confirm", 0)
        if conf > 0:
            run_ = 0; onc = np.zeros(T, bool)
            for t in range(T):
                run_ = run_ + 1 if on[t] else 0
                onc[t] = run_ >= conf
            on = onc
        hser = hf * on.astype(float)
        hsleeve_prev = np.concatenate([[0.0], hser[:-1]])
        port = port + hsleeve_prev * (-gret) * hl
        invested = invested | (hsleeve_prev > 1e-6)
        turn = turn + np.abs(np.diff(hser, prepend=0)) * hl
    # 거래비용 차감 (편도 bps)
    cost_bps = cfg.get("cost_bps", 5)
    port = port - turn * (cost_bps / 1e4)
    return port, invested, yrs


def run(cfg, D):
    T, n = D["T"], D["n"]; pret = D["pret"]; yrs = D["yrs"]; lastday = D["lastday"]; vixarr = D["vixarr"]
    W = np.zeros((T, n))   # full-position 단위 (0..1) — 각 시점 종가후 보유량
    ma_win = cfg["ma"]; rsi_th = cfg["rsi"]; vix_th = cfg["vix"]
    mt = cfg["maxtranche"]; gap = cfg["gap"]; stop = cfg["stop"]; trail = cfg["trail"]
    tp = cfg["tp_rsi"]; yend = cfg["yearend"]; reentry = cfg["reentry"]
    mkt_ma = cfg.get("mkt_ma", 0); mkt_exit = cfg.get("mkt_exit", False)
    tp_pct = cfg.get("tp_pct", 0); vixmax = cfg.get("vixmax", 0)

    # SPX 추세 게이트 (전날 기준): gspc > MA(mkt_ma)
    gspc = D["gspc"]
    if cfg.get("mkt_gc", False):
        s50 = pd.Series(gspc).rolling(50).mean().values
        s200 = pd.Series(gspc).rolling(200).mean().values
        mkt_ok = np.array([(t >= 1 and not np.isnan(s200[t - 1]) and s50[t - 1] > s200[t - 1])
                           for t in range(T)])
    elif mkt_ma > 0:
        gma = pd.Series(gspc).rolling(mkt_ma).mean().values
        mkt_ok = np.array([(t >= 1 and not np.isnan(gma[t - 1]) and gspc[t - 1] > gma[t - 1])
                           for t in range(T)])
    else:
        mkt_ok = np.ones(T, bool)

    for tk in D["cols"]:
        st = D["stocks"][tk]; ci = st["ci"]; c = st["c"]; R = st["R"]
        ma = pd.Series(c).rolling(ma_win).mean().values
        # 진입신호(전날 확정): RSI<=th & close>ma & VIX>=th
        sig = np.zeros(T, bool)
        for t in range(1, T):
            j = t - 1
            ok = (not np.isnan(R[j]) and R[j] <= rsi_th and not np.isnan(ma[j]) and c[j] > ma[j]
                  and not np.isnan(vixarr[j]) and vixarr[j] >= vix_th)
            if ok and vixmax > 0 and vixarr[j] > vixmax:
                ok = False
            if ok and not mkt_ok[t]:
                ok = False
            sig[t] = ok
        in_pos = False; tr = 0; sum_cost = 0.0; peak = 0.0; last_add = -10**9; w_end = 0.0
        for t in range(1, T):
            if in_pos:
                peak = max(peak, c[t])
                avg = sum_cost / tr
                exit_now = False
                if stop > 0 and c[t] <= avg * (1 - stop / 100):
                    exit_now = True
                elif trail > 0 and c[t] <= peak * (1 - trail / 100):
                    exit_now = True
                elif tp > 0 and R[t] >= tp:
                    exit_now = True
                elif tp_pct > 0 and c[t] >= avg * (1 + tp_pct / 100):
                    exit_now = True
                elif mkt_exit and not mkt_ok[t]:
                    exit_now = True
                elif yend and lastday[t]:
                    exit_now = True
                if exit_now:
                    in_pos = False; tr = 0; sum_cost = 0.0; w_end = 0.0
                else:
                    if sig[t] and tr < mt and (t - last_add) >= gap:
                        tr += 1; sum_cost += c[t]; last_add = t
                    w_end = tr / mt
            if not in_pos:
                if sig[t] and (reentry or True):
                    in_pos = True; tr = 1; sum_cost = c[t]; peak = c[t]; last_add = t; w_end = 1.0 / mt
            W[t, ci] = w_end

    # full-position 단위 → NAV 비중 (1포지션=1/slots, 총합 1 상한)
    slots = cfg["slots"]
    dollar = W / slots
    rowsum = dollar.sum(axis=1)
    scale = np.where(rowsum > 1.0, 1.0 / np.maximum(rowsum, EPS), 1.0)
    dollar = dollar * scale[:, None]
    # 일수익: 전일 보유비중 × 당일 종목수익
    held_prev = np.vstack([np.zeros((1, n)), dollar[:-1]])
    port = (held_prev * pret).sum(axis=1)
    invested = held_prev.sum(axis=1) > 1e-6
    # 유휴현금 SPX 파킹 (상승장에서만)
    gret = pd.Series(gspc).pct_change().fillna(0.0).values
    if cfg.get("cash_in_spx", False):
        ca = cfg.get("core_asset", "spx")
        if ca == "m7":
            core_ret = D["m7_ret"]
        elif ca == "ewall":
            core_ret = D["ewall_ret"]
        elif ca.startswith("mom"):
            core_ret = D["mom"][int(ca[3:])]
        else:
            core_ret = gret
        frac = cfg.get("cash_spx_frac", 1.0)
        resid = np.clip(1.0 - dollar.sum(axis=1), 0, 1) * mkt_ok.astype(float) * frac
        resid_prev = np.concatenate([[0.0], resid[:-1]])
        port = port + resid_prev * core_ret
        invested = invested | (resid_prev > 1e-6)
    # 헤지(실현가능만): 전일 레짐으로 결정된 인버스 SPX 슬리브. 룩어헤드 없음.
    hf = cfg.get("hedge_frac", 0); hl = cfg.get("hedge_lev", 1)
    if hf > 0:
        hmode = cfg.get("hedge_mode", "inverse")
        if hmode == "deep":   # SPX<200 그리고 SPX<50MA (전일)
            s50 = pd.Series(gspc).rolling(50).mean().values
            s200b = pd.Series(gspc).rolling(200).mean().values
            on = np.array([(t >= 1 and not np.isnan(s200b[t-1]) and gspc[t-1] < s200b[t-1] and gspc[t-1] < s50[t-1])
                           for t in range(T)])
        else:                 # 기본: not mkt_ok (전일 SPX<MA)
            on = (~mkt_ok)
        conf = cfg.get("hedge_confirm", 0)
        if conf > 0:          # N일 연속 하락확인시에만
            onc = on.copy()
            run_ = 0
            for t in range(T):
                run_ = run_ + 1 if on[t] else 0
                onc[t] = run_ >= conf
            on = onc
        hsleeve = hf * on.astype(float)
        hsleeve_prev = np.concatenate([[0.0], hsleeve[:-1]])
        port = port + hsleeve_prev * (-gret) * hl   # 인버스: 전일결정 보유 → 당일 -SPX수익
        invested = invested | (hsleeve_prev > 1e-6)
    return port, invested, yrs


def metrics(port, invested, yrs, ppy=252):
    r = port; eq = np.cumprod(1 + r); T = len(r); y = T / ppy
    cagr = (eq[-1] ** (1 / y) - 1) * 100
    vol = r.std(ddof=1) * np.sqrt(ppy) * 100
    sharpe = (r.mean() * ppy) / (r.std(ddof=1) * np.sqrt(ppy) + EPS)
    downs = r[r < 0]
    sortino = (r.mean() * ppy) / (downs.std(ddof=1) * np.sqrt(ppy) + EPS) if len(downs) > 1 else 0
    peak = np.maximum.accumulate(eq); mdd = ((eq / peak) - 1).min() * 100
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    expo = invested.mean() * 100
    # 연도별 수익
    yret = {}
    for yr in np.unique(yrs):
        m = yrs == yr
        yret[int(yr)] = (np.prod(1 + r[m]) - 1) * 100
    # 강건성: 최고의 해 제거 후 CAGR 유지율
    best = max(yret, key=yret.get)
    rr = r[yrs != best]
    eq2 = np.cumprod(1 + rr); y2 = len(rr) / 252.0
    cagr_exbest = (eq2[-1] ** (1 / y2) - 1) * 100 if len(rr) > 10 else cagr
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino, mdd=mdd, calmar=calmar,
                expo=expo, mult=eq[-1], yret=yret, cagr_exbest=cagr_exbest, best_year=int(best))


def score(m):
    s = {}
    s["sharpe"] = float(np.clip(m["sharpe"] / 1.5 * 25, 0, 25))
    s["beat_ew"] = float(np.clip((m["sharpe"] - EW_BH_SHARPE) / 0.30 * 15, 0, 15))
    # MDD: 20%이내=20pt, 45%이상=0
    a = abs(m["mdd"])
    s["mdd"] = float(np.clip((45 - a) / (45 - 20) * 20, 0, 20))
    s["calmar"] = float(np.clip(m["calmar"] / 1.0 * 10, 0, 10))
    s["sortino"] = float(np.clip(m["sortino"] / 1.7 * 10, 0, 10))
    b18 = m["yret"].get(2018, -99); b22 = m["yret"].get(2022, -99)
    s["bear"] = (5 if b18 >= 0 else max(0, 5 + b18)) + (5 if b22 >= 0 else max(0, 5 + b22))
    s["bear"] = float(np.clip(s["bear"], 0, 10))
    # 강건성: ex-best CAGR이 full의 60%이상 유지
    keep = m["cagr_exbest"] / m["cagr"] if m["cagr"] > 0 else 0
    s["robust"] = float(np.clip(keep / 0.60 * 10, 0, 10))
    s["total"] = sum(v for k, v in s.items() if k != "total")
    return s


def main():
    D = load_all()
    print(f"종목 {D['n']} · 거래일 {D['T']} ({D['cal'][0].date()}~{D['cal'][-1].date()})\n")

    base = dict(vix=26, rsi=30, ma=200, slots=20, maxtranche=1, gap=5,
                stop=0, trail=0, tp_rsi=0, yearend=True, reentry=True)

    # ITER 목록 (레드팀 제안) — 실행 후 결과보고 다음 배치 결정
    iters = json.load(open(os.path.join(HERE, "rb_iters.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(HERE, "rb_iters.json")) else [
            dict(name="I46 M7코어(헤지無)", **{**base, "mkt_ma": 200, "mkt_exit": True, "cash_in_spx": True, "cash_spx_frac": 0.3, "slots": 10, "core_asset": "m7"}),
            dict(name="I51 M7+테일헤지50", **{**base, "mkt_ma": 200, "mkt_exit": True, "cash_in_spx": True, "cash_spx_frac": 0.3, "slots": 10, "core_asset": "m7", "hedge_frac": 0.5, "hedge_crashonly": True}),
            dict(name="I52 M7+테일헤지100", **{**base, "mkt_ma": 200, "mkt_exit": True, "cash_in_spx": True, "cash_spx_frac": 0.3, "slots": 10, "core_asset": "m7", "hedge_frac": 1.0, "hedge_crashonly": True}),
            dict(name="I53 M7+테일헤지100x2", **{**base, "mkt_ma": 200, "mkt_exit": True, "cash_in_spx": True, "cash_spx_frac": 0.3, "slots": 10, "core_asset": "m7", "hedge_frac": 1.0, "hedge_lev": 2, "hedge_crashonly": True}),
            dict(name="I54 M7파킹40+테일100", **{**base, "mkt_ma": 200, "mkt_exit": True, "cash_in_spx": True, "cash_spx_frac": 0.4, "slots": 10, "core_asset": "m7", "hedge_frac": 1.0, "hedge_crashonly": True}),
            dict(name="I55 M7파킹35+테일100", **{**base, "mkt_ma": 200, "mkt_exit": True, "cash_in_spx": True, "cash_spx_frac": 0.35, "slots": 10, "core_asset": "m7", "hedge_frac": 1.0, "hedge_crashonly": True}),
            dict(name="I56 M7파킹40+테일100x2", **{**base, "mkt_ma": 200, "mkt_exit": True, "cash_in_spx": True, "cash_spx_frac": 0.4, "slots": 10, "core_asset": "m7", "hedge_frac": 1.0, "hedge_lev": 2, "hedge_crashonly": True}),
        ]

    results = []
    print(f"{'전략':>20s} | {'CAGR':>7s} {'샤프':>5s} {'소르':>5s} {'MDD':>7s} {'칼마':>5s} {'노출':>5s} | {'점수':>6s}")
    print("-" * 92)
    for cfg in iters:
        name = cfg.pop("name")
        port, inv, yrs = run(cfg, D)
        m = metrics(port, inv, yrs); sc = score(m)
        results.append((name, cfg, m, sc))
        print(f"{name:>20s} | {m['cagr']:>+6.2f}% {m['sharpe']:>5.2f} {m['sortino']:>5.2f} "
              f"{m['mdd']:>+6.1f}% {m['calmar']:>5.2f} {m['expo']:>4.0f}% | {sc['total']:>5.1f}")

    # 최고 점수 상세
    best = max(results, key=lambda x: x[3]["total"])
    name, cfg, m, sc = best
    print(f"\n■ 최고: {name}  (총 {sc['total']:.1f}/100)")
    print(f"  배점: 샤프 {sc['sharpe']:.1f}/25 · EW추월 {sc['beat_ew']:.1f}/15 · MDD {sc['mdd']:.1f}/20 · "
          f"칼마 {sc['calmar']:.1f}/10 · 소르티노 {sc['sortino']:.1f}/10 · 베어 {sc['bear']:.1f}/10 · 강건 {sc['robust']:.1f}/10")
    print(f"  연도별: " + " ".join(f"{y}:{v:+.0f}%" for y, v in m['yret'].items()))
    print(f"  설정: {cfg}")

    json.dump([(nm, cfg, {k: (v if not isinstance(v, dict) else v) for k, v in m.items()}, sc)
               for nm, cfg, m, sc in results],
              open(os.path.join(HERE, "rb_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n저장: rb_results.json")


if __name__ == "__main__":
    main()
