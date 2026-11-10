# -*- coding: utf-8 -*-
"""선별·사이징 개선 실험 — 포트폴리오 워크포워드 하네스.
================================================================================
청산·진입은 확정(알파 아님). 여기선 ★선별(무엇을 고르나)과 ★사이징(얼마 거나)만 변주.
월 리밸런스: 매월초 직전데이터로 점수→TOPN 선별(+레짐필터)→사이징 비중→터틀이 진입/청산 관장.
포트 일별수익 = Σ 비중_i × 터틀포지션_i × 수익_i − 수수료.  미래참조 없음(리밸 시점 이전 데이터만).

토글:
  선별  = 모멘텀 룩백(mom_lb) · vol/mom 가중(vw,mw) · TOPN · 레짐필터(regime: SPY/BTC 200MA 위만)
  사이징 = equal / risk(1.5N ATR·R%) / zweight(z점수 비례) / sectorcap(반도체 상한)
평가  = 전체복리·CAGR·MDD·Calmar + OOS(Train≤2023 / Test 2024~) Calmar.
정직성: 같은 데이터 다중비교라 과최적 위험 → OOS·연도별 일관성으로 판정. 생존편향·슬리피지 미반영.
"""
import os, sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 선별 엔진은 같은 폴더
import batto_screener_live as B

MIN_HIST = 504
SEMI = {"NVDA", "AMD", "MU", "AVGO", "LRCX", "AMAT", "MRVL", "QCOM", "TXN", "INTC", "SMCI", "ASML"}
TEST_START = pd.Timestamp("2024-01-01")


def prep():
    series, ohlc = B.load_all()
    tpos = {s: B.turtle_pos(ohlc[s], *B.turtle_params(s)) for s in ohlc}
    # 마스터 일별 인덱스(코인+미국 합집합)
    idx = sorted(set().union(*[set(c.index) for c in series.values()]))
    idx = pd.DatetimeIndex(idx)
    ret = {}; pos = {}; atrp = {}
    for s in series:
        c = series[s].reindex(idx)
        ret[s] = c.pct_change().fillna(0.0)
        pos[s] = tpos[s].reindex(idx).ffill().fillna(0.0)
        a = pd.Series(B.atr(ohlc[s]["h"].values, ohlc[s]["l"].values, ohlc[s]["c"].values, 20), index=ohlc[s].index)
        atrp[s] = (a / ohlc[s]["c"]).reindex(idx).ffill()   # ATR/price
    # 레짐: SPY·BTC 200일 SMA
    def sma200(s):
        c = series[s].reindex(idx).ffill()
        return (c > c.rolling(200).mean())
    regime = {"us": sma200("SPY") if "SPY" in series else pd.Series(True, index=idx),
              "coin": sma200("BTC") if "BTC" in series else pd.Series(True, index=idx)}
    return series, idx, ret, pos, atrp, regime


def score_at(series, d, mom_lb, vw, mw):
    sc = {}
    for s, c in series.items():
        prev = c[c.index < d]
        if len(prev) < MIN_HIST: continue
        r = prev.pct_change().dropna()
        vol = r.tail(252).std() * np.sqrt(252)
        mom = prev.iloc[-1] / prev.iloc[-mom_lb] - 1 if len(prev) > mom_lb else np.nan
        if mom != mom: continue
        sc[s] = (vol, mom)
    if len(sc) < 3: return {}, {}
    vols = np.array([sc[s][0] for s in sc]); moms = np.array([sc[s][1] for s in sc])
    zv = (vols - vols.mean()) / (vols.std() or 1); zm = (moms - moms.mean()) / (moms.std() or 1)
    z = {s: vw * zv[i] + mw * zm[i] for i, s in enumerate(sc)}
    return z, sc


def weights(top, z, atrp, d, sizing, R, maxw):
    w = {}
    if sizing == "equal":
        for s in top: w[s] = 1.0 / len(top)
    elif sizing == "zweight":
        zz = {s: max(z[s] - min(z[s] for s in top) + 0.1, 0.01) for s in top}
        tot = sum(zz.values())
        for s in top: w[s] = zz[s] / tot
    else:  # risk / sectorcap
        for s in top:
            ap = atrp[s].asof(d)
            w[s] = min(R / (1.5 * ap), maxw) if ap and ap > 0 else 0.0
        if sizing == "sectorcap":
            semi_w = sum(w[s] for s in top if s in SEMI)
            if semi_w > 0.40:
                sc = 0.40 / semi_w
                for s in top:
                    if s in SEMI: w[s] *= sc
    # 레버리지 금지
    tot = sum(w.values())
    if tot > 1.0:
        for s in w: w[s] /= tot
    return w


def run(cfg, series, idx, ret, pos, atrp, regime):
    mom_lb = cfg.get("mom_lb", 252); vw = cfg.get("vw", 1.0); mw = cfg.get("mw", 1.0)
    TOPN = cfg.get("topn", 8); use_reg = cfg.get("regime", False)
    sizing = cfg.get("sizing", "risk"); R = cfg.get("R", 0.01); maxw = cfg.get("maxw", 0.20)
    rebals = pd.date_range(idx[0], idx[-1], freq="MS")
    rebals = [d for d in rebals if d >= idx[0] + pd.Timedelta(days=MIN_HIST + 40)]
    # 종목별 비중 시계열(스텝)
    W = pd.DataFrame(0.0, index=idx, columns=list(series.keys()))
    for k, d in enumerate(rebals):
        z, sc = score_at(series, d, mom_lb, vw, mw)
        if not z: continue
        cand = sorted(z, key=lambda s: -z[s])
        if use_reg:  # 레짐: 해당 자산군이 200MA 위일 때만
            cand = [s for s in cand if regime["coin" if B.is_coin(s) else "us"].asof(d)]
        top = cand[:TOPN]
        w = weights(top, z, atrp, d, sizing, R, maxw) if top else {}
        d1 = rebals[k + 1] if k + 1 < len(rebals) else idx[-1] + pd.Timedelta(days=1)
        seg = (W.index >= d) & (W.index < d1)
        for s, wi in w.items(): W.loc[seg, s] = wi
    # 노출 = 비중 × 터틀포지션
    P = pd.DataFrame(pos)[W.columns].reindex(idx)
    Rt = pd.DataFrame(ret)[W.columns].reindex(idx)
    E = W * P
    fee = pd.Series([0.0007] * len(W.columns), index=W.columns)   # 혼합 평균 수수료 근사
    port = (E.shift(1) * Rt).sum(axis=1) - (E.diff().abs() * fee).sum(axis=1)
    eq = (1 + port.fillna(0)).cumprod()
    return eq


def metrics(eq, lo=None, hi=None):
    e = eq.copy()
    if lo is not None: e = e[(e.index >= lo)]
    if hi is not None: e = e[e.index < hi]
    if len(e) < 60: return None
    e = e / e.iloc[0]
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1 if e.iloc[-1] > 0 else -1
    pk = np.maximum.accumulate(e.values); mdd = (1 - e.values / pk).max() * 100
    cal = cagr / (mdd / 100) if mdd > 0 else np.nan
    return e.iloc[-1], cagr * 100, mdd, cal


def yearly(eq):
    out = {}
    for yr in range(2016, 2027):
        e = eq[(eq.index >= pd.Timestamp(f"{yr}-01-01")) & (eq.index <= pd.Timestamp(f"{yr}-12-31"))]
        if len(e) > 30: out[yr] = (e.iloc[-1] / e.iloc[0] - 1) * 100
    return out


def main():
    print("데이터 준비...")
    series, idx, ret, pos, atrp, regime = prep()
    base = {"mom_lb": 252, "vw": 1, "mw": 1, "topn": 8, "regime": False, "sizing": "risk"}
    CFGS = [
        ("기준(현재: mom252·N8·risk)", base),
        ("+레짐필터(200MA)", {**base, "regime": True}),
        ("모멘텀 126일(6mo)", {**base, "mom_lb": 126}),
        ("모멘텀만(vw0)", {**base, "vw": 0, "mw": 1}),
        ("변동성만(mw0)", {**base, "vw": 1, "mw": 0}),
        ("모멘텀가중(vw0.5)", {**base, "vw": 0.5, "mw": 1}),
        ("TOPN 4", {**base, "topn": 4}),
        ("TOPN 12", {**base, "topn": 12}),
        ("사이징 등가중", {**base, "sizing": "equal"}),
        ("사이징 z가중", {**base, "sizing": "zweight"}),
        ("사이징 섹터캡(반도체40%)", {**base, "sizing": "sectorcap"}),
        ("★레짐+섹터캡", {**base, "regime": True, "sizing": "sectorcap"}),
        ("★레짐+모멘텀가중", {**base, "regime": True, "vw": 0.5, "mw": 1}),
    ]
    L = ["선별·사이징 개선 실험 — 포트폴리오 워크포워드 2016~2026 (월 리밸·터틀 매매·미래참조 없음)",
         "청산/진입 확정, 선별·사이징만 변주. 전체복리·MDD·Calmar + OOS(2024~).\n",
         f"  {'구성':30s}{'복리배수':>8s}{'CAGR%':>7s}{'MDD%':>6s}{'Calmar':>7s}{'OOS_Cal':>8s}"]
    rows = []
    for name, cfg in CFGS:
        eq = run(cfg, series, idx, ret, pos, atrp, regime)
        m = metrics(eq); oos = metrics(eq, lo=TEST_START)
        rows.append((name, m, oos[3] if oos else np.nan, eq, cfg))
        L.append(f"  {name:30s}{m[0]:>8.1f}{m[1]:>7.0f}{m[2]:>6.0f}{m[3]:>7.2f}{(oos[3] if oos else float('nan')):>8.2f}")
    # 연도별: 기준 vs 레짐필터
    L.append("\n[연도별 수익률 %] 기준 vs +레짐필터 (2022·2025 약점 개선 확인)")
    yb = yearly(rows[0][3]); yr_ = yearly(rows[1][3])
    L.append("  " + f"{'연도':6s}" + "".join(f"{y:>7d}" for y in sorted(yb)))
    L.append("  " + f"{'기준':6s}" + "".join(f"{yb.get(y,0):>7.0f}" for y in sorted(yb)))
    L.append("  " + f"{'+레짐':6s}" + "".join(f"{yr_.get(y,0):>7.0f}" for y in sorted(yb)))
    L.append("")
    best = max(rows, key=lambda r: (r[1][3] if r[1] else -9))
    bestoos = max(rows, key=lambda r: (r[2] if r[2] == r[2] else -9))
    L.append(f"  ★전체 Calmar 최고: {best[0]} (Calmar {best[1][3]:.2f}·복리 {best[1][0]:.1f}배·MDD {best[1][2]:.0f}%)")
    L.append(f"  ★OOS Calmar 최고: {bestoos[0]} (OOS {bestoos[2]:.2f})")
    L.append("  ※ 전체·OOS 둘 다 상위면 신뢰↑. 한쪽만이면 과최적/국면운 의심. 다음 단계서 robust 재확인 필요.")
    txt = "\n".join(L); print(txt)
    open(os.path.join(B.DATA_DIR, "SCREENER_SELECT_SIZE_LOG.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
