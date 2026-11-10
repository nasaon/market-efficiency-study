# -*- coding: utf-8 -*-
"""[비용·체결 현실화] 확정 시스템(스크리너 z-TOP8 + 터틀 C안)의 백테스트→실계좌 간극 3종 정량화.
① 세후: 미국주식 양도세 22%(연 250만 공제·손익통산·KRW 환산 실현손익) — 절반익절 구조의 세금 드래그
② 갭 손절: stop을 갭으로 뚫을 때 시가 체결 — 빈도·슬리피지·'리스크 1%'의 실제값
③ 환율: 원화 기준 성과·MDD (환노출 무헤지)
+ 부록: 평균 현금 비중(파킹 규칙 근거)
전제: screener_turtle_improve.py의 C안(조합시 SMA½) 하네스와 동일 로직·수수료(코인0.1%/주식0.05% 편도).
정직성: 연 1회 선별·등가중 1/N 근사(리스크사이징 아님)·생존편향 포함 낙관상한. 세금은 연말 일괄 납부 근사.
"""
import os, sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
COINS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "TRX", "AVAX", "LINK"]
US = ["NVDA", "AMD", "SMCI", "TSLA", "AVGO", "MU", "META", "AMZN", "PLTR", "AAPL", "MSFT", "GOOGL",
      "QQQ", "SPY", "JPM", "XOM", "WMT", "PG", "KO", "JNJ", "VZ", "COST", "HD", "UNH", "V", "MA",
      "BAC", "DIS", "NFLX", "CRM", "ADBE", "ORCL", "CSCO", "INTC", "QCOM", "TXN", "AMAT", "LRCX",
      "MRVL", "COIN", "MSTR", "PYPL", "SHOP", "UBER", "ABNB", "CRWD", "PANW", "NOW", "LLY", "TMO"]
YEARS = list(range(2016, 2027))
TOPN = 8; MIN_HIST = 504
TAX_RATE = 0.22; TAX_DEDUCT = 2_500_000  # KRW
ACCOUNTS = [50_000_000, 100_000_000]     # 5천만·1억


def load_stock(tk):
    return pd.read_csv(os.path.join(HERE, f"yf_{tk}.csv"), parse_dates=["date"], index_col="date")[["o", "h", "l", "c"]]


def load_coin(sym):
    h1 = pd.read_csv(os.path.join(HERE, f"bn_{sym}_1h.csv"), parse_dates=["date"], index_col="date")
    return h1.resample("1D").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()


def atr(h, l, c, n=20):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().values


def params_C(sym):
    if sym not in COINS: return (30, 10, 50, True)
    if sym == "ETH": return (20, 20, 25, True)
    return (20, 10, 30, True)


def turtle(df, eN, xN, smaN, use_sma, slN=1.5, trailN=2.0, gap_fill=False):
    """포지션 시계열 + 실현 이벤트 목록. 이벤트=(bar_i, 청산비중, entry_i, entry_px, fill_px, kind, 이론가eff)"""
    o = df["o"].values; h = df["h"].values; l = df["l"].values; c = df["c"].values; n = len(c)
    hh = pd.Series(h).rolling(eN).max().shift(1).values
    ll = pd.Series(l).rolling(xN).min().shift(1).values
    a = atr(h, l, c, 20)
    sma = pd.Series(c).rolling(max(smaN, 2)).mean().values
    pos = 0.0; stop = 0.0; peakh = 0.0; half = False
    entry_px = 0.0; entry_i = -1
    arr = np.zeros(n); events = []
    start = max(eN, smaN if use_sma else eN) + 1
    for i in range(start, n):
        price = c[i]
        if pos == 0:
            oksma = (not use_sma) or (not np.isnan(sma[i]) and price > sma[i])
            if not np.isnan(hh[i]) and price > hh[i] and oksma:
                pos = 1.0; stop = price - slN * a[i]; peakh = h[i]; half = False
                entry_px = price; entry_i = i
        else:
            peakh = max(peakh, h[i])
            eff = stop if not half else max(stop, peakh - trailN * a[i])
            if l[i] <= eff:
                fill = min(o[i], eff) if gap_fill else eff   # 갭이면 시가 체결
                events.append((i, pos, entry_i, entry_px, fill, "stop", eff))
                pos = 0.0
            elif not half and not np.isnan(ll[i]) and c[i] < ll[i]:
                events.append((i, 0.5, entry_i, entry_px, c[i], "half", c[i]))
                pos = 0.5; half = True
        arr[i] = pos
    return pd.Series(arr, index=df.index), events


def mdd(eq):
    pk = np.maximum.accumulate(eq); return (1 - eq / pk).max() * 100


def cagr_of(eq, ndays):
    yrs = ndays / 252.0
    return eq[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan


def main():
    series = {}; ohlc = {}
    for s in COINS:
        try: d = load_coin(s); ohlc[s] = d; series[s] = d["c"]
        except Exception: pass
    for s in US:
        try: d = load_stock(s); ohlc[s] = d; series[s] = d["c"]
        except Exception: pass

    # 환율 (일별 종가·미국 달력에 ffill)
    fxp = os.path.join(HERE, "fx_usdkrw.csv")
    fx = pd.read_csv(fxp, parse_dates=["date"], index_col="date")["c"] if os.path.exists(fxp) else None

    # 연도별 선별 (improve.py와 동일)
    picks = {}
    for yr in YEARS:
        t0 = pd.Timestamp(f"{yr}-01-01")
        scores = {}
        for s, cc in series.items():
            prev = cc[cc.index < t0]
            cur = cc[(cc.index >= t0) & (cc.index <= pd.Timestamp(f"{yr}-12-31"))]
            if len(prev) < MIN_HIST or len(cur) < 30: continue
            ret = prev.pct_change().dropna()
            vol = ret.tail(252).std() * np.sqrt(252); mom = prev.iloc[-1] / prev.iloc[-252] - 1
            scores[s] = (vol, mom)
        if len(scores) < 8: continue
        vols = np.array([scores[s][0] for s in scores]); moms = np.array([scores[s][1] for s in scores])
        zv = (vols - vols.mean()) / (vols.std() or 1); zm = (moms - moms.mean()) / (moms.std() or 1)
        z = {s: zv[j] + zm[j] for j, s in enumerate(scores)}
        picks[yr] = sorted(z, key=lambda s: -z[s])[:TOPN]

    def fee_of(s): return 0.001 if s in COINS else 0.0005

    def run_port(gap_fill):
        """전 기간 일별 포트 수익(USD) + 연도별 실현이익(USD, 원금 1가정 비율) + 갭 이벤트 + 투자비중."""
        daily = []; dates = []
        realized_by_year = {}   # yr -> list[(date, frac_gain_usd_per_alloc, s)]  alloc=연초 자산/8
        gap_events = []; invested = []
        for yr in YEARS:
            if yr not in picks: continue
            t0 = pd.Timestamp(f"{yr}-01-01"); t1 = pd.Timestamp(f"{yr}-12-31")
            curves = []; idxs = None
            for s in picks[yr]:
                cc = series[s]; cyr = cc[(cc.index >= t0) & (cc.index <= t1)]
                r = cyr.pct_change().fillna(0).values
                eN, xN, smaN, use_sma = params_C(s)
                pos, evs = turtle(ohlc[s], eN, xN, smaN, use_sma, gap_fill=gap_fill)
                p = pos.reindex(cyr.index).fillna(0).shift(1).fillna(0).values
                dchg = np.abs(np.diff(p, prepend=0))
                net = p * r - dchg * fee_of(s)
                curves.append(net)
                if idxs is None or len(cyr.index) < len(idxs): idxs = cyr.index
                # 이 해에 청산된 실현 이벤트 수집
                di = ohlc[s].index
                for (i, frac, ei, epx, fpx, kind, eff) in evs:
                    d = di[i]
                    if not (t0 <= d <= t1): continue
                    fee = fee_of(s)
                    g = frac * ((fpx / epx) - 1.0) - frac * fee * 2   # 왕복 수수료 근사
                    realized_by_year.setdefault(yr, []).append((d, g, s, di[ei]))
                    if kind == "stop":
                        slip = (eff - fpx) / epx  # 갭이면 >0
                        gap_events.append((s, d, epx, eff, fpx, slip, frac))
            m = min(len(x) for x in curves)
            pr = np.mean([x[:m] for x in curves], axis=0)
            daily.append(pr); dates.append(idxs[:m])
            # 투자비중(현금 파킹 근거): 해당 해 평균 포지션 합/8
            inv = np.mean([np.mean(np.abs(x) > 1e-12) for x in curves])  # 근사: 수익≠0 일수 비중
            invested.append(inv)
        allr = np.concatenate(daily); alld = pd.DatetimeIndex(np.concatenate([d.values for d in dates]))
        return allr, alld, realized_by_year, gap_events, invested

    L = ["[비용·체결 현실화] 스크리너 z-TOP8 + 터틀 C안 — 세후·갭·환율 (2016~2026)"]
    L.append("  전제: 등가중 1/8 근사·연1회 선별·수수료 반영·생존편향 포함 낙관상한. 코인+미국 60종목 유니버스.")
    L.append("")

    # ── 기준(이론 체결) vs 갭 체결 ─────────────────────────────
    r0, d0, real0, _, inv0 = run_port(gap_fill=False)
    r1, d1, real1, gaps, _ = run_port(gap_fill=True)
    eq0 = np.cumprod(1 + r0); eq1 = np.cumprod(1 + r1)
    c0, m0 = cagr_of(eq0, len(eq0)), mdd(eq0)
    c1, m1 = cagr_of(eq1, len(eq1)), mdd(eq1)
    L.append("② 갭 손절 실측 — '일일 배치 사이 stop 관통' 크기 (stop가 체결 vs 다음 평가시점 시가 체결)")
    L.append("  ※하네스 수익률은 청산일 '종가' 반영이라 갭 손실은 이미 성과에 포함 — 여기선 체결가 가정의 크기만 계량.")
    for grp, name in [(lambda s: s not in COINS, "미국주식(오버나이트 갭)"), (lambda s: s in COINS, "코인(24/7이나 봇은 일1회 평가 — 평가간 급락)")]:
        stops = [g for g in gaps if grp(g[0])]
        n_stop = len(stops)
        gapped = [g for g in stops if g[5] > 1e-9]
        L.append(f"  [{name}] stop 청산 {n_stop}건 · 갭 관통 {len(gapped)}건 ({len(gapped)/max(n_stop,1)*100:.0f}%)")
        if gapped:
            slips = np.array([g[5] for g in gapped]) * 1e4
            L.append(f"    갭 추가손실(진입가 대비 bp): 중앙 {np.median(slips):.0f} · 평균 {slips.mean():.0f} · p95 {np.percentile(slips,95):.0f}")
            worst = sorted(gapped, key=lambda g: -g[5])[:3]
            for s, d, epx, eff, fpx, slip, frac in worst:
                L.append(f"      최악례 {s} {str(d.date())}: stop {eff:,.2f} → 체결 {fpx:,.2f} (추가 −{slip*100:.1f}%p)")
        # '리스크 1%' 실제 배율: 손실성 초기손절(eff<entry)만으로 계산 — 트레일 익절(eff>entry) 제외
        loss_stops = [g for g in stops if g[3] < g[2]]
        if loss_stops:
            theo = np.array([(g[2] - g[3]) / g[2] for g in loss_stops])
            act = np.array([(g[2] - g[4]) / g[2] for g in loss_stops])
            L.append(f"    손실성 손절 {len(loss_stops)}건: 이론 손실 평균 {theo.mean()*100:.1f}% → 갭 반영 {act.mean()*100:.1f}% (배율 ×{act.mean()/theo.mean():.2f})")
            L.append(f"    → 사이징 '손절시 계좌 −1%' 가정의 실제 기대값 ≈ −{act.mean()/theo.mean():.2f}% (p95 단일건 ×{np.percentile(act/theo,95):.2f})")
    L.append("")

    # ── 환율(원화 기준) ────────────────────────────────────────
    L.append("③ 환율 — 원화 기준 성과 (무헤지·환노출 그대로)")
    if fx is not None:
        fxs = fx.reindex(d1).ffill().bfill().values
        fxr = np.concatenate([[0.0], np.diff(fxs) / fxs[:-1]])
        rk = (1 + r1) * (1 + fxr) - 1
        eqk = np.cumprod(1 + rk)
        ck, mk = cagr_of(eqk, len(eqk)), mdd(eqk)
        L.append(f"  USD 기준: 복리 {eq1[-1]:.1f}x · CAGR {c1*100:.1f}% · MDD {m1:.0f}%")
        L.append(f"  KRW 기준: 복리 {eqk[-1]:.1f}x · CAGR {ck*100:.1f}% · MDD {mk:.0f}%")
        # 2022 원화 완충 확인
        msk22 = (d1 >= "2022-01-01") & (d1 <= "2022-12-31")
        u22 = np.cumprod(1 + r1[msk22]); k22 = np.cumprod(1 + rk[msk22])
        L.append(f"  2022: USD {((u22[-1]-1)*100):.0f}% vs KRW {((k22[-1]-1)*100):.0f}% (환이 완충하면 KRW 우위)")
    else:
        L.append("  fx_usdkrw.csv 없음 — kr_download.py 먼저 실행")
        rk = r1; eqk = eq1
    L.append("")

    # ── 세후 (KRW 실현손익·연 250만 공제·22%) ────────────────────
    L.append("① 세후 알파 — 양도세 22%·연 250만 공제·손익통산 (실현 시점 KRW 환산)")
    L.append("  ※코인 과세는 유예 반복으로 미반영(주식분만 과세)·연말 일괄 납부 근사·이월공제 없음(한국 세법).")
    for acct in ACCOUNTS:
        # KRW 계좌 시뮬: 일별 rk 복리 + 연말 세금 차감
        eq = acct; peak = acct; maxdd = 0.0; eq_series = []
        # 연도 경계 인덱스
        years_d = pd.Series(d1.year, index=np.arange(len(d1)))
        tax_paid_total = 0.0; tax_by_year = {}
        yr_start_eq = {}
        for i in range(len(rk)):
            yr = d1[i].year
            if yr not in yr_start_eq: yr_start_eq[yr] = eq
            eq *= (1 + rk[i])
            # 연말(해의 마지막 봉)에 과세
            last_of_year = (i == len(rk) - 1) or (d1[i + 1].year != yr)
            if last_of_year:
                evs = real1.get(yr, [])
                # 주식만 과세: 코인 제외
                gain_krw = 0.0
                for (dd, g, s, ed) in evs:
                    if s in COINS: continue
                    alloc = yr_start_eq[yr] / TOPN
                    # KRW 실현손익 ≈ alloc × g × (환율 변화 포함해야 하나 근사: 청산일 환율로 환산된 alloc 기준)
                    gain_krw += alloc * g
                tax = max(0.0, gain_krw - TAX_DEDUCT) * TAX_RATE
                eq -= tax; tax_paid_total += tax; tax_by_year[yr] = tax
            peak = max(peak, eq); maxdd = max(maxdd, 1 - eq / peak)
            eq_series.append(eq)
        yrs = len(rk) / 252.0
        cagr_at = (eq / acct) ** (1 / yrs) - 1
        cagr_bt = (np.cumprod(1 + rk)[-1]) ** (1 / yrs) - 1
        L.append(f"  계좌 {acct/1e8:.1f}억: 세전 CAGR {cagr_bt*100:.1f}% → 세후 {cagr_at*100:.1f}% (드래그 {-(cagr_at-cagr_bt)*100:.1f}%p/년) · 총 납부세액 {tax_paid_total/1e6:.0f}백만원 · 세후 MDD {maxdd*100:.0f}%")
        top_tax = sorted(tax_by_year.items(), key=lambda kv: -kv[1])[:3]
        L.append(f"    납세 상위 해: " + " · ".join(f"{y} {t/1e6:.0f}백만" for y, t in top_tax if t > 0))
    L.append("")

    # ── 부록: 현금 비중 ────────────────────────────────────────
    inv_arr = []
    for yr in YEARS:
        if yr not in picks: continue
        t0 = pd.Timestamp(f"{yr}-01-01"); t1 = pd.Timestamp(f"{yr}-12-31")
        vals = []
        for s in picks[yr]:
            cc = series[s]; cyr = cc[(cc.index >= t0) & (cc.index <= t1)]
            eN, xN, smaN, use_sma = params_C(s)
            pos, _ = turtle(ohlc[s], eN, xN, smaN, use_sma)
            p = pos.reindex(cyr.index).fillna(0).values
            vals.append(p)
        m = min(len(x) for x in vals)
        inv_arr.append(np.mean([x[:m] for x in vals], axis=0))
    inv_all = np.concatenate(inv_arr)
    cash = 1 - inv_all
    L.append("부록) 현금 비중 (등가중 1/8·포지션 크기 반영)")
    L.append(f"  평균 현금 {cash.mean()*100:.0f}% · 중앙 {np.median(cash)*100:.0f}% · 25/75분위 {np.percentile(cash,25)*100:.0f}/{np.percentile(cash,75)*100:.0f}%")
    for rate in (0.03, 0.04, 0.05):
        L.append(f"  단기 무위험 {rate*100:.0f}% 파킹 시 기대 기여 ≈ +{cash.mean()*rate*100:.1f}%p/년")
    L.append("")
    L.append("정직성: ①은 등가중·연말 일괄과세 근사(실제는 결제일 환율·건별 환산으로 다소 다름) ②갭 체결도 시가 근사(장중 급락 miss 가능)")
    L.append("③생존편향·슬리피지 미반영은 기존 결론과 동일. 숫자의 '방향과 자릿수'를 믿고 소수점은 믿지 말 것.")

    txt = "\n".join(L)
    print(txt)
    open(os.path.join(HERE, "COST_REALITY_LOG.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
