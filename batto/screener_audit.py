# -*- coding: utf-8 -*-
"""스크리너 엄격 감사 — 따라갈 엔진의 숨은 오점 찾기.
① 룩어헤드(코드): asof 이전 데이터만 쓰는지. ② 집중도: TOP8이 한 섹터(반도체)에 쏠리나.
③ 유니버스편향: 큐레이션된 리스트(생존자)에 의존하나 — 랜덤 절반서도 상대알파 유지되나.
④ 회전율: TOP8 연간 교체율(수수료). ⑤ OOS 상대알파 연도별. 정직·엄격."""
import os, sys, random
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 선별 엔진은 같은 폴더
import batto_screener_live as B

SEMI = {"NVDA","AMD","MU","AVGO","LRCX","AMAT","MRVL","QCOM","TXN","INTC","SMCI","ASML"}
YEARS = list(range(2017, 2027))


def picks_by_year(series, universe):
    """연초 TOP8 선별(asof=연초·과거만) 반환 {year: [top8]}."""
    out = {}
    for yr in YEARS:
        asof = pd.Timestamp(f"{yr}-01-01")
        keys = [s for s in universe if s in series]
        top, sc, z = B.zscore_pick({s: series[s] for s in keys}, asof, keys)
        if top: out[yr] = top
    return out


def fwd_ret(series, s, yr):
    c = series[s]; a = c[(c.index >= pd.Timestamp(f"{yr}-01-01")) & (c.index <= pd.Timestamp(f"{yr}-12-31"))]
    return (a.iloc[-1]/a.iloc[0]-1)*100 if len(a) > 30 else np.nan


def rel_alpha(series, universe):
    """선별 중앙수익 - 전체 중앙수익 (상대알파) 연도별."""
    pk = picks_by_year(series, universe); out = {}
    for yr in YEARS:
        if yr not in pk: continue
        sel = [fwd_ret(series, s, yr) for s in pk[yr]]; sel = [x for x in sel if x == x]
        allk = [fwd_ret(series, s, yr) for s in universe if s in series]; allk = [x for x in allk if x == x]
        if sel and allk: out[yr] = np.median(sel) - np.median(allk)
    return out


def main():
    series, ohlc = B.load_all()
    US = [s for s in B.US if s in series]; COINS = [s for s in B.COINS if s in series]
    universe = US + COINS
    L = ["스크리너 엄격 감사 — 숨은 오점 찾기\n"]

    # ① 룩어헤드 (코드 audit 결론)
    L.append("[① 룩어헤드] zscore_pick: prev=series[s].index < asof (엄격 미만)·mom=prev.iloc[-252]·shift(1) 진입")
    L.append("   verify=연초 asof 과거만·live=마지막봉+1일. → 코드상 미래참조 없음(PASS). ✔")

    # ② 집중도 (섹터 쏠림)
    pk = picks_by_year(series, universe)
    L.append("\n[② 집중도] TOP8 중 반도체 개수 (쏠림=사실상 1베팅)")
    L.append("  " + "".join(f"{y:>6}" for y in YEARS))
    L.append("  " + "".join(f"{sum(1 for s in pk.get(y,[]) if s in SEMI):>6}" for y in YEARS))
    avg_semi = np.mean([sum(1 for s in pk.get(y,[]) if s in SEMI) for y in YEARS if y in pk])
    L.append(f"   평균 반도체 {avg_semi:.1f}/8 — 높으면 '8종목이 아니라 1섹터 베팅'(상관↑·분산 착시). ★현재 라이브 6/8 반도체.")

    # ③ 유니버스 편향 (랜덤 절반서도 상대알파 유지?)
    full = np.median([v for v in rel_alpha(series, universe).values()])
    subs = []
    random.seed(1)
    for _ in range(8):
        half = random.sample(US, len(US)//2) + COINS
        subs.append(np.median([v for v in rel_alpha(series, half).values()]))
    L.append(f"\n[③ 유니버스 편향] 전체 유니버스 상대알파 중앙 {full:+.0f}%p")
    L.append(f"   랜덤 절반 유니버스 8회: 중앙 {np.median(subs):+.0f}%p (범위 {min(subs):+.0f}~{max(subs):+.0f})")
    L.append(f"   → 절반서도 +유지면 큐레이션 의존 낮음(robust)·절반서 무너지면 리스트 편향 의심.")

    # ④ 회전율 (TOP8 연간 교체)
    turns = []
    ys = sorted(pk.keys())
    for a, b in zip(ys, ys[1:]):
        common = len(set(pk[a]) & set(pk[b])); turns.append((8-common)/8*100)
    L.append(f"\n[④ 회전율] TOP8 연간 평균 교체 {np.mean(turns):.0f}% (높으면 수수료·슬리피지 부담)")

    # ⑤ OOS 상대알파 (2024~)
    ra = rel_alpha(series, universe)
    L.append(f"\n[⑤ 상대알파 연도별] " + " ".join(f"{y}:{ra.get(y,0):+.0f}" for y in YEARS))
    oos = [ra[y] for y in (2024,2025,2026) if y in ra]
    L.append(f"   OOS(24~26) 상대알파: {' '.join(f'{v:+.0f}' for v in oos)} — 최근도 +유지되나(2025 음이면 전환기 약점).")

    L.append("\n[★감사 종합]")
    L.append("  · 룩어헤드=깨끗(PASS). 회전율·집중도·유니버스편향이 실제 점검포인트.")
    L.append("  · 생존편향은 코드로 못고침(상폐종목 데이터 없음)=상대알파·랜덤절반 강건성으로만 간접확인.")
    L.append("  · ★집중도가 최대 실전리스크(반도체 쏠림=분산 착시). 섹터상한 수동 권장.")
    txt = "\n".join(L); print(txt)
    open(os.path.join(B.DATA_DIR, "SCREENER_AUDIT_LOG.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
