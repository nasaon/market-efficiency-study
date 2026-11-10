# -*- coding: utf-8 -*-
"""정제 전략 구성 — 추세선을 A스펙(K12+45+완화)으로 교체 후, 여러 필터 조합 비교.
양수 핵심(추세선A · OB+FVG겹침)만 남기는 최적 정제 전략 도출. 5종목 풀, N일 방향수익."""
import os, sys, warnings
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
warnings.filterwarnings("ignore")
import numpy as np
import golden_strategy as G
import trend_compare as TC

SYMS = [("BTC","BTC-USD"), ("S&P","^GSPC"), ("코스피","^KS11"), ("삼성","005930.KS"), ("하이닉스","000660.KS")]


def trendA(o, h, l, c):
    """A스펙 추세선(K12+45+완화) → (j, d, '추세선', tlv)."""
    out = []
    for j, d, s1, s2 in TC.tl_swings(h, l, c, 0.003, 45, 12):
        if d == "L":
            slope = (l[s2]-l[s1])/(s2-s1); tlv = l[s2]+slope*(j-s2)
        else:
            slope = (h[s2]-h[s1])/(s2-s1); tlv = h[s2]+slope*(j-s2)
        out.append((j, d, "추세선", tlv))
    return out


def sigs_refined(df):
    """build_signals와 동일하되 추세선만 A스펙으로 교체."""
    o, h, l, c = [df[x].values for x in ["o","h","l","c"]]; sh, sl = G.swings(h, l)
    obs = G.detect_ob(o, h, l, c, sh, sl); fvs = G.detect_fvg(o, h, l, c)
    trigs = G.trig_sweep(h,l,c,sh,sl) + G.trig_trap(h,l,c,sh,sl) + trendA(o,h,l,c) + G.trig_channel(c,h,l,c)
    sigs = []
    for j, d, tn, lv in trigs:
        hasOB = any(G._conf_one(j, d, o, h, l, c, e, "strict") for e in obs)
        hasFV = any(G._conf_one(j, d, o, h, l, c, e, "strict") for e in fvs)
        if not (hasOB or hasFV): continue
        grade = "OB+FVG" if (hasOB and hasFV) else ("OB만" if hasOB else "FVG만")
        sigs.append(dict(j=j, dir=d, trig=tn, grade=grade))
    return sigs


def stat(rs):
    a = np.array(rs); n = len(a)
    if n < 3: return n, np.nan, np.nan, np.nan
    m = a.mean(); t = m/(a.std(ddof=1)/np.sqrt(n)); wr = (a > 0).mean()*100
    return n, m, wr, t


FILTERS = {
    "전체(A추세선 포함)":   lambda s: True,
    "추세선A 단독":         lambda s: s["trig"] == "추세선",
    "OB+FVG겹침 단독":      lambda s: s["grade"] == "OB+FVG",
    "추세선A ∪ 겹침":       lambda s: s["trig"] == "추세선" or s["grade"] == "OB+FVG",
    "스윕·숏 제외":         lambda s: not (s["trig"] == "스윕" and s["dir"] == "S"),
    "손실원2개 제외":       lambda s: not ((s["trig"] == "스윕" and s["dir"] == "S")),  # +추세선롱은 A로 이미 정제
}


def main():
    # 종목별 sigs 미리 계산
    data = []
    for kor, tk in SYMS:
        df = G.load_price(tk); data.append((df, df["c"].values, sigs_refined(df)))
    for N in [10, 20]:
        print(f"\n{'='*54}\n■ 보유 {N}일 · 5종목 풀 · 추세선=A스펙 교체본\n{'='*54}")
        print(f"  {'정제 필터':22s}{'타점':>6s}{'평균%':>8s}{'승률%':>7s}{'t값':>7s}")
        for fname, ff in FILTERS.items():
            rs = []
            for df, c, sigs in data:
                for s in sigs:
                    if not ff(s): continue
                    j = s["j"]
                    if j+N >= len(c): continue
                    rs.append((1 if s["dir"] == "L" else -1)*(c[j+N]/c[j]-1)*100)
            n, m, wr, t = stat(rs)
            f = lambda v, p: f"{v:{p}}" if v == v else "   -  "
            print(f"  {fname:22s}{n:6d}{f(m,'+8.2f')}{f(wr,'7.0f')}{f(t,'+7.2f')}")


if __name__ == "__main__":
    main()
