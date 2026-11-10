# -*- coding: utf-8 -*-
"""
Batto Model v1 — 백테스트
규칙:
  · FVG: ATR14×0.7 이상 갭만 유효 (작은 갭 제거)
  · OB: 분출 배수 2.0 유지
  · 추세선: 30봉 vs 45봉 비교
  · 스윕: 롱만 (숏 삭제)
  · Trap: 완전 삭제
  · 채널: 유지
  · strict 컨플루언스
S&P500 50종목 × 무작위 4년도 × 보유 20일 방향수익
"""
import os, sys, random, warnings
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import golden_strategy as G
import trend_compare as TC

random.seed(42); np.random.seed(42)
N = 20
HERE = os.path.dirname(os.path.abspath(__file__))
SNP = open(os.path.join(HERE, "snp50.txt")).read().strip().split()
FVG_ATR = 0.7   # Batto v1 FVG 최소 갭
OB_ATR  = 0.3   # Batto v1 OB 최소 존 폭 (극소형 OB 제거)


def load(tk):
    fp = os.path.join(HERE, f"px_{tk.replace('.','_').replace('-','_')}.csv")
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")


def atr14(h, l, c):
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    a = pd.Series(tr).rolling(14).mean().fillna(pd.Series(tr).expanding().mean()).values
    return np.concatenate([[a[0]], a])


def detect_fvg_filtered(o, h, l, c, atr_arr):
    """FVG + ATR 최소 갭 필터 (Batto v1)."""
    n = len(o); body = np.abs(c-o); fv = []
    for m in range(1, n-1):
        a, b = m-1, m+1
        if body[m] < 2.0*max(body[a], body[b]): continue
        if h[a] < l[b] and c[m] > o[m]:
            gap = l[b]-h[a]
            if gap >= FVG_ATR*atr_arr[m]: fv.append(dict(form=b, lo=h[a], hi=l[b], dir="L"))
        elif l[a] > h[b] and c[m] < o[m]:
            gap = l[a]-h[b]
            if gap >= FVG_ATR*atr_arr[m]: fv.append(dict(form=b, lo=h[b], hi=l[a], dir="S"))
    return fv


def detect_ob_filtered(o, h, l, c, atr_arr):
    """OB + ATR 최소 존 폭 필터 (Batto v1 — 극소형 OB 제거)."""
    sh, sl = G.swings(h, l)
    raw = G.detect_ob(o, h, l, c, sh, sl)
    return [e for e in raw if (e["hi"]-e["lo"]) >= OB_ATR*atr_arr[e["form"]]]


def trig_fake_reentry(h, l, c, obs_filtered, fvs_filtered):
    """AMD형 Fake-Down Reentry 규칙:
    OB/FVG 존을 꼬리(저가)로 뚫고 내려갔지만 종가는 존 위로 복귀 = 가짜 하락
    → 다음 봉에서 롱 진입. (롱 방향만, 숏 미러는 별도)"""
    n = len(c); ev = []
    for e in obs_filtered + fvs_filtered:
        if e["dir"] != "L": continue
        zone_lo, zone_hi = e["lo"], e["hi"]
        for j in range(e["form"]+1, min(e["form"]+G.LOOK, n-1)):
            if G.invalidated(e, c, j): break             # 이전에 완전 관통 → 중단
            # 꼬리가 존 하단을 뚫었지만 종가는 존 위 복귀 = Fake Down
            if l[j] < zone_lo and c[j] >= zone_lo:
                # 다음 봉이 진입봉
                entry = j+1
                if entry < n:
                    ev.append((entry, "L", "Fake재진입", zone_lo))
                break
    return ev


def trig_sweep_long_only(h, l, c, sh, sl):
    """스윕 롱만 반환 (숏 제거)."""
    return [(j, d, tn, lv) for j, d, tn, lv in G.trig_sweep(h, l, c, sh, sl) if d == "L"]


def build_batto(df, tl_minbars, include_fake=False):
    """Batto v1 신호 생성.
    · OB: ATR×0.3 최소 폭 필터 적용
    · FVG: ATR×0.7 최소 갭 필터 적용
    · 진입봉 종가가 존을 관통하면 컨플루언스 무효
    · include_fake=True 시 Fake재진입 트리거 추가"""
    o, h, l, c = [df[x].values for x in ["o","h","l","c"]]
    atr = atr14(h, l, c)
    sh, sl = G.swings(h, l)
    obs = detect_ob_filtered(o, h, l, c, atr)
    fvs = detect_fvg_filtered(o, h, l, c, atr)
    tl_trigs = [(j, d, "추세선", (l if d=="L" else h)[j])
                for j, d, s1, s2 in TC.tl_swings(h, l, c, 0.003, tl_minbars, 12)]
    trigs = (trig_sweep_long_only(h, l, c, sh, sl) +
             tl_trigs +
             G.trig_channel(c, h, l, c))
    if include_fake:
        trigs += trig_fake_reentry(h, l, c, obs, fvs)
    sigs = []
    for j, d, tn, lv in trigs:
        hasOB = any(G._conf_one(j,d,o,h,l,c,e,"strict") for e in obs)
        hasFV = any(G._conf_one(j,d,o,h,l,c,e,"strict") for e in fvs)
        if tn == "Fake재진입":      # Fake는 컨플루언스가 이미 내장
            hasOB = any(G._conf_one(j,d,o,h,l,c,e,"strict") for e in obs)
            hasFV = any(G._conf_one(j,d,o,h,l,c,e,"strict") for e in fvs)
            if not (hasOB or hasFV): continue
        elif not (hasOB or hasFV): continue
        grade = "OB+FVG" if (hasOB and hasFV) else ("OB만" if hasOB else "FVG만")
        sigs.append(dict(j=j, dir=d, trig=tn, grade=grade))
    return sigs


def rand_years(idx, k=4):
    ys = list(range(idx.year.min()+1, idx.year.max()))
    return random.sample(ys, min(k, len(ys))) if len(ys) >= k else ys


def stat(rs):
    a = np.array(rs); n = len(a)
    if n < 5: return n, np.nan, np.nan, np.nan
    m = a.mean(); t = m/(a.std(ddof=1)/np.sqrt(n))
    return n, m, (a > 0).mean()*100, t


def run(tl_mb, label):
    rs_all = []; byT = {}; byG = {}; byD = {}
    for tk in SNP:
        df = load(tk); c = df["c"].values
        sigs = build_batto(df, tl_mb)
        yrs = rand_years(df.index)
        for yr in yrs:
            ym = df.index.year == yr; a = int(np.argmax(ym)); b = a+int(ym.sum())-1
            for s in sigs:
                j = s["j"]
                if not (a <= j <= b): continue
                if j+N >= len(c): continue
                r = (1 if s["dir"] == "L" else -1)*(c[j+N]/c[j]-1)*100
                rs_all.append(r)
                byT.setdefault(s["trig"], []).append(r)
                byG.setdefault(s["grade"], []).append(r)
                byD.setdefault("롱" if s["dir"] == "L" else "숏", []).append(r)
    print(f"\n{'='*60}")
    print(f"■ Batto Model v1 — {label}")
    print(f"{'='*60}")
    print(f"  {'구분':20s}{'타점':>6s}{'평균%':>8s}{'승률%':>7s}{'t':>7s}")
    n,m,wr,t = stat(rs_all)
    f = lambda v,p: f"{v:{p}}" if v==v else "   -  "
    print(f"  {'[전체]':20s}{n:6d}{f(m,'+8.2f')}{f(wr,'7.0f')}{f(t,'+7.2f')}")
    print(f"  -- 트리거별 --")
    for k in ["스윕","추세선","채널"]:
        if k in byT:
            n,m,wr,t = stat(byT[k])
            print(f"  {k:20s}{n:6d}{f(m,'+8.2f')}{f(wr,'7.0f')}{f(t,'+7.2f')}")
    print(f"  -- 방향별 --")
    for k in ["롱","숏"]:
        if k in byD:
            n,m,wr,t = stat(byD[k])
            print(f"  {k:20s}{n:6d}{f(m,'+8.2f')}{f(wr,'7.0f')}{f(t,'+7.2f')}")
    print(f"  -- 등급별 --")
    for k in ["OB만","FVG만","OB+FVG"]:
        if k in byG:
            n,m,wr,t = stat(byG[k])
            print(f"  {k:20s}{n:6d}{f(m,'+8.2f')}{f(wr,'7.0f')}{f(t,'+7.2f')}")
    return stat(rs_all)


if __name__ == "__main__":
    print(f"Batto Model v1 백테스트")
    print(f"S&P500 {len(SNP)}종목 × 무작위 4년도 × 보유{N}일")
    print(f"FVG 최소 갭 ATR×{FVG_ATR} · OB 배수2.0 · 스윕롱만 · Trap없음")

    r30 = run(30, "추세선 30봉")
    r45 = run(45, "추세선 45봉")

    print(f"\n{'='*60}")
    print(f"■ 추세선 30봉 vs 45봉 직접 비교")
    print(f"{'='*60}")
    f = lambda v,p: f"{v:{p}}" if v==v else "  -  "
    for lb,(n,m,wr,t) in [("30봉",r30),("45봉",r45)]:
        print(f"  {lb}: {n}타점  평균{f(m,'+.2f')}%  승률{f(wr,'.0f')}%  t={f(t,'+.2f')}")
