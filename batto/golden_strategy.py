# -*- coding: utf-8 -*-
"""
황금비율 전략 v0 (0부터 새로 작성) — 5도구 통합 타점.
- 유동성 레벨 트리거: 스윕 · Trap(쌍바닥/쌍고점) · 추세선 터치 · 채널 밴드
- 진입 근거(컨플루언스 코어): OB만 / FVG만 / OB+FVG (3등급)
- 방향 = 트리거대로(롱/숏). 시장 게이트 없음.
- 종목: 엔비디아·삼성전자·BTC. 벤치마크: BTC=Crypto F&G<30, 주식=VIX≥30(공포).
- 산출: 타점 + '왜 진입했는지' 도형 시각화.
"""
import os, sys, warnings
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf, requests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(HERE, exist_ok=True)
SYMS = [("엔비디아", "NVDA", "vix"), ("삼성전자", "005930.KS", "vix"), ("BTC", "BTC-USD", "crypto")]
K = 5; W = 40; KSD = 2.0; CW = 5; LOOK = 50; TOL = 0.01; TLPEN = 0.005   # TLPEN=추세선 사이 캔들 관통 허용(0.5% 이상 뚫으면 무효)

# ===================== 데이터 =====================
def load_price(tk):
    fp = os.path.join(HERE, f"px_{tk.replace('.','_')}.csv")
    if not os.path.exists(fp):
        r = yf.download(tk, start="2018-01-01", auto_adjust=True, progress=False)
        if isinstance(r.columns, pd.MultiIndex): r.columns = r.columns.get_level_values(0)
        r = r[["Open","High","Low","Close"]].dropna(); r.columns = ["o","h","l","c"]; r.index.name = "date"
        r.to_csv(fp, encoding="utf-8-sig")
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")

def load_cryptofng():
    fp = os.path.join(HERE, "fng_crypto.csv")
    if not os.path.exists(fp):
        j = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=20).json()
        d = pd.DataFrame(j["data"]); d["date"] = pd.to_datetime(d["timestamp"].astype(int), unit="s").dt.normalize()
        d["fng"] = d["value"].astype(int); d = d[["date","fng"]].set_index("date").sort_index()
        d.to_csv(fp, encoding="utf-8-sig")
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")

def load_vix():
    fp = os.path.join(HERE, "vix.csv")
    if not os.path.exists(fp):
        v = yf.download("^VIX", start="2018-01-01", auto_adjust=True, progress=False)
        if isinstance(v.columns, pd.MultiIndex): v.columns = v.columns.get_level_values(0)
        v = v[["Close"]].dropna(); v.columns = ["vix"]; v.index.name = "date"
        v.to_csv(fp, encoding="utf-8-sig")
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")

# ===================== 스윙 =====================
def swings(h, l):
    n = len(h); sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(K, n-K):
        if h[i] == h[i-K:i+K+1].max(): sh[i] = True
        if l[i] == l[i-K:i+K+1].min(): sl[i] = True
    return sh, sl

# ===================== OB (엥걸프 + BOS, 존=첫캔들 몸통) =====================
OBMULT = 2.0   # 위키 OB1: 둘째 캔들(분출)이 첫 캔들의 2배 이상

def detect_ob(o, h, l, c, sh, sl):
    # 위키 OB1 충실 (윗꼬리 조건 폐기 — 가짜 OB 양산 원인, 2026-06-03 사용자 재지적):
    # 상승 OB: 음봉(c1)→양봉(c2). ① 음봉종가=양봉시가 ② 양봉몸통 > 음봉몸통×2 ③ 양봉종가 > 음봉고가(분출). 존 = 음봉 몸통.
    # 하락 OB: 양봉(c1)→음봉(c2). ① 양봉종가=음봉시가 ② 음봉몸통 > 양봉몸통×2 ③ 음봉종가 < 양봉저가(분출). 존 = 양봉 몸통.
    n = len(o); obs = []; EPS = 0.003
    for p in range(1, n-1):
        c1, c2 = p, p+1
        # 상승 OB (음봉 → 큰 양봉 분출)
        if c[c1] < o[c1] and c[c2] > o[c2] and abs(o[c2]-c[c1]) <= c[c1]*EPS:
            bC1 = o[c1]-c[c1]; bC2 = c[c2]-o[c2]
            if bC2 > bC1*OBMULT and c[c2] > h[c1]:
                obs.append(dict(form=c2, lo=c[c1], hi=o[c1], dir="L"))
        # 하락 OB (양봉 → 큰 음봉 분출)
        if c[c1] > o[c1] and c[c2] < o[c2] and abs(o[c2]-c[c1]) <= c[c1]*EPS:
            bC1 = c[c1]-o[c1]; bC2 = o[c2]-c[c2]
            if bC2 > bC1*OBMULT and c[c2] < l[c1]:
                obs.append(dict(form=c2, lo=o[c1], hi=c[c1], dir="S"))
    return obs

# ===================== FVG (3캔들 갭) =====================
def detect_fvg(o, h, l, c):
    n = len(o); body = np.abs(c-o); fv = []
    for m in range(1, n-1):
        a, b = m-1, m+1
        if body[m] < 2.0*max(body[a], body[b]): continue
        if h[a] < l[b] and c[m] > o[m]: fv.append(dict(form=b, lo=h[a], hi=l[b], dir="L"))
        elif l[a] > h[b] and c[m] < o[m]: fv.append(dict(form=b, lo=h[b], hi=l[a], dir="S"))
    return fv

# ===================== 트리거 (유동성 레벨) =====================
def trig_sweep(h, l, c, sh, sl):
    n = len(c); ev = []
    for s in np.where(sl)[0]:
        lv = l[s]
        for j in range(s+K, min(s+LOOK, n)):
            if c[j] < lv: break
            if l[j] < lv and c[j] > lv: ev.append((j, "L", "스윕", lv)); break
    for s in np.where(sh)[0]:
        hv = h[s]
        for j in range(s+K, min(s+LOOK, n)):
            if c[j] > hv: break
            if h[j] > hv and c[j] < hv: ev.append((j, "S", "스윕", hv)); break
    return ev

def trig_trap(h, l, c, sh, sl):
    n = len(c); G = 20; BR = 0.01; NR = 0.015; ev = []
    for s in np.where(sl)[0]:
        lv = l[s]; j1 = None
        for j in range(s+K, min(s+2*G, n)):
            if c[j] < lv*(1-BR): break
            if l[j] < lv: j1 = j; break
        if j1 is None: continue
        reb = None
        for k in range(j1+1, min(j1+G, n)):
            if c[k] > lv: reb = k; break
            if c[k] < lv*(1-BR): break
        if reb is None: continue
        j2 = None
        for k in range(reb+1, min(reb+G, n)):
            if l[k] <= lv*(1+NR) and l[k] >= l[j1]: j2 = k; break
            if c[k] < lv*(1-BR) or c[k] < l[j1]: break
        if j2 is None: continue
        for k in range(j2+1, min(j2+G, n)):
            if c[k] > lv: ev.append((k, "L", "Trap", lv)); break
            if c[k] < l[j1]: break
    for s in np.where(sh)[0]:
        hv = h[s]; j1 = None
        for j in range(s+K, min(s+2*G, n)):
            if c[j] > hv*(1+BR): break
            if h[j] > hv: j1 = j; break
        if j1 is None: continue
        reb = None
        for k in range(j1+1, min(j1+G, n)):
            if c[k] < hv: reb = k; break
            if c[k] > hv*(1+BR): break
        if reb is None: continue
        j2 = None
        for k in range(reb+1, min(reb+G, n)):
            if h[k] >= hv*(1-NR) and h[k] <= h[j1]: j2 = k; break
            if c[k] > hv*(1+BR) or c[k] > h[j1]: break
        if j2 is None: continue
        for k in range(j2+1, min(j2+G, n)):
            if c[k] < hv: ev.append((k, "S", "Trap", hv)); break
            if c[k] > h[j1]: break
    return ev

def trig_trendline(h, l, c, sh, sl, tol=TOL):
    n = len(c); sli = np.where(sl)[0]; shi = np.where(sh)[0]; ev = []
    for i in range(1, len(sli)):
        s1, s2 = sli[i-1], sli[i]
        if l[s2] <= l[s1] or s2-s1 < 3: continue
        slope = (l[s2]-l[s1])/(s2-s1)
        if any(l[k] < (l[s1]+slope*(k-s1))*(1-TLPEN) for k in range(s1+1, s2)): continue   # 사이 캔들이 선 0.5%+ 관통 → 무효
        for j in range(s2+K, min(s2+LOOK, n)):
            tlv = l[s2]+slope*(j-s2)
            if l[j] <= tlv*(1+tol) and c[j] > tlv and c[j-1] >= tlv*(1-tol): ev.append((j,"L","추세선",tlv)); break
            if c[j] < tlv*(1-tol): break
    for i in range(1, len(shi)):
        s1, s2 = shi[i-1], shi[i]
        if h[s2] >= h[s1] or s2-s1 < 3: continue
        slope = (h[s2]-h[s1])/(s2-s1)
        if any(h[k] > (h[s1]+slope*(k-s1))*(1+TLPEN) for k in range(s1+1, s2)): continue   # 사이 캔들이 선 0.5%+ 관통 → 무효
        for j in range(s2+K, min(s2+LOOK, n)):
            tlv = h[s2]+slope*(j-s2)
            if h[j] >= tlv*(1-tol) and c[j] < tlv and c[j-1] <= tlv*(1+tol): ev.append((j,"S","추세선",tlv)); break
            if c[j] > tlv*(1+tol): break
    return ev

def trig_channel(c_s, h, l, c):
    ma = pd.Series(c).rolling(W).mean().values; sd = pd.Series(c).rolling(W).std().values
    up = ma+KSD*sd; lo = ma-KSD*sd; n = len(c); ev = []; last = {}
    for j in range(W, n):
        if np.isnan(up[j]): continue
        d = None; lv = None
        if l[j] <= lo[j] and c[j] > lo[j]: d, lv = "L", lo[j]
        elif h[j] >= up[j] and c[j] < up[j]: d, lv = "S", up[j]
        if d is None: continue
        if d in last and j-last[d] < 5: continue
        last[d] = j; ev.append((j, d, "채널", lv))
    return ev

# ===================== 컨플루언스 (트리거 + OB/FVG 3등급) =====================
def invalidated(e, c, j):
    """OB/FVG 무효화: 형성(form) 이후 ~ 진입봉(j) 전에 어떤 캔들 종가가
    존을 완전 관통하면 폐기 (위키 무효화 ②). 상승존=lo 하향 / 하락존=hi 상향."""
    for k in range(e["form"]+1, j):
        if e["dir"] == "L" and c[k] < e["lo"]: return True
        if e["dir"] == "S" and c[k] > e["hi"]: return True
    return False


OB_MIN_ATR = 0.3   # OB 최소 크기: 존 폭 >= ATR × 0.3 (극소형 OB 제거)


def _conf_one(j, d, o, h, l, c, e, mode):
    """진입봉 j(방향 d)에 OB/FVG e가 컨플루언스로 유효한가.
    loose : |form-j| ≤ CW (시간 윈도우, 미래/위치 무시 — 구형)
    causal: form ≤ j 이고 j-form ≤ CW (미래참조 제거, 과거 CW일)
    strict: form ≤ j(수명 LOOK 내) + 미관통(무효화 X) + 진입봉이 존 도달(겹침) + 방향정합
            + 진입봉 종가가 존을 완전 관통하지 않음 (꼬리 관통은 허용, 종가 관통은 무효)"""
    if e["dir"] != d: return False
    if mode == "loose":  return abs(e["form"]-j) <= CW
    if mode == "causal": return e["form"] <= j and (j-e["form"]) <= CW
    # strict
    if not (e["form"] <= j and (j-e["form"]) <= LOOK): return False
    if invalidated(e, c, j): return False                       # 이전 봉들이 종가로 관통 → 폐기
    # 진입봉 종가 자체가 존을 뚫으면 무효
    if d == "L" and c[j] < e["lo"]: return False               # 롱인데 종가가 존 아래 = 뚫림
    if d == "S" and c[j] > e["hi"]: return False               # 숏인데 종가가 존 위 = 뚫림
    # 숏: 진입봉 고가(wick)가 존 상단을 뚫으면 무효 (저항 돌파 = 존 소멸)
    if d == "S" and h[j] > e["hi"]: return False
    # 숏: 진입봉 종가가 존 하단 아래 = 이미 가격이 저항존 아래 = 숏 근거 소멸
    if d == "S" and c[j] < e["lo"]: return False
    overlap = (l[j] <= e["hi"]) and (h[j] >= e["lo"])          # 진입봉 꼬리가 존에 닿음
    return overlap


def build_signals(df, conf_mode="loose", tol=TOL):
    o, h, l, c = [df[x].values for x in ["o","h","l","c"]]; n = len(df)
    sh, sl = swings(h, l)
    obs = detect_ob(o, h, l, c, sh, sl); fvs = detect_fvg(o, h, l, c)
    trigs = trig_sweep(h,l,c,sh,sl) + trig_trap(h,l,c,sh,sl) + trig_trendline(h,l,c,sh,sl,tol) + trig_channel(c,h,l,c)
    sigs = []
    for j, d, tname, lv in trigs:
        hasOB = any(_conf_one(j, d, o, h, l, c, e, conf_mode) for e in obs)
        hasFV = any(_conf_one(j, d, o, h, l, c, e, conf_mode) for e in fvs)
        if not (hasOB or hasFV): continue
        grade = "OB+FVG" if (hasOB and hasFV) else ("OB만" if hasOB else "FVG만")
        sigs.append(dict(j=j, dir=d, trig=tname, lv=lv, grade=grade))
    return sigs, obs, fvs


OBC = "#7b3fa0"   # OB = 보라
FVC = "#f2c200"   # FVG = 노랑
YEARS = [2021, 2022, 2023]
UPC, DNC = "#2e8b6f", "#d64545"


def page(pdf, kor, df, sigs, obs, fvs, yr):
    ym = np.asarray(df.index.year == yr)
    if ym.sum() < 20: return
    i0 = int(np.argmax(ym)); i1 = i0 + int(ym.sum()) - 1
    sub = df.iloc[i0:i1+1]; n = len(sub)
    o, h, l, c = [sub[x].values for x in ["o","h","l","c"]]
    fig, ax = plt.subplots(figsize=(16, 8.5))
    for k in range(n):
        col_ = UPC if c[k] >= o[k] else DNC
        ax.plot([k, k], [l[k], h[k]], color=col_, lw=0.8, zorder=4)
        a, b = min(o[k], c[k]), max(o[k], c[k])
        ax.add_patch(Rectangle((k-0.3, a), 0.6, max(b-a, 1e-9), facecolor=col_, edgecolor=col_, lw=0.3, zorder=5))
    drawn = set()
    for s in [x for x in sigs if i0 <= x["j"] <= i1]:
        j = s["j"]; x = j - i0; d = s["dir"]
        # OB 박스(보라) — 위키대로 '첫 캔들(c1=form-1) 몸통'
        for e in obs:
            if e["dir"] == d and abs(e["form"]-j) <= CW and i0 <= e["form"]-1 <= i1 and ("O", e["form"]) not in drawn:
                drawn.add(("O", e["form"])); x0 = (e["form"]-1)-i0
                ax.add_patch(Rectangle((x0-0.4, e["lo"]), (j-(e["form"]-1))+1.4, e["hi"]-e["lo"], facecolor=OBC, alpha=0.30, edgecolor=OBC, lw=0.8, zorder=2))
        # FVG 박스(노랑) — 위키대로 '1번 캔들(form-2) 고가 ~ 3번 저가'
        for e in fvs:
            if e["dir"] == d and abs(e["form"]-j) <= CW and i0 <= e["form"]-2 <= i1 and ("F", e["form"]) not in drawn:
                drawn.add(("F", e["form"])); x0 = (e["form"]-2)-i0
                ax.add_patch(Rectangle((x0-0.4, e["lo"]), (j-(e["form"]-2))+1.4, e["hi"]-e["lo"], facecolor=FVC, alpha=0.30, edgecolor="#b8860b", lw=0.8, zorder=2))
        # 겹칠 때만 '겹침' 표기 (방향은 알아서 판단 → 마커 없음)
        if s["grade"] == "OB+FVG":
            ax.text(x, h[x]*1.035, "겹침", fontsize=8.5, weight="bold", ha="center", va="bottom", color="#d62728", zorder=9)
    ax.set_title(f"[{kor}] {yr}년 · OB/FVG 표시", fontsize=15, weight="bold")
    ax.text(0.005, 1.04, "OB = 보라 띠     FVG = 노랑 띠     '겹침' = OB·FVG 둘 다 (분할매수 자리)",
            transform=ax.transAxes, fontsize=11, va="bottom", ha="left", color="#333")
    ax.set_ylabel("가격"); ax.grid(alpha=0.12)
    t_ = range(0, n, max(1, n//11)); ax.set_xticks(list(t_)); ax.set_xticklabels([sub.index[i].strftime("%m-%d") for i in t_], fontsize=8)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    from collections import Counter
    OUT = os.path.join(HERE, "황금전략_타점_2021_2023.pdf")
    with PdfPages(OUT) as pdf:
        f0 = plt.figure(figsize=(16, 9))
        f0.text(0.5, 0.6, "황금비율 전략 — 5도구 통합 타점", ha="center", fontsize=24, weight="bold", color="#1F3A5F")
        f0.text(0.5, 0.5, "엔비디아·삼성전자·BTC · 2021~2023", ha="center", fontsize=13)
        f0.text(0.5, 0.42, "OB = 보라 띠   FVG = 노랑 띠   '겹침' = OB·FVG 둘 다 (분할매수 자리)", ha="center", fontsize=11, color="#555")
        pdf.savefig(f0); plt.close(f0)
        for kor, tk, bench in SYMS:
            df = load_price(tk); sigs, obs, fvs = build_signals(df)
            g = Counter(s["grade"] for s in sigs)
            print(f"[{kor}] 타점 {len(sigs)}  등급{dict(g)}")
            for yr in YEARS:
                page(pdf, kor, df, sigs, obs, fvs, yr)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
