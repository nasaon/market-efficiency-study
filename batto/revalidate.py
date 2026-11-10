# -*- coding: utf-8 -*-
"""터틀·VIX 재검증 — 3개 유니버스(S&P500·나스닥100·러셀=S&P600소형)로. 손픽 9종목 편향 제거.
① 스크리너+터틀 워크포워드: 상대알파 + 선별종목 터틀 중앙 Calmar (터틀 작동?)
② VIX 바닥신호: 레벨(≥28)/깊이(패닉40+)/백워데이션(VIX>VIX3M) 신호후 1년 수익 (VIX 작동?)
각 유니버스서 결론(터틀 청산·VIX 깊이/백워데이션)이 유지되나. 종가/OHLC 보유분만. 생존편향 잔존."""
import os, sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 선별 엔진은 같은 폴더
import batto_screener_live as B

HERE = B.DATA_DIR
YEARS = list(range(2017, 2027))
MIN_HIST = 504; TOPN = 8


def list_of(fn):
    p = os.path.join(HERE, fn)
    return [t.strip() for t in open(p, encoding="utf-8").read().split(",") if t.strip()] if os.path.exists(p) else []


def load_ohlc(tks):
    R = {}
    for tk in tks:
        fp = os.path.join(HERE, f"yf_{tk}.csv")
        if os.path.exists(fp):
            try: R[tk] = pd.read_csv(fp, parse_dates=["date"], index_col="date")[["o","h","l","c"]].dropna()
            except Exception: pass
    return R


def turtle_cal(df):
    """단일종목 터틀(채택안·30/10/SMA100) 전체기간 Calmar."""
    pos = B.turtle_pos(df, 30, 10, 100).values; c = df["c"].values
    cret = np.concatenate([[0], c[1:]/c[:-1]-1])
    net = np.concatenate([[0], pos[:-1]])*cret - np.abs(np.diff(pos,prepend=0))*0.0005
    eq = np.cumprod(1+np.nan_to_num(net)); pk = np.maximum.accumulate(eq)
    mdd = (1-eq/pk).max()*100; yrs=len(eq)/252
    cagr = eq[-1]**(1/yrs)-1 if eq[-1]>0 else -1
    return cagr/(mdd/100) if mdd>0 else np.nan


def screener_turtle(ohlc):
    """워크포워드: 연초 TOP8 선별 → 그해 터틀수익·상대알파·선별터틀Calmar."""
    series = {k: v["c"] for k, v in ohlc.items()}
    alphas=[]; tcals=[]
    for yr in YEARS:
        asof = pd.Timestamp(f"{yr}-01-01")
        top, sc, z = B.zscore_pick(series, asof, list(series.keys()))
        if not top: continue
        def fwd(s):
            a=series[s][(series[s].index>=asof)&(series[s].index<=pd.Timestamp(f"{yr}-12-31"))]
            return (a.iloc[-1]/a.iloc[0]-1)*100 if len(a)>30 else np.nan
        sel=[fwd(s) for s in top]; sel=[x for x in sel if x==x]
        univf=[fwd(s) for s in series]; univf=[x for x in univf if x==x]
        if sel and univf: alphas.append(np.median(sel)-np.median(univf))
        for s in top:
            cyr = ohlc[s][(ohlc[s].index>=asof-pd.Timedelta(days=400))&(ohlc[s].index<=pd.Timestamp(f"{yr}-12-31"))]
            if len(cyr)>150:
                cc=cyr["c"].values; pos=B.turtle_pos(cyr,30,10,100).values
                cr=np.concatenate([[0],cc[1:]/cc[:-1]-1]); m=(cyr.index>=asof)
                net=(np.concatenate([[0],pos[:-1]])*cr)[m]
                if net.sum()==net.sum(): tcals.append(net.sum()*100)
    return (np.median(alphas) if alphas else np.nan), (np.median(tcals) if tcals else np.nan)


def vix_signals(ohlc, vix, v3):
    """유니버스 종목들 신호후 1년 중앙수익 — 레벨/패닉40+/백워데이션."""
    B_={'레벨28':[], '패닉40+':[], '백워데이션':[]}
    for tk, df in ohlc.items():
        c=df["c"]; r=B.rsi(c.values,14)
        vv=vix.reindex(df.index,method="ffill"); v3v=v3.reindex(df.index,method="ffill")
        for i in range(len(c)-260):
            if r[i]>34: continue
            lvl=vv.iloc[i]>=28; pan=vv.iloc[i]>=40; bw=(vv.iloc[i]>v3v.iloc[i]) and not np.isnan(v3v.iloc[i])
            if not (lvl or bw): continue
            fwd=c.iloc[i+252]/c.iloc[i]-1
            if pan: B_['패닉40+'].append(fwd)
            if lvl: B_['레벨28'].append(fwd)
            if bw: B_['백워데이션'].append(fwd)
    return {k:(len(v), np.median(v)*100 if v else np.nan, (np.array(v)>0).mean()*100 if v else 0) for k,v in B_.items()}


def main():
    vix=B.load_vix(); v3=pd.read_csv(os.path.join(HERE,"vix3m.csv"),parse_dates=["date"],index_col="date")["vix3m"]
    univs = {"S&P500": list_of("_sp500_ok.txt"), "나스닥100": list_of("_nq100.txt"), "러셀(S&P600)": list_of("_sp600_ok.txt")}
    L=["터틀·VIX 3개 유니버스 재검증 (손픽 9종목 편향 제거·2017~2026)\n"]
    L.append("[① 스크리너+터틀 워크포워드]")
    L.append(f"  {'유니버스':14s}{'종목수':>6s}{'상대알파중앙':>11s}{'선별터틀수익중앙':>15s}")
    ohlc_cache={}
    for nm, tks in univs.items():
        oh = load_ohlc(tks); ohlc_cache[nm]=oh
        a, tc = screener_turtle(oh)
        L.append(f"  {nm:14s}{len(oh):>6d}{a:>+10.0f}%p{tc:>+14.0f}%")
    L.append("  → 상대알파 +·선별터틀 + 유지되면 '스크리너+터틀'이 유니버스 무관 작동(robust).")
    L.append("\n[② VIX 바닥신호 — 신호후 1년 중앙수익%(승률%)]")
    L.append(f"  {'유니버스':14s}{'레벨28':>16s}{'패닉40+':>16s}{'백워데이션':>16s}")
    for nm in univs:
        v = vix_signals(ohlc_cache[nm], vix, v3)
        def cell(k): return f"{v[k][1]:+.0f}%({v[k][2]:.0f}%)"
        L.append(f"  {nm:14s}{cell('레벨28'):>16s}{cell('패닉40+'):>16s}{cell('백워데이션'):>16s}")
    L.append("  → 깊을수록(패닉)·백워데이션서 수익/승률↑ 유지되면 VIX 결론 robust(유니버스 무관).")
    L.append("\n  ※ 러셀 소형은 생존편향 최악·모멘텀 약. 종가/OHLC 보유분만. 슬리피지/수수료 미반영.")
    txt="\n".join(L); print(txt)
    open(os.path.join(HERE,"REVALIDATE_LOG.md"),"w",encoding="utf-8").write(txt+"\n")


if __name__=="__main__":
    main()
