# -*- coding: utf-8 -*-
"""숏 타점 100R 진화 — 엄격·정직 평가. 하락필터(종가<SMA200) 고정 + 과매수 트리거 6종 진화.
★다중 하락장 OOS: Train 하락장(2018Q4·2020코로나)서 진화 → Test 하락장(2022·홀드아웃) 검증.
★불장(19·21·23·24)서 새면(-5%↓) 자동 기각 = '하락장 전용 알파'만 통과.
100R '조금이라도 좋으면 채택'. 정직성: 표본작음·숏비용/차입비 미반영·마진필요. 미래참조 없음."""
import os, sys, random, copy
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import turtle_exit_evolve as EV
import batto_screener_live as B

NAMES = ["SPY","QQQ","AAPL","MSFT","AMD","MU","AVGO","META","TSLA","GOOGL","AMZN"]
TRAIN_BEAR = [("18Q4","2018-10-01","2019-01-01"), ("20코로나","2020-02-15","2020-05-01")]
TEST_BEAR  = [("22약세","2022-01-01","2023-01-01")]
BULLS = [("19","2019-01-01","2020-01-01"),("21","2021-01-01","2022-01-01"),
         ("23","2023-01-01","2024-01-01"),("24","2024-01-01","2025-01-01")]
TRIGS = ["rsi2","rsi14","stoch","boll","willr","ext"]
FEE = 0.0005


def prep(df):
    h,l,c = df["h"].values, df["l"].values, df["c"].values; n=len(c)
    d={"h":h,"l":l,"c":c,"idx":df.index,"n":n}
    d["rsi2"]=B.rsi(c,2); d["rsi14"]=B.rsi(c,14)
    hh14=pd.Series(h).rolling(14).max().values; ll14=pd.Series(l).rolling(14).min().values
    d["stoch"]=100*(c-ll14)/np.where(hh14-ll14==0,np.nan,hh14-ll14)
    d["willr"]=-100*(hh14-c)/np.where(hh14-ll14==0,np.nan,hh14-ll14)  # -100~0, 0=과매수
    s20=pd.Series(c).rolling(20).mean().values; std20=pd.Series(c).rolling(20).std().values
    d["bollz"]=(c-s20)/np.where(std20==0,np.nan,std20)               # 몇 표준편차 위
    s5=pd.Series(c).rolling(5).mean().values
    d["ext"]=c/np.where(s5==0,np.nan,s5)-1                            # SMA5 대비 과확장%
    d["sma200"]=pd.Series(c).rolling(200).mean().values
    for m in range(3,11): d[f"sma{m}"]=pd.Series(c).rolling(m).mean().values
    d["atr"]=EV.S.atr(h,l,c,20)
    return d


def trig_ok(D,i,trig,th):
    v={"rsi2":D["rsi2"][i]>=th,"rsi14":D["rsi14"][i]>=th,"stoch":D["stoch"][i]>=th,
       "boll":D["bollz"][i]>=th,"willr":D["willr"][i]>=-th,"ext":D["ext"][i]>=th}[trig]
    return bool(v==v and v)


def short_ret(D,g,lo,hi):
    c=D["c"];h=D["h"];n=D["n"];s200=D["sma200"];sC=D[f"sma{g['coverMA']}"];atr=D["atr"]
    pos=0.0;ent=0.0;stop=0.0;held=0;arr=np.zeros(n)
    for i in range(201,n):
        if pos==0:
            if not np.isnan(s200[i]) and c[i]<s200[i] and trig_ok(D,i,g["trig"],g["th"]):
                pos=-1.0;ent=c[i];stop=c[i]+g["slN"]*atr[i];held=0
        else:
            held+=1
            if h[i]>=stop or (not np.isnan(sC[i]) and c[i]<sC[i]) or held>=g["maxd"]: pos=0.0
        arr[i]=pos
    cret=np.concatenate([[0],c[1:]/c[:-1]-1])
    net=np.concatenate([[0],arr[:-1]])*cret-np.abs(np.diff(arr,prepend=0))*FEE
    eq=pd.Series(np.cumprod(1+np.nan_to_num(net)),index=D["idx"])
    e=eq[(eq.index>=pd.Timestamp(lo))&(eq.index<pd.Timestamp(hi))]
    if len(e)<20: return None
    return (e.iloc[-1]/e.iloc[0]-1)*100


def wins_med(DATA,g,wins):
    out=[]
    for wn,lo,hi in wins:
        vs=[short_ret(D,g,lo,hi) for D in DATA]; vs=[v for v in vs if v is not None]
        out.append(np.median(vs) if vs else np.nan)
    return out


def fitness(DATA,g):
    tr=wins_med(DATA,g,TRAIN_BEAR); bl=wins_med(DATA,g,BULLS)
    tr=[v for v in tr if v==v]; bl=[v for v in bl if v==v]
    if not tr: return -99
    bull_worst=min(bl) if bl else 0
    base=float(np.mean(tr))
    return base-50 if bull_worst<-5 else base   # 불장 -5%↓면 기각(휴면 아님)


def default_th(trig):
    return {"rsi2":90,"rsi14":72,"stoch":85,"boll":1.6,"willr":15,"ext":0.06}[trig]


def clampth(trig,th):
    rng={"rsi2":(80,98),"rsi14":(60,85),"stoch":(75,97),"boll":(0.8,2.8),"willr":(5,35),"ext":(0.02,0.15)}[trig]
    return round(min(max(th,rng[0]),rng[1]), 3 if trig in ("boll","ext") else 0)


def mutate(g):
    g=copy.deepcopy(g); op=random.random()
    if op<0.4: g["th"]=clampth(g["trig"],g["th"]*random.uniform(0.85,1.15))
    elif op<0.6: g["trig"]=random.choice(TRIGS); g["th"]=default_th(g["trig"])
    elif op<0.75: g["coverMA"]=random.randint(3,10)
    elif op<0.9: g["maxd"]=random.randint(3,15)
    else: g["slN"]=round(random.uniform(1.5,3.5),2)
    return g


SEEDS=[{"trig":t,"th":default_th(t),"coverMA":5,"maxd":10,"slN":2.0} for t in TRIGS]


def gstr(g): return f"{g['trig']}≥{g['th']}·커버SMA{g['coverMA']}·{g['maxd']}일·{g['slN']}N"


def main():
    rounds=100
    for a in sys.argv[1:]:
        if a.startswith("--rounds="): rounds=int(a.split("=")[1])
    random.seed(42); np.random.seed(42)
    print("데이터 로드·지표 사전계산...")
    DATA=[]
    for tk in NAMES:
        try: DATA.append(prep(pd.read_csv(os.path.join(EV.HERE,f"yf_{tk}.csv"),parse_dates=["date"],index_col="date")))
        except Exception as e: print(f"  {tk} 스킵 {e}")
    L=["숏 타점 100R 진화 — 다중 하락장 OOS 엄격평가 (하락필터 종가<SMA200 고정)",
       f"미국 {len(DATA)}종목·Train하락장(18Q4·20코로나)진화→Test하락장(22)검증·불장4개 휴면검증\n",
       "[시드 6종] Train하락장평균 / Test(22) / 불장최악"]
    arch=[]
    for sd in SEEDS:
        tr=np.mean([v for v in wins_med(DATA,sd,TRAIN_BEAR) if v==v]); te=wins_med(DATA,sd,TEST_BEAR)[0]
        bw=min([v for v in wins_med(DATA,sd,BULLS) if v==v])
        arch.append({"g":sd,"f":fitness(DATA,sd)})
        L.append(f"  {gstr(sd):34s} {tr:>6.1f}% /{te:>6.1f}% / {bw:>5.1f}%")
    arch.sort(key=lambda x:-x["f"]); champ=copy.deepcopy(arch[0]); cg=champ["g"]
    L.append(f"\n[100R 진화] 시작챔프 {gstr(cg)} fit {champ['f']:.1f}")
    acc=0
    for rd in range(1,rounds+1):
        cand=mutate(cg); f=fitness(DATA,cand)
        if f>champ["f"]+1e-6:
            champ={"g":cand,"f":f};cg=cand;acc+=1
            te=wins_med(DATA,cand,TEST_BEAR)[0]
            L.append(f"  R{rd:3d} ★fit {f:5.1f} Test(22) {te:5.1f}% | {gstr(cand)}")
    # 최종 엄격판정
    ctr=wins_med(DATA,cg,TRAIN_BEAR); cte=wins_med(DATA,cg,TEST_BEAR)[0]; cbl=wins_med(DATA,cg,BULLS)
    # 22 종목별 승률
    v22=[short_ret(D,cg,"2022-01-01","2023-01-01") for D in DATA]; v22=[v for v in v22 if v is not None]
    win22=sum(1 for v in v22 if v>0)
    L.append(f"\n[★엄격 최종판정] 챔피언 {gstr(cg)}")
    L.append(f"  Train 하락장: "+" ".join(f"{w[0]}{v:+.0f}%" for w,v in zip(TRAIN_BEAR,ctr)))
    L.append(f"  ★Test 하락장(홀드아웃 2022): {cte:+.0f}% · 종목승 {win22}/{len(v22)}")
    L.append(f"  불장(휴면 확인): "+" ".join(f"{w[0]}{v:+.0f}%" for w,v in zip(BULLS,cbl)))
    ok = cte>0 and win22>len(v22)*0.5 and min(cbl)>-5
    L.append(f"  ★판정: {'통과 — OOS 하락장 +·종목과반·불장 휴면(하락장전용 숏알파 robust)' if ok else '불합격 — OOS 하락장 or 종목과반 or 불장휴면 미달(과최적/표본운)'}")
    L.append("  ※ 표본=하락장 3개뿐(적음). 숏비용·차입비·슬리피지 미반영(실현↓). 마진/선물 필요·코인불가. 헤지용이지 수익엔진 아님.")
    txt="\n".join(L); print(txt)
    open(os.path.join(EV.HERE,"SHORT_EVOLVE_LOG.md"),"w",encoding="utf-8").write(txt+"\n")


if __name__=="__main__":
    main()
