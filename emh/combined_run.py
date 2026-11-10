# -*- coding: utf-8 -*-
"""
SMC 5종 종합(컨플루언스) 분석. FVG·OB·추세선·채널·페이크아웃을 한 판에.
- 방법별 방향부호: FVG_s, OB_s, TL_s, CH_s, FO_s ∈ {-1,0,+1}
- net = 5종 합 (-5..+5) = 합의 점수
- (1) 다변량 회귀: ex_N ~ 5부호 + 통제 (서로 통제 후 기여)
- (2) net 회귀: ex_N ~ net + 통제 (합의가 예측?)
- (3) 컨플루언스 버킷: |net| 클수록 방향조정 초과수익 커지나? (위키 핵심 가설)
같은 3시장, 개별종목 이중클러스터.
"""
import os, sys, glob
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import signals as S
from smc3_signals import trend_channel_fakeout

RES = os.path.join(S.HERE, "results"); os.makedirs(RES, exist_ok=True)
S2 = os.path.join(S.HERE, "data", "stage2")
NS = [5, 10, 20]; CTRL = ["vol20", "logvol"]
MAP = [("us", "미국", "sp500"), ("kr", "한국", "kospi200"), ("crypto", "가상자산", "btc")]
PARTS = ["FVG_s", "OB_s", "TL_s", "CH_s", "FO_s"]


def with_all(df):
    df = S.add_basics(df)
    sh, sl = S.swing_points(df)
    fvg, _ = S.fvg_signals(df)
    ob, _ = S.ob_signals(df, sh, sl)
    smc3 = trend_channel_fakeout(df, sh, sl)
    n = len(df)
    def col(idxs):
        v = np.zeros(n, int)
        for j in idxs: v[j] = 1
        return v
    FVG_L = col(fvg["L"]); FVG_S = col(fvg["S"])
    OB_L = ((col(ob[("A", "L")]) | col(ob[("C", "L")]))).astype(int)
    OB_S = ((col(ob[("A", "S")]) | col(ob[("C", "S")]))).astype(int)
    TL_L, TL_S = col(smc3["TL_L"]), col(smc3["TL_S"])
    CH_L, CH_S = col(smc3["CH_L"]), col(smc3["CH_S"])
    FO_L, FO_S = col(smc3["FO_L"]), col(smc3["FO_S"])
    df["FVG_s"] = FVG_L - FVG_S
    df["OB_s"] = OB_L - OB_S
    df["TL_s"] = TL_L - TL_S
    df["CH_s"] = CH_L - CH_S
    df["FO_s"] = FO_L - FO_S
    df["net"] = df[PARTS].sum(axis=1)
    return df


def build(folder, idxkey):
    idx = pd.read_csv(os.path.join(S.OUT, f"{idxkey}_panel.csv"), parse_dates=["date"], index_col="date")
    idx = idx[[f"fwd_{N}" for N in NS]].rename(columns={f"fwd_{N}": f"idx_{N}" for N in NS})
    frames = []
    for fp in glob.glob(os.path.join(S2, folder, "*.csv")):
        tic = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        if len(df) < 250: continue
        df = with_all(df)
        m = df.join(idx, how="inner")
        for N in NS: m[f"ex_{N}"] = m[f"fwd_{N}"] - m[f"idx_{N}"]
        m["ticker"] = tic
        frames.append(m[["ticker"] + [f"ex_{N}" for N in NS] + CTRL + PARTS + ["net"]].reset_index())
    return pd.concat(frames, ignore_index=True)


def clreg(p, ycol, xcols):
    d = p[[ycol, "date", "ticker"] + xcols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200: return None
    y = d[ycol].values; X = sm.add_constant(d[xcols].values)
    g = np.column_stack([pd.factorize(d["date"])[0], pd.factorize(d["ticker"])[0]])
    try:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    except Exception:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g[:, 0]})
    return m, d


def main():
    multi, netr, buck = [], [], []
    for folder, kor, idxkey in MAP:
        p = build(folder, idxkey)
        print(f"[{folder}] {p['ticker'].nunique()}종목 {len(p):,}행  net분포: " +
              ", ".join(f"|{k}|={int((p['net'].abs()==k).sum())}" for k in [1,2,3]))
        for N in NS:
            # (1) 다변량
            out = clreg(p, f"ex_{N}", PARTS + CTRL)
            if out:
                m, d = out
                for i, part in enumerate(PARTS, start=1):
                    multi.append(dict(market=folder, kor=kor, N=N, part=part,
                                      coef=m.params[i], p=m.pvalues[i]))
            # (2) net
            out = clreg(p, f"ex_{N}", ["net"] + CTRL)
            if out:
                m, d = out
                netr.append(dict(market=folder, kor=kor, N=N, coef=m.params[1], t=m.tvalues[1], p=m.pvalues[1]))
            # (3) 컨플루언스 버킷: 방향조정 초과수익 by |net|
            for k in [1, 2, 3]:
                sub = p[(p["net"].abs() == k) if k < 3 else (p["net"].abs() >= 3)].copy()
                if len(sub) < 20:
                    buck.append(dict(market=folder, kor=kor, N=N, bucket=k, n=len(sub), dret=np.nan)); continue
                dret = (np.sign(sub["net"]) * sub[f"ex_{N}"]).dropna()
                buck.append(dict(market=folder, kor=kor, N=N, bucket=k, n=len(dret), dret=dret.mean()))
    pd.DataFrame(multi).to_csv(os.path.join(RES, "combined_multi.csv"), index=False, encoding="utf-8-sig")
    nr = pd.DataFrame(netr); nr.to_csv(os.path.join(RES, "combined_net.csv"), index=False, encoding="utf-8-sig")
    bk = pd.DataFrame(buck); bk.to_csv(os.path.join(RES, "combined_bucket.csv"), index=False, encoding="utf-8-sig")

    print("\n=== (2) 합의점수 net 회귀 (ex_N ~ net) ===")
    nr["star"] = np.where(nr["p"] < .05, "*", "")
    print(nr.assign(cf=lambda x:(x.coef*100).round(3))[["kor","N","cf","p","star"]].to_string(index=False))
    print("\n=== (3) 컨플루언스 버킷: |net|별 방향조정 초과수익(%, N=10) ===")
    b10 = bk[bk["N"] == 10]
    for folder, kor, _ in MAP:
        s = b10[b10["market"] == folder].set_index("bucket")
        vals = ", ".join(f"|net|{k}: {s.loc[k,'dret']*100:+.3f}%(n{int(s.loc[k,'n'])})" if pd.notna(s.loc[k,'dret']) else f"|net|{k}:NA" for k in [1,2,3])
        print(f"  {kor:7s}: {vals}")
    print("\n=== (1) 다변량 — N=10 각 신호 계수(%p) & 유의 ===")
    m10 = pd.DataFrame(multi); m10 = m10[m10["N"] == 10]
    piv = m10.pivot(index="part", columns="market", values="coef").reindex(PARTS)
    pp = m10.pivot(index="part", columns="market", values="p").reindex(PARTS)
    for part in PARTS:
        cells = " | ".join(f"{piv.loc[part,m]*100:+.3f}{'*' if pp.loc[part,m]<.05 else ' '}" for m in ['us','kr','crypto'])
        print(f"  {part:6s}: {cells}")
    print("\n완료. results/combined_*.csv")


if __name__ == "__main__":
    main()
