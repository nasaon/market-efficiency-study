# -*- coding: utf-8 -*-
"""
Part C: 1시간봉 SMC + 일봉 바이어스(MTF). 진짜 하위 타임프레임 검정.
- 일봉 바이어스: 1h→일봉 리샘플, 일봉 MA20 위/아래(전일 확정값) → +1/-1.
- 1h에서 FVG·OB 신호(방향부호). '정렬' = 신호부호와 일봉바이어스 일치 시만 채택.
- fwd_h = 향후 h개 1시간봉 누적수익. raw vs aligned 비교(바이어스 필터 효과).
- 풀 패널, 시간·종목 이중클러스터.
"""
import os, sys, glob
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import signals as S

RES = os.path.join(S.HERE, "results"); os.makedirs(RES, exist_ok=True)
INTRA = os.path.join(S.HERE, "data", "intraday")
HS = [6, 12, 24]; CTRL = ["vol20", "logvol"]
MK = [("us", "미국"), ("kr", "한국"), ("crypto", "가상자산")]


def prep(df):
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    df = S.add_basics(df)   # body, atr, ret1, vol20, logvol (1h 기준)
    c = df["close"]
    for h in HS:
        df[f"fwd_{h}"] = c.shift(-h) / c - 1
    # 일봉 바이어스
    dates = df.index.normalize()
    dclose = c.groupby(dates).last()
    dma = dclose.rolling(20, min_periods=20).mean()
    dbias = np.sign(dclose - dma).shift(1)   # 전일 확정 바이어스
    bias_map = dbias.to_dict()
    df["bias"] = pd.Series(dates, index=df.index).map(bias_map).values
    return df


def signed(df):
    sh, sl = S.swing_points(df)
    fvg, _ = S.fvg_signals(df)
    ob, _ = S.ob_signals(df, sh, sl)
    n = len(df)
    def col(idxs):
        v = np.zeros(n, int)
        for j in idxs: v[j] = 1
        return v
    df["FVG_s"] = col(fvg["L"]) - col(fvg["S"])
    OBL = (col(ob[("A", "L")]) | col(ob[("C", "L")])).astype(int)
    OBS = (col(ob[("A", "S")]) | col(ob[("C", "S")])).astype(int)
    df["OB_s"] = OBL - OBS
    # 정렬: 신호부호 × 바이어스 > 0 일 때만
    for s in ["FVG_s", "OB_s"]:
        b = df["bias"].fillna(0).values
        al = np.where(df[s].values * b > 0, df[s].values, 0)
        df[s + "_al"] = al
    return df


def build(folder):
    frames = []
    for fp in glob.glob(os.path.join(INTRA, folder, "*.csv")):
        tic = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, parse_dates=["dt"], index_col="dt")
        if len(df) < 500: continue
        df = signed(prep(df)); df["ticker"] = tic
        keep = ["ticker"] + [f"fwd_{h}" for h in HS] + CTRL + ["FVG_s", "OB_s", "FVG_s_al", "OB_s_al"]
        f = df[keep].reset_index().rename(columns={"dt": "date"})
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def clreg(p, ycol, xcol):
    d = p[[ycol, "date", "ticker", xcol] + CTRL].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200 or (d[xcol] != 0).sum() < 20: return None
    y = d[ycol].values; X = sm.add_constant(d[[xcol] + CTRL].values)
    g = np.column_stack([pd.factorize(d["date"])[0], pd.factorize(d["ticker"])[0]])
    try:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    except Exception:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g[:, 0]})
    return m


def main():
    rows = []
    for folder, kor in MK:
        p = build(folder)
        print(f"[{folder}] {p['ticker'].nunique()}종목 {len(p):,}행(1h)")
        for base, name in [("FVG_s", "FVG"), ("OB_s", "OB")]:
            for variant, col in [("raw", base), ("aligned", base + "_al")]:
                for h in HS:
                    m = clreg(p, f"fwd_{h}", col)
                    if m is None: continue
                    rows.append(dict(market=folder, kor=kor, sig=name, variant=variant, h=h,
                                     coef=m.params[1], t=m.tvalues[1], p=m.pvalues[1], nsig=int((p[col] != 0).sum())))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(RES, "partC.csv"), index=False, encoding="utf-8-sig")
    r["star"] = np.where(r["p"] < .05, "*", "")
    print("\n=== Part C: 1시간봉 SMC + 일봉바이어스 (fwd h시간 수익, %, *p<.05) ===")
    for folder, kor in MK:
        print(f"\n[{kor}]")
        for name in ["FVG", "OB"]:
            for variant in ["raw", "aligned"]:
                ss = r[(r["market"] == folder) & (r["sig"] == name) & (r["variant"] == variant)].sort_values("h")
                if len(ss) == 0: continue
                cells = " ".join(f"h{int(x.h)}:{x.coef*100:+.3f}{x.star}" for _, x in ss.iterrows())
                print(f"  {name} {variant:7s}: {cells}")
    print("\n완료. results/partC.csv")


if __name__ == "__main__":
    main()
