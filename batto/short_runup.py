# -*- coding: utf-8 -*-
"""X% 올랐을때 숏 진입 → 5% 고정 청산, 연도별(2021~2026) 백테스트 (사용자 요청).
진입: 종가가 20일 저점 대비 X% 위(최근저점서 X% 상승=과열). X=5·10·15·20%.
청산(5% 고정): 진입가 -5% 도달=이익 커버 · +5% 도달=손절 커버 · 최대 20일 보유. 먼저 닿는 쪽.
미국 숏가능 11종목·연도별 중앙값 수익%·승률. ★숏=마진/차입비 미반영·미래참조없음. 정직."""
import os, sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import turtle_exit_evolve as EV

NAMES = ["SPY","QQQ","AAPL","MSFT","AMD","MU","AVGO","META","TSLA","GOOGL","AMZN"]
XS = [0.05, 0.10, 0.15, 0.20]
EXITPCT = 0.05; MAXD = 20; FEE = 0.0005
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def load(tk):
    df = pd.read_csv(os.path.join(EV.HERE, f"yf_{tk}.csv"), parse_dates=["date"], index_col="date")
    return df


def trades_by_year(df, X):
    """숏 트레이드별 (청산연도, 수익%) 리스트. 진입=종가>20일저점*(1+X)·플랫. 청산=±5%/20일."""
    h = df["h"].values; l = df["l"].values; c = df["c"].values; idx = df.index; n = len(c)
    ll20 = pd.Series(l).rolling(20).min().shift(1).values
    out = []
    i = 25
    while i < n:
        if not np.isnan(ll20[i]) and c[i] >= ll20[i] * (1 + X):
            ent = c[i]; tp = ent * (1 - EXITPCT); sl = ent * (1 + EXITPCT)
            exitpx = None; j = i + 1
            while j < n and j <= i + MAXD:
                if l[j] <= tp: exitpx = tp; break        # 이익(하락 5%)
                if h[j] >= sl: exitpx = sl; break        # 손절(상승 5%)
                j += 1
            if exitpx is None:
                j = min(n - 1, i + MAXD); exitpx = c[j]   # 시간청산
            ret = (ent - exitpx) / ent - 2 * FEE          # 숏 수익 - 왕복수수료
            out.append((idx[j].year, ret * 100))
            i = j + 1                                      # 청산 후 다음
        else:
            i += 1
    return out


def main():
    print("데이터 로드...")
    data = {tk: load(tk) for tk in NAMES}
    L = ["X% 올랐을때 숏 → 5% 고정 청산 · 연도별 (미국11·20일저점대비 X%위 진입)",
         "청산: 진입가 -5%=이익 / +5%=손절 / 20일. 숏 수익%(왕복수수료반영). ★마진/차입비 미반영.\n"]
    for X in XS:
        # 종목별·연도별 총수익 → 연도별 중앙값 + 전체 승률
        by_year_asset = {y: [] for y in YEARS}
        allrets = []
        for tk, df in data.items():
            tr = trades_by_year(df, X)
            allrets += [r for _, r in tr]
            yr_sum = {y: 0.0 for y in YEARS}; yr_has = {y: False for y in YEARS}
            for yy, r in tr:
                if yy in yr_sum: yr_sum[yy] += r; yr_has[yy] = True
            for y in YEARS:
                if yr_has[y]: by_year_asset[y].append(yr_sum[y])
        ntr = len(allrets); wr = (np.array(allrets) > 0).mean() * 100 if allrets else 0
        avg = np.mean(allrets) if allrets else 0
        L.append(f"[{X*100:.0f}% 올랐을때 숏]  총 {ntr}트레이드 · 트레이드 승률 {wr:.0f}% · 평균 {avg:+.1f}%/트레이드")
        L.append("  " + f"{'연도':6s}" + "".join(f"{y:>8d}" for y in YEARS) + f"{'전체평균':>9s}")
        cells = []
        for y in YEARS:
            v = np.median(by_year_asset[y]) if by_year_asset[y] else np.nan
            cells.append(v)
        L.append("  " + f"{'수익%':6s}" + "".join((f"{v:>+8.0f}" if v == v else f"{'-':>8s}") for v in cells)
                  + f"{np.nanmean([x for x in cells if x==x]):>+9.0f}")
        L.append("")
    L.append("  ★해석: 대부분 연도 -면 강세장서 숏 역행(예상). 특정 하락연도(22)만 +면 하락장 전용.")
    L.append("  ±5% 대칭청산=상승장선 +5%손절(상승지속)이 -5%이익보다 자주 → 净손실 경향. 종목별 편차 큼.")
    L.append("  ※ 숏=마진/선물/인버스·차입비·슬리피지 미반영(실현 더 낮음). 2026 부분연도. 생존편향.")
    txt = "\n".join(L); print(txt)
    open(os.path.join(EV.HERE, "SHORT_RUNUP_LOG.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
