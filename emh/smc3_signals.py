# -*- coding: utf-8 -*-
"""
추가 SMC 3종 신호 (위키 3·4·5장 충실). 모두 꼬리(wick) 기준, look-ahead 차단.
- 추세선(TL): 확정 스윙 저점2개(상승선)/고점2개(하락선) 연결. 선 터치 후 홀드 = 추세추종 진입.
    TL_L: 상승추세선(저점연결, slope>0)에 저가가 닿고 종가가 선 위 → 롱.
    TL_S: 하락추세선(고점연결, slope<0)에 고가가 닿고 종가가 선 아래 → 숏.
- 채널(CH): 추세선 + 평행선. 하단밴드 터치=롱, 상단밴드 터치=숏 (4포인트 역추세).
- 페이크아웃(FO): 직전 확정 스윙 저점/고점을 꼬리로 이탈했다가 종가 복귀(거짓돌파/스윕).
    FO_L: low<직전스윙저점 & close>직전스윙저점 → 롱.
    FO_S: high>직전스윙고점 & close<직전스윙고점 → 숏.
유효창 WIN(추세선·채널), 스윙 k=5.
"""
import numpy as np
from signals import SWING_K

WIN = 60


def trend_channel_fakeout(df, is_sh, is_sl):
    o = df["open"].values; h = df["high"].values; l = df["low"].values; c = df["close"].values
    n = len(df)
    out = {k: set() for k in ["TL_L", "TL_S", "CH_L", "CH_S", "FO_L", "FO_S"]}

    lows = []   # 확정 스윙 저점 인덱스(시간순)
    highs = []  # 확정 스윙 고점 인덱스
    up = None   # 상승 채널: dict(s1,s2,slope,off_up)
    dn = None   # 하락 채널: dict(s1,s2,slope,off_dn)
    prior_sl = None  # 직전 확정 스윙 저점 가격
    prior_sh = None

    for t in range(n):
        j = t - SWING_K   # j 시점 스윙이 t에 확정됨
        if j >= 0:
            if is_sl[j]:
                lows.append(j); prior_sl = l[j]
                if len(lows) >= 2:
                    s1, s2 = lows[-2], lows[-1]
                    if s2 > s1:
                        slope = (l[s2] - l[s1]) / (s2 - s1)
                        if slope > 0:
                            off = max((h[k] - (l[s1] + slope * (k - s1))) for k in range(s1, s2 + 1))
                            up = dict(s1=s1, s2=s2, slope=slope, off=max(off, 0))
            if is_sh[j]:
                highs.append(j); prior_sh = h[j]
                if len(highs) >= 2:
                    s1, s2 = highs[-2], highs[-1]
                    if s2 > s1:
                        slope = (h[s2] - h[s1]) / (s2 - s1)
                        if slope < 0:
                            off = max(((h[s1] + slope * (k - s1)) - l[k]) for k in range(s1, s2 + 1))
                            dn = dict(s1=s1, s2=s2, slope=slope, off=max(off, 0))

        # ── 추세선 (상승) ──
        if up and 0 < t - up["s2"] <= WIN:
            line = l[up["s2"]] + up["slope"] * (t - up["s2"])
            if l[t] <= line and c[t] > line:
                out["TL_L"].add(t)
            # 채널 상단 밴드(숏)
            upper = line + up["off"]
            if h[t] >= upper and c[t] < upper:
                out["CH_S"].add(t)
            # 채널 하단 밴드(롱) = 추세선 터치
            if l[t] <= line and c[t] > line:
                out["CH_L"].add(t)
        # ── 추세선 (하락) ──
        if dn and 0 < t - dn["s2"] <= WIN:
            line = h[dn["s2"]] + dn["slope"] * (t - dn["s2"])
            if h[t] >= line and c[t] < line:
                out["TL_S"].add(t)
            lower = line - dn["off"]
            if l[t] <= lower and c[t] > lower:
                out["CH_L"].add(t)
            if h[t] >= line and c[t] < line:
                out["CH_S"].add(t)

        # ── 페이크아웃 (직전 스윙 스윕 후 복귀) ──
        if prior_sl is not None and l[t] < prior_sl and c[t] > prior_sl:
            out["FO_L"].add(t)
        if prior_sh is not None and h[t] > prior_sh and c[t] < prior_sh:
            out["FO_S"].add(t)

    return out


COLS = ["TL_L", "TL_S", "CH_L", "CH_S", "FO_L", "FO_S"]
