# -*- coding: utf-8 -*-
"""CNN 주식 Fear & Greed 일별 (2021~) 수집 -> fng_stock.csv."""
import os, sys, datetime as dt
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import requests, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
H = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
    "Origin": "https://www.cnn.com",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124"',
    "sec-ch-ua-platform": '"Windows"',
}
url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2021-01-01"
r = requests.get(url, headers=H, timeout=30)
r.raise_for_status()
hist = r.json()["fear_and_greed_historical"]["data"]
rows = [(dt.datetime.fromtimestamp(p["x"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d"),
         round(float(p["y"]), 2)) for p in hist]
df = pd.DataFrame(rows, columns=["date", "fng"]).drop_duplicates("date").sort_values("date")
df.to_csv(os.path.join(HERE, "fng_stock.csv"), index=False)
u30 = int((df["fng"] < 30).sum())
# 30 하향 교차 이벤트 수
s = df["fng"].values
cross = int(sum(1 for i in range(1, len(s)) if s[i] < 30 <= s[i - 1]))
print(f"F&G 저장 {len(df)}일  {df.date.min()} ~ {df.date.max()}")
print(f"F&G<30 거래일수 = {u30}  /  30 하향교차 이벤트 = {cross}")
