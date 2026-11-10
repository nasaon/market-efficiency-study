# -*- coding: utf-8 -*-
"""심화 감사 — ①룩어헤드 민감도(코어 모멘텀 1일 추가지연 시 성과 유지 여부)
②현실적 비용(KR 거래세·코인 수수료 반영) 재측정.
코드 정독 결론: 모든 신호 t-1 이하 정보(condv[:-1]·gspc[t-1]·prices[t-1]·shift(1)·rv[t-1]) = 룩어헤드 없음.
민감도로 실증: 진짜 강건하면 1일 더 밀어도 성과 유지."""
import sys, json
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import rb_sim as rb, mkt_sim as ms
from trend_strat import evalc as trend_evalc

US = {"vix":31,"rsi":31,"ma":200,"slots":5,"mkt_ma":220,"mkt_exit_ma":100,"mkt_confirm":0,
      "cash_spx_frac":0.3,"core_asset":"mom5","hedge_frac":0,"hedge_mode":"inverse","hedge_confirm":0,
      "hedge_lev":1,"vixmax":0,"stop":0,"trail":0,"tp_rsi":0,"tp_pct":0,"maxtranche":1,"gap":5,
      "cost_bps":5,"yearend":True,"reentry":True,"mkt_exit":True,"cash_in_spx":True}
KR = {**US, "rsi_p":10,"ma":150,"slots":6,"mkt_exit_ma":175,"mkt_confirm":5,"core_asset":"mom5_ens"}


def lag1(D):
    """코어 모멘텀·위성신호를 1일 추가 지연한 D 사본(룩어헤드 민감도)."""
    D2 = dict(D)
    D2["mom"] = {k: np.concatenate([[0.0], v[:-1]]) for k, v in D["mom"].items()}
    # vixarr·closemat·Rmats도 1일 지연 → 위성 신호도 하루 더 밀림
    D2["vixarr"] = np.concatenate([[np.nan], D["vixarr"][:-1]])
    D2["closemat"] = np.vstack([D["closemat"][:1], D["closemat"][:-1]])
    D2["Rmats"] = {k: np.vstack([v[:1], v[:-1]]) for k, v in D["Rmats"].items()}
    return D2


def us_sc(cfg, D):
    p,i,y = rb.run_fast(cfg, D); m = rb.metrics(p,i,y)
    h1=rb.score(rb.metrics(p[y<=2020],i[y<=2020],y[y<=2020]))["total"]
    h2=rb.score(rb.metrics(p[y>2020],i[y>2020],y[y>2020]))["total"]
    return min(h1,h2), m["sharpe"], m["cagr"]


def kr_sc(cfg, D):
    p,i,y = rb.run_fast(cfg, D)
    mask=(y>=D["y0"])&(y<=D["y1"]); p,i,y=p[mask],i[mask],y[mask]; mid=(D["y0"]+D["y1"])//2
    m=rb.metrics(p,i,y,ppy=D["ppy"])
    h1=rb.score(rb.metrics(p[y<=mid],i[y<=mid],y[y<=mid],ppy=D["ppy"]))["total"]
    h2=rb.score(rb.metrics(p[y>mid],i[y>mid],y[y>mid],ppy=D["ppy"]))["total"]
    return min(h1,h2), m["sharpe"], m["cagr"]


def main():
    print("="*60); print("① 룩어헤드 민감도 — 코어/신호 1일 추가지연 시 성과"); print("="*60)
    Du = rb.load_all()
    a = us_sc(US, Du); b = us_sc(US, lag1(Du))
    print(f"US     기준 강건{a[0]:.1f} 샤프{a[1]:.2f} | +1일지연 강건{b[0]:.1f} 샤프{b[1]:.2f}  Δ강건{b[0]-a[0]:+.1f}")

    top80 = [c.strip() for c in open("kr_universe_top80.txt",encoding="utf-8").read().split() if c.strip()]
    Dk_all = ms.load_market("KR"); exc = set(Dk_all["cols"]) - set(top80)
    Dk = ms.load_market("KR", exclude=exc)
    a = kr_sc(KR, Dk); b = kr_sc(KR, lag1(Dk))
    print(f"KR     기준 강건{a[0]:.1f} 샤프{a[1]:.2f} | +1일지연 강건{b[0]:.1f} 샤프{b[1]:.2f}  Δ강건{b[0]-a[0]:+.1f}")

    Dc = ms.load_market("CRYPTO")
    CR = dict(core_asset="mom3_ensw", mkt_ma=130, mkt_exit_ma=130, mkt_confirm=0, alloc=0.7, target_vol=0.20, vol_win=30, cost_bps=5)
    m,s,h1,h2,_,_ = trend_evalc(CR, Dc); base=(min(h1,h2), m["sharpe"])
    m,s,h1,h2,_,_ = trend_evalc(CR, lag1(Dc)); lag=(min(h1,h2), m["sharpe"])
    print(f"CRYPTO 기준 강건{base[0]:.1f} 샤프{base[1]:.2f} | +1일지연 강건{lag[0]:.1f} 샤프{lag[1]:.2f}  Δ강건{lag[0]-base[0]:+.1f}")

    print("\n" + "="*60); print("② 현실적 비용 재측정 (편도 bps)"); print("="*60)
    print("가정: US 5(대형주 스프레드) / KR 15(거래세 매도18-23bps 실효환산) / CRYPTO 10(거래소 taker)")
    for cb in (5, 10, 15, 20, 30):
        u = us_sc({**US,"cost_bps":cb}, Du)[0]
        k = kr_sc({**KR,"cost_bps":cb}, Dk)[0]
        mm,ss,h1,h2,_,_ = trend_evalc({**CR,"cost_bps":cb}, Dc); c = min(h1,h2)
        print(f"  {cb:>2d}bps: US {u:5.1f} | KR {k:5.1f} | CRYPTO {c:5.1f}")
    print("\n실효비용 시나리오: US@5 / KR@15 / CRYPTO@10")
    print(f"  US     {us_sc({**US,'cost_bps':5}, Du)[0]:.1f}")
    print(f"  KR     {kr_sc({**KR,'cost_bps':15}, Dk)[0]:.1f}")
    mm,ss,h1,h2,_,_ = trend_evalc({**CR,'cost_bps':10}, Dc)
    print(f"  CRYPTO {min(h1,h2):.1f}")


if __name__ == "__main__":
    main()
