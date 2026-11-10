# -*- coding: utf-8 -*-
"""터틀 청산(익절) 진화 루프 — '저점이탈 근처'가 아닌 '진짜 고점 근처'에서 익절하도록.
================================================================================
사용자 문제: 터틀 청산이 추세종료(저점이탈)라 고점서 꺾인 만큼 반납. 절반익절 도입했으나
  그 익절도 저점이탈 라인 근처서 일어남 → 진짜 고점 근처서 익절하게 청산규칙을 진화.

방법(100R 진화): 청산만 최적화(진입 30/10+SMA·1.5N손절 고정). 매 라운드 청산 genome을
  변이/교배해 백테스트, '조금이라도 좋으면' 챔피언 갱신·병합·발전. 반복.

정직성 장치(과최적화 방어):
  1) 체결가 정밀 회계: 한계가(익절타깃)·트레일선 실제 체결가 + 일일 MTM 자산(MDD 정확).
     → 'close 기준' 근사가 못 잡는 '고점 익절' 이득을 실제로 측정.
  2) 바스켓 중앙값 Calmar로 적합도(단일종목 운빨 배제). 코인3 + 미국강추세6.
  3) ★Train(~2023말)서만 진화 → Test(2024~, OOS)로 최종 판정. in-sample 100R은 낙관상한.

청산 메커니즘(트랜치별): trail(고점-k·ATR) / ptarget(진입+m·ATR 한계익절) /
  donch(L일 저점이탈) / rsi(RSI≥r 과열 익절·강세에 팖=고점근처) / giveback(고점×(1-g) %반납).
"""
import os, sys, random, json, copy
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 선별 엔진은 같은 폴더
import batto_screener_live as S

HERE = S.DATA_DIR
TEST_START = pd.Timestamp("2024-01-01")     # Train ≤2023말 / Test 2024~ (OOS)
# 바스켓: 고변동 추세자산(익절 연구가 의미있는 자산). (티커, 종류, eN, smaN, fee)
BASKET = [
    ("BTC", "coin", 20, 60, 0.001), ("ETH", "coin", 20, 50, 0.001), ("SOL", "coin", 20, 60, 0.001),
    ("NVDA", "us", 30, 100, 0.0005), ("AMD", "us", 30, 100, 0.0005), ("MU", "us", 30, 100, 0.0005),
    ("AVGO", "us", 30, 100, 0.0005), ("TSLA", "us", 30, 100, 0.0005), ("LRCX", "us", 30, 100, 0.0005),
]
SL_N = 1.5
MECHS = ["trail", "ptarget", "donch", "rsi", "giveback"]


# ── 데이터 + 지표 사전계산 ────────────────────────────────────────────────────
def load_asset(tk, kind, eN, smaN, fee):
    if kind == "coin":
        df = S.load_coin(tk)
    else:
        df = S.load_stock(tk)
    df = df.dropna()
    h = df["h"].values; l = df["l"].values; c = df["c"].values
    o = df["o"].values if "o" in df else c.copy()
    a = S.atr(h, l, c, 20)
    sma = pd.Series(c).rolling(smaN).mean().values
    hh_e = pd.Series(h).rolling(eN).max().shift(1).values
    r = S.rsi(c, 14)
    low_L = {L: pd.Series(l).rolling(L).min().shift(1).values for L in range(3, 31)}
    return {"tk": tk, "kind": kind, "fee": fee, "eN": eN, "smaN": smaN,
            "o": o, "h": h, "l": l, "c": c, "a": a, "sma": sma, "hh_e": hh_e, "r": r,
            "low_L": low_L, "idx": df.index, "n": len(c)}


# ── 백테스트 엔진 (트랜치·체결가 정밀·일일 MTM) ───────────────────────────────
def simulate(D, genome):
    """청산 genome으로 백테스트 → 일일 equity 시리즈(pd.Series) + 트레이드 리스트.
    genome = {"tranches": [{"frac":, "type":, "param":}, ...]} (frac 합 1.0)."""
    o, h, l, c, a, sma, hh_e, r, low_L = (D["o"], D["h"], D["l"], D["c"], D["a"],
                                          D["sma"], D["hh_e"], D["r"], D["low_L"])
    n = D["n"]; fee = D["fee"]; smaN = D["smaN"]; eN = D["eN"]
    start = max(eN, smaN, 20) + 1
    eq = 1.0; equity = np.full(n, np.nan)
    in_pos = False; trades = []
    tr = []          # 활성 트랜치: {frac, type, param, shares, target}
    E = atr_e = stop = peak = 0.0; cash = 0.0; tmax = 0.0

    for i in range(n):
        if i < start:
            equity[i] = eq; continue
        if not in_pos:
            # 진입: 고점돌파 + SMA위
            if not np.isnan(hh_e[i]) and c[i] > hh_e[i] and not np.isnan(sma[i]) and c[i] > sma[i]:
                E = c[i]; atr_e = a[i]; stop = E - SL_N * atr_e; peak = h[i]; tmax = h[i]
                total_sh = eq * (1 - fee) / E
                tr = []
                for t in genome["tranches"]:
                    tgt = E + t["param"] * atr_e if t["type"] == "ptarget" else None
                    tr.append({"frac": t["frac"], "type": t["type"], "param": t["param"],
                               "shares": total_sh * t["frac"], "target": tgt})
                cash = 0.0; in_pos = True
                equity[i] = eq * (1 - fee)   # 진입봉 MTM(=매수직후, 수수료 반영)
            else:
                equity[i] = eq
            continue

        # 보유 중 (진입봉 다음부터 청산 평가)
        for t in tr:
            if t["shares"] <= 0:
                continue
            typ = t["type"]; pm = t["param"]
            # 하방 청산선(하드스톱 + trail/giveback 메커니즘선 중 높은 쪽이 먼저 맞음)
            ldown = stop
            if typ == "trail":
                ldown = max(stop, peak - pm * a[i])
            elif typ == "giveback":
                ldown = max(stop, peak * (1 - pm))
            exit_px = None
            if o[i] <= ldown:            # 갭하락 관통 → 시가 체결
                exit_px = o[i]
            elif l[i] <= ldown:          # 장중 도달 → 청산선 체결
                exit_px = ldown
            elif typ == "ptarget":       # 상방 한계익절(고점근처서 팖)
                if o[i] >= t["target"]:  exit_px = o[i]
                elif h[i] >= t["target"]: exit_px = t["target"]
            elif typ == "rsi":           # 과열 익절(강세에 팖=고점근처)
                if not np.isnan(r[i]) and r[i] >= pm: exit_px = c[i]
            elif typ == "donch":         # 저점이탈(원전)
                L = int(round(pm)); L = min(max(L, 3), 30)
                if not np.isnan(low_L[L][i]) and c[i] < low_L[L][i]: exit_px = c[i]
            if exit_px is not None:
                cash += t["shares"] * exit_px * (1 - fee)
                t["shares"] = 0.0
        # 트레이드 종료?
        rem = sum(t["shares"] for t in tr)
        if rem <= 0:
            eq = cash; in_pos = False
            trades.append({"E": E, "tmax": tmax})
            equity[i] = eq
        else:
            equity[i] = cash + rem * c[i]    # 일일 MTM(미실현=종가)
        # 다음 바를 위한 peak 갱신(현재 바 high 반영 — trail선은 직전까지 peak 사용했음)
        peak = max(peak, h[i]); tmax = max(tmax, h[i])

    return pd.Series(equity, index=D["idx"])


# ── 지표 ──────────────────────────────────────────────────────────────────────
def metrics_slice(equity, lo, hi):
    """[lo,hi) 구간 재기준(1.0) → (최종배수, MDD%, CAGR, Calmar)."""
    e = equity[(equity.index >= lo) & (equity.index < hi)].dropna()
    if len(e) < 60 or e.iloc[0] <= 0:
        return None
    e = e / e.iloc[0]
    pk = np.maximum.accumulate(e.values); mdd = (1 - e.values / pk).max() * 100
    yrs = len(e) / 252.0
    cagr = e.iloc[-1] ** (1 / yrs) - 1 if e.iloc[-1] > 0 else -1
    calmar = cagr / (mdd / 100) if mdd > 0 else np.nan
    return e.iloc[-1], mdd, cagr, calmar


def fitness(DATA, genome, window):
    """바스켓 중앙값 Calmar (window='train' or 'test'). 부수: 평균배수·평균MDD."""
    cals, mults, mdds = [], [], []
    for D in DATA:
        eq = simulate(D, genome)
        if window == "train":
            m = metrics_slice(eq, eq.index[0], TEST_START)
        else:
            m = metrics_slice(eq, TEST_START, eq.index[-1] + pd.Timedelta(days=1))
        if m is None: continue
        mult, mdd, cagr, cal = m
        if cal == cal: cals.append(cal)
        mults.append(mult); mdds.append(mdd)
    if not cals: return -9.0, 0.0, 99.0
    return float(np.median(cals)), float(np.mean(mults)), float(np.mean(mdds))


# ── genome 유틸 ───────────────────────────────────────────────────────────────
def norm(g):
    """frac 합 1.0 정규화 + 최소 0.1·최대 3트랜치 정리."""
    ts = [t for t in g["tranches"] if t["frac"] > 0.001][:3]
    s = sum(t["frac"] for t in ts) or 1.0
    for t in ts: t["frac"] = t["frac"] / s
    return {"tranches": ts}


def rand_mech():
    m = random.choice(MECHS)
    p = {"trail": lambda: round(random.uniform(0.5, 6), 2),
         "ptarget": lambda: round(random.uniform(2, 30), 1),
         "donch": lambda: random.randint(3, 30),
         "rsi": lambda: random.randint(65, 95),
         "giveback": lambda: round(random.uniform(0.05, 0.5), 3)}[m]()
    return {"type": m, "param": p}


def clamp_param(typ, p):
    if typ == "trail": return round(min(max(p, 0.3), 8), 2)
    if typ == "ptarget": return round(min(max(p, 1.5), 40), 1)
    if typ == "donch": return int(min(max(round(p), 3), 30))
    if typ == "rsi": return int(min(max(round(p), 55), 97))
    if typ == "giveback": return round(min(max(p, 0.03), 0.6), 3)
    return p


def mutate(g, archive):
    g = copy.deepcopy(g); ts = g["tranches"]
    op = random.random()
    if op < 0.40 and ts:                       # 파라미터 섭동
        t = random.choice(ts); t["param"] = clamp_param(t["type"], t["param"] * random.uniform(0.6, 1.5))
    elif op < 0.60 and ts:                     # 메커니즘 교체
        t = random.choice(ts); m = rand_mech(); t["type"] = m["type"]; t["param"] = m["param"]
    elif op < 0.75 and len(ts) >= 2:           # frac 재분배
        i, j = random.sample(range(len(ts)), 2); d = random.uniform(0.05, 0.3)
        ts[i]["frac"] = max(0.1, ts[i]["frac"] - d); ts[j]["frac"] += d
    elif op < 0.88:                            # 트랜치 추가/삭제
        if len(ts) < 3:
            m = rand_mech(); m["frac"] = 0.33; ts.append(m)
        elif len(ts) > 1:
            ts.pop(random.randrange(len(ts)))
    else:                                      # 교배(아카이브 우량과 병합)
        if archive:
            other = random.choice(archive)["g"]["tranches"]
            mix = random.sample(ts, max(1, len(ts) - 1)) + random.sample(other, 1)
            g["tranches"] = [copy.deepcopy(t) for t in mix]
    return norm(g)


def gstr(g):
    def one(t):
        u = {"trail": f"트레일{t['param']}", "ptarget": f"익절타깃{t['param']}ATR",
             "donch": f"저점이탈{t['param']}일", "rsi": f"RSI{t['param']}과열",
             "giveback": f"반납{t['param']*100:.0f}%"}[t["type"]]
        return f"{t['frac']*100:.0f}%·{u}"
    return " + ".join(one(t) for t in g["tranches"])


# ── 시드(아이디어 메뉴) ───────────────────────────────────────────────────────
SEEDS = [
    {"name": "원전(저점이탈 전량)", "tranches": [{"frac": 1.0, "type": "donch", "param": 10}]},
    {"name": "현재채택(저점이탈½+트레일2½)", "tranches": [{"frac": 0.5, "type": "donch", "param": 10}, {"frac": 0.5, "type": "trail", "param": 2.0}]},
    {"name": "전량트레일2", "tranches": [{"frac": 1.0, "type": "trail", "param": 2.0}]},
    {"name": "타이트1+와이드3", "tranches": [{"frac": 0.5, "type": "trail", "param": 1.0}, {"frac": 0.5, "type": "trail", "param": 3.0}]},
    {"name": "3단(1/2/4)", "tranches": [{"frac": 0.34, "type": "trail", "param": 1.0}, {"frac": 0.33, "type": "trail", "param": 2.0}, {"frac": 0.33, "type": "trail", "param": 4.0}]},
    {"name": "익절타깃6+트레일3", "tranches": [{"frac": 0.5, "type": "ptarget", "param": 6}, {"frac": 0.5, "type": "trail", "param": 3.0}]},
    {"name": "RSI80익절+트레일3", "tranches": [{"frac": 0.4, "type": "rsi", "param": 80}, {"frac": 0.6, "type": "trail", "param": 3.0}]},
    {"name": "반납15%+트레일3", "tranches": [{"frac": 0.5, "type": "giveback", "param": 0.15}, {"frac": 0.5, "type": "trail", "param": 3.0}]},
    {"name": "익절타깃5/10+트레일3", "tranches": [{"frac": 0.34, "type": "ptarget", "param": 5}, {"frac": 0.33, "type": "ptarget", "param": 10}, {"frac": 0.33, "type": "trail", "param": 3.0}]},
    {"name": "RSI85익절+익절타깃8+트레일4", "tranches": [{"frac": 0.34, "type": "rsi", "param": 85}, {"frac": 0.33, "type": "ptarget", "param": 8}, {"frac": 0.33, "type": "trail", "param": 4.0}]},
]


def main():
    rounds = 100; seed = 42
    for a in sys.argv[1:]:
        if a.startswith("--rounds="): rounds = int(a.split("=")[1])
        if a.startswith("--seed="): seed = int(a.split("=")[1])
    random.seed(seed); np.random.seed(seed)
    print("데이터 로드·지표 사전계산...")
    DATA = []
    for tk, kind, eN, smaN, fee in BASKET:
        try: DATA.append(load_asset(tk, kind, eN, smaN, fee))
        except Exception as e: print(f"  {tk} 스킵: {e}")
    print(f"  바스켓 {len(DATA)}종목 · Train≤{TEST_START.date()} / Test 이후\n")

    log = ["터틀 청산(익절) 진화 루프 — Train 중앙값 Calmar 최대화, Test=OOS 판정",
           f"바스켓 {len(DATA)}종목(코인3+미국강추세6) · 100R · 청산만 최적화(진입·1.5N손절 고정)\n",
           "[시드 평가] (Train Calmar / Test Calmar)"]
    archive = []
    base_genome = None
    for sd in SEEDS:
        g = norm({"tranches": [dict(t) for t in sd["tranches"]]})
        ftr = fitness(DATA, g, "train"); fte = fitness(DATA, g, "test")
        archive.append({"g": g, "name": sd["name"], "train": ftr[0], "test": fte[0]})
        log.append(f"  {sd['name']:30s} Train {ftr[0]:5.2f} / Test {fte[0]:5.2f}  | {gstr(g)}")
        if sd["name"].startswith("현재채택"): base_genome = g
    archive.sort(key=lambda x: -x["train"])
    champ = copy.deepcopy(archive[0]); champ_g = champ["g"]
    log.append(f"\n[시작 챔피언] {champ['name']} Train {champ['train']:.2f}")
    log.append("\n[진화 100R] (개선 채택만 기록)")
    print("\n".join(log[-3:]))

    accepted = 0
    for rd in range(1, rounds + 1):
        cand = mutate(champ_g, archive)
        ftr, mult, mdd = fitness(DATA, cand, "train")
        if ftr > champ["train"] + 1e-4:        # '조금이라도 좋으면' 채택·발전
            fte = fitness(DATA, cand, "test")[0]
            champ = {"g": cand, "name": f"R{rd}진화", "train": ftr, "test": fte}
            champ_g = cand; accepted += 1
            archive.append(champ); archive.sort(key=lambda x: -x["train"]); archive = archive[:12]
            line = f"  R{rd:3d} ★채택 Train {ftr:5.2f}/Test {fte:5.2f} 배수{mult:5.1f} MDD{mdd:3.0f}% | {gstr(cand)}"
            log.append(line); print(line)

    # ── 최종 판정: 챔피언 vs 현재채택 baseline, Train·Test ──
    log.append(f"\n[진화 결과] 100R 중 {accepted}회 개선 채택")
    base_tr = fitness(DATA, base_genome, "train"); base_te = fitness(DATA, base_genome, "test")
    ch_tr = fitness(DATA, champ_g, "train"); ch_te = fitness(DATA, champ_g, "test")
    log.append("\n[★최종 OOS 판정] Train(in-sample) → Test(out-of-sample) 중앙값 Calmar")
    log.append(f"  {'전략':28s}{'Train':>8s}{'Test':>8s}{'배수(Te)':>9s}{'MDD(Te)':>9s}")
    log.append(f"  {'현재채택(저점½+트레일2½)':28s}{base_tr[0]:>8.2f}{base_te[0]:>8.2f}{base_te[1]:>9.1f}{base_te[2]:>8.0f}%")
    log.append(f"  {'진화챔피언':28s}{ch_tr[0]:>8.2f}{ch_te[0]:>8.2f}{ch_te[1]:>9.1f}{ch_te[2]:>8.0f}%")
    log.append(f"  챔피언 청산: {gstr(champ_g)}")
    verdict = ("진화 청산이 OOS에서도 baseline 초과 → 진짜 엣지 가능" if ch_te[0] > base_te[0] + 0.02
               else "진화 청산이 OOS에선 baseline 못 넘음 → in-sample 과최적(흔한 결과)")
    log.append(f"  ★판정: {verdict}")
    log.append("  ※ Train 100R는 낙관상한(과최적). 믿을건 Test(OOS). 생존편향·슬리피지/세금 미반영.")

    txt = "\n".join(log); print("\n" + "\n".join(log[-9:]))
    open(os.path.join(HERE, "TURTLE_EXIT_EVOLVE_LOG.md"), "w", encoding="utf-8").write(txt + "\n")
    json.dump({"champion": champ_g, "base_test": base_te[0], "champ_test": ch_te[0]},
              open(os.path.join(HERE, "turtle_exit_evolve_champ.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n로그: TURTLE_EXIT_EVOLVE_LOG.md · 챔피언: turtle_exit_evolve_champ.json")


if __name__ == "__main__":
    main()
