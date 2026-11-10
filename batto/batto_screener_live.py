# -*- coding: utf-8 -*-
# =============================================================================
#  BATTO 실전 스크리너+터틀 통합 — single source of truth (모의 아닌 실전 투자용)
# =============================================================================
#  모든 검증 결론 반영(기존 누더기 스크립트 대체):
#   · 편향완화: 상장후 2년(504일) 버퍼 + 매년 동적 유니버스 + 중앙값/상대알파 (a/b/c 대응)
#   · 선별: 직전 252일 변동성+모멘텀 z 상위 TOPN (시점의존 낮음·강건 검증)
#   · 터틀 진입 = SMA완화 C (스크리너가 모멘텀 선별 → 터틀 SMA필터 중복이라 절반):
#       주식 30/10/SMA50 · BTC·알트 20/10/SMA30 · ETH 20/20/SMA25
#       손절 3.0N · 저점이탈 절반익절 · 절반 트레일 2ATR  (2026 전수검증: 1.5N→3.0N)
#   · 수수료 코인 0.1% / 주식 0.05% 편도 반영 (실전)
#   · 무레버리지 (코인 자체가 S&P 1.8~2.7배 레버리지)
#  모드: --verify(2016~2026 검증) / --live(현재 후보+진입상태) / 둘다(기본)
#  [정직성] 절대수익=생존편향+과최적화 낀 낙관상한. 믿을건 상대알파+추세장한정+낙폭관리.
#   선별은 종목추천 아님(정량 출력). 소액 라이브 1~2분기 검증 후 비중↑. 슬리피지/세금/생존편향 미반영.
# =============================================================================
import os, sys
import json as _json
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

# 데이터(yf_*.csv·bn_*.csv)는 golden 폴더에 유지. 이 스크립트는 MAIN INVEST로 옮겨도
# 아래 절대경로로 golden 데이터를 읽으므로 자체 완결. golden 이동·개명 시 이 경로만 수정.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))   # 공개본: 데이터는 이 폴더에 받는다(README 재현)
BOT_DIR = DATA_DIR   # 공개본: 원래는 개인 자동매매 봇 폴더(이 저장소에는 봇 코드 없음)

# ── 유니버스 (보유 데이터) ─────────────────────────────────────────────────────
#  ★유니버스 = S&P500 (규칙기반·큐레이션 편향↓·robust). universe_compare.py 검증:
#   S&P500 상대알파 +20%p·OOS 2025 +56(손픽 50종목은 -19)·소형/러셀은 알파0이라 제외.
#   선별 풀만 500으로 넓히고 실진입은 여전히 TOP8(소액 관리부담 무증가).
COINS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "TRX", "AVAX", "LINK"]
_US_FALLBACK = ["NVDA", "AMD", "SMCI", "TSLA", "AVGO", "MU", "META", "AMZN", "PLTR", "AAPL", "MSFT",
      "GOOGL", "QQQ", "SPY", "JPM", "XOM", "WMT", "PG", "KO", "JNJ", "VZ", "COST", "HD", "UNH", "V",
      "MA", "BAC", "DIS", "NFLX", "CRM", "ADBE", "ORCL", "CSCO", "INTC", "QCOM", "TXN", "AMAT",
      "LRCX", "MRVL", "COIN", "MSTR", "PYPL", "SHOP", "UBER", "ABNB", "CRWD", "PANW", "NOW", "LLY", "TMO"]


def _load_us_universe():
    """S&P500 구성종목(다운로드 성공분) 로딩. 없으면 기존 50종목 폴백."""
    p = os.path.join(DATA_DIR, "_sp500_ok.txt")
    if os.path.exists(p):
        tks = [t.strip() for t in open(p, encoding="utf-8").read().split(",") if t.strip()]
        if len(tks) >= 100:
            return tks
    return _US_FALLBACK


US = _load_us_universe()


# ── ★시점별 구성종목(PIT) — 미래참조 제거 ────────────────────────────────────
#  _sp500_ok.txt 는 '오늘의' S&P500 스냅샷이라, 과거 검증에 쓰면 그 시점엔 편입도 안 된
#  종목(2024년 검증에 PLTR·COIN·SMCI 등)을 고르는 미래참조가 된다. 실측 -66%.
#  _sp500_pit.csv = 일별 구성종목 이력(1996~). 없으면 자동으로 기존 동작(폴백).
_PIT_PATH = os.path.join(DATA_DIR, "_sp500_pit.csv")
_PIT = None
if os.path.exists(_PIT_PATH):
    try:
        _PIT = pd.read_csv(_PIT_PATH, parse_dates=["date"]).sort_values("date")
    except Exception:
        _PIT = None


def members_at(asof):
    """asof 시점에 실제로 지수에 편입돼 있던 종목 집합. 이력 없거나 커버 범위 밖이면 None(=필터 안 함)."""
    if _PIT is None:
        return None
    s = _PIT[_PIT["date"] <= asof]
    if len(s) == 0:
        return None                      # 이력 시작 이전 → 필터 불가
    if asof > _PIT["date"].iloc[-1] + pd.Timedelta(days=400):
        return None                      # 이력이 너무 낡음 → 현재 유니버스 사용(라이브)
    return set(t.strip() for t in s["tickers"].iloc[-1].split(",") if t.strip())


# ★TOPN=2 — 자금이 한정적이라 실제로 1~2종목만 잡는다. 매일 재선별 전수검증(2016~2026,
#   PIT·ATR3.0·거래비 0.15%): TOP8 배수 4.11(CAGR 14.3%) → TOP2 배수 8.12(CAGR 21.9%).
#   TOP1은 금지 — 승률 39%라 연속손실을 그대로 맞아 배수 1.23·MDD 56%로 붕괴한다.
TOPN = 2
MIN_HIST = 504          # (c) 상장 후 2년 버퍼
VOL_WIN = 252; MOM_WIN = 252
YEARS = list(range(2016, 2027))

# ── 포지션 사이징 (리스크 기반) ────────────────────────────────────────────────
# ★2.5% — STOP_N을 1.5→3.0N으로 넓히면서 비중이 절반으로 눌렸다(실측: 종목당 3~4%,
#   총노출 6%). 리스크 1%로는 백테스트가 가정한 노출(종목당 ~12%)에 도달하지 못한다.
#   2.5%면 손절 1회당 계좌 손실 2.5%, 종목당 비중 ~10%. 그 이상은 D안(노출 40%·MDD 41%) 영역.
RISK_PER_TRADE = 0.025  # 트레이드당 계좌 리스크: 손절시 손실이 계좌의 2.5%
MAX_WEIGHT = 0.20       # 종목당 최대 비중 상한 (저변동주 과대비중 방지·분산)
STOP_N = 3.0            # 손절폭 = 3.0 × ATR. 1.5N은 헛손절 64%(2026 전수 85건) → 3.0N으로 확대.
                        #   검증: 2022~2026 5개년 전부 1.5N 대비 우세(승률 40→55%·총수익↑). 비중은 자동 축소(리스크 동일)
VOL_TARGET = True       # ★변동성 타게팅: 최근 변동성이 '자기 평시'보다 높으면 노출↓ (하락장 방어)
VOL_FLOOR = 0.4         # 변동성 스케일 하한(과도축소 방지). 검증: 22MDD 17→14%·전체Calmar 0.56→0.77(turtle_bear.py)

# ── 위성(병합) — 본체=터틀 추세추종, 위성=역추세 보조(연구상 알파 없음·약세장 참여·심리용) ──
#   기본 OFF(연구 최적=터틀 단독). --coin-dip / --vix-sat / --satellites 로 켬.
SAT_COIN_W = 0.15       # 코인 저점매수 위성 '총' 비중 상한 (코인병합 터틀85/저점15)
SAT_VIX_W = 0.15        # 미국 VIX저점 위성 '총' 비중 상한 (초우량주 홀드)
COIN_DIP_UNIV = ["BTC", "ETH", "SOL"]         # 우량만(약세장 칼날 방어·알트 제외)
# ★VIX 위성 유니버스 = 지수 ETF만. 개별주 하드코딩(구: NVDA·AAPL)은 '지금 시점에 우량인 걸
#   알고 고른 것'이라 미래참조+생존편향이다. 실측: 그 4종목 CAGR 36.7%인데 NVDA 한 종목이
#   기여 75.6% — 2016년에 NVDA를 고를 근거는 없었다. ETF는 상장폐지가 없어 생존편향 0,
#   구성종목 교체가 지수 안에서 자동 처리돼 시점별 유니버스 문제도 없다.
#   검증(2016~2026·동일비중·거래비 반영): ETF만 CAGR 19.6%·MDD 30.9%·Calmar 0.63.
VIX_UNIV = ["SPY", "QQQ"]
# 확장 옵션 — PIT 구성종목 중 저변동 상위 x%를 추가(0이면 ETF만). 하락장 방어는 좋으나
#   수익이 떨어진다(하위25%: 2022 -6.3%로 방어↑ / CAGR 12.4%로 수익↓).
VIX_LOWVOL_PCT = float(os.getenv("BATTO_VIX_LOWVOL_PCT", "0"))
DIP_RSI = 25; DIP_TRAIL_N = 4.0               # 코인 저점: 약세&RSI≤25 진입, 트레일4 청산
VIX_TH = 28.0; VIX_FEAR_RSI = 34; VIX_REV_RSI = 30  # 장기일봉 방식(RSI 과매도 동반 요구)

# ── ★분할(개별) 저점 타점 — 초거대기업 한정 ───────────────────────────────────
#  기존 분할 조건(4중 역배열 s20<s60<s120<s200 & RSI≤30)은 '장기 상승추세 우량주'를
#  구조적으로 배제한다. 실측: AAPL 2026-01-20 RSI 17.9·52주 -13.8%였는데 SMA120/200이
#  상승 중이라 미발동 → 저점 246.63→311.00(+26%)을 통째로 놓쳤다.
#  그래서 3중(s20<s60<s120)으로 완화하되, 유니버스를 시총 상위 100으로 좁혀 무손절 홀드의
#  치명상(전 종목 적용 시 최악 -50.8%)을 막는다.
#  2026 실측(시총100·3중·RSI≤30·재장전 RSI>50): 28건·승률 78.6%·평균 +31.29%·최악 -10.9%.
#    · 전 종목 원본: 165건·64.8%·+9.01%·최악 -50.8%  → 범위 축소가 핵심
#    · 2중까지 풀면 57건·+22.40%(AAPL 포착)이나 관리부담↑ → 3중 채택(AAPL은 포기)
#  ★재장전은 반드시 RSI>50. 기존 '종가>SMA200'은 3중 역배열과 양립해 같은 조정을
#    매일 재발동시킨다(실측: AAPL 1월에만 7회 중복).
SPLIT_TOP = 100          # 시총 상위 N위까지만 (_us_mcap.csv)
SPLIT_RSI = 30           # 진입 RSI 임계
SPLIT_REARM_RSI = 50     # 재장전: RSI가 이 값을 넘으면 다시 무장


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
def load_stock(tk):
    return pd.read_csv(os.path.join(DATA_DIR, f"yf_{tk}.csv"), parse_dates=["date"], index_col="date")[["o", "h", "l", "c"]]


def load_coin(sym):
    h1 = pd.read_csv(os.path.join(DATA_DIR, f"bn_{sym}_1h.csv"), parse_dates=["date"], index_col="date")
    return h1.resample("1D").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()


def load_all():
    series, ohlc = {}, {}
    for s in COINS:
        try: d = load_coin(s); ohlc[s] = d; series[s] = d["c"]
        except Exception: pass
    for s in US:
        try: d = load_stock(s); ohlc[s] = d; series[s] = d["c"]
        except Exception: pass
    return series, ohlc


# ── ★효율 로드 (라이브용): 선별=종가매트릭스(500·0.06초) · 매매=TOP8+위성 OHLC만 ──────
def load_us_closes():
    """S&P500 종가 매트릭스(_us_close.pkl·us_update가 갱신). 없으면 CSV 종가만 폴백."""
    p = os.path.join(DATA_DIR, "_us_close.pkl")
    if os.path.exists(p):
        try:
            M = pd.read_pickle(p)
            return {c: M[c].dropna() for c in M.columns if c in US}
        except Exception:
            pass
    out = {}
    for tk in US:
        fp = os.path.join(DATA_DIR, f"yf_{tk}.csv")
        if os.path.exists(fp):
            try: out[tk] = pd.read_csv(fp, usecols=["date", "c"], parse_dates=["date"], index_col="date")["c"]
            except Exception: pass
    return out


def load_live():
    """라이브 선별용: 코인 OHLC(10·싼값) + US 종가매트릭스(500). series + coin_ohlc 반환."""
    coin_ohlc = {}
    for s in COINS:
        try: coin_ohlc[s] = load_coin(s)
        except Exception: pass
    series = {s: d["c"] for s, d in coin_ohlc.items()}
    series.update(load_us_closes())
    return series, coin_ohlc


def ohlc_of(s, coin_ohlc):
    """단일 종목 OHLC — 코인은 사전로드분·미국은 CSV. (TOP8+위성만 호출)"""
    if s in coin_ohlc: return coin_ohlc[s]
    if is_coin(s): return load_coin(s)
    return load_stock(s)


def load_ohlc_subset(names, coin_ohlc):
    out = {}
    for s in names:
        try: out[s] = ohlc_of(s, coin_ohlc)
        except Exception: pass
    return out


def is_coin(s): return s in COINS
def fee_of(s): return 0.001 if is_coin(s) else 0.0005


# ── 터틀 파라미터 (SMA완화 C = 실전 확정) ──────────────────────────────────────
def turtle_params(sym):
    if not is_coin(sym): return (30, 10, 50)     # 주식
    if sym == "ETH":     return (20, 20, 25)     # 이더
    return (20, 10, 30)                          # BTC·알트


# ── 지표 ──────────────────────────────────────────────────────────────────────
def atr(h, l, c, n=20):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().values


def turtle_pos(df, eN, xN, smaN, slN=3.0, trailN=2.0):
    """롱온리 터틀: 고점돌파+SMA진입 · 3.0N손절 · 저점이탈 절반익절 · 절반 트레일2ATR. 미래참조 없음(shift1)."""
    h = df["h"].values; l = df["l"].values; c = df["c"].values; n = len(c)
    hh = pd.Series(h).rolling(eN).max().shift(1).values
    ll = pd.Series(l).rolling(xN).min().shift(1).values
    a = atr(h, l, c, 20); sma = pd.Series(c).rolling(smaN).mean().values
    pos = 0.0; stop = 0.0; peakh = 0.0; half = False; arr = np.zeros(n)
    for i in range(max(eN, smaN) + 1, n):
        price = c[i]
        if pos == 0:
            if not np.isnan(hh[i]) and price > hh[i] and not np.isnan(sma[i]) and price > sma[i]:
                pos = 1.0; stop = price - slN * a[i]; peakh = h[i]; half = False
        else:
            peakh = max(peakh, h[i]); trail = peakh - trailN * a[i]
            if not half:
                if l[i] <= stop: pos = 0.0
                elif not np.isnan(ll[i]) and c[i] < ll[i]: pos = 0.5; half = True
            else:
                if l[i] <= max(stop, trail): pos = 0.0
        arr[i] = pos
    return pd.Series(arr, index=df.index)


def rsi(c, n=14):
    """Wilder RSI. 미래참조 없음(현재봉까지)."""
    c = np.asarray(c, float)
    if len(c) == 0: return np.array([])
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1/n, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1/n, adjust=False).mean().values
    rs = up / np.where(dn == 0, 1e-9, dn)
    return 100 - 100 / (1 + rs)


def sma(c, n):
    return pd.Series(c).rolling(n).mean().values


def dip_pos(df, rsi_th=DIP_RSI, trailN=DIP_TRAIL_N):
    """코인 저점매수 위성: 약세구간(SMA33<SMA200 또는 종가<SMA200) & RSI≤25 진입 · 트레일4ATR 청산.
    역추세(칼날)라 손절 없이 고점추적 트레일만. 미래참조 없음. 반환 0/1 시리즈."""
    h = df["h"].values; l = df["l"].values; c = df["c"].values; n = len(c)
    if n < 200: return pd.Series(np.zeros(n), index=df.index)
    s33 = sma(c, 33); s200 = sma(c, 200); r = rsi(c, 14); a = atr(h, l, c, 20)
    weak = (s33 < s200) | (c < s200)
    pos = 0.0; peak = 0.0; arr = np.zeros(n)
    for i in range(201, n):
        if pos == 0:
            if weak[i] and not np.isnan(r[i]) and r[i] <= rsi_th:
                pos = 1.0; peak = h[i]
        else:
            if arr[i-1] == 0: arr[i] = pos; continue   # 진입봉 청산제외
            peak = max(peak, h[i])
            if l[i] <= peak - trailN * a[i]: pos = 0.0   # 트레일4 익절
        arr[i] = pos
    return pd.Series(arr, index=df.index)


def vix_pos(df, vix, vix_th=VIX_TH, fear_rsi=VIX_FEAR_RSI, rev_rsi=VIX_REV_RSI):
    """미국 VIX저점 위성: (VIX≥28 & RSI≤34) 또는 (MA역배열 SMA20<60<120<200 & RSI≤30) 진입 · 홀드(청산없음).
    초우량주 한정·손절없이 회복까지 보유(연구: 미국 초우량주는 홀드가 절대수익 우위). 미래참조 없음."""
    c = df["c"].values; n = len(c)
    if n < 200: return pd.Series(np.zeros(n), index=df.index)
    r = rsi(c, 14)
    s20 = sma(c, 20); s60 = sma(c, 60); s120 = sma(c, 120); s200 = sma(c, 200)
    vrev = (s20 < s60) & (s60 < s120) & (s120 < s200)
    v = vix.reindex(df.index, method="ffill").values if vix is not None else np.full(n, np.nan)
    vfear = ~np.isnan(v) & (v >= vix_th)
    sig = (vfear & (r <= fear_rsi)) | (vrev & (r <= rev_rsi))
    pos = 0.0; arr = np.zeros(n)
    for i in range(201, n):
        if pos == 0 and sig[i]: pos = 1.0   # 진입
        arr[i] = pos                          # 홀드: 한번 사면 청산 없음
    return pd.Series(arr, index=df.index)


def load_vix():
    """VIX 일봉 로드(긴 것 우선). 없으면 None → VIX 위성 비활성."""
    for fn in ("vix_long.csv", "vix_full.csv", "vix.csv"):
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            try:
                return pd.read_csv(p, parse_dates=["date"], index_col="date")["vix"]
            except Exception:
                pass
    return None


def dip_entry_active(df, rsi_th=DIP_RSI):
    """코인 저점 '신규 진입 트리거'가 마지막 봉에서 활성인가(약세 & RSI≤25). 갓 배포 봇의 늦은매수 방지."""
    c = df["c"].values
    if len(c) < 200: return False
    s33 = sma(c, 33); s200 = sma(c, 200); r = rsi(c, 14)
    weak = (s33[-1] < s200[-1]) or (c[-1] < s200[-1])
    return bool(weak and not np.isnan(r[-1]) and r[-1] <= rsi_th)


def vix_entry_active(df, vix, vix_th=VIX_TH, fear_rsi=VIX_FEAR_RSI, rev_rsi=VIX_REV_RSI):
    """미국 VIX저점 '신규 진입 트리거'가 마지막 봉에서 활성인가. (과거에 켜진 홀드 잔존과 구분 — 스테일 매수 방지)"""
    c = df["c"].values
    if len(c) < 200: return False
    r = rsi(c, 14)
    s20 = sma(c, 20); s60 = sma(c, 60); s120 = sma(c, 120); s200 = sma(c, 200)
    vrev = (s20[-1] < s60[-1]) and (s60[-1] < s120[-1]) and (s120[-1] < s200[-1])
    v = vix.reindex(df.index, method="ffill").values[-1] if vix is not None else np.nan
    vfear = (not np.isnan(v)) and (v >= vix_th)
    return bool((vfear and r[-1] <= fear_rsi) or (vrev and r[-1] <= rev_rsi))


def _bot_held_symbols():
    """봇이 현재 보유 중인 심볼(state_*.json 합집합). 위성 '홀드 지속 vs 신규 진입' 구분용.
    role 미기록이라 자산군으로만 구분(코인 보유=dip 후보·주식 보유=vix 후보)."""
    held = set()
    try:
        for fn in os.listdir(BOT_DIR):
            if fn.startswith("state_") and fn.endswith(".json"):
                try:
                    d = _json.load(open(os.path.join(BOT_DIR, fn), encoding="utf-8"))
                    pos = d.get("positions", d)
                    for sym, p in (pos or {}).items():
                        if float(p.get("qty", 0) or 0) > 0: held.add(sym)
                except Exception:
                    pass
    except Exception:
        pass
    return held


def compute_satellites(ohlc, coin_dip, us_vix, account, held=None, exclude=None,
                       asof=None, series=None):
    """위성 신호 → {symbol: {role, weight, stop_price, price, atr}}. 심볼당 1개.
    ★신규 매수는 '진입 트리거가 지금 활성'일 때만(스테일 방지). 이미 보유 중이면 청산규칙까지 홀드 지속.
    코인 저점(트레일4·총 SAT_COIN_W 상한) + 미국 VIX(홀드·총 SAT_VIX_W 균등·손절없음)."""
    held = held or set(); exclude = exclude or set()
    sats = {}
    # 1) 코인 저점매수 위성 (BTC/ETH/SOL) — 신규=진입활성·보유=트레일4 살아있으면 지속
    if coin_dip:
        cand = []
        for s in COIN_DIP_UNIV:
            if s not in ohlc or s in exclude or len(ohlc[s]) < 200: continue
            on = dip_pos(ohlc[s]).iloc[-1] > 0          # 규칙상 현재 보유(트레일4 미이탈)
            if s in held:
                if on: cand.append(s)                    # 보유 지속
            elif dip_entry_active(ohlc[s]) and on:
                cand.append(s)                           # 신규 저점 진입(트리거 지금 활성)
        for s in cand:
            df = ohlc[s]; price = float(df["c"].iloc[-1])
            a_val = float(atr(df["h"].values, df["l"].values, df["c"].values, 20)[-1])
            w, _stop, _ = size_position(price, a_val)
            w = min(w, SAT_COIN_W / max(1, len(cand)))
            sats[s] = {"role": "dip", "weight": w, "stop_price": None, "price": price, "atr": a_val}
    # 2) 미국 VIX저점 위성 (지수ETF + 옵션: PIT 저변동주) — 신규=진입활성·보유=영구홀드
    if us_vix:
        vix = load_vix()
        cand = []
        for s in vix_universe(asof, series):
            if s not in ohlc or s in exclude or len(ohlc[s]) < 200: continue
            if s in held:
                cand.append(s)                           # 홀드 지속(청산 없음)
            elif vix_entry_active(ohlc[s], vix):
                cand.append(s)                           # 신규 VIX저점 진입(트리거 지금 활성)
        for s in cand:
            df = ohlc[s]; price = float(df["c"].iloc[-1])
            a_val = float(atr(df["h"].values, df["l"].values, df["c"].values, 20)[-1])
            w = min(SAT_VIX_W / max(1, len(cand)), MAX_WEIGHT)
            sats[s] = {"role": "vix", "weight": w, "stop_price": None, "price": price, "atr": a_val}
    return sats


def _load_mcap_top(n=SPLIT_TOP):
    """시총 상위 종목 목록(_us_mcap.csv). 없으면 빈 리스트 → 분할 타점 비활성."""
    p = os.path.join(DATA_DIR, "_us_mcap.csv")
    if not os.path.exists(p):
        return []
    try:
        return pd.read_csv(p)["symbol"].astype(str).tolist()[:n]
    except Exception:
        return []


def split_entry_active(df, rsi_th=SPLIT_RSI, rearm_rsi=SPLIT_REARM_RSI):
    """분할(개별) 저점 타점이 '마지막 봉에서 신규 활성'인가.
    조건: 3중 역배열(s20<s60<s120) & RSI<=rsi_th, 단 직전에 RSI>rearm_rsi 회복이 있어야 함
    (같은 조정을 매일 재발동하지 않도록 — 스테일/중복 매수 방지)."""
    c = df["c"].values
    if len(c) < 260:
        return False
    r = rsi(c, 14)
    s20 = sma(c, 20); s60 = sma(c, 60); s120 = sma(c, 120)
    if not (s20[-1] < s60[-1] < s120[-1]) or not (r[-1] <= rsi_th):
        return False
    # 재장전 확인: 마지막 'RSI>rearm' 이후에 이미 타점이 켜졌었다면 중복이므로 제외
    for i in range(len(c) - 2, max(0, len(c) - 400), -1):
        if r[i] > rearm_rsi:
            return True                      # 최근 회복 이후 첫 과매도 → 신규
        if (s20[i] < s60[i] < s120[i]) and r[i] <= rsi_th:
            return False                     # 회복 없이 이미 켜졌던 구간 → 중복
    return True


def split_taps(coin_ohlc, exclude=None):
    """시총 상위 SPLIT_TOP 중 분할 타점이 지금 활성인 종목 → [(symbol, price)]."""
    exclude = exclude or set()
    out = []
    for s in _load_mcap_top():
        if s in exclude or is_coin(s):
            continue
        try:
            df = ohlc_of(s, coin_ohlc)
        except Exception:
            continue
        if len(df) < 260:
            continue
        if split_entry_active(df):
            out.append((s, float(df["c"].iloc[-1])))
    return out


def vix_universe(asof=None, series=None):
    """VIX 위성 유니버스 — 규칙기반(미래참조 없음).
    기본: 지수 ETF만(생존편향 0). VIX_LOWVOL_PCT>0 이면 asof 시점 PIT 구성종목 중
    직전 252일 변동성 하위 x% 를 추가한다(하락장 방어↑·수익↓)."""
    univ = list(VIX_UNIV)
    if not VIX_LOWVOL_PCT or series is None or asof is None:
        return univ
    mem = members_at(asof)
    vols = {}
    for s, ser in series.items():
        if is_coin(s) or s in univ:
            continue
        if mem is not None and s not in mem:
            continue
        prev = ser[ser.index < asof].dropna()
        if len(prev) < MIN_HIST:
            continue
        vols[s] = prev.pct_change().dropna().tail(VOL_WIN).std() * np.sqrt(252)
    if not vols:
        return univ
    n = max(1, int(len(vols) * VIX_LOWVOL_PCT / 100.0))
    return univ + sorted(vols, key=lambda s: vols[s])[:n]


def zscore_pick(series, asof, universe_keys):
    """asof 시점(미래참조 없음, asof 이전 데이터만) 변동성+모멘텀 z 상위 TOPN 선별.
    ★유니버스도 asof 시점 실제 구성종목으로 제한(편입 전 종목 선택 = 미래참조 차단)."""
    mem = members_at(asof)
    scores = {}
    for s in universe_keys:
        if mem is not None and not is_coin(s) and s not in mem: continue
        prev = series[s][series[s].index < asof]
        if len(prev) < MIN_HIST: continue
        ret = prev.pct_change().dropna()
        vol = ret.tail(VOL_WIN).std() * np.sqrt(252)
        mom = prev.iloc[-1] / prev.iloc[-MOM_WIN] - 1
        scores[s] = (vol, mom)
    if len(scores) < TOPN: return [], scores, {}
    vols = np.array([scores[s][0] for s in scores]); moms = np.array([scores[s][1] for s in scores])
    zv = (vols - vols.mean()) / (vols.std() or 1); zm = (moms - moms.mean()) / (moms.std() or 1)
    z = {s: zv[i] + zm[i] for i, s in enumerate(scores)}
    top = sorted(z, key=lambda s: -z[s])[:TOPN]
    return top, scores, z


def mdd(eq):
    pk = np.maximum.accumulate(eq); return (1 - eq / pk).max() * 100


# ── 모드 1: 워크포워드 검증 (sanity) ──────────────────────────────────────────
def verify(series, ohlc):
    tpos = {s: turtle_pos(ohlc[s], *turtle_params(s)) for s in ohlc}
    L = ["[검증] 스크리너+터틀(SMA완화 C) 워크포워드 2016~2026 (수수료 반영)\n",
         f"  {'연도':6s}{'터틀수익':>9s}{'터틀MDD':>9s}{'선별중앙':>9s}{'전체중앙':>9s}{'상대%p':>8s}"]
    daily = []; rel = []
    for yr in YEARS:
        t0 = pd.Timestamp(f"{yr}-01-01"); t1 = pd.Timestamp(f"{yr}-12-31")
        top, scores, z = zscore_pick(series, t0, series.keys())
        if not top: continue
        fwd = {}
        for s in scores:
            cur = series[s][(series[s].index >= t0) & (series[s].index <= t1)]
            if len(cur) >= 30: fwd[s] = (cur.iloc[-1] / cur.iloc[0] - 1) * 100
        smed = np.median([fwd[s] for s in top if s in fwd]); amed = np.median(list(fwd.values()))
        curves = []
        for s in top:
            c = series[s]; cyr = c[(c.index >= t0) & (c.index <= t1)]
            r = cyr.pct_change().fillna(0).values
            p = tpos[s].reindex(cyr.index).fillna(0).shift(1).fillna(0).values
            dchg = np.abs(np.diff(p, prepend=0))
            curves.append(np.cumprod(1 + (p * r - dchg * fee_of(s))))
        m = min(len(x) for x in curves); eqy = np.mean([x[:m] for x in curves], axis=0)
        L.append(f"  {yr:<6d}{(eqy[-1]-1)*100:>8.0f}%{mdd(eqy):>8.0f}%{smed:>8.0f}%{amed:>8.0f}%{smed-amed:>7.0f}p")
        # 연속 곡선용 일별
        for s in top:
            pass
        rel.append(smed - amed)
        daily.append(eqy[-1])
    tot = np.prod(daily)
    L.append("")
    L.append(f"  전체기간 복리(연도체인) {tot:.1f}배 · 선별 상대알파(중앙값) 평균 {np.mean(rel):+.0f}%p · 양의알파 {sum(1 for x in rel if x>0)}/{len(rel)}년")
    return "\n".join(L)


# ── 포지션 사이징: 손절시 손실이 계좌의 RISK_PER_TRADE 넘지 않게 (ATR 기반) ──────
def size_position(price, atr_val):
    """리스크기반 비중 = R / (STOP_N × ATR/price). 변동성 클수록 비중↓. MAX_WEIGHT 상한.
    반환 (권장비중, 손절가, ATR%). 손절폭=STOP_N×ATR, 수량=계좌×R/손절폭 → 비중=R/(손절폭/price)."""
    stop_dist = STOP_N * atr_val
    stop_price = price - stop_dist
    atr_pct = atr_val / price
    raw_w = RISK_PER_TRADE / (stop_dist / price) if stop_dist > 0 else 0.0
    weight = min(raw_w, MAX_WEIGHT)
    return weight, stop_price, atr_pct


def vol_regime_scale(df, win=20, base=252):
    """변동성 타게팅 스케일(0.4~1.0) — 최근(win일) 실현변동성이 '평시'(trailing base일 중앙값)보다
    높으면 노출↓. 미래참조 없음(과거만·trailing median). 하락/급락 국면 자동 축소 → 하락장 방어."""
    if not VOL_TARGET: return 1.0
    c = df["c"].values
    if len(c) < base + win: return 1.0
    ret = pd.Series(c).pct_change()
    rv = (ret.rolling(win).std() * np.sqrt(252)).values
    cur = rv[-1]; med = np.nanmedian(rv[-base:])
    if not (cur and cur == cur and med and med == med and cur > 0): return 1.0
    return float(np.clip(med / cur, VOL_FLOOR, 1.0))


# ── 모드 2: 현재 시점 실전 후보 + 포지션 사이징 ───────────────────────────────
def live(series, coin_ohlc, account=None, coin_dip=False, us_vix=False):
    asof = max(s.index[-1] for s in series.values())
    asof = pd.Timestamp(asof.date()) + pd.Timedelta(days=1)   # 마지막 데이터일까지 포함
    top, scores, z = zscore_pick(series, asof, series.keys())
    ohlc = load_ohlc_subset(set(top) | set(COIN_DIP_UNIV) | set(vix_universe(asof, series)), coin_ohlc)  # 필요분만
    accs = f" · 계좌 {account:,.0f}" if account else ""
    L = [f"[실전] 현재 후보 + 포지션 사이징 — 기준일 {(asof - pd.Timedelta(days=1)).date()}{accs} (미래참조 없음)\n",
         f"  리스크규칙: 종목당 손절손실 ≤ 계좌의 {RISK_PER_TRADE*100:.1f}% · 종목 비중상한 {MAX_WEIGHT*100:.0f}% · 손절 {STOP_N}N\n",
         f"  {'순위':4s}{'종목':7s}{'자산':5s}{'z':>6s}{'모멘텀%':>8s}{'터틀상태':>13s}{'손절가':>10s}{'비중%':>7s}{'금액':>13s}"]
    sum_w = 0.0; coin_w = 0.0; stock_w = 0.0; rows_entry = 0
    for rk, s in enumerate(top, 1):
        vol, mom = scores[s]
        df = ohlc[s]; eN, xN, smaN = turtle_params(s)
        pos = turtle_pos(df, eN, xN, smaN).iloc[-1]
        c = df["c"]; sma = c.rolling(smaN).mean().iloc[-1]
        hh = df["h"].rolling(eN).max().shift(1).iloc[-1]
        price = c.iloc[-1]
        a_val = atr(df["h"].values, df["l"].values, df["c"].values, 20)[-1]
        if pos >= 1.0: state = "보유중(추세)"
        elif pos == 0.5: state = "절반보유"
        elif price > sma and price >= hh * 0.97: state = "진입임박"
        elif price > sma: state = "관심(SMA위)"
        else: state = "관망(추세下)"
        entryable = state != "관망(추세下)"
        if entryable:
            w, stop_price, atr_pct = size_position(price, a_val)
            w *= vol_regime_scale(df)     # ★변동성 타게팅(하락장 방어)
            sum_w += w; rows_entry += 1
            if is_coin(s): coin_w += w
            else: stock_w += w
            wstr = f"{w*100:.1f}%"; spstr = f"{stop_price:,.2f}"
            amtstr = f"{w*account:,.0f}" if account else "-"
        else:
            wstr = "-"; spstr = "-"; amtstr = "-"
        asset = "코인" if is_coin(s) else "주식"
        L.append(f"  {rk:<4d}{s:7s}{asset:5s}{z[s]:>6.2f}{mom*100:>7.0f}%{state:>13s}{spstr:>10s}{wstr:>7s}{amtstr:>13s}")
    L.append("")
    L.append(f"  진입대상 {rows_entry}종목 합산비중 {sum_w*100:.0f}% (주식 {stock_w*100:.0f}% · 코인 {coin_w*100:.0f}%) · 현금 {max(0,1-sum_w)*100:.0f}%")
    if sum_w > 1.0:
        L.append(f"  ⚠ 합산비중 100% 초과 → 전 종목 비중을 {1/sum_w:.2f}배로 비례 축소(레버리지 금지)")
    if stock_w > 0.6 or coin_w > 0.6:
        L.append("  ⚠ 한 자산군 비중 60% 초과 = 쏠림. 섹터(예: 반도체)까지 겹치면 추가로 비중 제한 권장.")
    L.append("  진입규칙: '진입임박/관심'에서 신고가 돌파+SMA위 시 진입 · 손절가 이탈시 청산 · 저점이탈 절반익절 · 절반 트레일2ATR")
    # ── 위성(병합) 타점 — 신규 진입 트리거가 '지금' 활성인 것만(스테일 매수 방지) ──
    if coin_dip or us_vix:
        sats = compute_satellites(ohlc, coin_dip, us_vix, account,
                                  held=_bot_held_symbols(), exclude=set(top))
        L.append("")
        L.append(f"  [위성·병합] 코인저점={coin_dip}·미국VIX={us_vix} (본체=터틀, 위성=역추세 보조·알파없음·약세장참여용)")
        if sats:
            for sym, v in sorted(sats.items(), key=lambda kv: -kv[1]["weight"]):
                amt = f"{v['weight']*account:,.0f}" if account else "-"
                role = "코인저점(트레일4)" if v["role"] == "dip" else "VIX저점(홀드)"
                L.append(f"    {sym:6s} {role:16s} 비중 {v['weight']*100:.1f}% 금액 {amt}")
        else:
            L.append("    (현재 신규 진입 트리거 없음 — 약세장 바닥/VIX공포 아님. 강세장엔 본체 터틀만 작동.)")
    sp = split_taps(coin_ohlc, exclude=set(top))
    L.append(f"  [분할저점] 시총상위 {SPLIT_TOP} · 3중역배열 · RSI≤{SPLIT_RSI} (표시전용·수동 판단)")
    if sp:
        for sym, px in sp:
            L.append(f"    ★ {sym:6s} 현재가 {px:,.2f}")
    else:
        L.append("    (현재 분할 타점 없음)")
    L.append("  ※ 정량 스크리닝·사이징 출력일 뿐 종목추천 아님. 최종판단 본인. 무레버리지·BTC·ETH·우량만. 슬리피지/세금 미반영.")
    return "\n".join(L)


def get_signals(series, coin_ohlc, account=None, coin_dip=False, us_vix=False):
    """봇 실행기용 구조화 신호 — signals.json 스키마(broker_base.py 참조)로 반환.
    ★효율: 종가(series)로 선별 → TOP8+위성 OHLC만 로드(전체 500 OHLC 로드 안 함).
    본체=터틀(TOP8 추세) + 위성 taps. 심볼당 1신호로 비중 합산. 자산군별 무레버리지 축소."""
    asof_date = max(s.index[-1] for s in series.values())
    asof = pd.Timestamp(asof_date.date()) + pd.Timedelta(days=1)
    top, scores, z = zscore_pick(series, asof, series.keys())
    ohlc = load_ohlc_subset(set(top) | set(COIN_DIP_UNIV) | set(vix_universe(asof, series)), coin_ohlc)  # 필요분만

    # 심볼당 누적 컨트리뷰션: {sym: {weight, stop_price, price, atr, z, params, state, roles[]}}
    acc = {}
    def _add(sym, w, stop_price, price, a_val, zval, params, state, role):
        e = acc.setdefault(sym, {"weight": 0.0, "stop_price": None, "price": price,
                                 "atr": a_val, "z": zval, "params": params,
                                 "states": [], "roles": []})
        e["weight"] += w
        if stop_price is not None: e["stop_price"] = stop_price  # 터틀 손절이 우선(위성은 None)
        e["states"].append(state); e["roles"].append(role)

    # 1) 터틀 본체 (TOP8) ----------------------------------------------------
    for s in top:
        df = ohlc[s]; eN, xN, smaN = turtle_params(s)
        pos = turtle_pos(df, eN, xN, smaN).iloc[-1]
        c = df["c"]; smav = c.rolling(smaN).mean().iloc[-1]
        hh = df["h"].rolling(eN).max().shift(1).iloc[-1]
        price = float(c.iloc[-1])
        a_val = float(atr(df["h"].values, df["l"].values, df["c"].values, 20)[-1])
        if pos >= 1.0: state = "hold"
        elif pos == 0.5: state = "half"
        elif price > smav and price >= hh * 0.97: state = "entry_soon"
        elif price > smav: state = "watch"
        else: state = "flat"
        if state != "flat":
            w, stop_price, _ = size_position(price, a_val)
            w *= vol_regime_scale(df)     # ★변동성 타게팅: 고변동 국면 노출↓(하락장 방어)
            _add(s, w, round(stop_price, 4), round(price, 4), round(a_val, 4),
                 round(float(z[s]), 2), {"eN": eN, "xN": xN, "smaN": smaN}, state, "turtle")

    # 2) 종목당 상한(MAX_WEIGHT) + 자산군별 무레버리지 정규화 (터틀만) ----------
    for e in acc.values():
        e["weight"] = min(e["weight"], MAX_WEIGHT)
    for grp in ("coin", "stock"):
        tot = sum(e["weight"] for sym, e in acc.items()
                  if (is_coin(sym) == (grp == "coin")))
        if tot > 1.0:                       # 합산 100% 초과 → 비례 축소(레버리지 금지)
            for sym, e in acc.items():
                if is_coin(sym) == (grp == "coin"):
                    e["weight"] *= 1.0 / tot

    # 3) 거래 신호 직렬화 (★순수 터틀만 — 봇이 실제 매매) ----------------------
    sigs = []
    for sym, e in acc.items():
        st = next((s for s in e["states"] if s in ("hold", "half", "entry_soon", "watch")), e["states"][0])
        sigs.append({
            "symbol": sym, "asset": "coin" if is_coin(sym) else "stock",
            "price": e["price"], "state": st, "entryable": e["weight"] > 0,
            "target_weight": round(e["weight"], 4),
            "stop_price": round(e["stop_price"], 4) if e["stop_price"] else None,
            "atr": e["atr"], "z": e["z"], "role": "turtle", "params": e["params"],
        })
    sigs.sort(key=lambda x: -x["target_weight"])

    # 4) 위성 = ★표시/참고 타점만 (별도 taps 필드·executor는 무시·매매 안 함) ----
    #    감사 결과 위성 실매매는 알파X + 무청산홀드·role혼동 버그 → 봇은 순수 터틀. VIX/코인저점은 참고.
    taps = []
    if coin_dip or us_vix:
        sats = compute_satellites(ohlc, coin_dip, us_vix, account,
                                  held=set(), exclude=set(acc.keys()),
                                  asof=asof, series=series)  # held=∅: 표시용은 '신규 트리거'만
        for sym, sv in sats.items():
            taps.append({"symbol": sym, "asset": "coin" if is_coin(sym) else "stock",
                         "role": sv["role"], "price": round(sv["price"], 4),
                         "ref_weight": round(sv["weight"], 4), "note": "표시전용·봇 매매 안함"})
    # ★분할(개별) 저점 타점 — 시총 상위 초거대기업만. 표시전용(수동 판단 후 진입).
    for sym, px in split_taps(coin_ohlc, exclude=set(acc.keys())):
        taps.append({"symbol": sym, "asset": "stock", "role": "split",
                     "price": round(px, 4), "ref_weight": 0.0,
                     "note": "분할저점(시총상위·3중역배열·RSI≤30)·표시전용·봇 매매 안함"})
    # 5) ★보유종목 최신가 — executor의 손절 판정/청산에 실가격이 필요하다.
    #    없으면 executor가 평단으로 폴백해 '손익 0 청산'이라는 가짜 결과가 나온다(실측 버그).
    prices = {s["symbol"]: s["price"] for s in sigs}
    for st_file in ("state_kis.json", "state_binance.json"):
        p = os.path.join(BOT_DIR, st_file)
        if not os.path.exists(p):
            continue
        try:
            held = _json.load(open(p, encoding="utf-8")).get("positions", {})
        except Exception:
            continue
        for sym in held:
            if sym in prices:
                continue
            try:                       # 신호에서 빠진 보유종목도 최신 종가를 실어 보낸다
                prices[sym] = round(float(series[sym].iloc[-1]), 4)
            except Exception:
                pass

    return {"asof": str(asof_date.date()), "account": account,
            "risk_per_trade": RISK_PER_TRADE, "max_weight": MAX_WEIGHT, "stop_n": STOP_N,
            "signals": sigs, "taps": taps, "prices": prices}


def main():
    args = sys.argv[1:]
    account = None
    for a in args:
        if a.startswith("--account="):
            account = float(a.split("=")[1])
    # 위성(병합): --satellites = 둘 다, 개별 --coin-dip / --vix-sat
    coin_dip = ("--coin-dip" in args) or ("--satellites" in args)
    us_vix = ("--vix-sat" in args) or ("--satellites" in args)
    # ★라이브(emit/live)는 load_live(종가매트릭스·빠름)·verify(백테스트)는 load_all(전체 OHLC)
    if "--emit-signals" in args:
        import json
        series, coin_ohlc = load_live()
        sig = get_signals(series, coin_ohlc, account, coin_dip=coin_dip, us_vix=us_vix)
        path = os.path.join(BOT_DIR, "signals.json")
        json.dump(sig, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"signals.json 생성: {path} (★거래신호=순수터틀 {len(sig['signals'])}종목 · "
              f"참고 taps {len(sig.get('taps', []))}개[표시전용·매매X], 기준일 {sig['asof']})")
        return
    out = []
    if "--live" in args and "--verify" not in args:
        series, coin_ohlc = load_live()
        out = [live(series, coin_ohlc, account, coin_dip, us_vix)]
    elif "--verify" in args and "--live" not in args:
        series, ohlc = load_all()
        out = [verify(series, ohlc)]
    else:  # 둘 다(기본): verify엔 전체 OHLC 필요 → load_all(ohlc를 live의 coin_ohlc로도 재사용)
        series, ohlc = load_all()
        out = [verify(series, ohlc), "", "=" * 70, "", live(series, ohlc, account, coin_dip, us_vix)]
    txt = "\n".join(out); print(txt)
    open(os.path.join(DATA_DIR, "BATTO_SCREENER_LIVE_LOG.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
