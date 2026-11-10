# -*- coding: utf-8 -*-
"""BATTO 일봉장기 VIX 매수타점 — RSI×VIX 그리드 서치 (골든크로스 추세필터 추가).

매수 신호 = (RSI <= rth) AND (VIX >= vth) [AND 골든크로스상태]
 - 골든크로스상태 = 4h200MA(일별 정렬) > 일봉200MA
 - min_gap=5봉, forward EV/alpha 측정
목표: 수익(alpha) 높고 타점 적당(너무 적으면 제외).
출력: GC ON/OFF 그리드 표 + 히트맵 PDF + 최적 추천.
"""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import golden_strategy as G
import backtest_4h as B4
import fib_backtest as FB

plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = G.HERE
RSIS = [33, 34, 35, 36, 37]
VIXS = [16, 18, 20, 22, 25, 28, 30]
FWD = [10, 20, 40, 60]
MIN_GAP = 5
MIN_COUNT = 40          # '타점 적당히' — 이 미만이면 추천 제외
SCORE_N = 20            # 추천 기준 forward 기간


def rsi_wilder(c, n=14):
    c = np.asarray(c, float); d = np.zeros_like(c); d[1:] = np.diff(c)
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().values
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().values
    rs = np.divide(ru, rd, out=np.full_like(ru, np.nan), where=rd > 0)
    out = 100 - 100/(1 + rs); out[:n] = np.nan
    return out


def precompute(uni, vix):
    data = {}
    for t in uni:
        try:
            df = G.load_price(t); h4 = B4.load_4h(t)
        except Exception:
            continue
        if len(h4) < 220 or len(df) < 260:
            continue
        c = df["c"].values.astype(float); n = len(c)
        r = rsi_wilder(c)
        dma = pd.Series(c).rolling(200).mean().values
        h4d = FB.daily_4h_ma(h4); h4_al = h4d.reindex(df.index, method="ffill").values
        vix_al = vix["vix"].reindex(df.index, method="ffill").values
        gc = np.where(np.isnan(h4_al) | np.isnan(dma), False, h4_al > dma)
        valid = ~(np.isnan(r) | np.isnan(vix_al) | np.isnan(dma) | np.isnan(h4_al))
        fwd = {N: np.full(n, np.nan) for N in FWD}
        for N in FWD:
            if n > N:
                fwd[N][:n-N] = (c[N:]/c[:n-N] - 1)*100
        base = {}
        for N in FWD:
            m = valid & ~np.isnan(fwd[N])
            base[N] = float(np.nanmean(fwd[N][m])) if m.sum() >= 20 else np.nan
        data[t] = dict(c=c, r=r, vix=vix_al, gc=gc, valid=valid, fwd=fwd, base=base, n=n,
                       dates=df.index, dma=dma, h4_al=h4_al)
    return data


def evaluate(data, rth, vth, gc_on):
    perN = {N: [] for N in FWD}; perN_a = {N: [] for N in FWD}; count = 0; entries = []
    for t, d in data.items():
        cond = d["valid"] & (d["r"] <= rth) & (d["vix"] >= vth)
        if gc_on:
            cond = cond & d["gc"]
        last = -10**9
        for i in np.where(cond)[0]:
            if i - last < MIN_GAP:
                continue
            last = i; count += 1; entries.append((t, i))
            for N in FWD:
                v = d["fwd"][N][i]
                if v == v:
                    perN[N].append(v)
                    if d["base"][N] == d["base"][N]:
                        perN_a[N].append(v - d["base"][N])
    res = {"count": count, "entries": entries}
    for N in FWD:
        a = np.array(perN[N]); aa = np.array(perN_a[N])
        res[f"ev{N}"] = float(a.mean()) if len(a) else np.nan
        res[f"wr{N}"] = float((a > 0).mean()*100) if len(a) else np.nan
        res[f"al{N}"] = float(aa.mean()) if len(aa) else np.nan
        res[f"t{N}"] = (float(a.mean()/(a.std(ddof=1)/np.sqrt(len(a))))
                        if len(a) > 2 and a.std(ddof=1) > 0 else np.nan)
    return res


def grid(data, gc_on):
    G_ = {}
    for rth in RSIS:
        for vth in VIXS:
            G_[(rth, vth)] = evaluate(data, rth, vth, gc_on)
    return G_


def fmt_table(G_, field, title):
    lines = [f"\n  [{title}]   (행 RSI≤ / 열 VIX≥)"]
    lines.append("  RSI\\VIX " + "".join(f"{v:>8d}" for v in VIXS))
    for rth in RSIS:
        row = f"  {rth:>6d} "
        for vth in VIXS:
            x = G_[(rth, vth)][field]
            row += f"{x:>8.2f}" if x == x else f"{'-':>8s}"
        lines.append(row)
    return "\n".join(lines)


def fmt_count(G_):
    lines = ["\n  [타점 수]   (행 RSI≤ / 열 VIX≥)"]
    lines.append("  RSI\\VIX " + "".join(f"{v:>8d}" for v in VIXS))
    for rth in RSIS:
        row = f"  {rth:>6d} "
        for vth in VIXS:
            row += f"{G_[(rth, vth)]['count']:>8d}"
        lines.append(row)
    return "\n".join(lines)


def heatmap(pdf, G_, field, title, cmap):
    M = np.full((len(RSIS), len(VIXS)), np.nan)
    for i, rth in enumerate(RSIS):
        for j, vth in enumerate(VIXS):
            M[i, j] = G_[(rth, vth)][field]
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(VIXS))); ax.set_xticklabels([f"≥{v}" for v in VIXS])
    ax.set_yticks(range(len(RSIS))); ax.set_yticklabels([f"≤{r}" for r in RSIS])
    ax.set_xlabel("VIX 임계"); ax.set_ylabel("RSI 임계")
    ax.set_title(title, fontsize=14, weight="bold")
    for i in range(len(RSIS)):
        for j in range(len(VIXS)):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}" if v == v else "-", ha="center", va="center",
                    fontsize=9, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    uni = json.load(open(os.path.join(HERE, "fib_universe.json"), encoding="utf-8"))["union"]
    vix = G.load_vix()
    data = precompute(uni, vix)
    print(f"분석종목 {len(data)}/{len(uni)}")

    gon = grid(data, True)
    goff = grid(data, False)

    # baseline 평균(시장)
    bavg = {N: np.nanmean([d["base"][N] for d in data.values()]) for N in FWD}
    bline = "시장 baseline: " + "  ".join(f"@{N}={bavg[N]:+.2f}%" for N in FWD)

    out_txt = os.path.join(HERE, "BATTO_VIX_GC_RSI_GRID.txt")
    blocks = []
    blocks.append(bline)
    blocks.append("\n############ 골든크로스 ON ############")
    blocks.append(fmt_count(gon))
    blocks.append(fmt_table(gon, f"al{SCORE_N}", f"ALPHA@{SCORE_N} (초과수익%)"))
    blocks.append(fmt_table(gon, f"ev{SCORE_N}", f"절대 EV@{SCORE_N} (%)"))
    blocks.append(fmt_table(gon, f"al60", "ALPHA@60 (초과수익%)"))
    blocks.append("\n############ 골든크로스 OFF (비교) ############")
    blocks.append(fmt_count(goff))
    blocks.append(fmt_table(goff, f"al{SCORE_N}", f"ALPHA@{SCORE_N} (초과수익%)"))

    # 추천: GC ON, count>=MIN_COUNT 중 alpha@SCORE_N 최대
    cand = [(k, v) for k, v in gon.items() if v["count"] >= MIN_COUNT and v[f"al{SCORE_N}"] == v[f"al{SCORE_N}"]]
    cand.sort(key=lambda kv: kv[1][f"al{SCORE_N}"], reverse=True)
    rec_lines = [f"\n############ 추천 (GC ON, 타점≥{MIN_COUNT}, ALPHA@{SCORE_N} 상위) ############"]
    for (rth, vth), v in cand[:5]:
        rec_lines.append(f"  RSI≤{rth}  VIX≥{vth}  | 타점 {v['count']:3d}  "
                         f"EV@20 {v['ev20']:+.2f}  ALPHA@20 {v['al20']:+.2f}  t@20 {v['t20']:+.2f}  "
                         f"ALPHA@60 {v['al60']:+.2f}  승률@20 {v['wr20']:.0f}%")
    if not cand:
        rec_lines.append("  (조건 충족 조합 없음 — MIN_COUNT 완화 필요)")

    txt = "\n".join(blocks + rec_lines)
    print(txt)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    out_pdf = os.path.join(HERE, "BATTO_VIX_GC_RSI_GRID.pdf")
    with PdfPages(out_pdf) as pdf:
        heatmap(pdf, gon, f"al{SCORE_N}", f"골든크로스 ON · ALPHA@{SCORE_N} (초과수익%)", "RdYlGn")
        heatmap(pdf, gon, "count", "골든크로스 ON · 타점 수", "Blues")
        heatmap(pdf, gon, "al60", "골든크로스 ON · ALPHA@60 (초과수익%)", "RdYlGn")
        heatmap(pdf, goff, f"al{SCORE_N}", f"골든크로스 OFF · ALPHA@{SCORE_N} (초과수익%)", "RdYlGn")
    print("\n저장:", out_txt, "\n     ", out_pdf)

    # 추천 결과를 json으로 (Pine 반영용)
    if cand:
        (rth, vth), v = cand[0]
        json.dump({"rsi": rth, "vix": vth, "count": v["count"], "al20": v["al20"], "ev20": v["ev20"]},
                  open(os.path.join(HERE, "batto_gc_reco.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
