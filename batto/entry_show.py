# -*- coding: utf-8 -*-
"""진입 기준 — 개념 + 실제 NVDA 비교.
컨플루언스 causal(중간) vs strict(엄격) · 추세선 TOL 0.3% vs 0.5% 4조합.
숏=2022 · 롱=2024. 각 차트에 '추세선 닿음 / 존 도달' 정렬 진단 명시."""
import os, sys, warnings
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import golden_strategy as G

plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
UP, DN, OBC, FVC, TLC = G.UPC, G.DNC, G.OBC, G.FVC, "#2563eb"
OUT = os.path.join(G.HERE, "진입기준_개념과실제.pdf")


def candle(ax, x, o, h, l, c, w=0.4):
    col = UP if c >= o else DN
    ax.plot([x, x], [l, h], color=col, lw=1.2, zorder=4)
    ax.add_patch(Rectangle((x-w/2, min(o, c)), w, max(abs(c-o), 1e-9), facecolor=col, edgecolor=col, lw=0.4, zorder=5))


def concept(pdf):
    LD = [(5.0,5.05,4.7,4.75),(4.75,4.8,4.4,4.5),(4.5,4.55,4.2,4.25),(4.25,4.3,4.0,4.05),
          (4.05,4.1,3.7,3.78),(3.78,3.95,3.74,3.9),(3.9,3.95,3.78,3.82),
          (3.82,4.15,3.6,4.1),(4.1,4.4,4.05,4.35),(4.35,4.6,4.3,4.55)]
    fig, (aL, aS) = plt.subplots(1, 2, figsize=(16, 8))
    for i, (o,h,l,c) in enumerate(LD): candle(aL, i, o, h, l, c)
    aL.axhline(3.7, color="#555", ls="--", lw=1); aL.text(0, 3.72, " 전저점", fontsize=9, color="#555")
    aL.add_patch(Rectangle((5.6, 3.82), 3, 0.13, facecolor=OBC, alpha=0.3, edgecolor=OBC))
    aL.scatter([7], [3.55], marker="^", s=220, c="lime", ec="k", zorder=9)
    aL.annotate("①전저점 스윕\n(꼬리로 깨고 종가 복귀)", (7,3.62),(8.4,3.5), fontsize=9.5, ha="center", arrowprops=dict(arrowstyle="->"))
    aL.annotate("②OB/FVG 존 도달", (6,3.88),(3.3,4.45), fontsize=9.5, ha="center", color=OBC, arrowprops=dict(arrowstyle="->", color=OBC))
    aL.annotate("③롱 진입", (7,3.55),(8.7,3.95), fontsize=10, ha="center", color="green", arrowprops=dict(arrowstyle="->", color="green"))
    aL.set_title("롱 진입 개념", fontsize=13, weight="bold"); aL.set_xticks([]); aL.grid(alpha=0.12, axis="y")
    SD = [(10-o,10-l,10-h,10-c) for (o,h,l,c) in LD]
    for i, (o,h,l,c) in enumerate(SD): candle(aS, i, o, h, l, c)
    aS.axhline(10-3.7, color="#555", ls="--", lw=1); aS.text(0, 10-3.68, " 전고점", fontsize=9, color="#555")
    aS.add_patch(Rectangle((5.6, 10-3.95), 3, 0.13, facecolor=OBC, alpha=0.3, edgecolor=OBC))
    aS.scatter([7], [10-3.55], marker="v", s=220, c="red", ec="k", zorder=9)
    aS.annotate("①전고점 스윕", (7,10-3.62),(8.4,10-3.5), fontsize=9.5, ha="center", arrowprops=dict(arrowstyle="->"))
    aS.annotate("②OB/FVG 존 도달", (6,10-3.88),(3.3,10-4.45), fontsize=9.5, ha="center", color=OBC, arrowprops=dict(arrowstyle="->", color=OBC))
    aS.annotate("③숏 진입", (7,10-3.55),(8.7,10-3.95), fontsize=10, ha="center", color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
    aS.set_title("숏 진입 개념 (거울)", fontsize=13, weight="bold"); aS.set_xticks([]); aS.grid(alpha=0.12, axis="y")
    fig.suptitle("진입 기준 개념 — 트리거(추세선/스윕) 터치 + OB/FVG 존 도달 → 진입", fontsize=14, weight="bold")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def find_trendline(h, l, c, sh, sl, tj, d, tol):
    n = len(c); K, LOOK = G.K, G.LOOK
    if d == "L":
        sli = np.where(sl)[0]
        for i in range(1, len(sli)):
            s1, s2 = sli[i-1], sli[i]
            if l[s2] <= l[s1] or s2-s1 < 3: continue
            slope = (l[s2]-l[s1])/(s2-s1)
            if any(l[k] < (l[s1]+slope*(k-s1))*(1-G.TLPEN) for k in range(s1+1, s2)): continue
            for j in range(s2+K, min(s2+LOOK, n)):
                tlv = l[s2]+slope*(j-s2)
                if l[j] <= tlv*(1+tol) and c[j] > tlv and c[j-1] >= tlv*(1-tol):
                    if j == tj: return s1, s2, slope
                    break
                if c[j] < tlv*(1-tol): break
    else:
        shi = np.where(sh)[0]
        for i in range(1, len(shi)):
            s1, s2 = shi[i-1], shi[i]
            if h[s2] >= h[s1] or s2-s1 < 3: continue
            slope = (h[s2]-h[s1])/(s2-s1)
            if any(h[k] > (h[s1]+slope*(k-s1))*(1+G.TLPEN) for k in range(s1+1, s2)): continue
            for j in range(s2+K, min(s2+LOOK, n)):
                tlv = h[s2]+slope*(j-s2)
                if h[j] >= tlv*(1-tol) and c[j] < tlv and c[j-1] <= tlv*(1+tol):
                    if j == tj: return s1, s2, slope
                    break
                if c[j] > tlv*(1+tol): break
    return None


def pick_sig(df, conf, tol, yr, dw):
    sigs, obs, fvs = G.build_signals(df, conf_mode=conf, tol=tol)
    idx = df.index; ym = np.asarray(idx.year == yr); a = int(np.argmax(ym)); b = a+int(ym.sum())-1
    cand = [s for s in sigs if a <= s["j"] <= b and s["dir"] == dw and s["trig"] == "추세선"]
    if not cand: cand = [s for s in sigs if a <= s["j"] <= b and s["dir"] == dw]
    if not cand: return None, obs, fvs
    return next((s for s in cand if s["grade"] == "OB+FVG"), cand[len(cand)//2]), obs, fvs


def real(pdf, df, conf, tol, yr, dw):
    o, h, l, c = [df[x].values for x in ["o","h","l","c"]]; sh, sl = G.swings(h, l)
    pick, obs, fvs = pick_sig(df, conf, tol, yr, dw)
    confnm = "엄격(strict)" if conf == "strict" else "중간(causal)"
    if pick is None:
        fig, ax = plt.subplots(figsize=(16, 8.5)); ax.axis("off")
        ax.text(0.5, 0.5, f"[NVDA] {yr} {'롱' if dw=='L' else '숏'}\n컨플루언스={confnm}, 추세선 TOL={tol*100:.1f}%\n→ 조건 통과 타점 없음", ha="center", va="center", fontsize=16)
        pdf.savefig(fig); plt.close(fig); return
    j = pick["j"]; long_ = dw == "L"; idx = df.index
    tl = find_trendline(h, l, c, sh, sl, j, dw, tol)

    left = (tl[0] if tl else j-20); i0 = max(0, left-3); i1 = min(len(df)-1, j+18)
    sub = df.iloc[i0:i1+1]; n = len(sub); so, sh_, sl_, sc = [sub[x].values for x in ["o","h","l","c"]]
    RNG = float(np.ptp(sh_))
    fig, ax = plt.subplots(figsize=(16, 8.5))
    for k in range(n): candle(ax, k, so[k], sh_[k], sl_[k], sc[k], 0.6)

    # 추세선
    if tl:
        s1, s2, slope = tl; xs = np.arange(s1, j+4)
        ys = (l[s2] if long_ else h[s2]) + slope*(xs-s2)
        ax.plot(xs-i0, ys, color=TLC, ls="--", lw=1.8, zorder=6)
        for sx, lab in [(s1, "스윙①"), (s2, "스윙②")]:
            yv = l[sx] if long_ else h[sx]
            ax.scatter(sx-i0, yv, marker="o", s=80, c=TLC, ec="white", lw=1, zorder=8)
            ax.annotate(lab, (sx-i0, yv), (sx-i0, yv+(-RNG*0.06 if long_ else RNG*0.06)), fontsize=9, ha="center", color=TLC, weight="bold")
    else:
        x = j-i0; ax.plot([x-4, x+4], [pick["lv"]]*2, color="#777", lw=1.4, zorder=6)

    # 컨플루언스로 채택된 OB/FVG 존만 (conf 모드 기준)
    conf_es = [("OB", e) for e in obs if G._conf_one(j, dw, o, h, l, c, e, conf)] + \
              [("FVG", e) for e in fvs if G._conf_one(j, dw, o, h, l, c, e, conf)]
    zone_touch = False
    for tag, e in conf_es:
        x0pos = (e["form"]-1)-i0 if tag == "OB" else (e["form"]-2)-i0
        x0pos = max(x0pos, 0); bw = (j-i0)-x0pos+1.4
        col = OBC if tag == "OB" else FVC
        ax.add_patch(Rectangle((x0pos-0.4, e["lo"]), bw, e["hi"]-e["lo"], facecolor=col, alpha=0.38, edgecolor=col, lw=1.0, zorder=3))
        ax.annotate(f"{tag}", (x0pos+bw/2-0.4, e["hi"] if tag == "OB" else e["lo"]),
                    (x0pos+bw/2-0.4, (e["hi"]+RNG*0.05) if tag == "OB" else (e["lo"]-RNG*0.05)),
                    fontsize=8.5, ha="center", color=col if tag == "OB" else "#9a6b00", weight="bold",
                    arrowprops=dict(arrowstyle="->", color=col))
        if l[j] <= e["hi"] and h[j] >= e["lo"]: zone_touch = True

    # 진입봉
    x = j-i0; entry = c[j]; my = sl_[x]-RNG*0.04 if long_ else sh_[x]+RNG*0.04
    ax.scatter([x], [my], marker="^" if long_ else "v", s=260, c="lime" if long_ else "red", ec="k", lw=0.9, zorder=10)
    ax.annotate(f"진입 ${entry:.2f}", (x, my), (x+3, my+(-RNG*0.16 if long_ else RNG*0.16)),
                fontsize=10, ha="left", weight="bold", color="green" if long_ else "#d62728",
                arrowprops=dict(arrowstyle="->", color="green" if long_ else "#d62728"))

    # 정렬 진단
    if long_: tl_touch = l[j] <= pick["lv"]*(1+tol); dist = (l[j]/pick["lv"]-1)*100; near = "저가"
    else: tl_touch = h[j] >= pick["lv"]*(1-tol); dist = (h[j]/pick["lv"]-1)*100; near = "고가"
    ok = lambda b: "O" if b else "X"
    diag = (f"[진단]\n추세선 터치: {ok(tl_touch)}  ({near} {dist:+.1f}% vs 추세선)\n"
            f"OB/FVG 존 도달: {ok(zone_touch)}  (진입봉이 존과 겹침)")
    dcol = "#1a7f37" if (tl_touch and zone_touch) else "#d64545"
    bx = 0.015 if long_ else 0.985; ha = "left" if long_ else "right"
    ax.text(bx, 0.97, diag, transform=ax.transAxes, fontsize=11, va="top", ha=ha, weight="bold", color=dcol,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f2", ec=dcol))

    fe = min(j+30, len(df)-1)
    if long_: ext = h[j+1:fe+1].max(); res = f"진입 후 최고 ${ext:.2f}  ({(ext/entry-1)*100:+.0f}%)"
    else: ext = l[j+1:fe+1].min(); res = f"진입 후 최저 ${ext:.2f}  ({(1-ext/entry)*100:+.0f}% 하락)"
    ax.text(bx, 0.82, res, transform=ax.transAxes, fontsize=11, va="top", ha=ha, weight="bold", color="#333")

    gr = {"OB만":"OB", "FVG만":"FVG", "OB+FVG":"OB+FVG 겹침"}[pick["grade"]]
    ax.set_title(f"[NVDA] {yr} {'롱' if long_ else '숏'}  ·  컨플루언스={confnm}  ·  추세선 TOL={tol*100:.1f}%  ·  근거={gr}  ({idx[j].date()})",
                 fontsize=13.5, weight="bold", pad=12)
    ax.set_ylabel("가격"); ax.grid(alpha=0.12)
    t_ = range(0, n, max(1, n//12)); ax.set_xticks(list(t_)); ax.set_xticklabels([sub.index[i].strftime("%m-%d") for i in t_], fontsize=8)
    ax.set_xlim(-1, n+2)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def summary(pdf, df):
    rows = []
    for (yr, dw, nm) in [(2022, "S", "숏 2022"), (2024, "L", "롱 2024")]:
        for conf in ["causal", "strict"]:
            for tol in [0.003, 0.005]:
                pick, obs, fvs = pick_sig(df, conf, tol, yr, dw)
                o, h, l, c = [df[x].values for x in ["o","h","l","c"]]; idx = df.index
                if pick is None:
                    rows.append([nm, "엄격" if conf == "strict" else "중간", f"{tol*100:.1f}%", "타점 없음", "-", "-", "-"]); continue
                j = pick["j"]
                if dw == "L": tlt = l[j] <= pick["lv"]*(1+tol)
                else: tlt = h[j] >= pick["lv"]*(1-tol)
                zt = any((l[j] <= e["hi"] and h[j] >= e["lo"]) for e in (obs+fvs) if G._conf_one(j, dw, o, h, l, c, e, conf))
                rows.append([nm, "엄격" if conf == "strict" else "중간", f"{tol*100:.1f}%",
                             str(idx[j].date()), {"OB만":"OB","FVG만":"FVG","OB+FVG":"OB+FVG"}[pick["grade"]],
                             "O" if tlt else "X", "O" if zt else "X"])
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_title("4조합 비교 요약 — 컨플루언스(중간/엄격) × 추세선 TOL(0.3%/0.5%)", fontsize=15, weight="bold", pad=20)
    cols = ["포지션", "컨플루언스", "TOL", "진입일", "근거", "추세선\n터치", "존\n도달"]
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 2.2)
    for ci in range(len(cols)): tb[0, ci].set_facecolor("#1F3A5F"); tb[0, ci].set_text_props(color="white", weight="bold")
    for ri in range(1, len(rows)+1):
        td, zd = rows[ri-1][5], rows[ri-1][6]
        tb[ri, 5].set_text_props(color="#1a7f37" if td == "O" else "#d64545", weight="bold")
        tb[ri, 6].set_text_props(color="#1a7f37" if zd == "O" else "#d64545", weight="bold")
    ax.text(0.5, 0.06, "핵심: '존 도달 O'가 진짜 진입 가능. 중간(causal)은 미래참조만 막고 위치 부정합 남음 → 존 도달 X 흔함. 엄격(strict)만 O.",
            transform=ax.transAxes, ha="center", fontsize=11, color="#444")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    df = G.load_price("NVDA")
    with PdfPages(OUT) as pdf:
        f0 = plt.figure(figsize=(16, 9))
        f0.text(0.5, 0.58, "진입 기준 — 개념 + 실제(NVDA) · 기준 비교", ha="center", fontsize=22, weight="bold", color="#1F3A5F")
        f0.text(0.5, 0.48, "컨플루언스 중간 vs 엄격 · 추세선 TOL 0.3% vs 0.5% · 숏=2022 롱=2024", ha="center", fontsize=12)
        pdf.savefig(f0); plt.close(f0)
        concept(pdf)
        summary(pdf, df)
        for (yr, dw) in [(2022, "S"), (2024, "L")]:
            for conf in ["strict", "causal"]:
                for tol in [0.003, 0.005]:
                    real(pdf, df, conf, tol, yr, dw)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
