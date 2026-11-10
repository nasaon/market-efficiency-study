# -*- coding: utf-8 -*-
"""추세선 옵션 A(스윙K12+45봉) vs B(K5+45봉) 타점 시각화.
각 타점 확대: 추세선(스윙2점) + OB/FVG존 + 진입 + 추세선터치/존도달 진단 + N일 결과."""
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
import entry_show as E

plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
OBC, FVC, TLC = G.OBC, G.FVC, "#2563eb"
OUT = os.path.join(G.HERE, "추세선_A_vs_B.pdf")
SYMS = [("BTC","BTC-USD"), ("S&P","^GSPC"), ("코스피","^KS11"), ("삼성","005930.KS"), ("하이닉스","000660.KS")]
N = 10


def swings_k(h, l, K):
    n = len(h); sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(K, n-K):
        if h[i] == h[i-K:i+K+1].max(): sh[i] = True
        if l[i] == l[i-K:i+K+1].min(): sl[i] = True
    return sh, sl


def tl_swings(h, l, c, tol, minbars, swingK):
    sh, sl = swings_k(h, l, swingK); n = len(c); sli = np.where(sl)[0]; shi = np.where(sh)[0]; out = []
    for i in range(1, len(sli)):
        s1, s2 = sli[i-1], sli[i]
        if l[s2] <= l[s1] or s2-s1 < minbars: continue
        slope = (l[s2]-l[s1])/(s2-s1)
        if any(l[k] < (l[s1]+slope*(k-s1))*(1-G.TLPEN) for k in range(s1+1, s2)): continue   # 사이 캔들 선 0.5%+ 관통 → 무효
        for j in range(s2+swingK, min(s2+G.LOOK, n)):
            tlv = l[s2]+slope*(j-s2)
            if l[j] <= tlv*(1+tol) and c[j] > tlv and c[j-1] >= tlv*(1-tol): out.append((j, "L", s1, s2)); break
            if c[j] < tlv*(1-tol): break
    for i in range(1, len(shi)):
        s1, s2 = shi[i-1], shi[i]
        if h[s2] >= h[s1] or s2-s1 < minbars: continue
        slope = (h[s2]-h[s1])/(s2-s1)
        if any(h[k] > (h[s1]+slope*(k-s1))*(1+G.TLPEN) for k in range(s1+1, s2)): continue   # 사이 캔들 선 0.5%+ 관통 → 무효
        for j in range(s2+swingK, min(s2+G.LOOK, n)):
            tlv = h[s2]+slope*(j-s2)
            if h[j] >= tlv*(1-tol) and c[j] < tlv and c[j-1] <= tlv*(1+tol): out.append((j, "S", s1, s2)); break
            if c[j] > tlv*(1+tol): break
    return out


def collect(minbars, swingK):
    items = []
    for kor, tk in SYMS:
        df = G.load_price(tk); o, h, l, c = [df[x].values for x in ["o","h","l","c"]]
        sh, sl = G.swings(h, l); obs = G.detect_ob(o, h, l, c, sh, sl); fvs = G.detect_fvg(o, h, l, c)
        for j, d, s1, s2 in tl_swings(h, l, c, 0.003, minbars, swingK):
            hasC = any(G._conf_one(j, d, o, h, l, c, e, "strict") for e in obs) or \
                   any(G._conf_one(j, d, o, h, l, c, e, "strict") for e in fvs)
            if not hasC: continue
            items.append(dict(kor=kor, df=df, j=j, d=d, s1=s1, s2=s2, obs=obs, fvs=fvs))
    return items


def card(pdf, opt, it, idx_num, total):
    df = it["df"]; kor = it["kor"]; j = it["j"]; d = it["d"]; s1 = it["s1"]; s2 = it["s2"]
    obs = it["obs"]; fvs = it["fvs"]
    o, h, l, c = [df[x].values for x in ["o","h","l","c"]]; idx = df.index; long_ = d == "L"
    i0 = max(0, s1-3); i1 = min(len(df)-1, j+18)
    sub = df.iloc[i0:i1+1]; n = len(sub); so, sh_, sl_, sc = [sub[x].values for x in ["o","h","l","c"]]
    RNG = float(np.ptp(sh_))
    fig, ax = plt.subplots(figsize=(16, 8.5))
    for k in range(n): E.candle(ax, k, so[k], sh_[k], sl_[k], sc[k], 0.6)
    # 추세선
    slope = ((l[s2]-l[s1]) if long_ else (h[s2]-h[s1]))/(s2-s1)
    xs = np.arange(s1, j+4); ys = (l[s2] if long_ else h[s2]) + slope*(xs-s2)
    ax.plot(xs-i0, ys, color=TLC, ls="--", lw=1.8, zorder=6)
    tlv_j = (l[s2] if long_ else h[s2]) + slope*(j-s2)
    for sx, lab in [(s1, "스윙①"), (s2, "스윙②")]:
        yv = l[sx] if long_ else h[sx]
        ax.scatter(sx-i0, yv, marker="o", s=85, c=TLC, ec="white", lw=1, zorder=8)
        ax.annotate(lab, (sx-i0, yv), (sx-i0, yv+(-RNG*0.06 if long_ else RNG*0.06)), fontsize=9, ha="center", color=TLC, weight="bold")
    # OB/FVG 존 (strict 채택)
    zone_touch = False
    for tag, arr, col, ec in [("OB", obs, OBC, OBC), ("FVG", fvs, FVC, "#b8860b")]:
        for e in arr:
            if not G._conf_one(j, d, o, h, l, c, e, "strict"): continue
            x0 = (e["form"]-(1 if tag == "OB" else 2)) - i0; x0 = max(x0, -0.5); bw = (j-i0-x0)+1.2
            ax.add_patch(Rectangle((x0-0.3, e["lo"]), bw, e["hi"]-e["lo"], facecolor=col, alpha=0.33, edgecolor=ec, lw=0.9, zorder=3))
            ax.annotate(tag, (x0+bw/2-0.3, e["hi"] if tag == "OB" else e["lo"]),
                        (x0+bw/2-0.3, (e["hi"]+RNG*0.05) if tag == "OB" else (e["lo"]-RNG*0.05)),
                        fontsize=8.5, ha="center", color=col if tag == "OB" else "#9a6b00", weight="bold",
                        arrowprops=dict(arrowstyle="->", color=ec))
            if l[j] <= e["hi"] and h[j] >= e["lo"]: zone_touch = True
    # 진입
    x = j-i0; entry = c[j]; my = sl_[x]-RNG*0.04 if long_ else sh_[x]+RNG*0.04
    ax.scatter([x], [my], marker="^" if long_ else "v", s=240, c="lime" if long_ else "red", ec="k", lw=0.9, zorder=10)
    lx = x+3 if x < n*0.72 else x-3; lha = "left" if x < n*0.72 else "right"
    ax.annotate(f"진입 {entry:.2f}", (x, my), (lx, my+(-RNG*0.15 if long_ else RNG*0.15)),
                fontsize=10, weight="bold", ha=lha, color="green" if long_ else "#d62728",
                arrowprops=dict(arrowstyle="->", color="green" if long_ else "#d62728"))
    # 진단
    if long_: tlt = l[j] <= tlv_j*(1.003); dist = (l[j]/tlv_j-1)*100; nb = "저가"
    else: tlt = h[j] >= tlv_j*(0.997); dist = (h[j]/tlv_j-1)*100; nb = "고가"
    ok = lambda b: "O" if b else "X"
    diag = f"[진단]\n추세선 터치: {ok(tlt)} ({nb} {dist:+.1f}%)\n존 도달: {ok(zone_touch)}\n스윙거리: {s2-s1}봉"
    bx = 0.015 if long_ else 0.985; ha = "left" if long_ else "right"
    ax.text(bx, 0.97, diag, transform=ax.transAxes, fontsize=10.5, va="top", ha=ha, weight="bold",
            color="#1a7f37" if (tlt and zone_touch) else "#d64545", bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f2", ec="#bbb"))
    # 결과
    fe = min(j+N, len(df)-1); sgn = 1 if long_ else -1; ret = sgn*(c[fe]/entry-1)*100
    ax.text(bx, 0.74, f"{N}일 방향수익  {ret:+.1f}%", transform=ax.transAxes, fontsize=12, va="top", ha=ha,
            weight="bold", color="#1a7f37" if ret > 0 else "#d64545")
    ax.set_title(f"[옵션 {opt}] {idx_num}/{total} · {kor} {idx[j].date()} · {'롱' if long_ else '숏'} · 추세선 스윙거리 {s2-s1}봉",
                 fontsize=14, weight="bold", pad=10)
    ax.set_ylabel("가격"); ax.grid(alpha=0.12)
    t_ = range(0, n, max(1, n//12)); ax.set_xticks(list(t_)); ax.set_xticklabels([sub.index[i].strftime("%y-%m-%d") for i in t_], fontsize=8)
    ax.set_xlim(-1.5, n+1)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def summary_page(pdf, A, B):
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_title("추세선 옵션 A (스윙K12+45봉)  vs  B (K5+45봉)", fontsize=18, weight="bold", pad=20)
    def rows(items, opt):
        r = []
        for it in items:
            df = it["df"]; c = df["c"].values; j = it["j"]; d = it["d"]
            fe = min(j+N, len(df)-1); sgn = 1 if d == "L" else -1; ret = sgn*(c[fe]/c[j]-1)*100
            r.append([opt, it["kor"], str(df.index[j].date()), "롱" if d == "L" else "숏", f"{it['s2']-it['s1']}봉", f"{ret:+.1f}%"])
        return r
    data = rows(A, "A") + rows(B, "B")
    tb = ax.table(cellText=data, colLabels=["옵션","종목","진입일","방향","스윙거리","10일수익"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 1.8)
    for ci in range(6): tb[0, ci].set_facecolor("#1F3A5F"); tb[0, ci].set_text_props(color="white", weight="bold")
    for ri in range(1, len(data)+1):
        ret = float(data[ri-1][5].replace("%","")); tb[ri, 5].set_text_props(color="#1a7f37" if ret > 0 else "#d64545", weight="bold")
        tb[ri, 0].set_facecolor("#e8eef6" if data[ri-1][0] == "A" else "#fdeee8")
    aR = [float(rows([it],"")[0][5].replace("%","")) for it in A]; bR = [float(rows([it],"")[0][5].replace("%","")) for it in B]
    awr = (sum(1 for x in aR if x > 0)/len(aR)*100) if aR else 0
    bwr = (sum(1 for x in bR if x > 0)/len(bR)*100) if bR else 0
    ax.text(0.5, 0.08, f"A: {len(A)}타점 평균 {np.mean(aR):+.2f}% 승률 {awr:.0f}%   |   B: {len(B)}타점 평균 {np.mean(bR):+.2f}% 승률 {bwr:.0f}%",
            transform=ax.transAxes, ha="center", fontsize=12, weight="bold")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    A = collect(45, 12); B = collect(45, 5)
    print(f"A(K12+45) {len(A)}타점, B(K5+45) {len(B)}타점")
    with PdfPages(OUT) as pdf:
        f0 = plt.figure(figsize=(16, 9))
        f0.text(0.5, 0.56, "추세선 옵션 A vs B — 타점 시각화", ha="center", fontsize=23, weight="bold", color="#1F3A5F")
        f0.text(0.5, 0.46, f"A = 스윙K12 + 45봉 ({len(A)}타점)   ·   B = K5 + 45봉 ({len(B)}타점)   ·   5종목 풀", ha="center", fontsize=12)
        pdf.savefig(f0); plt.close(f0)
        summary_page(pdf, A, B)
        for k, it in enumerate(A, 1): card(pdf, "A", it, k, len(A))
        for k, it in enumerate(B, 1): card(pdf, "B", it, k, len(B))
    print("저장:", OUT)


if __name__ == "__main__":
    main()
