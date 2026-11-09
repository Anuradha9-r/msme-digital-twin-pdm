"""
Reproduce the four figures of the paper from results/*.json.

    Fig. 1  five-layer digital twin architecture
    Fig. 2  precision-recall curves and permutation importance
    Fig. 3  line performance under the four maintenance policies
    Fig. 4  sensitivity of the output gain to model quality and repair time

Usage:
    python src/make_figures.py         # writes figures/*.png
Requires results/pm_results.json (train_models.py) and
results/twin_results.json (line_twin.py).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def fig_architecture():
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7})
    fig, ax = plt.subplots(figsize=(3.45, 4.3), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(6, 121); ax.axis("off")
    INK = "#222222"; MUTED = "#555555"
    LAYERS = [
        (104, "L1  Physical shop floor", "CNC lathe · VMC · drilling · inspection\n(pump impeller / housing line)", "#cde2fb"),
        (82, "L2  Low-cost sensing & edge", "Current, vibration, temperature sensors;\nPLC/MES counters → edge gateway (MQTT)", "#b7d3f6"),
        (60, "L3  Digital twin core", "Physics-informed state: ΔT, power, wear×torque;\ndiscrete-event line model (synchronised)", "#9ec5f4"),
        (38, "L4  Predictive analytics", "Gradient-boosted failure classifier;\nalarm → lead-time window → PM request", "#86b6ef"),
        (16, "L5  Decision & what-if services", "Opportunistic PM in changeovers, buffer sizing,\nOEE dashboard for owner / supervisor", "#6da7ec"),
    ]
    for y, title, body, fc in LAYERS:
        ax.add_patch(FancyBboxPatch((4, y), 92, 15, boxstyle="round,pad=0.4,rounding_size=2",
                                    fc=fc, ec="#1c5cab", lw=0.8))
        ax.text(50, y + 11, title, ha="center", va="center", fontsize=7.4, weight="bold", color=INK)
        ax.text(50, y + 4.6, body, ha="center", va="center", fontsize=6.2, color=INK, linespacing=1.25)
    for y_top, y_bot in [(104, 97), (82, 75), (60, 53), (38, 31)]:
        ax.add_patch(FancyArrowPatch((40, y_top - 0.3), (40, y_bot + 0.3), arrowstyle="-|>", mutation_scale=9,
                                     color=INK, lw=0.9))
        ax.add_patch(FancyArrowPatch((60, y_bot + 0.3), (60, y_top - 0.3), arrowstyle="-|>", mutation_scale=9,
                                     color=MUTED, lw=1.1, ls=(0,(2,1.5))))
    ax.text(50, 9, "solid: data flow (L1 → L5)    dashed: decisions fed back (L5 → L1)", ha="center", fontsize=5.8, color=INK)
    fig.savefig(FIG / "fig1_architecture.png", bbox_inches="tight", pad_inches=0.03)


def fig_results():
    import json
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "axes.linewidth": 0.6,
                         "axes.edgecolor": "#888888", "xtick.color": "#444444", "ytick.color": "#444444",
                         "axes.labelcolor": "#222222", "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    INK = "#222222"; GRID = "#e4e4e4"
    W = 3.45
    pm = json.load(open(ROOT / "results" / "pm_results.json"))
    tw = json.load(open(ROOT / "results" / "twin_results.json"))

    def clean(ax):
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)

    # Fig 2: PR curves (a) + permutation importance (b)
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 4.4), dpi=300, gridspec_kw=dict(height_ratios=[1.15, 1]))
    for k, c, lab in [("LR-raw", S[3], "LR"), ("RF-raw", S[1], "RF"), ("GB-raw", S[0], "GB")]:
        rc, pr = pm["curves"][k]; a.plot(rc, pr, color=c, lw=1.0, ls=(0, (3, 1.5)))
        rc, pr = pm["curves"][k.replace("raw", "phys")]; a.plot(rc, pr, color=c, lw=1.6)
    for c, lab in zip([S[3], S[1], S[0]], ["LR", "RF", "GB"]):
        a.plot([], [], color=c, lw=1.6, label=lab)
    a.plot([], [], color="#666666", lw=1.6, label="physics-informed")
    a.plot([], [], color="#666666", lw=1.0, ls=(0, (3, 1.5)), label="raw sensors")
    a.set_xlabel("Recall"); a.set_ylabel("Precision"); a.set_xlim(0, 1); a.set_ylim(0, 1.03)
    a.legend(frameon=False, fontsize=5.8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), handlelength=1.8, columnspacing=0.9); clean(a); a.grid(axis="x", color=GRID, lw=0.5)
    a.set_title("(a) Precision-recall, 5-fold out-of-fold", fontsize=7, loc="left", color=INK)
    imp = [t for t in pm["importance"] if not t[0].startswith("type")][:7][::-1]
    names = {"dT_K": "ΔT (process−air)", "speed_rpm": "Rotational speed", "power_W": "Mechanical power",
             "strain_margin": "Overstrain margin", "tool_wear_min": "Tool wear", "strain_minNm": "Wear × torque",
             "torque_Nm": "Torque", "air_K": "Air temperature", "process_K": "Process temperature"}
    y = np.arange(len(imp))
    b.barh(y, [t[1] for t in imp], xerr=[t[2] for t in imp], color=S[0], height=0.62,
           error_kw=dict(lw=0.6, ecolor="#555555", capsize=1.5))
    b.set_yticks(y); b.set_yticklabels([names.get(t[0], t[0]) for t in imp])
    b.set_xlabel("Drop in PR-AUC when permuted"); clean(b); b.grid(axis="y", visible=False); b.grid(axis="x", color=GRID, lw=0.5)
    b.set_title("(b) Permutation importance, GB physics-informed", fontsize=7, loc="left", color=INK)
    fig.tight_layout(h_pad=2.2); fig.savefig(FIG / "fig2_model.png", bbox_inches="tight", pad_inches=0.03)

    # Fig 3: scenario comparison: good parts/month and unplanned downtime h/month
    pols = ["S0", "S1", "S2", "S3"]; labs = ["S0", "S1", "S2", "S3"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.0), dpi=300)
    for ax, key, ttl in [(a, "good_pm", "(a) Good parts / month"), (b, "unpl_h_pm", "(b) Unplanned downtime, h / month")]:
        m = [tw[p][key][0] for p in pols]; h = [tw[p][key][1] for p in pols]
        ax.bar(range(4), m, yerr=h, color=S, width=0.66, error_kw=dict(lw=0.6, ecolor="#555555", capsize=1.5))
        for i, v in enumerate(m):
            ax.text(i, v + max(m) * 0.02 + h[i], f"{v:,.0f}" if key == "good_pm" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=5.6, color=INK)
        ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=6.4); clean(ax)
        ax.set_title(ttl, fontsize=6.6, loc="left", color=INK); ax.set_ylim(0, max(m) * 1.15)
    fig.tight_layout(w_pad=1.0); fig.savefig(FIG / "fig3_scenarios.png", bbox_inches="tight", pad_inches=0.03)

    # Fig 4: sensitivity heatmap (sequential blue) + repair-delay line
    sens = tw["sens"]; G = np.array(sens["gain"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.95), dpi=300, gridspec_kw=dict(width_ratios=[1.15, 1]))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    im = a.imshow(G, cmap=cmap, origin="lower", aspect="auto")
    a.set_xticks(range(len(sens["precision"]))); a.set_xticklabels([f"{p:.1f}" for p in sens["precision"]], fontsize=5.4)
    a.set_yticks(range(len(sens["recall"]))); a.set_yticklabels([f"{r:.2f}" for r in sens["recall"]], fontsize=5.4)
    a.set_xlabel("Model precision"); a.set_ylabel("Model recall")
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            a.text(j, i, f"{G[i, j]:.1f}", ha="center", va="center", fontsize=4.6,
                   color="white" if G[i, j] > G.min() + 0.55 * (G.max() - G.min()) else INK)
    a.set_title("(a) Output gain vs S0, %", fontsize=6.6, loc="left", color=INK)
    for s in a.spines.values(): s.set_visible(False)
    R = np.array(tw["repair_sens"])
    b.plot(R[:, 0], R[:, 3], color=S[2], lw=1.6, marker="o", ms=3.2, mec="white", mew=0.6)
    for x, v in zip(R[:, 0], R[:, 3]): b.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=5.4, color=INK)
    b.set_xlabel("Mean breakdown repair time, h"); b.set_ylabel("S2 output gain vs S0, %")
    b.set_ylim(0, R[:, 3].max() * 1.25); clean(b)
    b.set_title("(b) Effect of repair delay", fontsize=6.6, loc="left", color=INK)
    fig.tight_layout(w_pad=1.2); fig.savefig(FIG / "fig4_sensitivity.png", bbox_inches="tight", pad_inches=0.03)
    print("ok")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_architecture()
    fig_results()
    print("Wrote figures to", FIG)
