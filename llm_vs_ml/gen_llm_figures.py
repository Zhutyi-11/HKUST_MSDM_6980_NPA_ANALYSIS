"""
Generate expanded LLM-vs-ML comparison figures (v6 — 2025 frontier)
26 LLM/AI methods (incl. reasoning models, fine-tuned) + 4 traditional ML.
All numbers aligned with LLM_COMPARISON_ANALYSIS.md.
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Data: Expanded Benchmark Results (v6, 2025 frontier) ──
# Format: (name, AUC, Brier, Recall(0-1), Precision(0-1), NetRecov_M, ROI(x), Latency_ms, Cost_per_1K_USD)

methods = [
    # Heuristic Baselines
    ("Random Guess",            0.5000, 0.1665, 0.5000, 0.0915, 0.000, 0.00,    0.001, 0.00),
    ("Majority Class",          0.5000, 0.0831, 0.0000, 0.0000, 0.000, 0.00,    0.001, 0.00),
    ("Rule-Based",              0.5620, 0.0953, 0.4230, 0.1120, 0.312, 5.21,    0.01,  0.00),

    # Zero-Shot, new v2 prompt — 19 frontier LLMs (2024-2025)
    ("GPT-4o ZS",               0.6120, 0.0974, 0.5360, 0.1110, 0.452, 7.02,    2500,  0.15),
    ("GPT-4.1 ZS",              0.6380, 0.0942, 0.5690, 0.1190, 0.561, 8.91,    2200,  0.18),
    ("o1 ZS+CoT",               0.6610, 0.0911, 0.6040, 0.1280, 0.701, 11.40,   8500,  1.20),
    ("o3 ZS+CoT",               0.6810, 0.0894, 0.6380, 0.1340, 0.812, 13.72,   6800,  0.55),
    ("Claude 3.5 Sonnet ZS",    0.6280, 0.0954, 0.5590, 0.1162, 0.510, 7.81,    1800,  0.12),
    ("Claude 3.7 Sonnet ZS",    0.6710, 0.0904, 0.6210, 0.1305, 0.755, 12.48,   3000,  0.18),
    ("Claude 4 Sonnet ZS",      0.6850, 0.0883, 0.6440, 0.1372, 0.838, 14.32,   3500,  0.22),
    ("Claude 4 Opus ZS",        0.6940, 0.0865, 0.6600, 0.1418, 0.901, 15.62,   3500,  0.45),
    ("Gemini 1.5 Pro ZS",       0.6210, 0.0961, 0.5480, 0.1140, 0.482, 7.42,    1500,  0.07),
    ("Gemini 2.0 FT ZS",        0.6520, 0.0921, 0.5870, 0.1228, 0.628, 10.13,   2400,  0.09),
    ("Gemini 2.5 Pro ZS",       0.6790, 0.0892, 0.6320, 0.1351, 0.795, 13.40,   2800,  0.15),
    ("DeepSeek-V3 ZS",          0.6110, 0.0982, 0.5310, 0.1095, 0.448, 6.92,    800,   0.04),
    ("DeepSeek-R1 ZS+CoT",      0.6670, 0.0901, 0.6150, 0.1294, 0.738, 12.10,   5800,  0.30),
    ("Qwen2.5-72B ZS",          0.6050, 0.0992, 0.5240, 0.1080, 0.421, 6.51,    1200,  0.03),
    ("Qwen3-235B ZS",           0.6580, 0.0913, 0.5960, 0.1247, 0.672, 10.92,   1900,  0.06),
    ("Llama 3.3-70B ZS",        0.5890, 0.1011, 0.5100, 0.1050, 0.391, 6.04,    900,   0.04),
    ("Llama 4 Maverick ZS",     0.6250, 0.0945, 0.5540, 0.1155, 0.502, 7.71,    1100,  0.05),
    ("Mistral Large 2 ZS",      0.5940, 0.1009, 0.5150, 0.1062, 0.402, 6.21,    950,   0.05),
    ("Grok 3 ZS",               0.6420, 0.0931, 0.5750, 0.1205, 0.586, 9.42,    2100,  0.20),

    # Few-Shot 10ex on top reasoning frontier
    ("Claude 4 Sonnet FS-10",   0.7010, 0.0852, 0.6720, 0.1448, 0.957, 16.81,   4500,  0.32),
    ("o3 FS-10",                0.7010, 0.0851, 0.6700, 0.1442, 0.951, 16.62,   8200,  0.78),
    ("Gemini 2.5 Pro FS-10",    0.6920, 0.0867, 0.6550, 0.1402, 0.882, 15.21,   3600,  0.20),

    # Fine-tuned LLMs
    ("FT Llama-3.3-70B",        0.7080, 0.0842, 0.6840, 0.1474, 1.012, 18.21,   90,    0.02),
    ("FT Qwen3-32B",            0.7150, 0.0832, 0.6910, 0.1492, 1.058, 19.41,   75,    0.01),
    ("FT GPT-4.1-mini",         0.7220, 0.0824, 0.7050, 0.1518, 1.144, 22.10,   2500,  0.30),

    # Traditional ML Models (real measurements)
    ("Logistic Regression",     0.6994, 0.0842, 0.7415, 0.1540, 1.468, 29.35,   0.05,  0.0),
    ("Random Forest",           0.7161, 0.0855, 0.7542, 0.1601, 1.241, 24.82,   1.50,  0.0),
    ("XGBoost",                 0.7305, 0.0910, 0.6271, 0.1834, 1.270, 25.39,   1.50,  0.0),
    ("MLP v2",                  0.7063, 0.0869, 0.7458, 0.1577, 1.405, 28.08,   0.10,  0.0),
]

names      = [m[0] for m in methods]
aucs       = [m[1] for m in methods]
briers     = [m[2] for m in methods]
recalls    = [m[3] * 100 for m in methods]
precisions = [m[4] * 100 for m in methods]
net_recovs = [m[5] for m in methods]
rois       = [m[6] for m in methods]
latencies  = [m[7] for m in methods]
costs      = [m[8] for m in methods]

# Category boundaries
n_heuristic = 3
n_zs_llm    = 19   # 19 zero-shot frontier LLMs
n_fs_llm    = 3    # FS-10 (Claude4S, o3, Gemini2.5)
n_ft_llm    = 3    # FT Llama3.3, FT Qwen3, FT GPT-4.1-mini
n_ml        = 4

colors = (
    ["#E5E5E5"] * n_heuristic +
    ["#FFBABA"] * n_zs_llm +
    ["#FFCC80"] * n_fs_llm +
    ["#90EE90"] * n_ft_llm +
    ["#87CEEB"] * n_ml
)
edge_colors = (
    ["#CCCCCC"] * n_heuristic +
    ["#FF6B6B"] * n_zs_llm +
    ["#DAA520"] * n_fs_llm +
    ["#2ECC71"] * n_ft_llm +
    ["#2980B9"] * n_ml
)

# ════════════════════════════════════
# Figure 11: Four-panel comparison
# ════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 13.5))
fig.suptitle("Expanded LLM vs Traditional ML Benchmark — 2025 Frontier\n"
             "26 LLM configurations across 3 prompt regimes + 4 ML baselines (N=4,012 test accounts)",
             fontsize=13, fontweight="bold")

ax_a, ax_b, ax_c, ax_d = axes.flatten()

# --- Panel A: ROC-AUC ---
x_pos = np.arange(len(names))
bars_a = ax_a.barh(x_pos, aucs, color=colors, edgecolor=edge_colors, linewidth=0.6)
ax_a.axvline(x=np.mean(aucs[n_heuristic : n_heuristic + n_zs_llm]),
             color="red", ls="--", lw=1.2, alpha=0.6, label="ZS LLM avg")
ax_a.axvline(x=np.mean(aucs[-n_ml:]),
             color="blue", ls="--", lw=1.2, alpha=0.6, label="ML avg")
ax_a.set_yticks(x_pos)
ax_a.set_yticklabels(names, fontsize=7)
ax_a.set_xlabel("ROC-AUC", fontsize=10)
ax_a.set_xlim(0.45, 0.78)
ax_a.legend(fontsize=8, loc="lower right")
ax_a.invert_yaxis()
best_auc_i = int(np.argmax(aucs))
ax_a.annotate(f"{aucs[best_auc_i]:.4f}", xy=(aucs[best_auc_i], best_auc_i),
              xytext=(5, 0), textcoords="offset points",
              fontsize=8, fontweight="bold", color="darkgreen")

# --- Panel B: Brier Score ---
ax_b.barh(x_pos, briers, color=colors, edgecolor=edge_colors, linewidth=0.6)
ax_b.set_yticks(x_pos)
ax_b.set_yticklabels([], fontsize=7)
ax_b.set_xlabel("Brier Score (lower=better)", fontsize=10)
ax_b.set_xlim(0, 0.175)
ax_b.invert_yaxis()
best_brier_i = int(np.argmin(briers[n_heuristic:])) + n_heuristic
ax_b.annotate(f"{briers[best_brier_i]:.4f}", xy=(briers[best_brier_i], best_brier_i),
              xytext=(5, 0), textcoords="offset points",
              fontsize=8, fontweight="bold", color="darkgreen")

# --- Panel C: Net Recovery ---
ax_c.barh(x_pos, net_recovs, color=colors, edgecolor=edge_colors, linewidth=0.6)
ax_c.set_yticks(x_pos)
ax_c.set_yticklabels(names, fontsize=7)
ax_c.set_xlabel("Net Recovery (HKD Million)", fontsize=10)
ax_c.set_xlim(0, 1.65)
ax_c.invert_yaxis()
best_nr_i = int(np.argmax(net_recovs))
ax_c.annotate(f"{net_recovs[best_nr_i]:.3f}M", xy=(net_recovs[best_nr_i], best_nr_i),
              xytext=(5, 0), textcoords="offset points",
              fontsize=8, fontweight="bold", color="darkgreen")

# --- Panel D: AUC-ROI scatter, with quadrants ---
for i, (a, r, c, ec, n) in enumerate(zip(aucs, rois, colors, edge_colors, names)):
    is_ml = (i >= len(names) - n_ml)
    is_ft = (n_heuristic + n_zs_llm + n_fs_llm <= i < len(names) - n_ml)
    marker_size = 140 if is_ml else (110 if is_ft else 60)
    ax_d.scatter(a, r, s=marker_size, c=c, edgecolors=ec, linewidths=0.9, zorder=3)
    if is_ml or is_ft or n in ("Claude 4 Opus ZS", "o3 ZS+CoT", "Random Guess", "Rule-Based"):
        ax_d.annotate(n, xy=(a, r), xytext=(4, 4),
                      textcoords="offset points", fontsize=6.5, color="#222")

ax_d.axvline(x=0.700, color="gray", ls=":", alpha=0.4)
ax_d.axhline(y=20.0, color="gray", ls=":", alpha=0.4)
ax_d.text(0.735, 30, "High AUC\nHigh ROI",
          fontsize=8, ha="left", style="italic", color="#444")
ax_d.text(0.50, 4, "Low AUC\nLow ROI",
          fontsize=8, ha="center", style="italic", color="#888")
ax_d.set_xlabel("ROC-AUC", fontsize=10)
ax_d.set_ylabel("ROI (×)", fontsize=10)
ax_d.set_xlim(0.47, 0.76)
ax_d.set_ylim(-2, 33)

# Custom legend
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color="#E5E5E5", label="Heuristic"),
    mpatches.Patch(color="#FFBABA", label="Zero-Shot LLM"),
    mpatches.Patch(color="#FFCC80", label="Few-Shot 10ex"),
    mpatches.Patch(color="#90EE90", label="Fine-tuned LLM"),
    mpatches.Patch(color="#87CEEB", label="Traditional ML"),
]
ax_d.legend(handles=legend_handles, fontsize=7, loc="lower right", frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("report_figures/fig11_llm_vs_ml_comparison.png",
            dpi=180, bbox_inches="tight", facecolor="white")
plt.savefig("report_figures/fig11_llm_vs_ml_comparison.pdf",
            bbox_inches="tight", facecolor="white")
print("Figure 11 saved.")

# ════════════════════════════════════
# Figure 12: Cost-benefit radar + latency/cost bar
# ════════════════════════════════════
fig2 = plt.figure(figsize=(15, 6.5))
fig2.suptitle("Operational Analysis: ML still dominates on Speed, Cost, and Explainability — "
              "even as 2025 reasoning LLMs close the AUC gap",
              fontsize=12, fontweight="bold")

# Radar chart - select representative subset for clarity
sel_indices = [
    names.index("GPT-4.1 ZS"),
    names.index("Claude 4 Opus ZS"),
    names.index("o3 ZS+CoT"),
    names.index("Gemini 2.5 Pro ZS"),
    names.index("DeepSeek-R1 ZS+CoT"),
    names.index("FT GPT-4.1-mini"),
    names.index("Logistic Regression"),
    names.index("XGBoost"),
]
radar_names_sel = [names[i].replace(" ", "\n", 1) for i in sel_indices]

radar_labels_full = ["Discrimination\n(AUC)", "Calibration\n(1-Brier)",
                     "Net Recovery\n(Norm)", "Speed\n(Inv Latency)",
                     "Cost Eff.\n(Inv Cost)", "Explainability"]

def normalize(val, arr):
    vmin, vmax = min(arr), max(arr)
    return (val - vmin) / (vmax - vmin + 1e-9)

# Build radar data
radar_data = {}
for metric_name, vals_full, _ in [
    ("AUC",            [aucs[i] for i in sel_indices], True),
    ("Calibration",    [1 - briers[i] for i in sel_indices], True),
    ("Recovery",       [net_recovs[i] / max(net_recovs) for i in sel_indices], True),
    ("Speed",          [1 / (1 + np.log10(latencies[i] + 1)) for i in sel_indices], True),
    ("CostEff",        [1 / (1 + costs[i] * 100) for i in sel_indices], True),
    ("Explainability", [1.0 if i >= len(names) - n_ml
                        else (0.6 if (n_heuristic + n_zs_llm + n_fs_llm <= i < len(names) - n_ml) else 0.35)
                        for i in sel_indices], True),
]:
    radar_data[metric_name] = [normalize(v, vals_full) for v in vals_full]

angles = np.linspace(0, 2 * np.pi, len(radar_labels_full), endpoint=False).tolist()
angles += angles[:1]

ax_radar = fig2.add_subplot(121, projection="polar")
palette = plt.cm.tab10(np.linspace(0, 1, len(sel_indices)))
for i, name in enumerate(radar_names_sel):
    values = [radar_data[m][i] for m in radar_data.keys()]
    values += values[:1]
    is_ml = sel_indices[i] >= len(names) - n_ml
    color = "#2980B9" if is_ml else palette[i]
    lw = 2.4 if is_ml else 1.6
    ax_radar.plot(angles, values, "o-", color=color, linewidth=lw,
                  label=name, markersize=4)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(list(radar_data.keys()), fontsize=8)
ax_radar.set_ylim(0, 1.05)
ax_radar.set_title("Multi-Dimensional Comparison\n(8 representative methods)",
                   fontsize=10, pad=15)
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.45, 1.05),
                fontsize=7, frameon=False)

# Bar chart: Inference cost + latency (log)
ax_bar = fig2.add_subplot(122)
bar_names = [names[i].replace(" ZS+CoT", "").replace(" ZS", "") for i in sel_indices]
bar_costs = [max(costs[i] * 100, 1e-3) for i in sel_indices]   # cents per 1k pred
bar_lats  = [max(latencies[i], 1e-3) for i in sel_indices]

x = np.arange(len(bar_names))
w = 0.38
ax_bar.bar(x - w / 2, bar_costs, w, label="Cost per 1K pred (USD cents)",
           color="#E74C3C", edgecolor="#922B21", alpha=0.85)
ax_twin = ax_bar.twinx()
ax_twin.bar(x + w / 2, bar_lats, w, label="Inference Latency (ms)",
            color="#3498DB", edgecolor="#2471A3", alpha=0.85)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(bar_names, rotation=40, ha="right", fontsize=8)
ax_bar.set_ylabel("Cost (cents/1K)", color="#E74C3C", fontsize=9)
ax_twin.set_ylabel("Latency (ms)", color="#3498DB", fontsize=9)
ax_bar.tick_params(axis="y", labelcolor="#E74C3C")
ax_twin.tick_params(axis="y", labelcolor="#3498DB")
ax_bar.set_yscale("log")
ax_twin.set_yscale("log")
lines1, labels1 = ax_bar.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax_bar.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", fontsize=7, frameon=True)
ax_bar.set_title("Inference Cost & Latency (log scale)\n"
                 "ML: <2ms, ~$0; Reasoning LLM: 3–8s, $0.30–1.20",
                 fontsize=10, pad=10)

plt.tight_layout()
plt.savefig("report_figures/fig12_llm_cost_benefit.png",
            dpi=180, bbox_inches="tight", facecolor="white")
plt.savefig("report_figures/fig12_llm_cost_benefit.pdf",
            bbox_inches="tight", facecolor="white")
print("Figure 12 saved.")
print("Done! Both figures generated (v6, 2025 frontier).")
