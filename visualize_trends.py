"""
visualize_trends.py
--------------------
Illustrates the core intuition of the model: CPI/SPI trajectories
diverge between eventually-overrun and on-track projects well before
project completion -- that divergence, 2-3 milestones out, is what the
classifiers learn to detect early.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(RESULTS, "figures")

df = pd.read_csv(os.path.join(DATA, "evm_full_timeseries.csv"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, metric, title, threshold in [
    (axes[0], "cpi", "Cost Performance Index (CPI)", 1.0),
    (axes[1], "spi", "Schedule Performance Index (SPI)", 1.0),
]:
    for outcome, color, label in [(0, "#4CAF50", "On-Track Projects"), (1, "#C0392B", "Eventually Overrun Projects")]:
        subset = df[df.final_overrun == outcome]
        pivot = subset.pivot_table(index="milestone", columns="project_id", values=metric)
        mean_line = pivot.mean(axis=1)
        std_line = pivot.std(axis=1)
        ax.plot(mean_line.index, mean_line.values, color=color, linewidth=2.5, label=f"{label} (mean)")
        ax.fill_between(mean_line.index, mean_line - std_line, mean_line + std_line, color=color, alpha=0.12)
    ax.axhline(threshold, color="black", linestyle="--", alpha=0.4, linewidth=1, label="Baseline (=1.0)")
    ax.axvline(11, color="#888", linestyle=":", alpha=0.6, linewidth=1.3)
    ax.text(11.15, ax.get_ylim()[1]*0.97 if ax.get_ylim()[1] > 0 else 1.05, "feature\ncutoff", fontsize=8, color="#666", va="top")
    ax.set_xlabel("Milestone")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{title} Trajectory\n(shaded = ±1 std dev)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Overrun Projects Show Diverging CPI/SPI Trends Well Before Completion\n"
             "(models are trained using only data up to the dotted line — 3 milestones before the finish)",
             fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "evm_trend_divergence.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Saved evm_trend_divergence.png")
