"""Plot all available pilot seed curves without selecting favorable episodes."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_pilot(output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = ("#1976a3", "#de7c32", "#549452")
    baseline_plotted = False
    plotted_seeds = 0
    for index, run in enumerate(sorted(output.glob("seed_*"))):
        path = run / "evaluation.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        curve = [row for row in records if row["kind"] in {"initial_policy", "validation"}]
        if not curve:
            continue
        plotted_seeds += 1
        steps = [row["training_timesteps"] / 1000 for row in curve]
        color = colors[index % len(colors)]
        axes[0].plot(steps, [row["success_rate_lower"] for row in curve], marker="o",
                     markersize=3, color=color, label=run.name)
        axes[1].plot(steps, [row["mean_return"] for row in curve], marker="o",
                     markersize=3, color=color, label=run.name)
        random = next((row for row in records if row["kind"] == "random_baseline"), None)
        if random is not None and not baseline_plotted:
            axes[0].axhline(random["success_rate_lower"], color="#555555", linestyle="--",
                           label="matched random actions")
            axes[1].axhline(random["mean_return"], color="#555555", linestyle="--")
            baseline_plotted = True
    if plotted_seeds == 0:
        plt.close(fig)
        raise ValueError("no seed evaluation records are available to plot")
    axes[0].set_ylabel("Success fraction on fixed validation episodes")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Mean screen-derived episode return")
    for axis in axes:
        axis.set_xlabel("Training decisions (thousands)")
        axis.grid(alpha=0.2)
        axis.spines[["right", "top"]].set_visible(False)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Navigation · scratch PPO/CNN · deterministic validation (all declared seeds)")
    png = output / "learning_curves.png"
    fig.savefig(png, dpi=170)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)
    return png


def pilot_statistics(output: Path) -> dict:
    """Describe observed seed spread; episodes are not independent training repeats."""
    results = []
    for run in sorted(output.glob("seed_*")):
        status_path = run / "status.json"
        if status_path.exists():
            results.append({"seed": run.name, **json.loads(status_path.read_text())})
    rates = [row["final_success_rate_lower"] for row in results if row["status"] == "completed"]
    return {
        "seeds_started": len(results), "seeds_completed": len(rates),
        "completed_seed_success_mean": float(np.mean(rates)) if rates else None,
        "completed_seed_success_min": float(np.min(rates)) if rates else None,
        "completed_seed_success_max": float(np.max(rates)) if rates else None,
        "seed_statuses": {row["seed"]: row["status"] for row in results},
        "interpretation": "descriptive development results; no confidence interval or superiority "
        "claim; failed/incomplete seeds retained separately",
    }
