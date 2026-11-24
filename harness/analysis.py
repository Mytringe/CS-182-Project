"""Summaries and plots for training runs evaluated on the test split."""

from pathlib import Path

import pandas as pd


REGIME_LABELS = {
    "clean": "Clean",
    "uniform": "Uniform FCM",
    "uniform_mrd": "Uniform FCM + MRD",
    "pos": "POS-weighted FCM",
    "pos_mrd": "POS + decay FCM",
}


def summarize_test(history: pd.DataFrame, run_root: str | Path) -> pd.DataFrame:
    if history.empty:
        raise ValueError("No test history is available to summarize")
    last = history.groupby("name")["step"].transform("max")
    final = history[history["step"] == last]
    grouped = final.groupby(["unique_train_tokens", "regime"])
    summary = grouped.agg(
        mean_test_loss=("test_loss", "mean"),
        mean_accuracy=("test_accuracy", "mean"),
        runs=("name", "nunique"),
    )
    if summary["runs"].max() > 1:
        summary["sd_test_loss"] = grouped["test_loss"].std()
    summary = summary.sort_values(["unique_train_tokens", "mean_test_loss"])
    summary.to_csv(Path(run_root) / "test_summary.csv")
    return summary


def plot_test_curves(history: pd.DataFrame, run_root: str | Path):
    import matplotlib.pyplot as plt

    if history.empty:
        raise ValueError("No test history is available to plot")
    run_root = Path(run_root)
    budgets = sorted(history["unique_train_tokens"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(budgets), figsize=(5 * len(budgets), 4),
                             sharex=True, sharey=True, layout="constrained")
    if len(budgets) == 1:
        axes = [axes]
    for ax, budget in zip(axes, budgets):
        subset = history[history["unique_train_tokens"] == budget]
        for regime, group in subset.groupby("regime"):
            group = group.sort_values("targets")
            ax.plot(group["targets"] / 1e6, group["test_loss"], marker="o",
                    label=REGIME_LABELS.get(regime, regime))
        ax.set_title(f"{budget / 2**20:.0f} Mi unique tokens")
        ax.set_xlabel("Training targets (millions)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Test loss (nats/token)")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Test loss at matched training exposure")
    fig.savefig(run_root / "test_curves.png", dpi=160)

    regimes = [r for r in REGIME_LABELS if r in set(history["regime"])]
    reuse_fig, reuse_axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4),
                                         sharey=True, layout="constrained")
    if len(regimes) == 1:
        reuse_axes = [reuse_axes]
    for ax, regime in zip(reuse_axes, regimes):
        subset = history[history["regime"] == regime]
        for budget, group in subset.groupby("unique_train_tokens"):
            group = group.sort_values("effective_epochs")
            ax.plot(group["effective_epochs"], group["test_loss"], marker="o",
                    label=f"{budget / 2**20:.0f} Mi unique tokens")
        ax.set_title(REGIME_LABELS[regime])
        ax.set_xlabel("Passes over training data")
        ax.grid(alpha=0.25)
    reuse_axes[0].set_ylabel("Test loss (nats/token)")
    reuse_axes[-1].legend(loc="best", fontsize=8)
    reuse_fig.suptitle("Test loss by training data reuse")
    reuse_fig.savefig(run_root / "test_by_reuse.png", dpi=160)
    return fig, reuse_fig
