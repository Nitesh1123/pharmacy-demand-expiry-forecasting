"""Write the technical report's figures from saved results and the executed EDA notebook."""

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
NOTEBOOK_FIGURES = {0: "eda_store_sales.png", 1: "eda_seasonality.png"}
DEMAND_LABELS = {"gradient_boosting_tuned": "Gradient boosting (served)", "seasonal_naive": "Seasonal naive",
                 "moving_average_28d": "28-day moving average", "ridge": "Ridge"}
EXPIRY_LABELS = {"rule": "Rule (served)", "gradient_boosting": "Gradient boosting",
                 "logistic_regression": "Logistic regression"}


def _style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def demand_figure(report, path):
    models = sorted(report["demand"]["models"].items(), key=lambda item: -item[1]["wape"])
    fig, ax = plt.subplots(figsize=(7.6, 2.8), facecolor=SURFACE)
    values = [100 * metrics["wape"] for _, metrics in models]
    ax.barh([DEMAND_LABELS[name] for name, _ in models], values, height=0.42, color=BLUE)
    for index, value in enumerate(values):
        ax.text(value + 0.1, index, f"{value:.2f}%", va="center", color=INK, fontsize=10)
    _style(ax, "Demand error on the held-out test set")
    ax.set_xlabel("WAPE, % of units sold (lower is better)", color=MUTED)
    ax.set_xlim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def expiry_figure(report, path):
    models = sorted(report["expiry"]["models"].items(), key=lambda item: item[1]["f1"])
    fig, ax = plt.subplots(figsize=(7.6, 2.5), facecolor=SURFACE)
    for index, (name, metrics) in enumerate(models):
        low, high = metrics["intervals_95"]["f1"]
        ax.plot([low, high], [index, index], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.scatter([metrics["f1"]], [index], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(high + 0.004, index, f"{metrics['f1']:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(models)), [EXPIRY_LABELS[name] for name, _ in models])
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlim(0.86, 1.0)
    _style(ax, "Expiry alert F1 on the held-out test set, with 95% intervals")
    ax.set_xlabel("F1 (dot) and bootstrap interval (line), resampling independent batches", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_notebook_figures(notebook, directory):
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    images = [output["data"]["image/png"] for cell in cells for output in cell.get("outputs", [])
              if "image/png" in output.get("data", {})]
    for index, name in NOTEBOOK_FIGURES.items():
        (directory / name).write_bytes(base64.b64decode(images[index]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/test-evaluation.json"))
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/01_eda.ipynb"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    demand_figure(report, args.output / "test_demand_wape.png")
    expiry_figure(report, args.output / "test_expiry_f1.png")
    export_notebook_figures(args.notebook, args.output)
    print(f"Wrote figures to {args.output}")


if __name__ == "__main__":
    main()
