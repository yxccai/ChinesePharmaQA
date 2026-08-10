from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    }
)


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Recreate Figures 2 and 3 from paper results.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=script.parent / "results" / "paper_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script.parent / "figures",
    )
    return parser.parse_args()


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def figure_answer_distribution(summary: dict, output_dir: Path) -> None:
    distribution = summary["dataset"]["answer_distribution"]
    labels = [item["range"] for item in distribution]
    counts = [item["count"] for item in distribution]

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    bars = ax.bar(
        range(len(labels)),
        counts,
        width=0.7,
        color="#4c78a8",
        edgecolor="#2f4b67",
        linewidth=0.7,
    )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("答案集合规模区间")
    ax.set_ylabel("问题数量")
    ax.set_ylim(0, max(counts) * 1.12)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(counts) * 0.02,
            str(value),
            ha="center",
            va="bottom",
            fontsize=8.3,
        )
    stats = summary["dataset"]["answer_set_size"]
    ax.text(
        0.99,
        0.96,
        f"n = {summary['dataset']['fusion_queries']:,}，中位数 = {stats['median']:g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color="#374151",
    )
    fig.tight_layout()
    save(fig, output_dir, "fig02_answer_distribution")


def series(curves: dict, method: str, key: str) -> list[float]:
    return [float(item[key]) for item in curves[method]]


def figure_topk_curves(summary: dict, output_dir: Path) -> None:
    curves = summary["retrieval_curves"]
    bm25 = curves["BM25-FullFields"]
    bge = curves["BGE"]
    k_bm25 = [item["k"] for item in bm25]
    k_bge = [item["k"] for item in bge]
    bm25_original_f1 = series(curves, "BM25-Original", "f1")

    colors = {
        "precision": "#4c78a8",
        "recall": "#e45756",
        "f1": "#2a9d8f",
        "original": "#6b7280",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharey=True)

    def setup(ax: plt.Axes, k_values: list[int], title: str) -> list[int]:
        x_values = list(range(len(k_values)))
        ax.set_xticks(x_values, [str(k) for k in k_values])
        ax.set_xlabel("Top-K")
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylim(0, 1.03)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        return x_values

    x_bm25 = setup(axes[0], k_bm25, "(a) BM25-FullFields")
    axes[0].set_ylabel("宏平均指标")
    axes[0].plot(
        x_bm25,
        series(curves, "BM25-FullFields", "precision"),
        marker="o",
        color=colors["precision"],
        linewidth=1.7,
        label="Precision",
    )
    axes[0].plot(
        x_bm25,
        series(curves, "BM25-FullFields", "recall"),
        marker="s",
        color=colors["recall"],
        linewidth=1.7,
        label="Recall",
    )
    bm25_f1 = series(curves, "BM25-FullFields", "f1")
    axes[0].plot(
        x_bm25,
        bm25_f1,
        marker="^",
        color=colors["f1"],
        linewidth=1.9,
        label="F1",
    )
    axes[0].plot(
        x_bm25,
        bm25_original_f1,
        marker="x",
        linestyle="--",
        color=colors["original"],
        linewidth=1.2,
        label="Original F1",
    )
    best_bm25 = max(range(len(bm25_f1)), key=bm25_f1.__getitem__)
    axes[0].annotate(
        f"F1={bm25_f1[best_bm25]:.4f}",
        xy=(best_bm25, bm25_f1[best_bm25]),
        xytext=(best_bm25 + 0.35, bm25_f1[best_bm25] + 0.10),
        arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 0.8},
        fontsize=8.2,
    )
    axes[0].legend(loc="center right", frameon=False)

    x_bge = setup(axes[1], k_bge, "(b) BGE")
    for key, marker, label in (
        ("precision", "o", "Precision"),
        ("recall", "s", "Recall"),
        ("f1", "^", "F1"),
    ):
        axes[1].plot(
            x_bge,
            series(curves, "BGE", key),
            marker=marker,
            color=colors[key],
            linewidth=1.9 if key == "f1" else 1.7,
            label=label,
        )
    bge_f1 = series(curves, "BGE", "f1")
    best_bge = max(range(len(bge_f1)), key=bge_f1.__getitem__)
    axes[1].annotate(
        f"F1={bge_f1[best_bge]:.4f}",
        xy=(best_bge, bge_f1[best_bge]),
        xytext=(best_bge + 0.25, bge_f1[best_bge] + 0.13),
        arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 0.8},
        fontsize=8.2,
    )
    axes[1].legend(loc="upper left", frameon=False)

    fig.tight_layout(w_pad=2.2)
    save(fig, output_dir, "fig03_topk_curves")


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    figure_answer_distribution(summary, args.output_dir)
    figure_topk_curves(summary, args.output_dir)
    print(f"saved Figures 2 and 3 to {args.output_dir}")


if __name__ == "__main__":
    main()
