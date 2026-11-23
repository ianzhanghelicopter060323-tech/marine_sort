"""从已保存的鱼类分类实验结果生成图表。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np


DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "output" / "further_exp_gpu0"
MODELS = ("cnn", "resnet18", "se", "cbam")
LABELS = {
    "cnn": "CNN",
    "resnet18": "ResNet18",
    "se": "ResNet18 + SE",
    "cbam": "ResNet18 + CBAM",
}
COLORS = {"cnn": "#63748C", "resnet18": "#3672B2", "se": "#168276", "cbam": "#A75226"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(figure, output_dir, name):
    path = output_dir / name
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"已保存 {path}")


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#263447",
        "axes.titleweight": "bold",
        "xtick.color": "#263447",
        "ytick.color": "#263447",
        "grid.color": "#DFE5EC",
        "axes.axisbelow": True,
    })


def plot_selection(preliminary, output_dir):
    """比较预实验的验证集 F1，标出入选的注意力配置。"""
    candidates = preliminary["candidates"]
    selected = {config["name"]: config for config in preliminary["selected"]}
    reference = next(row["macro_f1"] for row in candidates if row["config"]["name"] == "resnet18")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    figure.suptitle("预实验 · 验证集宏平均 F1（种子 42）", fontsize=16, fontweight="bold")
    all_scores = [row["macro_f1"] for row in candidates if row["config"]["name"] in ("se", "cbam")]
    lower = max(0, min(all_scores + [reference]) - 0.045)
    upper = min(1, max(all_scores + [reference]) + 0.055)

    for axis, name in zip(axes, ("se", "cbam")):
        axis.set_title(LABELS[name], loc="left")
        axis.axhline(reference, linestyle="--", color=COLORS["resnet18"], label="ResNet18 对照")
        kernels = (None,) if name == "se" else (3, 7)
        for index, kernel in enumerate(kernels):
            rows = sorted(
                (row for row in candidates if row["config"]["name"] == name and row["config"]["kernel"] == kernel),
                key=lambda row: row["config"]["ratio"],
            )
            color = COLORS[name] if index == 0 else "#886B9E"
            ratios = [row["config"]["ratio"] for row in rows]
            scores = [row["macro_f1"] for row in rows]
            label = "SE" if kernel is None else f"卷积核 {kernel}×{kernel}"
            axis.plot(ratios, scores, color=color, marker="o", linewidth=2, label=label)
            for row in rows:
                if row["config"] == selected[name]:
                    ratio = row["config"]["ratio"]
                    score = row["macro_f1"]
                    axis.scatter(ratio, score, s=165, facecolors="none", edgecolors=color, linewidth=1.8)
                    axis.annotate(f"入选 {score:.2%}", (ratio, score), xytext=(0, 10),
                                  textcoords="offset points", ha="center", color=color)
        axis.set(xlabel="通道压缩比", xticks=[8, 16, 32], xlim=(5, 35), ylim=(lower, upper))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y")
        axis.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2, frameon=False)
    axes[0].set_ylabel("宏平均 F1")
    save_figure(figure, output_dir, "preliminary_selection.png")


def plot_test_metrics(experiment, output_dir):
    """展示独立测试集的种子结果、均值与样本标准差。"""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    figure.suptitle("独立测试集表现 · 均值 ± 标准差（3 个种子）", fontsize=16, fontweight="bold")
    for axis, metric, title in zip(axes, ("macro_f1", "accuracy"), ("宏平均 F1", "总体准确率")):
        axis.set_title(title, loc="left")
        bounds = []
        for index, model in enumerate(MODELS):
            runs = sorted(
                (run for run in experiment["runs"] if run["config"]["model"]["name"] == model),
                key=lambda run: run["config"]["seed"],
            )
            values = [run["metrics"][metric] for run in runs]
            summary = experiment["groups"][model][metric]
            center, spread = summary["mean"], summary["std"]
            bounds.extend([*values, center - spread, center + spread])
            axis.scatter(values, np.linspace(index - 0.18, index + 0.18, len(values)),
                         color=COLORS[model], alpha=0.35, s=22, zorder=3)
            axis.errorbar(center, index, xerr=spread, fmt="D", color=COLORS[model],
                          capsize=4, linewidth=1.8, markersize=5, zorder=4)
            axis.annotate(f"{center:.2%}", (center + spread, index), xytext=(8, 0),
                          textcoords="offset points", va="center", fontsize=9)
        axis.set_xlim(max(0, min(bounds) - 0.025), min(1.03, max(bounds) + 0.045))
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_yticks(list(range(len(MODELS))))
        axis.set_yticklabels([LABELS[model] for model in MODELS])
        axis.grid(axis="x")
    axes[0].invert_yaxis()
    save_figure(figure, output_dir, "test_metrics.png")


def plot_training_time(experiment, output_dir):
    """比较各模型总训练耗时与单步耗时。"""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, constrained_layout=True)
    figure.suptitle("训练耗时 · 均值 ± 标准差（3 个种子）", fontsize=16, fontweight="bold")
    metrics = [
        ("train_seconds", "每次 5 轮训练", "分钟", 60),
        ("median_full_step_ms", "预热后完整批次中位耗时", "毫秒 / 步", 1),
    ]
    indices = list(range(len(MODELS)))
    for axis, (key, title, unit, divisor) in zip(axes, metrics):
        values = [experiment["groups"][model][key]["mean"] / divisor for model in MODELS]
        deviations = [experiment["groups"][model][key]["std"] / divisor for model in MODELS]
        axis.barh(indices, values, xerr=deviations, color=[COLORS[model] for model in MODELS],
                  height=0.58, error_kw={"capsize": 3, "ecolor": "#263447"})
        maximum = max(value + error for value, error in zip(values, deviations))
        axis.set(xlim=(0, maximum * 1.18), xlabel=unit)
        axis.set_title(title, loc="left")
        for index, (value, error) in enumerate(zip(values, deviations)):
            axis.annotate(f"{value:.1f}", (value + error, index), xytext=(6, 0),
                          textcoords="offset points", va="center", fontsize=9)
        axis.set_yticks(indices)
        axis.set_yticklabels([LABELS[model] for model in MODELS])
        axis.grid(axis="x")
    axes[0].invert_yaxis()
    save_figure(figure, output_dir, "training_time.png")


def plot_confusion_matrices(experiment, class_counts, output_dir):
    """合并同组种子的预测计数，按真实类别归一化。"""
    classes = [row["class_name"].split("_", 1)[1]
               for row in sorted(class_counts, key=lambda row: row["label"])]
    ticks = list(range(len(classes)))
    figure, axes = plt.subplots(2, 2, figsize=(11.3, 10), sharex=True, sharey=True, constrained_layout=True)
    figure.suptitle("独立测试集混淆矩阵 · 按真实类别归一化（汇总 3 个种子）", fontsize=16, fontweight="bold")
    scale_mapping = ScalarMappable(norm=Normalize(0, 1), cmap="Blues")
    for axis, model in zip(axes.flat, MODELS):
        counts = np.sum([run["metrics"]["confusion_matrix"] for run in experiment["runs"]
                         if run["config"]["model"]["name"] == model], axis=0)
        totals = counts.sum(axis=1, keepdims=True)
        # 对角线表示该类别的召回率，空白留给没有测试样本的类别。
        normalized = np.divide(counts, totals, out=np.full(counts.shape, np.nan), where=totals > 0)
        axis.imshow(normalized, cmap=scale_mapping.cmap, norm=scale_mapping.norm, interpolation="nearest")
        axis.set_title(LABELS[model], loc="left")
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.set_xticklabels(classes, fontsize=7)
        axis.set_yticklabels(classes, fontsize=7)
        axis.label_outer()
    scale = figure.colorbar(scale_mapping, ax=axes.ravel().tolist(), shrink=0.8, format=PercentFormatter(1))
    scale.set_label("占该真实类别的预测比例")
    figure.supxlabel("预测类别")
    figure.supylabel("真实类别")
    save_figure(figure, output_dir, "confusion_matrices.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS, help="已完成的实验结果目录")
    parser.add_argument("--output-dir", type=Path, help="图表目录，默认为结果目录下的 figures")
    args = parser.parse_args()

    preliminary = read_json(args.results_dir / "preliminary.json")
    experiment = read_json(args.results_dir / "experiment.json")
    class_counts = read_json(args.results_dir / "splits" / "class_counts.json")
    output_dir = args.output_dir or args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_selection(preliminary, output_dir)
    plot_test_metrics(experiment, output_dir)
    plot_training_time(experiment, output_dir)
    plot_confusion_matrices(experiment, class_counts, output_dir)


if __name__ == "__main__":
    main()
