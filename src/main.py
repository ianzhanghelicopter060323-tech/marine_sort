#!/usr/bin/env python
"""准备数据、训练和评估鱼类图像分类模型，并保存实验结果。"""

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from statistics import mean, median, stdev
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Dict, Optional

import matplotlib
import numpy as np
import paddle
from PIL import Image

from data import TRACK_SPLITS, FishDataset, prepare_data, prepare_track_data
from model import (
    Model_CNN, Model_CNN_CBAM, Model_ResNet18, Model_ResNet18_CBAM, Model_ResNet18_SE, TrainingTimer,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
GROUP_LABELS = {
    "cnn": "对照组 · CNN",
    "resnet18": "实验 1 · ResNet18",
    "se": "实验 2 · ResNet18 + SE",
    "cbam": "实验 3 · ResNet18 + CBAM",
}


def show_img(img, predict, path):
    """保存带有真实类别和预测类别的样例图。"""
    figure = plt.figure()
    plt.title(predict)
    plt.imshow(img, cmap="binary")
    plt.savefig(path)
    plt.close(figure)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def run_cnn(output_dir=OUTPUT_DIR):
    """运行原有 CNN 实验。"""
    print("\n=== 基础实验 · CNN ===", flush=True)
    set_seed(42)

    prepare_data()
    train_dataset = FishDataset(mode="train")
    eval_dataset = FishDataset(mode="eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Model_CNN()
    model.train(train_dataset)
    metrics = model.eval(eval_dataset)
    predict_result = model.predict(eval_dataset)

    # Paddle 2.6 返回最后一批的损失，这里按整个评估集计算平均交叉熵。
    labels = paddle.to_tensor([label for _, label in eval_dataset.data], dtype="int64")
    loss = paddle.nn.functional.cross_entropy(
        paddle.to_tensor(predict_result), labels, use_softmax=False
    )
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump({"acc": float(metrics["acc"]), "loss": float(loss.item())}, file, indent=4)

    for index, (image_path, real_class) in enumerate(eval_dataset.data[:10]):
        predict_class = np.argmax(predict_result[index])
        title = f"{index}   real_class: {real_class}   predict_class: {predict_class}"
        with Image.open(image_path) as image:
            show_img(image, title, output_dir / f"prediction_{index:02d}.png")

    model.save(output_dir / "inference_model")
    print(f"[SUCCEED] 基础实验已完成，结果写入 {output_dir}", flush=True)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    ratio: Optional[int] = None
    kernel: Optional[int] = None

    def build(self):
        if self.name == "cnn":
            return Model_CNN()
        if self.name == "cnn_cbam" and self.ratio == 16 and self.kernel == 3:
            return Model_CNN_CBAM(self.ratio, self.kernel)
        if self.name == "resnet18":
            return Model_ResNet18()
        if self.name == "se" and self.ratio in (8, 16, 32):
            return Model_ResNet18_SE(self.ratio)
        if self.name == "cbam" and self.ratio in (8, 16, 32) and self.kernel in (3, 7):
            return Model_ResNet18_CBAM(self.ratio, self.kernel)
        raise ValueError(f"无效模型配置：{self}")

    def run_id(self, seed):
        parts = [self.name]
        if self.ratio is not None:
            parts.append(f"r{self.ratio}")
        if self.kernel is not None:
            parts.append(f"k{self.kernel}")
        return "_".join([*parts, f"seed{seed}"])


def write_json(path, data):
    """先完整写出 JSON，再替换目标文件。"""
    with TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / path.name
        staged.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged.replace(path)


def run_gpu_worker(device, job, *args):
    """让每次 GPU 作业拥有独立的 CUDA 上下文和显存缓存。"""
    process = get_context("spawn").Process(target=run_on_device, args=(device, job, args))
    process.start()
    try:
        process.join()
    except KeyboardInterrupt:
        if process.is_alive():
            process.terminate()
        process.join()
        raise
    if process.exitcode != 0:
        raise RuntimeError(f"GPU 子进程退出，退出码 {process.exitcode}。请查看上方报错。")


def run_on_device(device, job, args):
    paddle.set_device(device)
    job(*args)


def train_configuration(root, config, seed, settings, train, val):
    """训练并保存一个配置，完整且匹配的记录可直接复用。"""
    run_dir = root / config.run_id(seed)
    expected = {"model": asdict(config), "seed": seed, "settings": settings}
    if run_dir.exists():
        result = json.loads((run_dir / "training.json").read_text(encoding="utf-8"))
        if result["config"] != expected or not (run_dir / "model.pdparams").is_file():
            raise ValueError(f"{run_dir} 的配置或模型文件不匹配，请为本次运行指定新的输出目录")
        print(f"复用已完成的训练：{run_dir.name}", flush=True)
        return result

    device = settings.get("device", "cpu")
    if device.startswith("gpu:"):
        run_gpu_worker(device, train_new_configuration, run_dir, config, seed, settings, train, val)
        return json.loads((run_dir / "training.json").read_text(encoding="utf-8"))
    return train_new_configuration(run_dir, config, seed, settings, train, val)


def train_new_configuration(run_dir, config, seed, settings, train, val):
    """在当前进程里完成一次训练、验证和权重保存。"""
    expected = {"model": asdict(config), "seed": seed, "settings": settings}
    print(f"开始训练 {run_dir.name}，共 {settings['epochs']} 轮，批大小 {settings['batch_size']}", flush=True)
    set_seed(seed)
    model = config.build()
    timer = TrainingTimer(settings.get("device", "cpu"))
    started = perf_counter()
    model.train(train, settings["epochs"], settings["batch_size"], callbacks=[timer])
    train_seconds = perf_counter() - started
    full_steps = [
        step["milliseconds"] for step in timer.steps[settings["warmup_steps"]:]
        if step["batch_size"] == settings["batch_size"]
    ]

    started = perf_counter()
    metrics, _ = model.eval(val, detailed=True)
    validation_seconds = perf_counter() - started
    with TemporaryDirectory(dir=run_dir.parent) as temp:
        staged = Path(temp) / run_dir.name
        staged.mkdir()
        started = perf_counter()
        model.save_weights(staged / "model.pdparams")
        result = {
            "config": expected,
            "validation": metrics,
            "validation_seconds": validation_seconds,
            "save_seconds": perf_counter() - started,
            "timing": {
                "steps": timer.steps,
                "epoch_seconds": timer.epochs,
                "train_seconds": train_seconds,
                "median_full_step_ms": median(full_steps) if full_steps else None,
            },
        }
        write_json(staged / "training.json", result)
        staged.rename(run_dir)
    print(f"[SUCCEED] {run_dir.name} 训练、验证和权重保存完成", flush=True)
    return result


def run_preliminary(root, settings, train, val):
    """在验证集上选出各注意力模型的最佳配置。"""
    configs = [ModelConfig("cnn"), ModelConfig("resnet18")]
    configs.extend(ModelConfig("se", ratio) for ratio in (8, 16, 32))
    configs.extend(ModelConfig("cbam", ratio, kernel) for ratio in (8, 16, 32) for kernel in (3, 7))
    print(f"\n=== 预实验 · 验证集选参（{len(configs)} 个配置）===", flush=True)
    candidates = []
    for index, config in enumerate(configs, start=1):
        print(
            f"\n[预实验 {index}/{len(configs)}] {GROUP_LABELS[config.name]} "
            f"({config.run_id(42)})", flush=True,
        )
        candidates.append(train_configuration(root, config, 42, settings, train, val))
    selected = [asdict(config) for config in configs[:2]]
    for name in ("se", "cbam"):
        group = [run for run in candidates if run["config"]["model"]["name"] == name]
        best = min(group, key=lambda run: (
            -run["validation"]["macro_f1"],
            -run["validation"]["accuracy"],
            float("inf") if run["timing"]["median_full_step_ms"] is None
            else run["timing"]["median_full_step_ms"],
        ))
        selected.append(best["config"]["model"])
    write_json(root / "preliminary.json", {
        "settings": settings,
        "selected": selected,
        "candidates": [
            {
                "config": run["config"]["model"],
                "macro_f1": run["validation"]["macro_f1"],
                "accuracy": run["validation"]["accuracy"],
                "median_full_step_ms": run["timing"]["median_full_step_ms"],
            }
            for run in candidates
        ],
    })
    selected_ids = "、".join(ModelConfig(**options).run_id(42) for options in selected[2:])
    print(f"[SUCCEED] 预实验完成，入选配置为 {selected_ids}，结果写入 {root / 'preliminary.json'}", flush=True)


def summarize_runs(rows):
    """汇总多个种子的分类和时间指标。"""
    summary = {}
    for name in rows[0]:
        values = [row[name] for row in rows if row[name] is not None]
        summary[name] = {
            "mean": mean(values) if values else None,
            "std": stdev(values) if len(values) > 1 else None,
            "count": len(values),
        }
    return summary


def evaluate_saved_model(run_dir, config, training_config, test):
    """重载一个配置的权重，保存独立测试指标和逐图预测。"""
    model = config.build()
    model.load_weights(run_dir / "model.pdparams")
    started = perf_counter()
    metrics, predictions = model.eval(test, detailed=True)
    evaluation = {
        "config": training_config,
        "metrics": metrics,
        "test_seconds": perf_counter() - started,
    }
    lines = ["image\tlabel\tprediction\n"]
    for (image, label), predicted in zip(test.data, predictions):
        lines.append(f"{image.relative_to(test.data_dir).as_posix()}\t{label}\t{predicted}\n")
    (run_dir / "test_predictions.tsv").write_text("".join(lines), encoding="utf-8")
    write_json(run_dir / "test.json", evaluation)


def train_and_test(root, config, seed, settings, train, val, test):
    """完成一个种子的训练、权重重载和独立测试。"""
    run_id = config.run_id(seed)
    device = settings["device"]
    result = train_configuration(root, config, seed, settings, train, val)
    run_dir = root / run_id
    evaluation_args = (run_dir, config, result["config"], test)
    if device.startswith("gpu:"):
        run_gpu_worker(device, evaluate_saved_model, *evaluation_args)
    else:
        evaluate_saved_model(*evaluation_args)
    evaluation = json.loads((run_dir / "test.json").read_text(encoding="utf-8"))
    metrics = evaluation["metrics"]
    print(
        f"[SUCCEED] {run_id} 独立测试完成，"
        f"宏平均 F1 {metrics['macro_f1']:.2%}，准确率 {metrics['accuracy']:.2%}", flush=True,
    )
    return evaluation, score_row(metrics, result["timing"])


def score_row(metrics, timing):
    """提取单次独立测试与训练的比较指标。"""
    return {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "train_seconds": timing["train_seconds"],
        "median_full_step_ms": timing["median_full_step_ms"],
        **{f"recall_{label}": value for label, value in enumerate(metrics["recall"])},
    }


def run_experiment(root, selection, train, val, test):
    """复用种子 42 权重，并用独立评估集进行配对比较。"""
    scores: Dict[str, dict] = {}
    per_run = []
    for options in selection["selected"]:
        config = ModelConfig(**options)
        if not (root / config.run_id(42) / "model.pdparams").is_file():
            raise FileNotFoundError(f"缺少预实验模型 {config.run_id(42)}，请重新运行预实验")
    print("\n=== 正式实验 · 独立测试 ===", flush=True)
    for group_index, options in enumerate(selection["selected"], start=1):
        config = ModelConfig(**options)
        scores[config.name] = {}
        print(f"\n[正式实验 {group_index}/{len(selection['selected'])}] {GROUP_LABELS[config.name]}", flush=True)
        for seed in (42, 43, 44):
            run_id = config.run_id(seed)
            print(f"[运行] {run_id}", flush=True)
            evaluation, scores[config.name][seed] = train_and_test(
                root, config, seed, selection["settings"], train, val, test
            )
            per_run.append(evaluation)
        print(f"[SUCCEED] {GROUP_LABELS[config.name]} 的三个种子均已完成", flush=True)

    comparisons = {}
    for name, reference in (("resnet18", "cnn"), ("se", "resnet18"), ("cbam", "resnet18")):
        paired = []
        for seed, row in scores[name].items():
            baseline = scores[reference][seed]
            paired.append({
                key: value - baseline[key] if value is not None and baseline[key] is not None else None
                for key, value in row.items()
            })
        comparisons[f"{name}_vs_{reference}"] = summarize_runs(paired)
    write_json(root / "experiment.json", {
        "runs": per_run,
        "groups": {name: summarize_runs(list(rows.values())) for name, rows in scores.items()},
        "paired_differences": comparisons,
    })
    print(f"[SUCCEED] 正式实验全部完成，结果写入 {root / 'experiment.json'}", flush=True)


def run_supplement(root, settings, train, val, test):
    """仅训练 CNN + CBAM，依据原始同种子结果计算两种骨干上的增益。"""
    config = ModelConfig("cnn_cbam", 16, 3)
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    references = {}
    for run in experiment["runs"]:
        saved = run["config"]
        name = saved["model"]["name"]
        if name not in ("cnn", "resnet18", "cbam"):
            continue
        if saved["settings"] != settings:
            raise ValueError("补充实验与原正式实验的训练设置或划分不一致")
        if name == "cbam" and saved["model"] != asdict(ModelConfig("cbam", 16, 3)):
            raise ValueError("原 ResNet18 + CBAM 的参数与补充实验不一致")
        seed = saved["seed"]
        run_dir = root / ModelConfig(**saved["model"]).run_id(seed)
        training = json.loads((run_dir / "training.json").read_text(encoding="utf-8"))
        references[name, seed] = score_row(run["metrics"], training["timing"])
    seeds = (42, 43, 44)
    if set(references) != {(name, seed) for name in ("cnn", "resnet18", "cbam") for seed in seeds}:
        raise ValueError("需要三个骨干对照配置在种子 42、43、44 下的完整结果")

    print("\n=== 补充实验 · CNN + CBAM（r=16，k=3）===", flush=True)
    runs, scores, paired = [], [], []
    for seed in seeds:
        print(f"[补充实验] {config.run_id(seed)}", flush=True)
        evaluation, score = train_and_test(root, config, seed, settings, train, val, test)
        runs.append(evaluation)
        scores.append(score)
        baseline = references["cnn", seed]
        delta = {
            key: value - baseline[key] if value is not None and baseline[key] is not None else None
            for key, value in score.items()
        }
        cnn_gain = 100 * (score["macro_f1"] - baseline["macro_f1"])
        resnet_gain = 100 * (references["cbam", seed]["macro_f1"] - references["resnet18", seed]["macro_f1"])
        paired.append({
            "seed": seed, "cnn_cbam_minus_cnn": delta,
            "cnn_f1_gain_pp": cnn_gain, "resnet18_f1_gain_pp": resnet_gain,
            "gain_difference_pp": cnn_gain - resnet_gain,
            "cnn_f1_relative_gain_pct": cnn_gain / baseline["macro_f1"] if baseline["macro_f1"] else None,
        })
    gain_keys = ("cnn_f1_gain_pp", "resnet18_f1_gain_pp", "gain_difference_pp", "cnn_f1_relative_gain_pct")
    write_json(root / "supplement.json", {
        "runs": runs, "group": summarize_runs(scores), "paired": paired,
        "cnn_cbam_minus_cnn": summarize_runs([row["cnn_cbam_minus_cnn"] for row in paired]),
        "gain_summary": summarize_runs([{key: row[key] for key in gain_keys} for row in paired]),
    })
    print(f"[SUCCEED] 补充实验完成，结果写入 {root / 'supplement.json'}", flush=True)


def benchmark_training(device, batch_size, steps):
    """用真实训练图像测量最大 CBAM 配置的显存需求和步耗时。"""
    with TemporaryDirectory() as temp:
        split_dir = Path(temp)
        prepare_track_data(split_dir)
        train = FishDataset("train", split_dir=split_dir)
        sample_count = batch_size * steps
        if sample_count > len(train):
            raise ValueError(f"测速需要 {sample_count} 张训练图，实际有 {len(train)} 张")
        set_seed(42)
        indices = np.random.default_rng(42).choice(len(train), size=sample_count, replace=False)
        sample = paddle.io.Subset(train, indices.tolist())
        timer = TrainingTimer(device)
        model = Model_ResNet18_CBAM(ratio=8, kernel_size=7)
        print(f"测速 {device}，CBAM r=8、k=7，{steps} 步，每批 {batch_size} 张", flush=True)
        model.train(sample, epochs=1, batch_size=batch_size, callbacks=[timer])
        milliseconds = median(step["milliseconds"] for step in timer.steps[1:])
        print(f"预热一步后的中位耗时 {milliseconds / 1000:.3f} 秒/步")
        if device.startswith("gpu:"):
            allocated = paddle.device.cuda.max_memory_allocated() / (1024 ** 2)
            reserved = paddle.device.cuda.max_memory_reserved() / (1024 ** 2)
            print(f"Paddle 峰值显存分配 {allocated:.0f} MiB，峰值保留 {reserved:.0f} MiB")


def main():
    parser = argparse.ArgumentParser(description="鱼类图像分类实验")
    parser.add_argument("--stage", choices=("cnn", "preliminary", "experiment", "supplement", "benchmark"), default="cnn")
    parser.add_argument("--batch-size", type=int, default=256, help="预实验或测速的训练批大小")
    parser.add_argument("--device", default="cpu", help="训练设备，例如 cpu 或 gpu:0")
    parser.add_argument("--benchmark-steps", type=int, default=4, help="测速批次数，包含一个预热批次")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    device = args.device
    if device == "gpu":
        device = "gpu:0"
    if device != "cpu":
        if not device.startswith("gpu:") or not device[4:].isdigit():
            parser.error("--device 使用 cpu、gpu 或 gpu:N")
        if not paddle.is_compiled_with_cuda():
            parser.error("当前解释器使用 CPU 版 Paddle，请用 GPU 版 Paddle 的虚拟环境")
        index = int(device[4:])
        if index >= paddle.device.cuda.device_count():
            parser.error(f"设备 {device} 不可用，请检查 NVIDIA 驱动和 CUDA_VISIBLE_DEVICES")
        device = f"gpu:{index}"
    if args.stage == "cnn":
        paddle.set_device(device)
        default_cnn_dir = OUTPUT_DIR if device == "cpu" else OUTPUT_DIR / f"cnn_gpu{device[4:]}"
        run_cnn(args.output_dir or default_cnn_dir)
        return
    if args.batch_size < 1:
        parser.error("--batch-size 必须为正整数")

    if args.stage == "benchmark":
        if args.benchmark_steps < 2:
            parser.error("--benchmark-steps 至少为 2，以便排除第一步预热")
        paddle.set_device(device)
        benchmark_training(device, args.batch_size, args.benchmark_steps)
        return

    root = args.output_dir or OUTPUT_DIR / (
        "further_exp" if device == "cpu" else f"further_exp_gpu{device[4:]}"
    )
    split_dir = root / "splits"
    selection_path = root / "preliminary.json"
    settings = {
        "batch_size": args.batch_size,
        "epochs": 5,
        "learning_rate": 0.0005,
        "device": device,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "1"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "warmup_steps": 1,
    }
    selection = None
    if args.stage == "preliminary":
        details = prepare_track_data(split_dir)
        write_json(split_dir / "class_counts.json", details)
    else:
        if not selection_path.is_file():
            parser.error("请先在同一输出目录完成预实验")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        # 正式实验沿用预实验实际使用的批大小。
        settings["batch_size"] = selection["settings"]["batch_size"]
    settings["split_sha256"] = {
        mode: hashlib.sha256((split_dir / f"{mode}.txt").read_bytes()).hexdigest()
        for mode in TRACK_SPLITS
    }
    train = FishDataset("train", split_dir=split_dir)
    val = FishDataset("val", split_dir=split_dir)
    if selection is None:
        run_preliminary(root, settings, train, val)
    else:
        if settings != selection["settings"]:
            parser.error("当前设备线程设置或数据清单与预实验不一致")
        test = FishDataset("test", split_dir=split_dir)
        if args.stage == "supplement":
            run_supplement(root, settings, train, val, test)
        else:
            run_experiment(root, selection, train, val, test)


if __name__ == "__main__":
    main()
