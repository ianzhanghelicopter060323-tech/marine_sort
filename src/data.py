"""鱼类图像的数据划分与预处理。"""

import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List
from zipfile import ZipFile

import numpy as np
import paddle
import paddle.vision.transforms as T
from PIL import Image


DATA_DIR = Path(__file__).resolve().parents[1] / "data_set"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
NUM_CLASSES = 23
TRACK_SPLITS = ("train", "val", "test")


def prepare_data(data_dir=DATA_DIR):
    """准备图像目录，按类别约九比一划分数据并生成清单。"""
    image_root = data_dir / "fish_image"
    if not image_root.is_dir():
        archive = data_dir / "fish_image23.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"请将鱼类数据放在 {image_root} 或 {archive}")
        with ZipFile(archive) as data_zip:
            data_zip.extractall(data_dir)

    # 固定目录和图像的遍历顺序，使标签和数据划分可以复现。
    class_dirs = sorted(path for path in image_root.glob("fish_*") if path.is_dir())
    if not class_dirs:
        raise ValueError(f"未找到鱼类图像目录：{image_root}")

    trainer_list = []
    eval_list = []
    class_detail = []
    for class_label, class_dir in enumerate(class_dirs):
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        eval_sum = 0
        for index, image_path in enumerate(images):
            record = f"{image_path.relative_to(data_dir).as_posix()}\t{class_label}\n"
            if index % 10 == 0:  # 每类按顺序每十张取一张用于评估。
                eval_list.append(record)
                eval_sum += 1
            else:
                trainer_list.append(record)

        class_detail.append({
            "class_name": class_dir.name,
            "class_label": class_label,
            "class_eval_images": eval_sum,
            "class_trainer_images": len(images) - eval_sum,
        })

    random.shuffle(trainer_list)
    random.shuffle(eval_list)
    (data_dir / "train.txt").write_text("".join(trainer_list), encoding="utf-8")
    (data_dir / "eval.txt").write_text("".join(eval_list), encoding="utf-8")

    readme = {
        "all_class_name": image_root.name,
        "all_class_sum": len(class_dirs),
        "all_class_images": len(trainer_list) + len(eval_list),
        "class_detail": class_detail,
    }
    with (data_dir / "readme.json").open("w", encoding="utf-8") as file:
        json.dump(readme, file, indent=4)


def prepare_track_data(split_dir, data_dir=DATA_DIR):
    """按轨迹划分三份清单，并返回各类的图像数与轨迹数。"""
    image_root = data_dir / "fish_image"
    class_dirs = sorted(path for path in image_root.glob("fish_*") if path.is_dir())
    if len(class_dirs) != NUM_CLASSES:
        raise ValueError(f"需要 {NUM_CLASSES} 类鱼图像，请检查 {image_root}")

    records = {mode: [] for mode in TRACK_SPLITS}
    details = []
    track_labels = {}
    for label, class_dir in enumerate(class_dirs):
        tracks = defaultdict(list)
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            prefix, separator, frame = path.stem.rpartition("_")
            if not separator or not prefix.startswith("fish_") or not prefix[5:].isdigit() or not frame.isdigit():
                raise ValueError(f"无法从文件名提取轨迹编号：{path}")
            if track_labels.setdefault(prefix, label) != label:
                raise ValueError(f"轨迹 {prefix} 对应多个类别")
            tracks[prefix].append(path.relative_to(data_dir).as_posix())
        if len(tracks) < len(TRACK_SPLITS):
            raise ValueError(f"{class_dir.name} 的轨迹不足以划分三个集合")

        total = sum(map(len, tracks.values()))
        targets: Dict[str, float] = dict(zip(TRACK_SPLITS, (total * 0.8, total * 0.1, total * 0.1)))
        image_counts: Dict[str, int] = dict.fromkeys(TRACK_SPLITS, 0)
        track_counts: Dict[str, int] = dict.fromkeys(TRACK_SPLITS, 0)
        groups = sorted(tracks.values(), key=len, reverse=True)
        for index, images in enumerate(groups):
            empty = [mode for mode in TRACK_SPLITS if track_counts[mode] == 0]
            # 留足轨迹以保证每个类别在三个集合中均有样本。
            candidates = empty if len(groups) - index == len(empty) else TRACK_SPLITS
            mode = max(candidates, key=lambda name: targets[name] - image_counts[name])
            records[mode].extend(f"{image}\t{label}\n" for image in images)
            image_counts[mode] += len(images)
            track_counts[mode] += 1
        details.append({
            "class_name": class_dir.name,
            "label": label,
            "images": image_counts,
            "tracks": track_counts,
        })

    split_dir.mkdir(parents=True, exist_ok=True)
    for mode, lines in records.items():
        (split_dir / f"{mode}.txt").write_text("".join(lines), encoding="utf-8")
    return details


class FishDataset(paddle.io.Dataset):
    """按清单读取鱼类图像与标签，训练集额外使用随机水平翻转。"""

    def __init__(self, mode="train", data_dir=DATA_DIR, split_dir=None):
        super().__init__()
        self.data_dir = data_dir
        if mode not in (*TRACK_SPLITS, "eval"):
            raise ValueError(f"未知的数据集模式：{mode}")

        self.data = []
        manifest_dir = data_dir if split_dir is None else split_dir
        with (manifest_dir / f"{mode}.txt").open(encoding="utf-8") as file:
            for line in file:
                image_file, label = line.rstrip("\n").split("\t")
                self.data.append((data_dir / image_file, int(label)))

        transforms: List[Callable] = [T.Resize((47, 47))]
        if mode == "train":
            transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.extend([
            T.ToTensor(),  # 图像维度从 HWC 转为 CHW。
            partial(T.normalize, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transforms = T.Compose(transforms)

    def __getitem__(self, index):
        """返回形状为 (3, 47, 47) 的图像张量及其整数标签。"""
        image_file, label = self.data[index]
        with Image.open(image_file) as image:
            image_tensor = self.transforms(image.convert("RGB"))
        return image_tensor, np.array(label, dtype="int64")

    def __len__(self):
        return len(self.data)
