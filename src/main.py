#!/usr/bin/env python
"""准备数据、训练和评估鱼类图像分类模型，并保存实验结果。"""

import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import paddle
from PIL import Image

from data import FishDataset, prepare_data
from model import Model_CNN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def show_img(img, predict, path):
    """保存带有真实类别和预测类别的样例图。"""
    figure = plt.figure()
    plt.title(predict)
    plt.imshow(img, cmap="binary")
    plt.savefig(path)
    plt.close(figure)


def main():
    random.seed(42)
    np.random.seed(42)
    paddle.seed(42)

    prepare_data()
    train_dataset = FishDataset(mode="train")
    eval_dataset = FishDataset(mode="eval")
    OUTPUT_DIR.mkdir(exist_ok=True)

    model = Model_CNN()
    model.train(train_dataset)
    metrics = model.eval(eval_dataset)
    predict_result = model.predict(eval_dataset)

    # Paddle 2.6 返回最后一批的损失，这里按整个评估集计算平均交叉熵。
    labels = paddle.to_tensor([label for _, label in eval_dataset.data], dtype="int64")
    loss = paddle.nn.functional.cross_entropy(
        paddle.to_tensor(predict_result), labels, use_softmax=False
    )
    with (OUTPUT_DIR / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump({"acc": float(metrics["acc"]), "loss": float(loss.item())}, file, indent=4)

    for index, (image_path, real_class) in enumerate(eval_dataset.data[:10]):
        predict_class = np.argmax(predict_result[index])
        title = f"{index}   real_class: {real_class}   predict_class: {predict_class}"
        with Image.open(image_path) as image:
            show_img(image, title, OUTPUT_DIR / f"prediction_{index:02d}.png")

    model.save(OUTPUT_DIR / "inference_model")


if __name__ == "__main__":
    main()
