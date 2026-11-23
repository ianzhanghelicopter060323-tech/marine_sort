"""鱼类图像分类的卷积网络及训练接口。"""

from time import perf_counter
from typing import Literal, Tuple, overload

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, Linear, MaxPool2D

from data import NUM_CLASSES


class MyCNN(paddle.nn.Layer):
    """用三组卷积和池化提取特征，再输出 23 类的概率。"""

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(in_channels=3, out_channels=20, kernel_size=5, stride=1)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = Conv2D(in_channels=50, out_channels=50, kernel_size=5, stride=1)
        self.pool3 = MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = Linear(in_features=200, out_features=23)

    def forward(self, inputs):
        x = self.pool1(F.relu(self.conv1(inputs)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # 池化后的特征图为 50 × 2 × 2，对应全连接层的 200 个输入。
        x = paddle.flatten(x, start_axis=1)
        x = self.linear1(x)
        return F.softmax(x)


class SE(paddle.nn.Layer):
    """从全局平均池化结果学习通道权重。"""

    def __init__(self, channels, ratio):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.mlp = paddle.nn.Sequential(
            Conv2D(channels, hidden, 1),
            paddle.nn.ReLU(),
            Conv2D(hidden, channels, 1),
        )

    def forward(self, inputs):
        weights = self.mlp(F.adaptive_avg_pool2d(inputs, 1))
        return inputs * F.sigmoid(weights)


class CBAM(paddle.nn.Layer):
    """依次用通道权重和空间权重调整残差特征。"""

    def __init__(self, channels, ratio, kernel_size):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.mlp = paddle.nn.Sequential(
            Conv2D(channels, hidden, 1, bias_attr=False),
            paddle.nn.ReLU(),
            Conv2D(hidden, channels, 1, bias_attr=False),
        )
        self.spatial = Conv2D(2, 1, kernel_size, padding=kernel_size // 2, bias_attr=False)

    def forward(self, inputs):
        average = self.mlp(F.adaptive_avg_pool2d(inputs, 1))
        maximum = self.mlp(F.adaptive_max_pool2d(inputs, 1))
        features = inputs * F.sigmoid(average + maximum)
        spatial = paddle.concat([
            paddle.mean(features, axis=1, keepdim=True),
            paddle.max(features, axis=1, keepdim=True),
        ], axis=1)
        return features * F.sigmoid(self.spatial(spatial))


class ResidualBlock(paddle.nn.Layer):
    """两个卷积组成残差分支，注意力位于分支相加之前。"""

    def __init__(self, in_channels, channels, stride, attention):
        super().__init__()
        self.conv1 = Conv2D(in_channels, channels, 3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm2D(channels)
        self.conv2 = Conv2D(channels, channels, 3, padding=1, bias_attr=False)
        self.bn2 = paddle.nn.BatchNorm2D(channels)
        self.attention = attention(channels)
        self.shortcut = paddle.nn.Identity()
        if stride != 1 or in_channels != channels:
            self.shortcut = paddle.nn.Sequential(
                Conv2D(in_channels, channels, 1, stride=stride, bias_attr=False),
                paddle.nn.BatchNorm2D(channels),
            )

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.bn2(self.conv2(features))
        return F.relu(self.shortcut(inputs) + self.attention(features))


class ResNet18(paddle.nn.Layer):
    """使用适配 47×47 输入的首层和四组残差块。"""

    def __init__(self, attention=None):
        super().__init__()
        if attention is None:
            attention = lambda channels: paddle.nn.Identity()
        self.stem = paddle.nn.Sequential(
            Conv2D(3, 64, 3, padding=1, bias_attr=False),
            paddle.nn.BatchNorm2D(64),
            paddle.nn.ReLU(),
        )
        blocks = []
        in_channels = 64
        for channels in (64, 128, 256, 512):
            stride = 1 if channels == 64 else 2
            blocks.append(ResidualBlock(in_channels, channels, stride, attention))
            blocks.append(ResidualBlock(channels, channels, 1, attention))
            in_channels = channels
        self.blocks = paddle.nn.Sequential(*blocks)
        self.pool = paddle.nn.AdaptiveAvgPool2D(1)
        self.classifier = Linear(512, NUM_CLASSES)

    def forward(self, inputs):
        features = self.blocks(self.stem(inputs))
        return self.classifier(paddle.flatten(self.pool(features), start_axis=1))


def classification_metrics(labels, predicted):
    """由固定为 23×23 的混淆矩阵计算分类指标。"""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    support = matrix.sum(axis=1)
    correct = matrix.diagonal()
    denominator = support + matrix.sum(axis=0)
    recall = np.divide(correct, support, out=np.zeros(NUM_CLASSES), where=support > 0)
    f1 = np.divide(2 * correct, denominator, out=np.zeros(NUM_CLASSES), where=denominator > 0)
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(correct.sum() / matrix.sum()),
        "recall": recall.tolist(),
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


class _Classifier:
    """封装四组网络共用的 Paddle 训练、评估和保存流程。"""

    def __init__(self, network, use_softmax):
        self.network = network
        inputs = paddle.static.InputSpec(shape=[None, 3, 47, 47], dtype="float32", name="inputs")
        self.model = paddle.Model(network, inputs=inputs)
        self.model.prepare(
            paddle.optimizer.Adam(learning_rate=0.0005, parameters=self.model.parameters()),
            paddle.nn.CrossEntropyLoss(use_softmax=use_softmax),
            paddle.metric.Accuracy(),
        )

    def train(self, dataset, epochs=5, batch_size=256, callbacks=None):
        """在训练集上更新网络，可传入 Paddle 回调记录耗时。"""
        self.model.fit(
            dataset, epochs=epochs, batch_size=batch_size, verbose=2,
            log_freq=1 if callbacks is not None else 10, callbacks=callbacks,
        )

    @overload
    def eval(self, dataset, *, detailed: Literal[False] = False) -> dict:
        ...

    @overload
    def eval(self, dataset, *, detailed: Literal[True]) -> Tuple[dict, np.ndarray]:
        ...

    def eval(self, dataset, *, detailed=False):
        """评估模型，可返回混淆矩阵指标和按清单排列的逐图预测。"""
        if detailed:
            labels = np.array([label for _, label in dataset.data])
            predicted = np.argmax(self.predict(dataset), axis=1)
            return classification_metrics(labels, predicted), predicted
        return self.model.evaluate(dataset, batch_size=64, verbose=2)

    def predict(self, dataset):
        return self.model.predict(dataset, batch_size=64, verbose=0, stack_outputs=True)[0]

    def save(self, path):
        self.model.save(str(path), training=False)

    def save_weights(self, path):
        paddle.save(self.network.state_dict(), str(path))

    def load_weights(self, path):
        self.network.set_state_dict(paddle.load(str(path)))


class Model_CNN(_Classifier):
    """现有三层 CNN，输出经过 Softmax 的概率。"""

    def __init__(self):
        super().__init__(MyCNN(), use_softmax=False)


class Model_ResNet18(_Classifier):
    """没有注意力模块的 ResNet18 对照模型。"""

    def __init__(self):
        super().__init__(ResNet18(), use_softmax=True)


class Model_ResNet18_SE(_Classifier):
    """在各残差块中使用指定压缩比的 SE。"""

    def __init__(self, ratio=16):
        super().__init__(ResNet18(lambda channels: SE(channels, ratio)), use_softmax=True)


class Model_ResNet18_CBAM(_Classifier):
    """在各残差块中使用指定压缩比和空间卷积核的 CBAM。"""

    def __init__(self, ratio=16, kernel_size=7):
        super().__init__(
            ResNet18(lambda channels: CBAM(channels, ratio, kernel_size)),
            use_softmax=True,
        )


class TrainingTimer(paddle.callbacks.Callback):
    """以相邻批次结束时刻计时，包含批次间的数据读取。"""

    def __init__(self, device="cpu"):
        super().__init__()
        self.steps = []
        self.epochs = []
        self.epoch = 0
        self.device = device
        self.epoch_start = self.previous_end = 0.0

    def on_epoch_begin(self, epoch=None, logs=None):
        if epoch is None:
            raise ValueError("训练计时需要轮次编号")
        self.epoch = epoch
        self.epoch_start = self.previous_end = perf_counter()

    def on_train_batch_end(self, step, logs=None):
        if logs is None:
            raise ValueError("训练计时需要批次日志")
        if self.device.startswith("gpu:"):
            paddle.device.cuda.synchronize()
        now = perf_counter()
        self.steps.append({
            "epoch": self.epoch + 1,
            "step": step + 1,
            "batch_size": logs["batch_size"],
            "milliseconds": (now - self.previous_end) * 1000,
        })
        self.previous_end = now

    def on_epoch_end(self, epoch=None, logs=None):
        self.epochs.append(perf_counter() - self.epoch_start)
