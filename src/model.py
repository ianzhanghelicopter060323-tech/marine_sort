"""鱼类图像分类的卷积网络及训练接口。"""

import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, Linear, MaxPool2D


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


class Model_CNN:
    """配置 Paddle 模型，并提供训练、评估、预测和导出接口。"""

    def __init__(self) -> None:
        inputs = paddle.static.InputSpec(shape=[None, 3, 47, 47], dtype="float32", name="inputs")
        self.model = paddle.Model(MyCNN(), inputs=inputs)
        self.model.prepare(
            paddle.optimizer.Adam(learning_rate=0.0005, parameters=self.model.parameters()),
            # 网络已经过 Softmax，交叉熵直接使用类别概率。
            paddle.nn.CrossEntropyLoss(use_softmax=False),
            paddle.metric.Accuracy(),
        )

    def train(self, dataset):
        self.model.fit(dataset, epochs=5, batch_size=256, verbose=2)

    def eval(self, dataset):
        return self.model.evaluate(dataset, batch_size=64, verbose=2)

    def predict(self, dataset):
        return self.model.predict(dataset, batch_size=64, verbose=0, stack_outputs=True)[0]

    def save(self, path):
        self.model.save(str(path), training=False)
