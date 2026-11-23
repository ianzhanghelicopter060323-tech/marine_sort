# 鱼类图像分类实验

这是一个基于 PaddlePaddle 的 AI 基础课程实验。数据集包含 23 类鱼、27,370 张图像。项目先用普通 CNN 完成分类，再用同一份按跟踪轨迹划分的数据，对照比较 CNN、ResNet18、ResNet18 + SE、ResNet18 + CBAM 和 CNN + CBAM。

## 查看已有结果

在项目根目录运行以下命令，可用已保存的 GPU 实验结果生成图表。绘图只依赖 NumPy 和 Matplotlib，输出位于 `output/further_exp_gpu0/figures/`。

```bash
.venv/bin/python src/img.py
```

扩展实验以宏平均 F1 为主指标，同时记录准确率、各类召回率和训练耗时。批大小为 128、每次训练 5 轮时，CNN 的平均宏 F1 为 55.96%，平均准确率为 93.08%，平均训练耗时为 80.8 秒。ResNet18 + CBAM 的两个分类指标均值最高（58.39%、93.56%），耗时约为 CNN 的 6.8 倍。同一 CBAM 配置加入普通 CNN 后，平均宏 F1 下降了 10.49 个百分点。[完整结果与结论](docs/FOR_HUMAN_further_exp.md#8-实验结论)

## 运行实验

训练使用 Python 3.8、PaddlePaddle 2.6.2、Pillow、NumPy 和 Matplotlib。CPU 环境路径为 `.venv/`，GPU 环境路径为 `.venv-gpu/`，需配置 GPU 版 PaddlePaddle 和对应运行库。图像位于 `data_set/fish_image/fish_*/`。[数据来源说明](data_set/fish_image/SOURCE.txt)

- **基础实验**。从项目根目录运行 `.venv/bin/python -u src/main.py --stage cnn`。程序按类别逐图划分数据，训练 CNN，并将评估结果、样例图和推理模型保存到 `output/`。
- **扩展实验**。依次运行 `preliminary`（在验证集选参）、`experiment`（四组正式对照）和 `supplement`（只增加 CNN + CBAM）。GPU 批大小 128 的完整命令及环境设置见[实验文档](docs/FOR_HUMAN_further_exp.md#5-gpu-全量实验批大小-128)，补充组命令见[第 7 节](docs/FOR_HUMAN_further_exp.md#7-补充实验cbam-在普通-cnn-上的贡献)。

## 项目入口

| 文件 | 职责 |
|---|---|
| [src/main.py](src/main.py) | 组织训练、选参、独立测试和结果保存 |
| [src/data.py](src/data.py) | 生成数据清单并预处理图像 |
| [src/model.py](src/model.py) | 定义网络和训练、评估、计时接口 |
| [src/img.py](src/img.py) | 读取已有实验结果并生成图表 |

实验设计和复现步骤见[实验文档](docs/FOR_HUMAN_further_exp.md)。接手分析或修改代码时，从 [AI 交接文档](docs/FOR_AI_further_exp.md) 开始。
