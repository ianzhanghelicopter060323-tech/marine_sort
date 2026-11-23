# 鱼类图像分类扩展实验设计

## 1 实验分组与研究问题

| 组别 | 模型 | 研究问题 |
|---|---|---|
| 对照组 | 现有 CNN | 在按鱼类跟踪轨迹划分的数据上，现有模型的分类效果和训练耗时是多少？ |
| 实验 1 | ResNet18 | 与现有 CNN 相比，更换为残差网络后，分类效果和训练耗时如何变化？ |
| 实验 2 | ResNet18 + SE | 与实验 1 相比，按图像调整特征通道的权重，能否改善识别？哪些鱼类受益，耗时增加多少？ |
| 实验 3 | ResNet18 + CBAM | 与实验 1 相比，同时调整通道和空间位置的权重，能否减少鱼类之间的误判？耗时如何变化？ |

四组采用同一评价标准。**宏平均 F1 为主指标**，即先计算每种鱼的 F1，再对 23 类等权平均。总体准确率衡量所有图像中认对的比例，每类召回率说明各类鱼有多少被认出。效率方面，记录预热后完整批次的每步耗时中位数、每轮耗时和训练总耗时。

## 2 实验步骤

1. **固定数据与训练条件。** 按跟踪轨迹划分训练集、验证集和独立评估集，使同一轨迹只进入一个集合，并检查各类样本数。四组共用数据清单、图像预处理、优化器、学习率、批大小和设备，每次训练 5 轮。
2. **预实验。** 用同一随机种子训练 11 个配置。CNN 和 ResNet18 各 1 个；SE 比较压缩比 8、16、32，共 3 个；CBAM 将这三个压缩比分别与 3×3、7×7 空间卷积核组合，共 6 个。按 **实验评价标准** 分别选出 SE 和 CBAM 在候选范围内的最佳配置。
3. **实验。** 固定选出的参数，对四组模型各补跑两个相同的新种子。复用预实验中对应的首次训练，合计训练 19 次。用独立评估集比较各组在三个种子下的指标均值和标准差。

## 3 结果分析

依次比较实验 1 与对照组、实验 2 与实验 1、实验 3 与实验 1。每次报告宏平均 F1 和总体准确率的变化、召回率变化明显的鱼类，以及训练耗时差异。结合混淆矩阵和同一批错例，分析哪些误判得到改善，哪些仍然存在。

## 4 启动全流程

确认项目的 `.venv/bin/python` 可用，且鱼类图像位于 `data_set/fish_image/`。在项目根目录执行以下命令。

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
.venv/bin/python -u src/main.py --stage preliminary && \
  .venv/bin/python -u src/main.py --stage experiment
```

预实验成功后，`&&` 会自动启动正式实验。正式实验读取已选出的 SE 和 CBAM 配置，并复用种子 42 的训练权重。

默认统一训练批大小为 256。若需调整，在预实验命令末尾加上 `--batch-size N`，正式实验会读取该设置。若已有不同批大小的结果，给两条命令指定同一个新目录，例如都加上 `--output-dir output/further_exp_batch64`。分开运行两个阶段时，使用相同的线程设置和输出目录。

默认结果目录为 `output/further_exp/`。其中 `preliminary.json` 保存预实验选参结果，`experiment.json` 保存正式实验的评估和配对比较。每次训练的权重和计时记录保存在各自的子目录中。

## 5 GPU 全量实验（批大小 128）

在项目根目录执行下面整段命令。预实验以批大小 128 训练并选参，成功后自动进入正式实验。正式实验读取预实验保存的批大小，四组模型均使用 128。

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
gpu_lib_dirs=$(find "$PWD/.venv-gpu/lib/python3.8/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="$PWD/.venv-gpu/lib/cuda-links:$gpu_lib_dirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv-gpu/bin/python -u src/main.py --stage preliminary --device gpu:0 --batch-size 128 && \
  .venv-gpu/bin/python -u src/main.py --stage experiment --device gpu:0
```

结果写入 `output/further_exp_gpu0/`。每个 GPU 配置的训练和独立测试各自在子进程中运行，退出时释放该进程的显存。需要停止时按 `Ctrl+C`。以相同设置重新执行命令时会复用已保存的训练结果，中断的配置从头开始。

如需单独复测速度，在设置好上面的环境变量后运行下面的命令。程序训练 4 个批次并报告预热一步后的中位耗时和显存峰值。

```bash
.venv-gpu/bin/python -u src/main.py --stage benchmark --device gpu:0 --batch-size 128
```

终端会显示预实验配置编号、正式实验组别和随机种子。预实验写完 `preliminary.json` 后打印 `[SUCCEED] 预实验完成`，四组独立测试均完成并写出 `experiment.json` 后打印 `[SUCCEED] 正式实验全部完成`。

## 6 生成结果图表

在项目根目录执行下列命令读取已保存的 GPU 实验结果。图表保存至 `output/further_exp_gpu0/figures/`。

```bash
.venv/bin/python src/img.py
```

图表包括验证集选参、独立测试集指标、训练耗时和四组混淆矩阵。独立测试指标图中的浅色点代表各随机种子，菱形和误差线表示三次运行的均值和样本标准差。混淆矩阵合并三个种子的预测计数，再按真实类别归一化。
