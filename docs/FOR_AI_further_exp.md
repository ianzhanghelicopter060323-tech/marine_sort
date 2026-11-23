# Fish image classification experiment handoff

The preliminary search, the four-group main experiment, and the supplementary CNN + CBAM experiment are complete. The five tested groups share one track-based dataset split and three seeds. The plain CNN gave the best balance of test performance and training time at the observed five-epoch budget. ResNet18 + CBAM gave the highest mean test macro F1 and accuracy. The same CBAM hyperparameters reduced both metrics when applied to the plain CNN.

## Where to start

1. Read [the experiment protocol and results below](#experiment-protocol) for the comparison groups, completed work, and interpretation.
2. For numerical claims, start with [the saved GPU results](#saved-artifacts-and-json-layout). All five groups can be compared offline. Use the prediction TSV files if you need to inspect individual mistakes or verify a confusion matrix.
3. For code changes, use [the source map](#source-map). For reruns, use [the reproduction commands](#reproduction). The [repository README](../README.md) introduces the project, and the [Chinese experiment guide](FOR_HUMAN_further_exp.md) records the human-facing design and conclusions.

## Experiment protocol

This AI foundations course project classifies 27,370 images into 23 fish classes. The project contains a standalone CNN exercise and a separate extension that compares architectures. The standalone entry point, `--stage cnn`, uses an image-level train/eval split. The CNN control in the extension uses the same **track-based** train/validation/test split as the other four extension groups. Use the extension's `cnn` results when comparing architectures.

### Dataset and preprocessing

- The dataset is under `data_set/fish_image/fish_*/`. Class labels are assigned from sorted class directories and range from 0 to 22. A filename such as `fish_003421605277_21905.png` identifies track `fish_003421605277` and frame `21905`. The original source is recorded alongside the data. [Dataset source](../data_set/fish_image/SOURCE.txt)
- `prepare_track_data()` groups frames by track and assigns whole tracks to one split, targeting approximately 80% training, 10% validation, and 10% test images in each class. All 23 classes are represented in each split. The saved manifests contain **21,892 training**, **2,745 validation**, and **2,733 test** images. A manifest row contains a path relative to `data_set/`, a tab, and the integer label.
- `FishDataset` resizes RGB images to 47×47 and returns a normalized float32 CHW tensor and an integer label. Training uses a horizontal flip with probability 0.5. Validation and test use deterministic preprocessing. Channel means are `[0.485, 0.456, 0.406]` and standard deviations are `[0.229, 0.224, 0.225]`. [Data implementation](../src/data.py)

### Models and comparisons

| Result key | Model wrapper | Architecture | Comparison |
|---|---|---|---|
| `cnn` | `Model_CNN` | `MyCNN` with three convolution, ReLU, and max-pooling blocks | Extension control |
| `resnet18` | `Model_ResNet18` | ResNet18 adapted to 47×47 input with a 3×3 stride-1 stem and four stages of two blocks | Compare against `cnn` |
| `se` | `Model_ResNet18_SE` | ResNet18 with Squeeze-and-Excitation on each residual branch | Compare against `resnet18` |
| `cbam` | `Model_ResNet18_CBAM` | ResNet18 with a Convolutional Block Attention Module on each residual branch | Compare against `resnet18`; also compare the observed performance and time against `cnn` |
| `cnn_cbam` | `Model_CNN_CBAM` | `CNNWithCBAM` adds CBAM after ReLU and before max pooling in all three convolution blocks | Compare against `cnn` and compare that paired change with `cbam` versus `resnet18` |

SE uses a global average-pooled descriptor to produce channel weights. CBAM first produces channel weights from global average and max pooling with a shared MLP. It then applies spatial weights derived from channel-wise mean and max maps and a spatial convolution. ResNet18 applies attention to the residual branch before adding the shortcut. In CNN + CBAM, the weighted activations feed directly into max pooling. Both CNN variants return softmax probabilities, and their cross-entropy uses `use_softmax=False`. The ResNet variants return logits and use `use_softmax=True`. [Model implementation](../src/model.py)

### What ran and what was reused

All extension runs used random initialization, Adam at learning rate `0.0005`, five epochs, and the same manifests. The saved GPU run used `gpu:0`, training batch size 128, `OMP_NUM_THREADS=4`, and `OPENBLAS_NUM_THREADS=1`. Detailed evaluation predicts in batches of 64.

1. **Preliminary search (`preliminary`).** Train 11 configurations with seed 42. Train one plain CNN and one plain ResNet18, test SE reduction ratios 8, 16, and 32, and test all six CBAM combinations of those ratios with 3×3 and 7×7 spatial kernels. Select SE and CBAM separately by highest validation macro F1, breaking ties by higher validation accuracy and then lower median full-batch step time. SE selected ratio 32 (64.47% validation macro F1). CBAM selected ratio 16 and kernel 3 (67.03% validation macro F1). [Selection records](../output/further_exp_gpu0/preliminary.json)
2. **Main experiment (`experiment`).** Keep the four selected configurations. Reuse their seed-42 weights and add seeds 43 and 44, for eight new training runs. Reload each saved model and evaluate it on the held-out test split. The main results contain 12 test records for four models and three seeds. [Main experiment records](../output/further_exp_gpu0/experiment.json)
3. **Supplementary experiment (`supplement`).** Train CNN + CBAM with ratio 16 and kernel 3 for seeds 42, 43, and 44. Reuse the saved main results for CNN, ResNet18, and ResNet18 + CBAM. This tests transfer of the selected CBAM configuration to the CNN. The supplement adds three training runs and three test records. The complete extension therefore contains 22 training runs and 15 final test records. [Supplementary experiment records](../output/further_exp_gpu0/supplement.json)

## Results and interpretation

Each mean is across seeds 42, 43, and 44. The uncertainty column for macro F1 is a **sample standard deviation in percentage points**. Per-step values are means of the three runs' median full-batch times.

| Model | Test macro F1 | Macro F1 SD | Test accuracy | Train time for five epochs | Full-step time |
|---|---:|---:|---:|---:|---:|
| CNN | 55.96% | 2.70 pp | 93.08% | 80.8 s | 94.17 ms |
| ResNet18 | 49.36% | 3.40 pp | 90.22% | 452.1 s | 528.99 ms |
| ResNet18 + SE | 53.36% | 0.64 pp | 92.19% | 501.1 s | 577.85 ms |
| ResNet18 + CBAM | 58.39% | 7.27 pp | 93.56% | 547.5 s | 639.92 ms |
| CNN + CBAM | 45.47% | 3.14 pp | 87.22% | 94.4 s | 110.25 ms |

- **CNN versus ResNet18.** At this budget, CNN led in mean test macro F1 by 6.60 pp and used about one fifth of the training time. CNN also led ResNet18 + SE in mean test macro F1, mean accuracy, and speed. SE improved over plain ResNet18 by 4.00 pp in mean macro F1 and had the smallest observed cross-seed macro F1 variation of the five groups.
- **ResNet18 + CBAM versus CNN.** ResNet18 + CBAM led on average by 2.43 pp in macro F1 and 0.48 pp in accuracy, with about 6.8× the training time. It led on both test metrics at seeds 42 and 44 and trailed on both at seed 43. Its highest observed accuracy was 94.29%, compared with CNN's 94.07%. A repeatable gain after further validation-set tuning remains an open question.
- **CBAM within each backbone.** ResNet18 gained 9.03 pp in mean macro F1 with CBAM. Plain CNN lost 10.49 pp in mean macro F1 and 5.86 pp in accuracy when CBAM was added. The macro F1 change had the same sign for all three seeds within each backbone. CNN + CBAM also took about 17% longer to train than CNN. Across the 23 classes, CNN + CBAM lowered mean recall in 16 classes, raised it in 4, and left 3 tied.
- **Scope of the evidence.** The comparison uses one split, three seeds, and a five-epoch budget. The test accuracy maxima are observed values under those conditions. The two backbones also use different insertion points, so the experiment estimates the effect of each implemented CBAM variant relative to its own backbone. The independent effects of the shortcut and insertion position remain unmeasured. The CNN + CBAM configuration was transferred from ResNet18. Its performance after a CNN-specific search remains unknown.

## Saved artifacts and JSON layout

Use `output/further_exp_gpu0/` for the results described here. Relative paths in the table refer to that directory.

| Artifact | What to read |
|---|---|
| [preliminary.json](../output/further_exp_gpu0/preliminary.json) | `settings`, `selected`, and `candidates`. The candidate metrics are from validation. Each selected entry gives `name`, `ratio`, and `kernel`. |
| [experiment.json](../output/further_exp_gpu0/experiment.json) | `runs` contains per-seed test records. `groups` contains summaries keyed by model. `paired_differences` contains `resnet18_vs_cnn`, `se_vs_resnet18`, and `cbam_vs_resnet18`. |
| [supplement.json](../output/further_exp_gpu0/supplement.json) | `runs` and `group` hold CNN + CBAM test records and summary. `paired` holds same-seed comparisons. `cnn_cbam_minus_cnn` and `gain_summary` hold their summaries. |
| `splits/` | `train.txt`, `val.txt`, `test.txt`, and `class_counts.json`. Match the integer labels to class names using the latter file. |
| `cnn_seed42/`, `se_r32_seed42/`, `cbam_r16_k3_seed42/`, `cnn_cbam_r16_k3_seed42/`, and other run directories | `model.pdparams` holds model weights. `training.json` holds configuration, validation metrics, and timing. A tested run also contains `test.json` and `test_predictions.tsv`. |
| `figures/` | `preliminary_selection.png`, `test_metrics.png`, `training_time.png`, `confusion_matrices.png`, and `supplement_cbam_gain.png`. |

A test record contains `config.model`, `config.seed`, `config.settings`, `metrics`, and `test_seconds`. The `metrics` object has `macro_f1`, `accuracy`, `recall`, `support`, and `confusion_matrix`. Group summaries use `{mean, std, count}` for each metric. The plotting script combines the supplementary results with the four main groups in memory. Its confusion matrix plots sum prediction counts across seeds before normalizing each true-class row.

### Metric and timing conventions

- Macro F1 is the mean of 23 per-class F1 values. The confusion matrix has true labels on rows and predictions on columns. For class `i`, recall is the diagonal count divided by its row sum. F1 is twice the diagonal count divided by the sum of its row and column totals. When a denominator is zero, the code returns zero for that class.
- Classification metrics and plain classification deltas are stored as proportions. Multiply a difference such as `0.026` by 100 for 2.6 percentage points. In `supplement.json`, fields ending in `_pp` already use percentage points, and `cnn_f1_relative_gain_pct` already uses percent. `gain_difference_pp` subtracts the ResNet18 CBAM gain from the CNN CBAM gain. The relative gain summary averages the individual seeds' relative changes.
- Training time is wall-clock time around model fitting. Validation, saving, and testing have separate timers. A step is timed using batch-end callbacks, with GPU synchronization and the interval covering data loading between batches. `median_full_step_ms` excludes the first warm-up step and batches smaller than the configured training batch size.

### Verify one run offline

Run this from the repository root with Python 3. It checks that prediction order matches the test manifest, then recomputes the confusion matrix, accuracy, and macro F1. Change the run directory to check another tested configuration.

```bash
python3 - <<'PY'
import csv
import json
import math
from pathlib import Path

root = Path("output/further_exp_gpu0")
run = root / "cnn_seed42"
with (run / "test_predictions.tsv").open() as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
manifest = (root / "splits" / "test.txt").read_text().splitlines()
assert [f"{row['image']}\t{row['label']}" for row in rows] == manifest

matrix = [[0] * 23 for _ in range(23)]
for row in rows:
    matrix[int(row["label"])][int(row["prediction"])] += 1
accuracy = sum(matrix[i][i] for i in range(23)) / len(rows)
actual = [sum(row) for row in matrix]
predicted = [sum(column) for column in zip(*matrix)]
macro_f1 = sum(
    2 * matrix[i][i] / (actual[i] + predicted[i]) if actual[i] + predicted[i] else 0
    for i in range(23)
) / 23

metrics = json.loads((run / "test.json").read_text())["metrics"]
assert matrix == metrics["confusion_matrix"]
assert math.isclose(accuracy, metrics["accuracy"])
assert math.isclose(macro_f1, metrics["macro_f1"])
print(f"{run.name}  n={len(rows)}  accuracy={accuracy:.2%}  macro F1={macro_f1:.2%}")
PY
```

## Source map

| Source | Entry points and responsibilities |
|---|---|
| [src/main.py](../src/main.py) | `main()` selects stage, device, and output directory. `ModelConfig` builds a model and derives its run directory name. `run_preliminary()`, `run_experiment()`, and `run_supplement()` orchestrate the three stages. `train_configuration()` checks whether a run can be reused. `evaluate_saved_model()` reloads the weights and writes test metrics and predictions. |
| [src/data.py](../src/data.py) | `prepare_data()` serves the standalone exercise. `prepare_track_data()` writes extension manifests. `FishDataset` reads either manifest format and applies preprocessing according to the mode. |
| [src/model.py](../src/model.py) | `MyCNN`, `CNNWithCBAM`, `ResNet18`, `ResidualBlock`, `SE`, and `CBAM` define the networks. `_Classifier` shares the training, evaluation, and weight I/O interfaces. `classification_metrics()` computes the final 23-class metrics. `TrainingTimer` records per-step and per-epoch timing. |
| [src/img.py](../src/img.py) | `main()` reads the saved results and generates PNG plots. It includes CNN + CBAM and the paired-gain plot when `supplement.json` is available. |

## Reproduction

Run commands from the repository root. The saved environment used Python 3.8 and PaddlePaddle 2.6.2. The training CLI also imports Pillow, NumPy, and Matplotlib. Rendering figures only requires NumPy and Matplotlib.

### Render figures from existing artifacts

```bash
.venv/bin/python src/img.py --results-dir output/further_exp_gpu0
```

The default output is the result directory's `figures/` subdirectory. Pass `--output-dir` to render into a separate directory.

### Run the GPU stages

These commands use the repository's GPU virtual environment layout. It needs the CUDA-enabled PaddlePaddle build, an accessible NVIDIA GPU, and the matching runtime libraries.

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
gpu_lib_dirs=$(find "$PWD/.venv-gpu/lib/python3.8/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="$PWD/.venv-gpu/lib/cuda-links:$gpu_lib_dirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

.venv-gpu/bin/python -u src/main.py --stage preliminary --device gpu:0 --batch-size 128 && \
  .venv-gpu/bin/python -u src/main.py --stage experiment --device gpu:0 && \
  .venv-gpu/bin/python -u src/main.py --stage supplement --device gpu:0
```

For a repeat of only the supplement, use the same environment and run the final command once the main artifacts exist. The supplement expects all three reference seeds and the saved ResNet18 + CBAM ratio-16, kernel-3 configuration. The training CLI currently restricts the supplementary CNN + CBAM configuration to those same parameters.

The default GPU output directory for `gpu:0` is `output/further_exp_gpu0/`. For a separate run, pass the same `--output-dir` at each stage. The main and supplementary stages inherit the preliminary batch size and verify device, thread settings, and SHA-256 hashes of the split manifests. Matching completed training directories and weight files can be reused. Test evaluation runs again when its stage is rerun. GPU training and testing each run in a spawned worker to release that worker's CUDA context between jobs. Each completed stage writes its summary and then prints a stage-specific `[SUCCEED]` line.
