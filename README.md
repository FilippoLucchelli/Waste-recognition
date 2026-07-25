# Waste Recognition

Training, evaluation and prediction for semantic segmentation of multispectral
waste datasets.

The project supports:

- UNet, UNet++, DeepLabV3 and DeepLabV3+ through
  `segmentation-models-pytorch`;
- ACNet for RGB + NIR inputs;
- MSNet for RGB plus one or more auxiliary bands;
- deterministic matching of per-band NumPy files;
- joint image/mask augmentation;
- dataset-level IoU and Dice metrics;
- resumable, versioned checkpoints;
- CPU and CUDA execution.

## Installation

Python 3.10 or newer is required.

```console
python -m venv .venv
.venv\Scripts\python -m pip install -e .
```

On Linux or macOS, activate the environment using the corresponding `bin`
directory. If a platform-specific PyTorch build is required, install PyTorch
first using the official instructions and then install this project with
`--no-deps`.

ImageNet encoder weights are not downloaded automatically. Enable them
explicitly with `--encoder_pretrained`.

## Dataset

The YAML file describes the dataset root and the split folders:

```yaml
root_dir: path/to/dataset
train_dir: train
val_dir: validation
test_dir: test
```

Relative `root_dir` values are resolved from the YAML location. Split paths are
resolved from `root_dir`.

Each split contains one folder per channel and, when ground truth is available,
a `masks` folder:

```text
dataset/
├── train/
│   ├── red/
│   │   └── img_000_red.npy
│   ├── green/
│   │   └── img_000_green.npy
│   ├── blue/
│   │   └── img_000_blue.npy
│   ├── nir/
│   │   └── img_000_nir.npy
│   └── masks/
│       └── img_000_masks.npy
├── validation/
└── test/
```

The channel suffix is optional. For example, `img_000.npy` is also accepted.
Every folder must contain exactly the same sample IDs; training stops with an
explicit validation error if a band or mask is missing.

## Training

The historical command remains valid:

```console
python train.py \
  --data_dir data.yaml \
  --model msnet \
  --channels nir \
  --save_folder experiment_01
```

After installation, the equivalent command is:

```console
waste-train --data_dir data.yaml --model msnet --channels nir --save_folder experiment_01
```

Useful low-resource options:

```console
waste-train \
  --data_dir data.yaml \
  --model unet \
  --channels nir \
  --save_folder cpu_test \
  --device cpu \
  --size 256 \
  --batch_size 1 \
  --num_workers 0 \
  --epochs 2 \
  --loss crossentropy \
  --scheduler none
```

Resume a run:

```console
waste-train \
  --data_dir data.yaml \
  --save_folder continued_run \
  --model_dir experiment_01 \
  --resume
```

`--pretrained` remains an alias for `--resume`. Use
`--encoder_pretrained` specifically for ImageNet initialization.

## Evaluation and prediction

Evaluate a test split containing masks:

```console
python test.py \
  --data_dir data.yaml \
  --model_dir experiment_01 \
  --ground_truth \
  --batch_size 1
```

Generate NPY prediction masks when ground truth is absent:

```console
waste-evaluate \
  --data_dir data.yaml \
  --model_dir experiment_01 \
  --predictions_folder predictions
```

Visdom is no longer required. The historical `--print_images` option is
accepted but only emits a migration warning.

## Run outputs

```text
results/experiment_01/
├── config.json
├── training_config.json
├── train_manifest.csv
├── validation_manifest.csv
├── metrics.csv
├── best.pt
├── last.pt
└── model.pth
```

- `config.json` contains typed model and normalization metadata.
- `training_config.json` contains the complete training CLI configuration.
- the manifest files record the exact band-to-sample mapping.
- `best.pt` is the best resumable checkpoint.
- `last.pt` is the latest resumable checkpoint.
- `model.pth` is a bare state dictionary retained for compatibility.

Historical bare `model.pth` checkpoints remain loadable.

## Lightweight validation

The repository tests do not train a neural network:

```console
python -m unittest discover -s tests -v
```

Manifest, configuration and statistics tests run without PyTorch. Small tensor
tests are automatically skipped when PyTorch is not installed.
