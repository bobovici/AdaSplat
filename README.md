# AdaSplat

Content-adaptive image representation with learnable generalized ℓp primitives, adaptive population growth, and post-training compression.

[中文说明](README.zh-CN.md) · [Installation](docs/INSTALL.md) · [Training](docs/TRAINING.md) · [Codec](docs/CODEC.md) · [Method](docs/ALGORITHM.md)

## Installation

Requirements: Python 3.8, PyTorch 2.4.1, a CUDA GPU, a matching CUDA toolkit and C++ compiler.

```bash
conda create -n adasplat python=3.8.20 -y
conda activate adasplat
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt -r requirements-codec.txt -r requirements-build.txt
python -m pip install -e . --no-deps --no-build-isolation
python tools/build_renderer.py --jobs 2
```

The CUDA extension is built locally. See [installation](docs/INSTALL.md) for Linux and Windows toolchain details.

## Training

Single image:

```bash
python train.py \
  --config configs/default.json \
  --input /path/to/image.png \
  --output runs/single \
  --gpu 0
```

Multiple images:

```bash
python train_batch.py \
  --config configs/default.json \
  --input-dir /path/to/images --recursive \
  --output runs/batch \
  --gpus 0,1 --jobs-per-gpu 2
```

The default configuration sets a 200,000-primitive upper bound and 50,000 main steps. CLI overrides include `--num`, `--density`, `--steps`, `--initial-ratio` and `--save-counts`. Use `configs/density.json` for a resolution-dependent capacity budget. `--dry-run` prints the resolved plan; `--smoke` runs a small integration profile.

Each image is optimized independently. Outputs include reconstructions, checkpoints, resume state, progress and complete console/event logs. See [training options and outputs](docs/TRAINING.md).

## Encoding and decoding

Encode a trained checkpoint:

```bash
python codec.py encode \
  --checkpoint runs/single/final.pt \
  --target /path/to/image.png \
  --output runs/single/model.adagc
```

Decode physical parameters on CPU:

```bash
python codec.py decode \
  --input runs/single/model.adagc \
  --output runs/single/fields.npz
```

Render directly from a complete stream:

```bash
python render.py \
  --input runs/single/model.adagc \
  --output runs/single/decoded.png
```

Encoding does not retrain or prune the representation. The complete stream contains the coded parameters and decoder metadata; decoding requires neither the source image nor a training checkpoint. GPU rasterization requires the compiled extension. See [codec settings and format](docs/CODEC.md).

## Evaluation

```bash
python tools/evaluate.py \
  --reference /path/to/image.png \
  --reconstruction runs/single/decoded.png \
  --device cuda
```

This computes full-resolution RGB8 PSNR, SSIM and MS-SSIM. Actual stream rate is `8 × file_bytes / (height × width)`.

## Project structure

```text
configs/                 training configurations
src/adasplat/core/       primitive model, optimizer, growth and CUDA sources
src/adasplat/training/   training, scheduling, logging and checkpoints
src/adasplat/codec/      encoder, decoder and entropy-coding dependencies
tools/                   build and evaluation utilities
examples/                command-line launch examples
tests/                   automated source-level tests
docs/                    usage and method documentation
```

## Tests

```bash
python -m pip install -r requirements-dev.txt -r requirements-codec.txt
python -m pytest -q tests
```

## License

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
