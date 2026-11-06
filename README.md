# AdaSplat

Training, compression and reconstruction code for AdaSplat.

[中文说明](README.zh-CN.md)

## Install

Requirements: Python 3.8, a CUDA GPU, CUDA toolkit 11.8, and a compatible C++ compiler. On Windows, run the build from an MSVC developer shell.

```bash
git clone https://github.com/bobovici/AdaSplat.git
cd AdaSplat
conda create -n adasplat python=3.8.20 -y
conda activate adasplat
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt -r requirements-codec.txt -r requirements-build.txt
python -m pip install -e . --no-deps --no-build-isolation
python tools/build_renderer.py --jobs 2
```

To inspect the environment or check the build prerequisites:

```bash
python tools/check_env.py
python tools/build_renderer.py --check
```

## Train

Provide your own images. Both image dimensions must exceed 160 pixels. Use a new output directory for each run.

Single image:

```bash
python -u train.py \
  --config configs/default.json \
  --input /path/to/image.png \
  --output runs/single \
  --gpu 0
```

Image directory:

```bash
python -u train_batch.py \
  --config configs/default.json \
  --input-dir /path/to/images --recursive \
  --output runs/batch \
  --gpus 0,1 --jobs-per-gpu 2
```

For one GPU, use `--gpus 0`. `--jobs-per-gpu` sets concurrent image tasks on each GPU.

The default config uses `--num 200000`, `--steps 50000` and `--initial-ratio 0.15`. Override these on the command line as needed. If lowering `--num`, also lower `--save-counts` so no requested snapshot exceeds the budget. `configs/density.json` is available for density-based budgets.

Use `--dry-run` to inspect the plan without training, or `--smoke` for a short cropped integration run. Check `progress.json` and `console.log` in the output directory for progress. The output also includes reconstruction PNGs, inference checkpoints, count snapshots and `resume.pt`.

To resume, keep the original input and configuration, pass `--resume /path/to/resume.pt`, and choose a new `--output` directory.

## Encode

After training a single image:

```bash
python codec.py encode \
  --checkpoint runs/single/final.pt \
  --target /path/to/image.png \
  --output runs/single/model.adagc \
  --decoded-output runs/single/decoded.png
```

Use the same input image as `--target`. This command requires the compiled CUDA renderer. It writes the stream, decoded image and a JSON report containing the actual file size and bpp.

## Decode and render

CPU-only parameter decoding:

```bash
python codec.py decode \
  --input runs/single/model.adagc \
  --output runs/single/fields.npz
```

Render an image from the stream:

```bash
python render.py \
  --input runs/single/model.adagc \
  --output runs/single/replayed.png
```

Rendering requires CUDA. Choose new output filenames. To replay a checkpoint produced with a different renderer build, explicitly add `--allow-renderer-change`.

## Evaluate

```bash
python tools/evaluate.py \
  --reference /path/to/image.png \
  --reconstruction runs/single/decoded.png \
  --device cuda
```

## More options

```bash
python train.py --help
python train_batch.py --help
python codec.py --help
python codec.py encode --help
python render.py --help
```

## Tests

```bash
python -m pip install -r requirements-dev.txt -r requirements-codec.txt
python -m pytest -q tests
```

## License

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
