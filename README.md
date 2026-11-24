# AdaSplat

Content-adaptive image representation with learnable generalized lp primitives,
validated population growth, and optional complete-stream compression.

## Quick start

Requires an NVIDIA GPU, CUDA Toolkit 11.8, a compatible C++ compiler, and Python 3.8--3.11.

```bash
conda create -n adasplat python=3.10 -y
conda activate adasplat
python tools/setup.py
python train.py --input /path/to/image.png --encode
```

For a shorter preview, add `--preset quick`. Default is a resolution-aware
50k-step recipe; quick is a 2k-step preview with lower capacity, not a paper result.
All detailed parameters have defaults and remain configurable.

```bash
# Directory: auto-detect GPUs, one image per GPU, automatically refill free slots.
python train_batch.py --input-dir /path/to/images --encode
# Inspect the plan without training.
python train.py --input /path/to/image.png --dry-run
# Optional overrides, in k primitives per megapixel.
python train.py --input /path/to/image.png --initial-density 1.5 --max-density 10
# Original 200k / 50k default recipe, with saved capacity states.
python train.py --input /path/to/image.png --preset paper --encode-milestones
```

`--encode` automatically encodes the final checkpoint, decodes it, checks RGB8
PSNR loss (default <=0.02 dB), and reports actual file bytes and bpp. Omit it or use
`--no-encode` for training only. Results are saved to a new timestamped `runs/`
directory. No image naming convention is required. No weights or images are included.

[Usage, defaults and advanced settings](docs/USAGE.md) ·
[中文说明](README.zh-CN.md) · [Anonymous distribution](docs/ANONYMOUS.md)

## License

See LICENSE and THIRD_PARTY_NOTICES.md. Original third-party notices are retained.
