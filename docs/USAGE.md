# Usage and defaults

## Installation

Use Python 3.8--3.11, an NVIDIA GPU, CUDA Toolkit 11.8 and a compatible C++ compiler.
Python 3.10 is recommended for a fresh environment. Linux requires a C++17 compiler;
Windows requires an x64 MSVC developer shell. The CUDA version displayed by
`nvidia-smi` is driver capability, not the installed toolkit version. Check `nvcc --version`.

```bash
conda create -n adasplat python=3.10 -y
conda activate adasplat
python tools/setup.py
```

The setup command installs pinned PyTorch/dependencies, installs this checkout,
and compiles the included CUDA extension. It downloads packages but no models or
sample images. Use `--skip-torch` only when a compatible PyTorch is already installed.
Use `--no-build` for dependency-only installation. Training requires the build.
The build checks that the toolkit matches PyTorch's CUDA version. Different
Python/CUDA/GPU environments need their own locally built extension.

## Minimal commands

```bash
# Train with defaults; output is a new timestamped runs/ directory.
python train.py --input /path/to/image.png

# Train, encode the final checkpoint and render/check the decoded stream.
python train.py --input /path/to/image.png --encode

# Short, full-resolution preview. This is not the paper-quality recipe.
python train.py --input /path/to/image.png --preset quick --encode

# Inspect resolved settings without CUDA, training or creating output.
python train.py --input /path/to/image.png --dry-run

# Folder: all visible GPUs, one active image per GPU, refill as workers finish.
python train_batch.py --input-dir /path/to/images --encode
```

PNG/JPEG/BMP/TIFF/WebP images with arbitrary filenames are supported. Native
orientation and resolution are retained, with RGB conversion. Both dimensions
must exceed 160 for the fixed five-level MS-SSIM definition. No image is bundled.
`--recursive` enables nested folders. `--gpus 0,2` selects specific GPUs;
`--jobs-per-gpu 2` explicitly enables concurrency on a GPU. Default is one.
`auto` preserves an integer CUDA_VISIBLE_DEVICES mask. UUID/MIG masks require
explicit environment-specific mapping and are not auto-remapped.

## Presets and units

| Setting | standard (default) | quick | paper |
|---|---|---|---|
| Main steps | 50,000 | 2,000 | 50,000 |
| Maximum capacity | 12.43395 k/MP, capped at 200k | 1.243395 k/MP, capped at 200k | 200,000 |
| Initial capacity | 15% of resolved maximum | 15% of resolved maximum | 30,000 |
| Growth window | first 80% of main schedule | first 80% | first 80% |
| Growth opportunities | 16 | 16 | 16 |
| Early stopping | disabled | disabled | disabled |
| Capacity snapshots | off | off | 40k, 50k, 70k, 100k, 150k, 200k targets |
| Automatic encoding | opt-in | opt-in | opt-in |

On a 4928x3264 input, standard resolves to an initial 30k and maximum 200k.
`quick` lowers capacity and updates; it is a usability preview, with no quality or
wall-time guarantee. Initialization and growth refinement add work beyond main steps.
Actual final N is an upper-bound outcome, not guaranteed to reach the maximum.

```bash
python train.py --input image.png --max-density 10 --initial-density 1.5 --steps 20000
python train.py --input image.png --num 100000 --initial-ratio 0.15
python train.py --input image.png --preset paper --encode-milestones
```

`--max-density` and `--initial-density` use **k primitives per megapixel**.
Legacy `--density` uses primitives per pixel; these are different units.
`--max-density` applies the documented 200k ceiling; explicit legacy `--density`
continues to reject requests above it. `--num` is a resolution-independent count.
Initial-density overrides the ratio and must fit within the resolved cap.
Use either initial-density or initial-ratio. Growth disabled via `--no-growth`
starts at the full cap. Advanced training/optimizer/loss settings remain available
through JSON (`--config`) and the existing CLI. Precedence is explicit config
instead of preset, then CLI overrides, then the bounded `--smoke` test profile.

`--preset paper` preserves the previously published default recipe; it is distinct
from the resolution-aware convenience preset. Customizing a paper cap also
requires compatible `--save-counts` (use an empty string to disable snapshots).

## Encoding and outputs

`--encode` runs **after complete training**, on `final.pt`, in a separate process
on the same GPU. It does not encode incomplete `--stop-after` sessions.
`--encode-milestones` additionally encodes each available snapshot. Target capacity
labels can refer to equal actual N; snapshot files are not necessarily unique states.
`--no-encode` explicitly disables encoding. `--codec-max-loss-db 0.02` controls
the original RGB8 PSNR guard. If no tested setting satisfies it, encoding fails
and leaves training outputs intact. This is reported as an encoding failure;
the batch scheduler does not silently repeat the training. Manual re-encoding is
available for recovery using `codec.py encode` and a new output path.

Each image run includes `config.json`, `resolved_plan.json`, `console.log`,
`events.jsonl`, `progress.json`, `result.json`, `summary.json`, `final.pt`,
`final.png`, `resume.pt`, and requested `milestones/`. Encoding adds:

```text
encoding.json                    encoding status and elapsed seconds
encoded/final.adagc               complete independently decodable stream
encoded/final.png                 reconstruction from that stream
encoded/final.adagc.json          actual bytes, bpp, RGB8 PSNR and guard attempts
encoded/final.console.log         codec output
```

Training summary metrics are float-domain metrics. Codec reports use full-image
RGB8 metrics. Do not mix the two or count checkpoint bytes as encoded stream bytes.
Stream bpp is `8 * actual_file_bytes / (height * width)` and includes the container.
The codec never retrains or prunes primitives. Continuous L2 normalization,
RGB-only coefficients and single pixel samples are required by this codec.

```bash
python render.py --input runs/my_run/encoded/final.adagc --output decoded.png
python codec.py decode --input runs/my_run/encoded/final.adagc --output fields.npz
```

Parameter decode is CPU-only and needs neither source image nor checkpoint.
Image rasterization requires CUDA. Resume with the original resolved config,
original image, and a new output directory:

```bash
python train.py --config runs/old/config.json --resume runs/old/resume.pt --output runs/resumed
```

Old renderer/implementation identities remain enforced; this release does not
promise cross-version training resume. Use inference checkpoints for rendering.

## Tests

```bash
python -m pip install -r requirements-dev.txt -r requirements-codec.txt
python -m pytest -q tests
```

Tests cover CPU configuration, scheduling, automatic codec dispatch, failure
handling, complete-stream round trips and fresh-process decoding. They do not
substitute for a full GPU training/encoding validation on a newly provisioned server.
