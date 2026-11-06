# Installation

## Environment

Use Python 3.8, PyTorch 2.4.1 and a matching CUDA toolkit. The commands below select CUDA 11.8:

```bash
conda create -n adasplat python=3.8.20 -y
conda activate adasplat
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt -r requirements-codec.txt -r requirements-build.txt
python -m pip install -e . --no-deps --no-build-isolation
python tools/check_env.py
python tools/build_renderer.py --jobs 2
```

Alternatively, create the environment from `environment.yml`, then perform the editable install and renderer build. Run these commands from the repository root.

## CUDA extension

Linux requires nvcc and a compatible C++ compiler. Windows requires an MSVC developer shell with CUDA available. The builder creates `native/` and saves a locally compiled `csrc.so` or `csrc.pyd`, together with `renderer.json` describing the build.

Use `python tools/build_renderer.py --check` to inspect the toolchain first. `--jobs` limits compilation concurrency; `--arch` selects explicit CUDA architectures. The driver-supported CUDA version displayed by `nvidia-smi` is not the installed nvcc toolkit version.

Configuration renderer hashes containing all zeroes are resolved against the local build manifest. Explicit nonzero hashes remain strict. Rebuild the extension when changing its source, Python/PyTorch ABI or CUDA toolchain.

## Source installation

Use this source checkout or an editable install, keeping the repository layout intact. The engine locates sources, configs and build outputs relative to the checkout. A relocated standalone wheel is not the supported deployment form.

CLI help and `--dry-run` do not load CUDA. CPU parameter decoding requires the codec dependencies but no compiled renderer. Training and image rasterization require CUDA.
