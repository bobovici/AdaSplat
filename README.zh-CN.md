# AdaSplat

[English](README.md)

## 安装

需要 Python 3.8、CUDA GPU、CUDA 11.8 toolkit 和兼容的 C++ 编译器。Windows 请在 MSVC 开发者终端中编译。

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

检查环境和编译条件：

```bash
python tools/check_env.py
python tools/build_renderer.py --check
```

## 训练

自行准备输入图片，两边尺寸均须大于 160 像素。每次运行使用新的输出目录。

单张图片：

```bash
python -u train.py \
  --config configs/default.json \
  --input /path/to/image.png \
  --output runs/single \
  --gpu 0
```

批量图片：

```bash
python -u train_batch.py \
  --config configs/default.json \
  --input-dir /path/to/images --recursive \
  --output runs/batch \
  --gpus 0,1 --jobs-per-gpu 2
```

只有一张 GPU 时使用 `--gpus 0`；`--jobs-per-gpu` 控制每张卡同时运行的图片任务数。

默认配置为 `--num 200000 --steps 50000 --initial-ratio 0.15`，可以在命令行覆盖。降低 `--num` 时，也需调整 `--save-counts`，避免快照目标超过预算。按图像分辨率设置密度可使用 `configs/density.json`。

`--dry-run` 只查看运行计划；`--smoke` 运行短流程裁剪测试。进度见输出目录的 `progress.json`，完整控制台日志见 `console.log`。运行结果还包括重建 PNG、模型权重、数量快照及恢复文件 `resume.pt`。

恢复训练时保持原输入和配置，传入 `--resume /path/to/resume.pt`，并指定一个新的 `--output` 目录。

## 编码

完成单图训练后执行：

```bash
python codec.py encode \
  --checkpoint runs/single/final.pt \
  --target /path/to/image.png \
  --output runs/single/model.adagc \
  --decoded-output runs/single/decoded.png
```

`--target` 使用训练时的同一张图片。此命令需要编译好的 CUDA 渲染器，输出码流、解码图以及包含实际文件大小和 bpp 的 JSON 报告。

## 解码和渲染

CPU 参数解码：

```bash
python codec.py decode \
  --input runs/single/model.adagc \
  --output runs/single/fields.npz
```

从码流渲染图片：

```bash
python render.py \
  --input runs/single/model.adagc \
  --output runs/single/replayed.png
```

渲染需要 CUDA。输出文件名必须未被使用。如果回放的 checkpoint 来自不同的渲染器构建，需显式增加 `--allow-renderer-change`。

## 评价

```bash
python tools/evaluate.py \
  --reference /path/to/image.png \
  --reconstruction runs/single/decoded.png \
  --device cuda
```

## 更多参数

```bash
python train.py --help
python train_batch.py --help
python codec.py --help
python codec.py encode --help
python render.py --help
```

## 测试

```bash
python -m pip install -r requirements-dev.txt -r requirements-codec.txt
python -m pytest -q tests
```

## 许可

见 [LICENSE](LICENSE) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。
