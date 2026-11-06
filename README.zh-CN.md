# AdaSplat

可学习广义 ℓp 基元、动态容量增长和后训练编码的公开源码实现。

本仓库只包含源码、必要配置、测试和使用文档，**不包含图像、数据集、训练权重、编码码流、实验结果或编译产物**。

## 安装

安装命令见 [README](README.md)；环境与 CUDA 编译说明见 [安装文档](docs/INSTALL.md)。使用前运行：

```bash
python tools/build_renderer.py --jobs 2
```

## 训练

```bash
python train_batch.py \
  --config configs/default.json \
  --input-dir /path/to/images --recursive \
  --output runs/batch \
  --gpus 0,1 --jobs-per-gpu 2
```

请自行提供输入图像。`--gpus 0,1 --jobs-per-gpu 2` 表示每张卡运行两个独立图像任务。默认配置采用 200k 基元上限、50k 主步骤；可通过命令行覆盖预算和调度。完整日志、模型状态和重建图在运行时写入指定输出目录。

## 编码与解码

```bash
python codec.py encode \
  --checkpoint runs/single/final.pt \
  --target /path/to/image.png \
  --output runs/single/model.adagc

python codec.py decode --input runs/single/model.adagc --output runs/single/fields.npz
python render.py --input runs/single/model.adagc --output runs/single/decoded.png
```

编码不重新训练、不删点。完整码流可以独立解码，不依赖原图或训练 checkpoint；参数解码可在 CPU 上执行，渲染图像需要 CUDA。实际 bpp 按完整码流文件大小计算。

进一步说明：[训练](docs/TRAINING.md)、[编码](docs/CODEC.md)、[算法](docs/ALGORITHM.md)。
