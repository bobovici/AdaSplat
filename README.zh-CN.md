# AdaSplat

本包只含源码、配置、说明和测试，不含任何图片、权重或 CUDA 二进制。

## 实现与上游代码归属

本仓库包含 AdaSplat 实现，以及复用和适配的上游组件。AdaSplat 特有的开发包括
广义 lp 基元参数化及其尺度/响应校准、经过验证的动态增点、训练流程集成，以及
完整码流的编码适配。这些实现建立在已有图像基元表示、渲染、优化和编码代码之上，
并非仓库内所有代码都从零编写。

根目录 [LICENSE](LICENSE) 保留了继承自 GaussianImage 的版权与许可声明。
**这一上游署名不代表整个仓库均由该上游作者编写，也不代表 AdaSplat 特有方法由其实现。**
复用的 gsplat/GLM、优化器和上游编码组件分别保留原有声明；具体归属见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和对应源码中的声明。

## 准备环境

需要 NVIDIA GPU、CUDA Toolkit 11.8 和 C++ 编译器；Windows 使用 x64 MSVC 开发终端。

```bash
conda create -n adasplat python=3.10 -y
conda activate adasplat
python tools/setup.py
```

安装脚本自动安装固定版本依赖、安装项目并编译渲染器。`nvidia-smi` 的 CUDA
数字是驱动支持上限，工具链版本请看 `nvcc --version`。不需要下载任何模型。

## 只给图片即可运行

```bash
python train.py --input /path/to/image.png --encode
```

`--encode` 自动执行训练后的编码、解码重建、RGB8 PSNR 损失检查，默认允许损失
不超过 0.02 dB；报告保存实际码流字节数和 bpp。不编码时去掉此开关或使用 `--no-encode`。
输出自动放在带时间戳的新 `runs/` 目录，已有结果不会覆盖。

想先快速看效果：

```bash
python train.py --input /path/to/image.png --preset quick --encode
```

quick 是全分辨率 2k 主步低容量预览，不代表论文实验；不保证具体耗时或画质。
默认 standard 为 50k 主步，最大密度约 12.434 k/MP、最高 200k 基元，初始为上限
的 15%。4928×3264 图片对应初始 30k、最大 200k。增长优化另有额外计算。

```bash
# 文件夹：自动识别可见显卡，每卡同时一张，完成后自动补任务。
python train_batch.py --input-dir /path/to/images --encode
# 仅看配置与预算，不训练。
python train.py --input /path/to/image.png --dry-run
# 修改密度；单位均为 k 基元/百万像素。
python train.py --input /path/to/image.png --initial-density 1.5 --max-density 10
# 原先的固定上限200k/50k配方，并自动编码保存的容量状态。
python train.py --input /path/to/image.png --preset paper --encode-milestones
```

其余优化器、损失和高级参数通过 JSON 调整；详见 [完整说明](docs/USAGE.md)。
任意图片名均可，支持常见格式，原始宽高均须超过160像素，不自动缩放。
训练 float 指标与编码 RGB8 指标分别报告。编码失败保留训练产物，不重新训练。
匿名发布见 [操作说明](docs/ANONYMOUS.md)，第三方许可证署名保留。
