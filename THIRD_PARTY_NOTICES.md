# Third-party notices

- The source project's MIT license and original copyright attribution to Xinjie Zhang are retained verbatim in `LICENSE`. Packaging does not replace the original attribution.
- The vendored gsplat code retains its Apache-2.0 license at `src/adasplat/core/gsplat/LICENSE`.
- GLM and bundled C++ headers retain their upstream notices under `src/adasplat/core/gsplat/gsplat/cuda/csrc/third_party/`.
- MesonGS++ RAHT and Laplace/range-coding helpers are pinned to upstream commit `0004153458b73387e0044b6f553f62116117388d`. The two unchanged source files and MIT license are included under `src/adasplat/codec/_upstream/`. Their SHA256 values are recorded and checked by each stream's decoder. The 2D position mapping, p sidecar, physical-field conversion, log-scale adaptation, complete container and quality-guard wrapper are AdaSplat-specific adaptations.
- PyTorch, NumPy, SciPy, Pillow, pytorch-msssim, Matplotlib, jaxtyping, rich, PyYAML, constriction, ninja and test/build dependencies retain their respective licenses.

No dataset or third-party pretrained network weights are redistributed in this source release. No pretrained neural network is required for AdaSplat training or this codec.
