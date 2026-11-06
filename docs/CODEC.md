# V6/V2 complete-stream codec

The codec provides population-preserving V6/V2 log-scale RAHT/Laplace coding of trained image primitives. It performs no parameter optimization or pruning.

## Commands

```bash
python codec.py encode --checkpoint final.pt --target input.png --output model.adagc --decoded-output decoded.png
python codec.py inspect --input model.adagc
python codec.py decode --input model.adagc --output fields.npz
python render.py --input model.adagc --output replay.png
python render.py --input final.pt --output raw.png
```

Outputs must be new files. Stream decoding needs neither the original image nor a training checkpoint. The NPZ stores decoded physical `xy`, `scale`, `angle`, `p`, `rgb` arrays and header JSON; it is not an optimizer-resume file. Rendering requires CUDA; CPU parameter decoding does not.

The stream's effective RGB already contains the fixed mass and continuous-kernel L2 amplitude. Do not apply L2 normalization a second time after decoding.

## Coding pipeline

1. Read an existing `adasplat_rebuild_v1` inference checkpoint. Reject unsupported moment-corrected, affine-color or pixel-quadrature representations.
2. Export FP32 physical fields: `scale=(softplus(scale_raw)+0.25)*q(p)` and `rgb=colors*mass/Z(p,scale)`.
3. Quantize positions at1/256 pixel and use deterministic spatial ordering, delta and zigzag coding. RAHT keys use rounded2D coordinates and a deterministic collision rank.
4. Quantize log physical scale, rotation and p at16bits. Fold amplitude into RGB before color coding, without an intermediate FP16 color payload.
5. Apply the pinned MesonGS++ RAHT transform and orthonormal luminance/chroma mapping. Encode blockwise quantized AC color coefficients with the original entropy helper. For24-bit colors, use the original high16/low8 extension.
6. Write all fields, DC values, quantizer metadata, dimensions, kernel settings, source checksum and section checksums into one `.adagc` container. Non-color sections choose the smaller zlib9/lzma6 representation.
7. Decode fields, render RGB8 and enforce the quality guard before publishing the output stream.

Default candidate order is `(blocks,bits)=(320,16),(1280,16),(4096,16),(320,24),(1280,24)`. This is a sequential quality fallback, not a search for the smallest byte count across all possible encoders. For small integration examples only, block count is capped at `N-1` so each AC block has data; ordinary manuscript populations retain the original settings.

The default guard is loss≤0.02dB relative to this checkpoint's unencoded RGB8 reconstruction, using the provided original image as GT. If no setting passes, retain a failure report and do not publish an accepted stream. `--max-loss-db` changes the requested policy explicitly. `--no-quality-guard` uses the first fixed setting and makes no decoded-quality guarantee; it can encode fields without CUDA when no decoded image is requested.

The pinned RAHT Morton helper supports at most14bits per axis, including the deterministic collision-rank coordinate. Unsupported very large images or excessive coordinate collisions are rejected explicitly rather than resized.

## Rate accounting

`bpp=8*Path(stream).stat().st_size/(H*W)` counts the entire container, including headers and decoding metadata. Sidecar quality reports and PNGs are not part of the bitstream. A `.pt`, `.npz` or gzip checkpoint is not this codec's compressed rate. Parameters decoded from an actual lossy bitstream should be scored after decoding.

## Build identity

Source checkpoint replay is strict about renderer SHA256. Different OS/compiler/GPU architectures can produce different binary hashes; for an explicit cross-build replay use `--allow-renderer-change`, which records both builds. The quality guard is then evaluated under the selected renderer; do not interpret it as a claim that every cross-platform pixel equals the original server output.

The stream includes enough state for independent decoding and verifies unchanged upstream helper SHA256 values. The renderer is a software dependency, not a hidden image-specific checkpoint. Install/build it on the target machine. Load historical streams with this decoder before claiming compatibility with other codec revisions.

## Output report

The `.adagc.json` sidecar records actual N, source step, dimensions, bytes/bpp, checkpoint and stream hashes, candidate precisions, unencoded/decoded RGB8 PSNR, quality loss, runtime and renderer identity. No primitive is pruned and no training update is performed. `inspect` reads the header; `decode` additionally verifies and reconstructs the coded fields.
