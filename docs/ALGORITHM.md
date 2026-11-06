# Algorithm and implementation protocol

## Morphology

A primitive has2D center, two raw scales, angle, p and three RGB coefficients:9 learnable scalars. Fixed p removes one degree of freedom. The raw-scale map is `nominal=softplus(r)+0.25`; the physical scale is `s=q(p)*nominal`. The numerical p domain is[0.1,8], with continuous, unsegmented scale calibration.

For the continuous untruncated kernel `K=exp(-0.5*(|u|^p+|v|^p))`, squared-response-weighted radial moment is `M(p)=2*Gamma(3/p)/Gamma(1/p)` and `q(p)=sqrt(M(2)/M(p))`. The L2 norm is `Z=2*Gamma(1+1/p)*sqrt(sx*sy)`. Physical-scale calibration preserves the RMS extent for a fixed nominal scale in this continuous definition; truncation and discrete sampling do not have an exact unit-energy guarantee.

The CUDA renderer applies the existing C1 support treatment and uses double-precision intermediate local coordinates in both forward/backward paths to reduce low-p axis cancellation. The final response remains an additive RGB image clipped to[0,1]. No opacity parameter is newly optimized; `mass` is a fixed buffer.

## Initialization and fitting

The effective initializer uses a65% coverage lattice and35% gradient/texture-guided unique detail cells, followed by a linear color CG solve. It does not run quadtree subdivision in the manuscript recipe.

The optimizer preserves the existing cohort-aware Adan updates. Each accepted birth cohort has its own optimization clock. The paper recipe uses MSE-only training for5000main steps, ramps SSIM over the next5000, and reaches0.035 structure weight. Single-scale Gaussian SSIM uses11×11/sigma1.5. Structure gradients are computed periodically with interval compensation.

The actual training objective additionally includes an unclipped out-of-range squared penalty(weight0.01), dark bright-leak penalty(weight0.5, threshold0.05, margin0.04), and an MSE-budget hinge on structure updates. The full implementation remains in `core/adasplat_full_image_objective.py` and `training/trainer.py`; these modules implement the numerical objective.

## Population adaptation

Residuals propose candidate regions. At each sampled location, shape competition evaluates p in{0.5,1,2,4,8}, multiple sizes and oriented geometries. A regularized local color solve estimates reconstruction gain. Correlated/redundant responses are suppressed, and proposed births receive color fitting, warmup and joint trial optimization.

An expansion is kept only after full-image validation against the pre-birth state. This is not an equal-extra-compute whole-image dual-branch counterfactual. The actual regional guard has tolerances: dark MSE≤old×1.20+1e-5 and bright-leak energy≤old×1.25+1e-6. Grid cells with fewer than256dark pixels are skipped as individual vetoes, while their pixels remain in global checks. Global PSNR/SSIM/MS-SSIM, tail and positive-gain conditions also apply.

The cap is fixed, the accepted final population is adaptive. Growth stops before the fixed refinement stage. Extra trial updates mean50k main steps is not50k total optimizer updates.

## Evaluation

Report actual N, PD=`1000N/(H*W)` and actual trainable scalar count. The current full-method200k model has1.8M scalars, while fixed-p200k has1.6M. Per-image PSNR is averaged across images; do not convert pooled image MSE into the reported mean PSNR. RGB8 metrics are evaluated after quantization, without implicit resize/rotation.

`tools/evaluate.py` computes full-resolution RGB8 PSNR, SSIM and five-scale MS-SSIM. It does not reproduce LPIPS or downstream feature-network metrics; those require their separately specified evaluation models/protocols. Training itself has no pretrained feature network.

Capacity snapshots along one dynamic trajectory are not independent converged runs. For multi-image curves, use identical image cohorts at every node, actual counts and checkpoint deduplication. For compression curves, score the decoded image and use actual complete-stream bytes.
