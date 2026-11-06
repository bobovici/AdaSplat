# Training, population snapshots and logs

## Manuscript recipe

Use `--config configs/default.json`. It defines the following training settings:

| Setting | Value |
|---|---:|
| Maximum N | 200000 |
| Initial ratio / N | 0.15 / 30000 |
| Main steps | 50000 |
| Growth cutoff | step40000 |
| Growth opportunities | 16 |
| Birth range | 2%–8% of cap per accepted opportunity |
| MSE-only / SSIM ramp | first5000 / next5000 main steps |
| Maximum SSIM weight | 0.035 |
| Dark-leak weight | 0.5 |
| Initial / new color CG | 64 / 16 |
| Birth warmup | 64–192 updates |
| Joint trial fitting | 256 updates |
| Main SSIM interval | 8, with interval compensation |
| Seed | 20261002 |
| Early stopping | disabled |

The cap is a hard maximum, not a guarantee of filling it. Trial/warmup updates are additional to main steps and are counted separately. Use an explicit configuration file to select the training schedule.

## CLI

`train.py --input IMAGE --output NEW_DIR --gpu 0`; batch uses `train_batch.py --input-dir DIR --output NEW_DIR --gpus 0,1 --jobs-per-gpu 2`.

Common overrides: `--num`/`--count` or `--density` (mutually exclusive), `--steps`, `--initial-ratio`, `--growth-fraction`, `--save-counts`, `--seed`, `--no-growth`, `--no-early-stop`, `--dry-run`, `--smoke`.

Density is primitives **per pixel**. The plotted unit PD(k primitives/MP) is `1000*N/(H*W)`. Do not pass a plotted PD value directly as a per-pixel density. For 4928×3264, N200000 corresponds to PD12.433950853.

`--recursive` enumerates nested images. `--expected-images 100` aborts a batch if the count differs; it is optional for other datasets. `--retries 1` permits one additional attempt, with original failed-attempt logs retained.

## Output

Each image attempt records `config.json`, `resolved_plan.json`, `progress.json`, `console.log`, `events.jsonl`, `result.json`, `summary.json`, inference checkpoints, PNG reconstructions and `resume.pt`. Preserve the whole attempt directory for auditing. Some outputs appear only after their corresponding stage has completed.

`console.log` captures Python and native stdout/stderr. `events.jsonl` records UTC timestamps and process IDs; `result.json` retains evaluation history and population trials. The batch root keeps job attempts, per-image state and summary files. A detached shell's launcher log is additional to these internal logs. See `examples/train_batch_linux.sh` for a nohup launcher.

Training quality records are floating-point native-render metrics. A saved RGB8 PNG has quantization error; use `tools/evaluate.py` for RGB8 PSNR/SSIM/MS-SSIM. Do not silently mix those domains.

## Six requested population milestones

The paper configuration requests 40000,50000,70000,100000,150000,200000 with a10% tolerance. These are near-count snapshots, not strict equal-count budgets. The manifest records actual N, actual step and source checkpoint. Accepted growth may jump across multiple targets; 40k and50k can both refer to one45360-point state. A final checkpoint can duplicate a200k milestone.

Deduplicate by actual tensor/state identity and count actual N when plotting. A snapshot saved early in a dynamic trajectory is not an independently trained50k-step model at that budget.

## Resume and progress

```bash
python train.py --config configs/default.json --input image.png --output runs/part1 --stop-after 10000
python train.py --config configs/default.json --input image.png --output runs/part2 --resume runs/part1/resume.pt
```

Keep the same total-step schedule, input and renderer hash for exact continuation. Choose a new session output directory. `final.pt` is an inference checkpoint, while `resume.pt` carries the optimizer/run state; they have different roles.

On Linux, read `progress.json` and `console.log`; on Windows use `tools/Show-TrainingProgress.ps1`. There is no background training daemon in this source package.
