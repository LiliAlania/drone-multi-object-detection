# Faster R-CNN two-stage experiments: Kaggle run

The Kaggle run of the Faster R-CNN notebook trained three variants and swept the test-time RPN proposals.
The baseline was best at mAP 0.2371. The notebook stopped at the summary cell, before the interpretability step.

The notebook with these outputs is
[`notebooks/02_faster_rcnn_experiments_kaggle.ipynb`](../notebooks/02_faster_rcnn_experiments_kaggle.ipynb). The
analysis of the results, including the failure cases (run afterwards), is in [`faster_rcnn.md`](faster_rcnn.md).

## Overview

Every run trains on every 5th train frame with one shared schedule and is evaluated on every 5th val frame. Each run
changes one parameter from the baseline (`--rpn-proposals 2000`, `--roi-pos-fraction 0.25`, `--box-loss smooth_l1`).

| Run | Change | Parameter | Planned cost | Ran? |
| --- | --- | --- | --- | --- |
| `base` | none | baseline | 1 training run (~2.8 h) | yes, 154 min |
| `giou` | `--box-loss giou` | box regression loss (stage 2) | 1 training run | yes, 154 min |
| `pos05` | `--roi-pos-fraction 0.5` | RoI sampling ratio (stage 2) | 1 training run, only if time is left | yes, 153 min |
| `base` @ 300 / 1000 proposals | test-time RPN proposals | number of RPN proposals (stage 1) | evaluation only | yes |
| `frcnn_explain` on `base` | none | where objects are lost; t-SNE; Grad-CAM | evaluation only | no |

The interpretability step was meant to look inside the `base` model: where each ground-truth object is lost (RPN vs
RoI head), a t-SNE of RoI features, and Grad-CAM over ResNet stages and FPN levels. Everything was meant to be logged
to Weights & Biases, one W&B run per model.

## Configuration and environment

The run used 5 epochs, batch 4 and every 5th frame. It ran on a Kaggle session with two Tesla T4 GPUs (15,360 MiB
each, driver 580.178.04, CUDA 13.0); training used one of them.

| Setting | Value | Note in the notebook |
| --- | --- | --- |
| `MOD` | `src.models.faster_rcnn` | training and evaluation module |
| `EPOCHS` | 5 | step LR drops after epochs 3 and 4; lower it if one epoch takes > 35 min |
| `SUBSET_STEP` | 5 | every 5th train frame: ~4.8k images, one epoch ~32 min on a T4 |
| `BATCH` | 4 | 2 if CUDA runs out of memory |
| `EVAL_STEP` | 5 | every 5th val frame |
| `BUDGET_MIN` | 690 | whole session; Kaggle stops at 12 h |
| `RESERVE_MIN` | 60 | kept free for the proposals sweep, the explain cells and the summary |
| `PROPOSALS` | 300, 1000 | test-time sweep on `base`; its own evaluation is the 2000 point |
| `RUNS` | `/kaggle/working/runs` | checkpoints and eval logs |
| Data workers | 4 | `--workers 4` |
| `WANDB_PROJECT` | `VisDrone_obj_det` | |
| `WANDB_ENTITY` | `irynkasv-ukrainian-catholic-university-org` | an organization, which W&B refuses (see below) |
| `WANDB_DIR` | `/kaggle/working` | offline runs land in the notebook output |

The repository was cloned fresh from GitHub into `/kaggle/working/repo`, and `pycocotools`, `wandb` and
`scikit-learn` were installed. The W&B API key came from the Kaggle secret `WANDB_API_KEY`; without it, logging
would have switched to offline mode.

Before running, the notebook asks for: GPU accelerator, Internet on, the competition data added as input, the W&B key
as a secret, and **Save Version → Save & Run All** so `/kaggle/working/runs` survives the 12 h session. Each run is
evaluated right after training, so a timeout loses at most the run in progress.

## Data

Training used 4,868 train frames (every 5th), built in 2 minutes from the competition data.

- **Competition data**: `/kaggle/input/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026`, with `test`,
  `test_images.csv`, `train` and `val`. The notebook finds it by searching `/kaggle/input` for
  `val/annotations_coco.json`.
- **Training subset**: `src.make_subset --step 5 --splits train val` wrote 5,439 frames to
  `/kaggle/working/visdrone-subset` (4,868 train + 571 val); its zip archive was deleted.
- **Evaluation**: every 5th frame of the original val split (`--step 5`), COCO metrics with up to 500 detections per
  image.

## Training runs

All three runs finished their 5 epochs in 153–154 minutes each (463 minutes for all three), and their loss fell every
epoch. Each epoch is 1,217 steps of batch 4; the learning rate drops ×0.1 after epochs 3 and 4.

| Mean training loss | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 |
| --- | --- | --- | --- | --- | --- |
| `base` | 0.7199 | 0.5983 | 0.5492 | 0.4846 | 0.4678 |
| `giou` | 0.8322 | 0.7119 | 0.6631 | 0.5975 | 0.5810 |
| `pos05` | 1.0003 | 0.8714 | 0.7954 | 0.6942 | 0.6695 |

The levels differ by design: GIoU and a 0.5 positive share change the loss itself, so compare the slopes, not the
heights.

At the end of training the box-regression loss of the second stage is the largest term and the RPN objectness loss
the smallest:

| Loss term at the end of training | `base` | `giou` | `pos05` |
| --- | --- | --- | --- |
| `loss_box_reg` (RoI head) | 0.2780 | 0.3891 | 0.4840 |
| `loss_classifier` (RoI head) | 0.1528 | 0.1540 | 0.1693 |
| `loss_rpn_box_reg` (RPN) | 0.0310 | 0.0271 | 0.0271 |
| `loss_objectness` (RPN) | 0.0069 | 0.0068 | 0.0073 |
| total (last logged step) | 0.4688 | 0.5770 | 0.6877 |
| wall time (train + evaluate) | 154 min | 154 min | 153 min |

## Evaluation of the trained runs

The three runs end within 0.0013 mAP of each other (0.2358–0.2371); `pos05` has the best small-object AP (0.102) and
`base` the best overall. Each run was evaluated right after training with `src.models.faster_rcnn evaluate` on every
5th val frame, batch 4.

| Metric (IoU 0.50:0.95 unless noted) | `base` | `giou` | `pos05` |
| --- | --- | --- | --- |
| mAP, maxDets 500 | **0.2371** | 0.2361 | 0.2358 |
| AP, maxDets 100 | 0.237 | 0.236 | 0.236 |
| AP50 | **0.487** | 0.482 | 0.481 |
| AP75 | 0.197 | 0.198 | 0.198 |
| AP_small | 0.091 | 0.086 | **0.102** |
| AP_medium | **0.225** | 0.224 | 0.221 |
| AP_large | 0.422 | 0.427 | **0.428** |
| AR, maxDets 1 | 0.107 | 0.114 | 0.118 |
| AR, maxDets 100 | 0.364 | 0.373 | 0.376 |
| AR, maxDets 500 | 0.365 | 0.374 | **0.378** |
| AR_small | 0.190 | 0.189 | **0.213** |
| AR_medium | 0.335 | 0.342 | 0.341 |
| AR_large | 0.562 | 0.572 | 0.573 |
| Inference, ms per image | 548.7 | 546.9 | 547.8 |
| Frames per second | 1.8 | 1.8 | 1.8 |

Classes differ far more than the three runs do: car is best (0.424 for `base`) and bus worst (0.051).

| AP by class | `base` | `giou` | `pos05` |
| --- | --- | --- | --- |
| car | 0.424 | 0.406 | 0.418 |
| person | 0.278 | 0.268 | 0.275 |
| van | 0.262 | 0.265 | 0.253 |
| truck | 0.262 | 0.233 | 0.270 |
| tricycle | 0.249 | 0.240 | 0.221 |
| bicycle | 0.238 | 0.246 | 0.238 |
| motor | 0.133 | 0.129 | 0.135 |
| bus | 0.051 | 0.102 | 0.077 |

The biggest per-run difference is bus, the weakest class: 0.051 for `base`, 0.077 for `pos05` and 0.102 for `giou`.

## Test-time RPN proposals sweep

1000 proposals give the same mAP as 2000 (0.2371) at 484 instead of 549 ms per image; 300 proposals lose 0.0036 mAP.
The sweep reuses the `base` weights and only changes how many proposals the RPN passes to the RoI head; the 2000
column is `base`'s own evaluation.

| Metric (IoU 0.50:0.95 unless noted) | 300 proposals | 1000 proposals | 2000 proposals |
| --- | --- | --- | --- |
| mAP, maxDets 500 | 0.2335 | **0.2371** | **0.2371** |
| AP50 | 0.475 | **0.487** | **0.487** |
| AP75 | 0.196 | 0.197 | 0.197 |
| AP_small | 0.090 | 0.091 | 0.091 |
| AP_medium | 0.221 | 0.225 | 0.225 |
| AP_large | 0.418 | 0.422 | 0.422 |
| AR, maxDets 500 | 0.348 | 0.361 | **0.365** |
| AR_small | 0.181 | 0.188 | **0.190** |
| AR_medium | 0.320 | 0.333 | **0.335** |
| AR_large | 0.548 | 0.557 | **0.562** |
| Inference, ms per image | **440.8** | 483.6 | 548.7 |
| Frames per second | **2.3** | 2.1 | 1.8 |

Per-class AP barely moves with the number of proposals: at 300 it is person 0.270, bicycle 0.234, car 0.417, van
0.262, truck 0.262, tricycle 0.246, bus 0.048, motor 0.129; at 1000 it matches 2000 within 0.001 (truck 0.263,
person 0.277).

## Summary step and the W&B error

The summary cell wrote `frcnn_experiments.csv` but then crashed on `wandb.init` with
`CommError: you may not log runs directly to your organization, please try using your team entity`, and Save & Run
All stopped there.

- **What the cell does**: reads mAP, AP50, AP_small and ms per image from each run's `eval.txt`, adds the proposal
  sweep, saves `runs/frcnn_experiments.csv`, prints the settings (`epochs 5, subset step 5, batch 4, eval step 5`)
  and logs one comparison table to W&B as `frcnn_experiments_summary`.
- **Why it failed**: `WANDB_ENTITY` was set to the organization `irynkasv-ukrainian-catholic-university-org`. The key
  logged in as user `irynkasv`, but W&B accepts runs only under a user or team entity.
- **Same error during training**: each training run also failed to start online (`W&B online init failed … logging
  offline`) and logged offline instead; at the end each offline run hit `run ID … is in use` (`64r1tvus`,
  `1bi7l287`, `18n5sb7u`). Training and evaluation were not affected.
- **Consequence**: no cell after the summary ran, so the interpretability step and the cleanup cell (`rm -rf` of the
  subset and the repo) were skipped.
- **Fix**: set `WANDB_ENTITY = ""` (the key's default entity) or a team name, and wrap the summary `wandb.init` in
  `try/except` so a W&B error cannot stop the notebook.

## Interpretability step (did not run)

The step meant to show where the `base` model fails did not run, because the summary cell had stopped the notebook;
no `explain/` folder is in the Kaggle output.

It calls `src.frcnn_explain` on `runs/frcnn_base/last.pt` twice:

| Command | Frames | What it produces |
| --- | --- | --- |
| `objects` | every 5th val frame | for each ground-truth object, the stage that lost it (no RPN proposal / proposal but score < 0.3 / wrong class); what the RoI head predicts when given the exact ground-truth box; a t-SNE of the 1024-d RoI features coloured by class, size and outcome |
| `gradcam` | every 50th val frame, 2 true + 2 false positives per frame | Grad-CAM of single detection scores on the ResNet stages C2–C5 and the FPN levels P2–P5 |

The cell then displays six plots from `runs/frcnn_base/explain/`: `outcomes`, `head_confusion`, `tsne`,
`gradcam_levels`, `gradcam_focus` and `gradcam_gallery`, which were also meant for the `frcnn_base` W&B run.

Rerunning it as written would also fail on the module path: the repository now has the script at
`src.analysis.frcnn_explain`, not `src.frcnn_explain`.
