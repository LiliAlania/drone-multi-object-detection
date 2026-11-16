---
title: Homework 1 -- Multiple Object Video Detection
author: Lili Alania, Iryna Svyrydenko
course: Artificial Intelligence · UCU
date: 2026-11-16
---

## 1. Methodology and motivation

### 1.1 What the data told us (EDA)

All numbers come from [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) and
[`notebooks/data_temporality.ipynb`](notebooks/data_temporality.ipynb).

| Split | Clips | Frames | Resolutions |
|---|---|---|---|
| train | 56 | 24,201 | 6 (1344×756 to 2720×1530) |
| val | 7 | 2,846 | 5 (incl. one 3840×2160 clip) |
| test | 48 | 3,811 | 6 (680×382 to 1920×1080) |

- **Objects are tiny.** At a 640 px input, 69% of boxes have a side < 16 px (22% at 1280, 13% at 1536).
  *What we did:* kept frames close to native resolution, used small anchors, raised the YOLO resolution, added a
  stride-4 P2 head, and implemented YOLO with tiles.
- **Small anchors are needed.** With default anchors (base 32 px), 92% of small boxes have no anchor with
  IoU ≥ 0.5; anchors from 16 px leave 19.6%.
  *What we did:* anchors 8–128 px for Faster R-CNN and from 16 px for RetinaNet.
- **Classes are very unequal.** Car has 505k train boxes, bus only 9.7k.
  *What we did:* reported per-class AP; class balancing is listed as future work.
- **Consecutive frames are near-duplicates.** Same-track IoU is 0.85 at lag 1; with every 5th frame only 0.7% of
  tracks are never seen.
  *What we did:* trained on every 5th frame of each clip.
- **Ignored regions are common.** 72% of frames have at least one; they cover 2.3% of the frame on average.
  *What we did:* handled them as `iscrowd` in evaluation (see 1.2).
- **Density is bounded.** At most 180 objects per frame (p99 140).
  *What we did:* nothing extra is needed; the 500-detection cap never limits recall.
- **Train and val clips are disjoint.** No clip is shared.
  *What we did:* used val as a test of generalisation to new scenes.

### 1.2 Data preparation

The steps below are the ones used for every model whose results are reported in Section 2.

1. **Ignored regions.** In the COCO file every ignored
   region is repeated several times per frame (0 "ignored region", 11 "others"; for example 64 entries but only 8 unique boxes in one frame). We removed 446,901 duplicates in train and 27,965 in val. On 20 sampled frames per split the COCO and YOLO boxes match. They are not used as training targets: the training labels contain only the 8
   classes.
2. **Frame subsampling.** We kept every 5th frame **within each clip**, in frame
   order, so every clip stays represented: 4,868 train frames (of 24,201) and 571 val frames. 
3. **Per-model input processing.**
   - *Faster R-CNN:* YOLO boxes are converted to pixel `x1 y1 x2 y2` and clipped to the image; boxes with a side
     under 1 px are dropped; labels are shifted by +1 (0 is background); pixels are scaled to [0, 1]. Frames
     are resized to a short side of 1080 (long side ≤ 1920), i.e. close to native size. The only augmentation
     is a horizontal flip with probability 0.5.
   - *RetinaNet:* the same dataset class and flip, but resized to a short side of 640 (long side ≤ 1280).
   - *YOLO11:* Ultralytics letterboxes to `imgsz` (640 / 960 / 1280) and applies its default augmentation
     (mosaic, ±50% zoom, HSV, flip); mosaic is switched off for the last 10 epochs.
   - *RT-DETR-L:* Ultralytics pipeline at 640 px, every 5th frame.

We also built a tiling pipeline (640 px tiles with 20% overlap, ignored regions painted gray, rare-class tiles
repeated, sliced inference with SAHI).

### 1.3 Validation design

- **Split:** we validate on the official val split. Its 7 clips do not occur in train, so neighbouring frames of
  one video never fall on both sides.
- **Frames:** every 5th val frame (570 frames). This is 5× faster than all 2,846 frames, and consecutive frames
  add little information.
- **Metric:** one COCO evaluation for all models: pycocotools, mAP@[.5:.95], AP50, AP75, AP by object size
  (small < 32², medium < 96², large), maxDets 500, ignored regions as `iscrowd`. Speed is reported as
  ms per image on a Tesla T4.
- **Temporal post-processing** needs consecutive frames, so it is evaluated on all frames. Its parameters were
  selected on development clips and the gain is reported only on held-out clips.
- One seed per setting. Differences below ~0.002 mAP are treated as noise.

### 1.4 Models

- **Two-stage, anchor-based: Faster R-CNN** ResNet-50 FPN v2 (torchvision, COCO-pretrained), anchors
  8/16/32/64/128 px on P2–P6 × ratios 0.5/1/2.
  *Why:* the second stage re-examines each candidate on aligned features, which should help on small, crowded objects.
- **One-stage, anchor-based: RetinaNet** ResNet-50 FPN v2 (COCO-pretrained), default vs smaller anchors.
  *Why:* dense prediction with focal loss; checks how much anchor size matters.
- **Anchor-free: YOLO11s** (Ultralytics, COCO-pretrained) at 640/960/1280, optional P2 head.
  *Why:* fast; tests resolution and grid density instead of anchors.
- **Query-based: RT-DETR-L** (Ultralytics, COCO-pretrained).
  *Why:* set prediction with Hungarian matching, no NMS.
- **Temporal:** causal tracking + rescoring on top of a frozen RetinaNet.
  *Why:* uses the video structure that single-frame detectors ignore.

### 1.5 Tuning strategy

Each experiment changes one parameter from a baseline with everything else fixed:

- **Faster R-CNN** (5 epochs, every 5th frame, SGD lr 0.01, momentum 0.9, wd 1e-4, ×0.1 after epochs 3 and 4,
  batch 4, AMP): box loss Smooth L1 vs GIoU; RoI positive fraction 0.25 vs 0.5; RPN proposals at test time
  300 / 1000 / 2000.
- **RetinaNet** (12 epochs): default vs small anchors; focal α = 0.5; `topk_candidates`, NMS and score
  thresholds at inference.
- **YOLO11s** (12 epochs): input size 640 → 960 → 1280, then the P2 head at the best size and at 640.
- **RT-DETR-L** (8 epochs, AdamW lr 1e-5): 300 vs 200 queries; matcher costs (class/bbox/GIoU) 2/5/2 vs 2/7/3.
- **Temporal:** rescoring weight and propagation settings, selected on development clips.

## 2. Results and analysis

### 2.1 Benchmark

Every 5th val frame (570 frames), COCO AP with maxDets 500. The best setting of each family, plus the initial one:

| Detector | Setting | Train frames / epochs | mAP | AP50 | AP_small | AP_medium | AP_large | ms / img |
|---|---|---|---|---|---|---|---|---|
| **Faster R-CNN** | `base` (Smooth L1, pos 0.25, 2000 proposals) | every 5th / 5 | **0.2371** | **0.4870** | **0.0910** | **0.225** | **0.422** | 548.7 |
| Faster R-CNN | initial run | every 20th / 6 | 0.1675 | 0.3611 | 0.0873 | 0.165 | 0.301 | 243.9* |
| YOLO11s | P2–P5 heads, 1280 | every 5th / 12 | 0.1791 | 0.3547 | 0.0769 | 0.166 | 0.314 | 57.4 |
| YOLO11s | initial, 640 | every 5th / 12 | 0.1239 | 0.2445 | 0.0613 | 0.116 | 0.247 | 37.7 |
| RetinaNet | small anchors | every 5th / 12 | 0.1459 | 0.3151 | 0.0608 | 0.136 | 0.278 | 140.2 |
| RetinaNet | default anchors | every 5th / 12 | 0.1468 | 0.2820 | 0.0423 | 0.134 | 0.310 | 126.6 |
| RT-DETR-L | 300 queries, matcher 2/7/3 | every 5th / 8 | 0.0189 | 0.0371 | 0.0093 | 0.019 | 0.044 | 86.3 |

\* the initial Faster R-CNN ran at a short side of 640, the tuned one at 1080.

![AP by object size](images/outcomes.png)

### 2.2 Incremental improvements

| Step | Change | mAP | Δ |
|---|---|---|---|
| Faster R-CNN | 640 px, every 20th frame, 6 ep → 1080 px, every 5th frame, 5 ep | 0.1675 → 0.2371 | **+0.070** |
| YOLO11s | 640 → 960 | 0.1239 → 0.1594 | +0.036 |
| YOLO11s | 960 → 1280 | 0.1594 → 0.1709 | +0.012 |
| YOLO11s | + P2 head at 1280 | 0.1709 → 0.1791 | +0.008 |
| RetinaNet | `topk_candidates` 1000 → 2000 at inference | 0.1468 → 0.1569 | +0.010 |
| RetinaNet | small anchors | 0.1468 → 0.1459 (AP_small 0.042 → 0.061) | 0.000 |
| RetinaNet + temporal | rescoring with past frames (held-out clips) | 0.1477 → 0.1505 | +0.003 |

The largest gains come from giving the network more pixels per object: native resolution for Faster R-CNN and
1280 px for YOLO. Changes to the loss or the anchors move AP between object sizes but barely change the total.

### 2.3 Feature experiments

**Faster R-CNN parameters** ([`docs/faster_rcnn.md`](docs/faster_rcnn.md)):

| Run | mAP | AP50 | AP75 | AP_small | AP_large | ms / img |
|---|---|---|---|---|---|---|
| `base` | **0.2371** | **0.4870** | 0.197 | 0.0910 | 0.422 | 548.7 |
| GIoU box loss | 0.2361 | 0.4825 | 0.198 | 0.0860 | 0.427 | 546.9 |
| RoI positive fraction 0.5 | 0.2358 | 0.4813 | 0.198 | **0.1018** | **0.428** | 547.8 |
| `base`, 1000 proposals | **0.2371** | 0.4866 | 0.197 | 0.0909 | 0.422 | 483.6 |
| `base`, 300 proposals | 0.2335 | 0.4752 | 0.196 | 0.0899 | 0.418 | 440.8 |

- GIoU does not help and lowers AP_small. For a box of a few pixels, a one-pixel shift changes the IoU a lot, so an
  IoU-based loss is noisiest exactly on small objects.
- A positive fraction of 0.5 gives the best AP_small (+0.011) at the cost of medium objects; the total is unchanged.
- 1000 proposals are as good as 2000 and 12% faster; 300 lose 0.004 mAP, because dense frames need several
  proposals per object.

**Where Faster R-CNN loses objects** (570 val frames, 22,899 objects; detected = right class, IoU ≥ 0.5, score ≥ 0.3):

| Size | RPN recall | RoI head correct on the exact GT box | Final recall |
|---|---|---|---|
| small | 0.843 | **0.340** | 0.419 |
| medium | 0.917 | 0.618 | 0.657 |
| large | 0.992 | 0.805 | 0.831 |

The RPN finds 91% of all objects. Most losses happen in the second stage: given the exact box of a small object,
the head calls 62% of them background. Grad-CAM shows that only 14–18% of the evidence for a small detection
lies inside its box, so the head mostly reads the surroundings.

**YOLO grid points.** At 640 px, 11.3% of small objects have no stride-8 grid point; at 1280 only 0.9%. A P2
head at 640 gives the same coverage as 1280 without P2, yet its AP_small is much lower (0.057 vs 0.082). So the
limit for small objects is the number of pixels the backbone sees, not only the label assignment.

### 2.4 Unsuccessful attempts

- **RT-DETR-L** reached only 0.019 mAP, about one eighth of RetinaNet, and plateaued from epoch 3. The 200-query
  run did not learn at all (mAP 0 at every epoch); we did not find the cause. With 8 epochs, lr 1e-5, batch 1
  and 640 px input, the setup under-performs rather than being unfinished.
- **GIoU loss** and a **positive fraction of 0.5** did not improve Faster R-CNN mAP.
- **Small anchors in RetinaNet:** AP_small +0.019 and AP50 +0.033, but AP75 and AP_large fell, so mAP stayed the same.
- **P2 head in YOLO at 1280** lowered AP_small (0.082 → 0.077). Only half of the weights could be loaded from
  COCO, the run used batch 4 instead of 8, and it was still improving at epoch 12.
- **Temporal post-processing** gave only +0.003 mAP on held-out clips. Gains were consistent on all 5 held-out
  clips (+0.0025 to +0.0055) and larger in dense frames (+0.005), but small in absolute terms.

### 2.5 Failure cases

1. **An unseen 4K clip.** The val clip `uav0000268_05773_v` is 3840×2160, a resolution that no train clip has,
   and it holds 34% of val frames. Faster R-CNN shrinks it by 2 to fit a short side of 1080: recall drops to
   0.28 and 37.5% of objects get no proposal (1.5–8.6% on the other clips). This clip alone pulls the val mAP down.
2. **Similar vehicles.** Given the exact box, the RoI head labels 61% of buses and 34% of vans as car, and 25%
   of bicycles as motor. From above these classes look alike, and the frequent class wins. Bus AP is 0.05.
3. **Small objects scored as background** (see 2.3).
4. **Many false positives are imprecise boxes or missing labels:** of 7,275 confident false positives, 35% are a
   real object with IoU < 0.5, 34% background, 28% wrong class. Some "background" persons are clearly visible
   but unannotated people.

![Faster R-CNN RoI-head confusion on exact GT boxes](images/head_confusion.png)

### 2.6 Conclusions

- **Resolution matters most.** Native-resolution input added +0.070 mAP to Faster R-CNN, and 640 → 1280 added
  +0.047 to YOLO11s. Loss and anchor changes stayed within ±0.002.
- **The two-stage detector is the most accurate** (0.237 mAP, 0.091 AP_small) but ~10× slower than YOLO11s-P2
  (0.179). Anchor-free YOLO beats anchor-based RetinaNet on every object size and is 2.4× faster.
- **The bottleneck is classification of small objects,** not finding them: the RPN proposes 84% of small objects,
  but the head keeps only about a third.
- **Query-based RT-DETR did not work** under our training budget.
- **Temporal context helps a little** (+0.003 mAP) and consistently.
- **Limits:** one seed per setting; unequal epochs across families (5 for Faster R-CNN, 8 for RT-DETR, 12 for
  YOLO and RetinaNet).

## 3. Final model and run instructions

**Selected model: YOLO11m at 1280 px + temporal post-processing**
([`notebooks/08_temporal_yolo.ipynb`](notebooks/08_temporal_yolo.ipynb)).

Why this model, and not Faster R-CNN, which has the best single-frame val mAP:
- The YOLO11s search (Section 2.2) showed that resolution is the main lever and that 1280 px is the best size.
  YOLO11m is the larger model of the same family at that size, and it is ~10× faster than Faster R-CNN. That
  makes it practical to predict every one of the 3,811 test frames in order, which tracking needs.
- The test data is video, and the temporal step (`src.analysis.temporal`) uses detections from earlier frames of
  the same clip to correct the current frame. On the frozen RetinaNet it gave a consistent gain on every held-out
  clip (Section 2.4).
- A 12-epoch YOLO11m run peaked at epoch 3 (Ultralytics val mAP50-95 0.238, 0.213 at epoch 12) and then overfit:
  train cls_loss kept falling while val cls_loss rose from epoch 5. So the final recipe trains **6 epochs** with
  stronger augmentation.

**Temporal post-processing** (causal: only earlier frames of the same clip are used):
1. *Camera motion*: a similarity transform between consecutive frames, from ORB keypoints and RANSAC. Track boxes
   are warped before matching; otherwise a panning drone moves a 10 px object out of IoU range.
2. *Tracking*: per class, tracks are matched to current detections by IoU (Hungarian matching, IoU ≥ 0.5).
3. *Rescoring*: a matched detection's score is blended with its track's running score (`weight`). Weak
   detections of objects seen confidently before go up, and one-frame false positives go down.
4. *Propagation*: a confident track (score ≥ 0.5) missed in the current frame adds its warped box, with the score
   decayed ×0.8 per frame, for up to `max_age` frames.

The parameters were chosen by a grid on 4 of the 7 val clips: `weight` ∈ {0.3, 0.5, 0.7} × `max_age` ∈ {0, 3, 6},
then motion compensation on/off at the best point. The gain is reported on the other 3 clips only.

**Dependencies.** Python 3.12+, CUDA PyTorch, `ultralytics==8.4.208`, `pycocotools`, `opencv-python`
(ORB features), `numpy`, `pandas`, `scipy` (Hungarian matching); optional `wandb`. All pins are in
[`requirements.txt`](requirements.txt). Run on Kaggle or Colab with one Tesla T4 (~3 h in total).

**Settings.**

| Setting | Value |
|---|---|
| Model | `yolo11m.pt` (COCO-pretrained), heads P3–P5 |
| Training frames | every 5th frame of each clip; the submission model uses **train + val** (`--train-val`) |
| Input size | 1280 |
| Epochs, batch | 6, 4 (Ultralytics accumulates gradients to a nominal batch of 64) |
| Augmentation | Ultralytics defaults + `mixup` 0.1, `copy_paste` 0.1, `scale` 0.7; mosaic in epochs 1–2 only (`--close-mosaic 4`) |
| Optimizer, schedule | Ultralytics defaults (`optimizer=auto`, linear LR decay, warm-up) |
| Weights used | `last.pt` (final epoch) |
| Inference | `imgsz` 1280, confidence ≥ 0.01, at most 500 boxes per frame |
| Temporal | `iou` 0.5, `ema` 0.7, `new_track` 0.3, `propagate` 0.5, `decay` 0.8; `weight`, `max_age` and `motion` as selected on the 4 tuning clips |
| Seed | Ultralytics defaults: `seed=0`, `deterministic=True` |

The submission model is retrained on train + val because the 7 val clips contain scenes the train split does not
have. Its val numbers would be fitted, so it is not evaluated. The same recipe trained on train only is what the
validation in the notebook measures.

**Steps** (from the repository root; the notebook runs the same on Kaggle or Colab):

```bash
pip install -r requirements.txt
# 1. data (needs Kaggle credentials and the accepted competition rules)
python -m src.data.get_data --output-dir data
python -m src.data.make_subset --data-root data --output-dir visdrone-subset --step 5 --splits train val

# 2. validation run: train on train only, COCO AP on every 5th val frame
python -m src.models.yolo_baseline train --data-root visdrone-subset --model yolo11m.pt --epochs 6 --imgsz 1280 \
    --batch 4 --close-mosaic 4 --mixup 0.1 --copy-paste 0.1 --scale 0.7 --project runs --name yolo11m_1280_e6_aug
python -m src.models.yolo_baseline evaluate --weights runs/yolo11m_1280_e6_aug/weights/last.pt \
    --split-root data/val --step 5 --imgsz 1280

# 3. temporal tuning: predict every val frame, then compare raw vs temporal
python -m src.analysis.temporal index --split-root data/val --out val_index
python -m src.models.yolo_baseline predict --weights runs/yolo11m_1280_e6_aug/weights/last.pt \
    --test-root val_index --out val_all_frames.csv --imgsz 1280 --conf 0.01 --batch 8
python -m src.analysis.temporal evaluate --pred val_all_frames.csv --images-csv val_index/test_images.csv \
    --split-root data/val --weight <W> --max-age <A> [--motion]

# 4. submission model: same settings on train + val
python -m src.models.yolo_baseline train --data-root visdrone-subset --model yolo11m.pt --epochs 6 --imgsz 1280 \
    --batch 4 --close-mosaic 4 --mixup 0.1 --copy-paste 0.1 --scale 0.7 --project runs \
    --name yolo11m_1280_e6_aug_trainval --train-val

# 5. predict all 3,811 test frames, then apply the selected temporal settings
python -m src.models.yolo_baseline predict --weights runs/yolo11m_1280_e6_aug_trainval/weights/last.pt \
    --test-root data --out submission_raw.csv --imgsz 1280 --conf 0.01 --batch 8
python -m src.analysis.temporal apply --pred submission_raw.csv --images-csv data/test_images.csv \
    --out submission_temporal.csv --weight <W> --max-age <A> [--motion]
```

`<W>`, `<A>` and `--motion` are the values selected in step 3 (the notebook passes them automatically).
`submission_temporal.csv` has the columns `image_id, PredictionString`, as the competition expects. After a
disconnect, add `--resume` to a `train` command to continue from `last.pt`.
