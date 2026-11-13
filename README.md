# UCU AI Homework Report Template

Your report is published as a website in UCU colors. The page is generated from `report.md`, and it shows the date of your repository's last push.

## How to fill

1. Click **Use this template** to create your own **public** repository. If you fork instead, also open the **Actions** tab and enable workflows.
2. Enable GitHub Pages (see [below](#enabling-github-pages)).
3. Edit **`report.md`**:
   - Fill in the header block at the top (`title`, `author`, `course`, `date`).
   - Write the report in Markdown. You can use tables, code blocks, and LaTeX math (`$...$` inline, `$$...$$` for display).
   - Put images in `images/` and reference them as `![caption](images/plot.png)`.
4. Commit and push to `main`. GitHub Actions deploys the site automatically within about a minute, at `https://<username>.github.io/<repo>/`.
5. Submit that link.

## Enabling GitHub Pages

This is a one-time step. GitHub does not copy it to your repository.

1. In your repository, open **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**. It saves as soon as you choose it.
3. Open the **Actions** tab, select the failed **Deploy report** run, and click **Re-run jobs**.

Until you do this, every deploy fails with the error "GitHub Pages is not set up".

## Local preview

```bash
python -m http.server 8000
# open http://localhost:8000
```

Opening `index.html` directly from disk will not work.

## VisDrone EDA and dataset subsets

The website/report instructions above apply to the course report. The EDA work lives in
separate `src/`, `notebooks/`, `tests/`, `results/`, `figures/`, and `splits/` folders;
it does not modify the report website.

### Run the EDA in Google Colab

1. Open [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) in Colab (upload it or open it
   from the public GitHub repository).
2. Set `MOUNT_DRIVE = True` in the setup cell if you want to mount Drive. The notebook
   clones this repository into `/content/drone-multi-object-detection` when needed.
3. Keep the dataset at `/content/data` to skip downloading. Otherwise the notebook calls
   `src/get_data.py`, which uses the Kaggle CLI. Install/configure Kaggle credentials in
   Colab and accept the competition rules before running the download cell; no credentials
   are stored in this repository.
4. Run all cells. Results and figures are saved under `results/`, `figures/eda/`, and
   `splits/`. The notebook intentionally leaves its observations for the student to fill in.

The helpers are in `src/data_utils.py`. Run their synthetic-data tests from the repository
root with `python -m unittest discover -s tests`.

### Create a reusable subsampled dataset

From a Python environment with pandas and NumPy installed, run:

```bash
python -m src.make_subset --data-root /content/data --output-dir /content/visdrone-subset --step 5 --splits train val
```

The command writes a YOLO-layout directory and a `.zip` beside it. To copy the zip to
mounted Drive, add, for example,
`--drive-copy /content/drive/MyDrive/visdrone-subset.zip`. The step is applied independently
to each clip in frame order; ignored-region labels remain in a separate `ignore/` tree.

### YOLO baseline (Ultralytics)

```bash
pip install ultralytics
# train on the subset (or pass /content/data for the full dataset)
python -m src.models.yolo_baseline train --data-root /content/visdrone-subset --epochs 30 --imgsz 1280
# write the Kaggle submission
python -m src.models.yolo_baseline predict --weights runs/detect/baseline/weights/best.pt \
    --test-root /content/data --out submission.csv
kaggle competitions submit -c multiple-object-video-detection-hw-2-ucu-ai-2026 -f submission.csv -m "yolo11s baseline"
```

### Faster R-CNN (two-stage, anchor-based)

Uses the same YOLO labels and writes the same submission format. Needs `torch` and `torchvision`
(preinstalled in Colab).

```bash
python -m src.models.faster_rcnn train --data-root /content/visdrone-subset --epochs 12 --batch 4
# after a Colab disconnect: same command + --resume (continues from <out-dir>/last.pt)
# COCO mAP on the labeled val split (--step 5 = every 5th frame, faster)
python -m src.models.faster_rcnn evaluate --weights runs/faster_rcnn/last.pt --split-root /content/data/val --step 5
python -m src.models.faster_rcnn predict --weights runs/faster_rcnn/last.pt \
    --test-root /content/data --out submission_frcnn.csv
```

### FCOS (one-stage, anchor-free)

Same data, CLI, checkpoint format and submission as Faster R-CNN; only the model differs.

```bash
python -m src.models.fcos train --data-root /content/visdrone-subset --epochs 12 --batch 4
python -m src.models.fcos evaluate --weights runs/fcos/last.pt --split-root /content/data/val --step 5
python -m src.models.fcos predict --weights runs/fcos/last.pt \
    --test-root /content/data --out submission_fcos.csv
```

### Initial experiment: tiled YOLO + sliced inference (EDA-driven)

Our first pipeline; the ensemble pipeline below builds on its tiling. Trains on 640 px tiles cut from full-resolution frames (ignored regions painted gray, rare-class
tiles repeated) and predicts on the whole frame at native resolution (same object scale as
training); `--tile 640` adds overlapping tile passes. See the module
docstring in `src/models/tiled_yolo.py` for which EDA finding drives each choice.

```bash
pip install ultralytics
python -m src.models.tiled_yolo prepare --data-root /content/visdrone-subset --out /content/visdrone-tiles
python -m src.models.tiled_yolo train --data-root /content/visdrone-tiles --epochs 20 \
    --project /content/drive/MyDrive/runs/tiled_yolo   # after a disconnect: same command + --resume
W=/content/drive/MyDrive/runs/tiled_yolo/tiled/weights/best.pt
python -m src.models.tiled_yolo evaluate --weights $W --split-root /content/data/val --step 5
python -m src.models.tiled_yolo evaluate --weights $W --split-root /content/data/val --step 5 --tile 640  # ablation: + tile passes
python -m src.models.tiled_yolo predict --weights $W --test-root /content/data --out submission_tiled.csv
```

### RT-DETR (query-based, transformer)

Same tiles, inference and COCO evaluation as the tiled YOLO pipeline; only the model differs.
Run `src.models.tiled_yolo prepare` first (the tiles are shared).

```bash
python -m src.models.rtdetr train --data-root /content/visdrone-tiles --epochs 20 \
    --project /content/drive/MyDrive/runs/rtdetr   # after a disconnect: same command + --resume
W=/content/drive/MyDrive/runs/rtdetr/rtdetr/weights/best.pt
python -m src.models.rtdetr evaluate --weights $W --split-root /content/data/val --step 5
python -m src.models.rtdetr predict --weights $W --test-root /content/data --out submission_rtdetr.csv
```

### Ensemble pipeline: YOLOv8s-P2 + Faster R-CNN, SAHI, WBF

Two detectors trained on the same tiles, run with sliced inference and fused with Weighted Boxes
Fusion. All settings and paths are in [`configs/pipeline.yaml`](configs/pipeline.yaml) (`envs.local`
for the RTX 4060 laptop, `envs.colab` for a T4; picked automatically). Every script takes
`--debug` (2 train clips, 1 val clip, 2 epochs, 8 val/test frames, separate `work/debug`, `runs/debug`).

```bash
pip install -r requirements.txt          # CUDA torch first, see the file header
python -m src.run_pipeline --debug       # smoke test: tiles -> train -> SAHI -> WBF -> submission (~12 min)
```

**Steps (each one is also its own CLI):**

| Step | Command | Output |
|---|---|---|
| 1. Tiles | `python -m src.prepare_tiles --force` | `work/tiles/` + `stats.json`, 5 previews in `vis/` |
| 2a. YOLO | `python -m src.train_yolo_p2` (`--resume` after a stop) | `runs/yolo_p2/weights/best.pt` |
| 2b. Faster R-CNN | `python -m src.train_frcnn_tiles` (`--resume`) | `runs/frcnn/best.pt`, losses in `log.csv` |
| 3. SAHI | `python -m src.infer_sahi --model {yolo,frcnn} --split {val,test}` | `work/preds/<model>_<split>.parquet` |
| 4. Single-model mAP | `python -m src.evaluate --pred work/preds/yolo_val.parquet` | mAP, AP50, AP_small, AP per class |
| 5. WBF | `python -m src.ensemble_wbf search`, then `apply --split test` | `work/ensemble/{grid.csv,best.json,comparison.md}` |
| 6. Submission | `python -m src.make_submission` | `work/submission.csv`, validated before writing |

`python -m src.run_pipeline --from infer` reruns everything after training.

**Data decisions**
- The competition data is already converted to 8 merged classes (`person` = pedestrian + people,
  `tricycle` = tricycle + awning-tricycle), and the submission is scored on them, so both models train
  on these 8. The mapping lives once in `src/pipeline_common.py` (class index = COCO `category_id` − 1).
- Ignored regions (VisDrone 0 "ignored" and 11 "others", `iscrowd=1` here) are painted (114,114,114)
  and the boxes centred in them dropped, before tiling, so the model is not taught that unlabeled
  objects are background.
- Every 5th frame (`data.frame_step`): consecutive frames are near-duplicates (see the EDA).
- 640 px tiles, 20% overlap; a box goes into a tile if ≥ 50% of it is inside (clipped to the edge)
  and both clipped sides are ≥ 2 px; only 10% of empty tiles are kept. Val tiles (every 25th frame)
  serve only training-time validation; model selection uses full val frames through SAHI.

**Models**
- **YOLOv8s-P2** (`yolov8s-p2.yaml` + `yolov8s.pt`): the extra stride-4 head sees 10–20 px objects
  that stride 8 merges into 2–3 cells. Ultralytics 8.4 ships official P2 configs only for YOLOv8 and
  YOLO26, not YOLO11; YOLOv8-P2 is the long-tested one (YOLO26-P2 is a one-line config change:
  `model: yolo26s-p2.yaml`, `weights: yolo26s.pt`). P2 layers start untrained; everything else
  comes from COCO. imgsz 640, AMP, `scale=0.3` (objects are already tiny; strong zoom-out would erase
  them), mosaic on, copy-paste off, seed 42 with deterministic mode.
- **Faster R-CNN** (ResNet-50 FPN v2, COCO): anchors 8–128 px, 2000 RPN proposals, 500 detections
  per image, new 8+1-class head, tiles fed at 640 px. AMP, batch 8 × accumulate 2 (effective 16,
  lr 0.02), 500-step warm-up, ×0.1 at 2/3 and 8/9 of the epochs, val mAP on the val tiles every
  2 epochs, `best.pt` by that mAP.

**Inference and ensemble**
- SAHI: 640 slices, 20% overlap, plus one full-frame pass (large objects). Slice results are merged
  class by class (`postprocess_class_agnostic=False`): a rider's `person` box sits inside its
  `bicycle`/`motor` box, and class-agnostic merging would delete one of them. GREEDYNMM on IOS 0.6,
  confidence 0.05; `force_postprocess_type=True` because SAHI otherwise silently switches to NMS/IOU
  below confidence 0.1. IOS merges same-class boxes that sit inside each other, which can cost recall
  in dense crowds: compare with `infer.postprocess_type: NMS` on val.
- Raw predictions of each model are saved, so the WBF grid (weights [1,1], [2,1], [3,1], [1,2] ×
  iou 0.5/0.6/0.7 × skip 0.01/0.05/0.1) never reruns a model. WBF fuses only boxes of the same
  class, in coordinates normalized by each frame's own size. The best config by val mAP@[.5:.95]
  (maxDets 500, the metric used throughout this repo) is applied to test; `comparison.md` lists
  YOLO / Faster R-CNN / WBF with person, bicycle and motor first.
- Final threshold 0.01 after the ensemble (mAP rewards recall), ≤ 500 boxes per frame.

**Time estimate for the full run** (RTX 4060 Laptop, measured in the smoke test and scaled up).
The same smoke test ran 1.5–2× slower in a second run (laptop power/thermal limits), so read these
as the fast case; keep the laptop plugged in. A T4 on Colab is in the same range.

| Step | Measured (debug) | Full run | Estimate |
|---|---|---|---|
| Tiles | 164 frames in ~10 s | 4868 train + 117 val frames → ~27k train tiles (~3.3 GB) | ~10 min |
| YOLOv8s-P2 | 32 tiles/s, 2.8 GB at batch 8 | ~15 min/epoch × 50 epochs (early stop at patience 15) | **~8–13 h** |
| Faster R-CNN | 7.4–10.3 tiles/s; 2.3 GB at batch 4, 3.9 GB at batch 8 | ~45–60 min/epoch × 12 epochs | **~9–12 h** |
| SAHI val (~570 frames) | YOLO 0.26–0.41 s, FRCNN 0.73–1.8 s per frame | | ~10–20 min |
| SAHI test (3811 frames) | YOLO 0.30–0.53 s, FRCNN 1.0–2.3 s per frame | | ~1.5–3 h |
| WBF grid (36 configs) | ~1.5 min per COCO evaluation | | ~1 h |

So about a day of GPU time; train YOLO and Faster R-CNN in separate sessions (on Colab each needs
a new session after ~12 h, so use `--resume`). For a quicker first result, set `yolo.epochs: 30`.

**On Kaggle:** [`notebooks/03_ensemble_kaggle.ipynb`](notebooks/03_ensemble_kaggle.ipynb) (GPU T4 x2,
Internet on, competition data as input). It clones this repository, so push first. Set `STAGE` and use
*Save & Run All*: `"smoke"` (debug run, ~20 min), then `"train"` (both models in parallel, one per T4,
stopped after `TRAIN_HOURS`; attach the previous version's output to resume, repeat until both are
`done`), then `"infer"` (SAHI, WBF, `/kaggle/working/submission.csv`). The `kaggle` env in the config
is picked automatically; the notebook points `PIPELINE_DATA_ROOT` at the competition input.

## ⚠️ Rules

You may change **only** `report.md` and the `images/` folder.

**Any change to the core website (`index.html`, `assets/`, `.github/`) will annul the grade for the homework.** This includes edits that change how the last-push date is shown.

The last-push date is read from GitHub and is used to check deadlines.
