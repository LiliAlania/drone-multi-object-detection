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
python -m src.yolo_baseline train --data-root /content/visdrone-subset --epochs 30 --imgsz 1280
# write the Kaggle submission
python -m src.yolo_baseline predict --weights runs/detect/baseline/weights/best.pt \
    --test-root /content/data --out submission.csv
kaggle competitions submit -c multiple-object-video-detection-hw-2-ucu-ai-2026 -f submission.csv -m "yolo11s baseline"
```

### Faster R-CNN (two-stage, anchor-based)

Uses the same YOLO labels and writes the same submission format. Needs `torch` and `torchvision`
(preinstalled in Colab).

```bash
python -m src.faster_rcnn train --data-root /content/visdrone-subset --epochs 12 --batch 4
# after a Colab disconnect: same command + --resume (continues from <out-dir>/last.pt)
# COCO mAP on the labeled val split (--step 5 = every 5th frame, faster)
python -m src.faster_rcnn evaluate --weights runs/faster_rcnn/last.pt --split-root /content/data/val --step 5
python -m src.faster_rcnn predict --weights runs/faster_rcnn/last.pt \
    --test-root /content/data --out submission_frcnn.csv
```

Two-stage training options (defaults in brackets). They are saved in the checkpoint, so
`evaluate`/`predict` rebuild the same model:

| Option | Stage | What it controls |
|---|---|---|
| `--rpn-proposals` [2000] | RPN | proposals kept after NMS and sent to the RoI head (pre-NMS: 2x in train, 1x in test) |
| `--roi-batch` [512] | RoI head | RoIs sampled per image for the second-stage loss |
| `--roi-pos-fraction` [0.25] | RoI head | foreground share of those sampled RoIs |
| `--box-loss` [`smooth_l1`] | RoI head | box regression loss: `smooth_l1` or `giou` |
| `--lr-schedule` [`step`] | training | `step` (x0.1 at 2/3 and 8/9 of epochs) or `cosine` |

Example ablation, one change from the default per run (separate `--out-dir` each):

```bash
python -m src.faster_rcnn train --data-root /content/visdrone-subset --rpn-proposals 500 --out-dir runs/frcnn_rpn500
python -m src.faster_rcnn train --data-root /content/visdrone-subset --roi-pos-fraction 0.5 --out-dir runs/frcnn_pos05
python -m src.faster_rcnn train --data-root /content/visdrone-subset --box-loss giou --out-dir runs/frcnn_giou
python -m src.faster_rcnn train --data-root /content/visdrone-subset --lr-schedule cosine --out-dir runs/frcnn_cosine
```

### FCOS (one-stage, anchor-free)

Same data, CLI, checkpoint format and submission as Faster R-CNN; only the model differs.

```bash
python -m src.fcos train --data-root /content/visdrone-subset --epochs 12 --batch 4
python -m src.fcos evaluate --weights runs/fcos/last.pt --split-root /content/data/val --step 5
python -m src.fcos predict --weights runs/fcos/last.pt     --test-root /content/data --out submission_fcos.csv
```

### Our pipeline: tiled YOLO + sliced inference (EDA-driven)

Trains on 640 px tiles cut from full-resolution frames (ignored regions painted gray, rare-class
tiles repeated) and predicts with overlapping tiles plus one full-image pass. See the module
docstring in `src/tiled_yolo.py` for which EDA finding drives each choice.

```bash
pip install ultralytics
python -m src.tiled_yolo prepare --data-root /content/visdrone-subset --out /content/visdrone-tiles
python -m src.tiled_yolo train --data-root /content/visdrone-tiles --epochs 20     --project /content/drive/MyDrive/runs/tiled_yolo   # after a disconnect: same command + --resume
W=/content/drive/MyDrive/runs/tiled_yolo/tiled/weights/best.pt
python -m src.tiled_yolo evaluate --weights $W --split-root /content/data/val --step 5
python -m src.tiled_yolo evaluate --weights $W --split-root /content/data/val --step 5 --tile 0  # ablation: no slicing
python -m src.tiled_yolo predict --weights $W --test-root /content/data --out submission_tiled.csv
```

### RT-DETR (query-based, transformer)

Same tiles, sliced inference and COCO evaluation as the tiled YOLO pipeline; only the model differs.
Run `src.tiled_yolo prepare` first (the tiles are shared).

```bash
python -m src.rtdetr train --data-root /content/visdrone-tiles --epochs 20     --project /content/drive/MyDrive/runs/rtdetr   # after a disconnect: same command + --resume
W=/content/drive/MyDrive/runs/rtdetr/rtdetr/weights/best.pt
python -m src.rtdetr evaluate --weights $W --split-root /content/data/val --step 5
python -m src.rtdetr predict --weights $W --test-root /content/data --out submission_rtdetr.csv
```

## ⚠️ Rules

You may change **only** `report.md` and the `images/` folder.

**Any change to the core website (`index.html`, `assets/`, `.github/`) will annul the grade for the homework.** This includes edits that change how the last-push date is shown.

The last-push date is read from GitHub and is used to check deadlines.
