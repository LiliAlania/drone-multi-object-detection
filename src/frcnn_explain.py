"""Look inside a trained Faster R-CNN on labeled val frames: what it learns and where it fails.

objects  One row per ground-truth object and per confident false positive: the stage that lost it
         (no RPN proposal / proposal but no confident detection / wrong class), what the RoI head predicts
         when handed the perfect GT box, the FPN level its RoI is pooled from, and its 1024-d RoI feature.
         Plots: outcomes by size and class, RoI-head confusion on GT boxes, t-SNE of RoI features.
gradcam  Grad-CAM of individual detection scores on the ResNet stages C2..C5 and the FPN levels P2..P5:
         which levels explain small vs large objects, and whether false positives look at the object.

    python -m src.frcnn_explain objects --weights runs/frcnn_base/last.pt --split-root /content/data/val
    python -m src.frcnn_explain gradcam --weights runs/frcnn_base/last.pt --split-root /content/data/val
Outputs go to <weights dir>/explain/; with --wandb-project they are also logged to the W&B run that trained
these weights (see src.models.faster_rcnn.wandb_run).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.data_utils import CLASS_NAMES

IOU_MATCH = 0.5
SIZES = ("small", "medium", "large")  # COCO: < 32^2, < 96^2, larger (original pixels)
GT_OUTCOMES = ("detected", "wrong class", "missed: proposal, low score", "missed: no proposal")
FP_OUTCOMES = ("FP: duplicate", "FP: wrong class", "FP: poor box", "FP: background")
OUTCOME_COLORS = dict(zip(GT_OUTCOMES + FP_OUTCOMES, ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
                                                     "#8d99ae", "#b5838d", "#6d597a", "#264653"]))
CAM_MAPS = ("C2", "C3", "C4", "C5", "P2", "P3", "P4", "P5")


def size_bucket(area: float) -> str:
    return SIZES[0] if area < 32 ** 2 else SIZES[1] if area < 96 ** 2 else SIZES[2]


def ioa(boxes, regions):
    """Share of each box's area inside each region: (N, M)."""
    import torch

    lt = torch.max(boxes[:, None, :2], regions[None, :, :2])
    rb = torch.min(boxes[:, None, 2:], regions[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(-1).clamp(min=1e-6)
    return inter / area[:, None]


def match(det_boxes, det_labels, gt_boxes, gt_labels, crowd_boxes) -> tuple[list[int], list[str]]:
    """Greedy COCO-style matching of score-sorted detections.

    Returns (gt_match: index of the matching detection or -1 per GT, det_status per detection).
    det_status: 'TP', one of FP_OUTCOMES, or 'ignored' (inside an ignored region, not counted)."""
    from torchvision.ops import box_iou

    gt_match = [-1] * len(gt_boxes)
    iou = box_iou(det_boxes, gt_boxes) if len(gt_boxes) else det_boxes.new_zeros(len(det_boxes), 0)
    in_crowd = ioa(det_boxes, crowd_boxes).amax(1) if len(crowd_boxes) else det_boxes.new_zeros(len(det_boxes))
    status = []
    for d in range(len(det_boxes)):
        same = gt_labels == det_labels[d]
        free = same & det_boxes.new_tensor([m < 0 for m in gt_match], dtype=bool)
        best_free = iou[d].where(free, 0).max().item() if len(gt_boxes) else 0.0
        if best_free >= IOU_MATCH:
            gt_match[int(iou[d].where(free, 0).argmax())] = d
            status.append("TP")
        elif len(gt_boxes) and iou[d].where(same, 0).max().item() >= IOU_MATCH:
            status.append("FP: duplicate")
        elif len(gt_boxes) and iou[d].max().item() >= IOU_MATCH:
            status.append("FP: wrong class")
        elif in_crowd[d].item() >= 0.5:
            status.append("ignored")
        elif len(gt_boxes) and iou[d].max().item() >= 0.1:
            status.append("FP: poor box")
        else:
            status.append("FP: background")
    return gt_match, status


def load_frames(split_root: Path, step: int, max_images: int | None):
    from pycocotools.coco import COCO

    coco = COCO(str(split_root / "annotations_coco.json"))
    return coco, sorted(coco.getImgIds())[::step][:max_images]


def ground_truth(coco, image_id: int, device):
    """(boxes xyxy, 1-based labels, ignored-region boxes) of one frame."""
    import torch

    anns = coco.imgToAnns[image_id]

    def xyxy(items):
        return torch.tensor([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in items)],
                            dtype=torch.float32, device=device).reshape(-1, 4)

    objects = [a for a in anns if not a.get("iscrowd")]
    labels = torch.tensor([a["category_id"] for a in objects], dtype=torch.int64, device=device)
    return xyxy(objects), labels, xyxy([a for a in anns if a.get("iscrowd")])


def staged_forward(model, image):
    """GeneralizedRCNN.forward split open so each stage's output is visible.

    Returns (resized ImageList, ResNet stages C2..C5, FPN levels, RPN proposals, final detections)."""
    images, _ = model.transform([image])
    stages = model.backbone.body(images.tensors)
    levels = model.backbone.fpn(stages)
    proposals, _ = model.rpn(images, levels)
    detections, _ = model.roi_heads(levels, proposals, images.image_sizes)
    detections = model.transform.postprocess(detections, images.image_sizes, [tuple(image.shape[-2:])])[0]
    return images, stages, levels, proposals[0], detections


def to_resized(boxes, image, images):
    """Original-pixel boxes -> coordinates of the model's resized input."""
    (h, w), (rh, rw) = image.shape[-2:], images.image_sizes[0]
    return boxes * boxes.new_tensor([rw / w, rh / h, rw / w, rh / h])


def roi_head_on(model, levels, boxes_resized, images):
    """Run only the second stage on given boxes: (1024-d features, class probabilities, FPN level 0..3)."""
    heads = model.roi_heads
    pooled = heads.box_roi_pool(levels, [boxes_resized], images.image_sizes)
    features = heads.box_head(pooled)
    logits, _ = heads.box_predictor(features)
    level = heads.box_roi_pool.map_levels([boxes_resized])  # set by the pooling call above
    return features, logits.softmax(-1), level


# --------------------------------------------------------------------------- objects + embeddings


def collect_objects(model, device, coco, image_ids, split_root: Path, score_thr: float):
    import numpy as np
    import pandas as pd
    import torch
    from torchvision.ops import box_iou

    from src.data_utils import image_path_for
    from src.models.faster_rcnn import _read_images

    rows, features = [], []
    with torch.inference_mode():
        for n, image_id in enumerate(image_ids, 1):
            image = _read_images([image_path_for(split_root, coco.imgs[image_id]["file_name"])], device)[0]
            gt_boxes, gt_labels, crowd = ground_truth(coco, image_id, device)
            images, _, levels, proposals, dets = staged_forward(model, image)
            keep = dets["scores"] >= score_thr
            det_boxes, det_labels, det_scores = dets["boxes"][keep], dets["labels"][keep], dets["scores"][keep]
            gt_match, det_status = match(det_boxes, det_labels, gt_boxes, gt_labels, crowd)
            fp = [i for i, s in enumerate(det_status) if s.startswith("FP")]

            boxes = torch.cat([gt_boxes, det_boxes[fp]])
            if not len(boxes):
                continue
            feats, probs, level = roi_head_on(model, levels, to_resized(boxes, image, images), images)
            rpn_iou = (box_iou(to_resized(gt_boxes, image, images), proposals).amax(1)
                       if len(gt_boxes) and len(proposals) else gt_boxes.new_zeros(len(gt_boxes)))
            det_iou = box_iou(gt_boxes, det_boxes) if len(det_boxes) and len(gt_boxes) else None
            for i, box in enumerate(boxes.tolist()):
                is_gt = i < len(gt_boxes)
                area = (box[2] - box[0]) * (box[3] - box[1])
                if is_gt:
                    label = int(gt_labels[i])
                    if gt_match[i] >= 0:
                        outcome = "detected"
                    elif det_iou is not None and bool(((det_iou[i] >= IOU_MATCH) & (det_labels != label)).any()):
                        outcome = "wrong class"
                    elif rpn_iou[i] >= IOU_MATCH:
                        outcome = "missed: proposal, low score"
                    else:
                        outcome = "missed: no proposal"
                    score = float(det_scores[gt_match[i]]) if gt_match[i] >= 0 else float("nan")
                else:
                    d = fp[i - len(gt_boxes)]
                    label, outcome, score = int(det_labels[d]), det_status[d], float(det_scores[d])
                rows.append({
                    "image_id": image_id, "kind": "gt" if is_gt else "fp", "class": CLASS_NAMES[label - 1],
                    "area": area, "size": size_bucket(area), "fpn_level": f"P{int(level[i]) + 2}",
                    "rpn_iou": float(rpn_iou[i]) if is_gt else float("nan"),
                    "head_pred": "background" if int(probs[i].argmax()) == 0
                    else CLASS_NAMES[int(probs[i].argmax()) - 1],
                    "head_prob_true": float(probs[i, label]), "score": score, "outcome": outcome,
                })
            features.append(feats.float().cpu().numpy().astype(np.float16))
            print(f"{n}/{len(image_ids)} frames, {len(rows)} objects", end="\r")
    print()
    return pd.DataFrame(rows), np.concatenate(features)


def stage_summary(objects):
    """Per size bucket: how many GT objects survive each stage."""
    import pandas as pd

    gt = objects[objects["kind"] == "gt"]
    groups = list(gt.groupby("size", sort=False)) + [("all", gt)]
    return pd.DataFrame([{
        "size": name, "objects": len(g),
        "rpn_recall@0.5": (g["rpn_iou"] >= IOU_MATCH).mean(),
        "head_correct_on_gt_box": (g["head_pred"] == g["class"]).mean(),
        "final_recall": (g["outcome"] == "detected").mean(),
        "fp_per_gt": (objects["kind"].eq("fp") & (objects["size"].eq(name) | (name == "all"))).sum() / len(g),
    } for name, g in groups]).set_index("size").reindex([*SIZES, "all"]).dropna(how="all")


def plot_outcomes(objects, path: Path) -> Path:
    import matplotlib.pyplot as plt

    gt = objects[objects["kind"] == "gt"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 2.2]})
    for ax, key, order in ((axes[0], "size", SIZES), (axes[1], "class", CLASS_NAMES)):
        share = gt.groupby(key)["outcome"].value_counts(normalize=True).unstack(fill_value=0)
        share = share.reindex(index=[o for o in order if o in share.index],
                              columns=[o for o in GT_OUTCOMES if o in share.columns], fill_value=0)
        share.plot.bar(stacked=True, ax=ax, color=[OUTCOME_COLORS[c] for c in share.columns], width=0.8,
                       legend=ax is axes[1])
        counts = gt[key].value_counts()
        ax.set_xticklabels([f"{i}\n(n={counts.get(i, 0)})" for i in share.index], rotation=0)
        ax.set_ylabel("share of GT objects")
        ax.set_xlabel("")
        ax.set_title(f"Where GT objects are lost, by {key}")
    axes[1].legend(title=None, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_head_confusion(objects, path: Path) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd

    gt = objects[objects["kind"] == "gt"]
    columns = ["background", *CLASS_NAMES]
    table = pd.crosstab(gt["class"], gt["head_pred"], normalize="index")
    table = table.reindex(index=[c for c in CLASS_NAMES if c in table.index], columns=columns, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.imshow(table.values, cmap="Blues", vmin=0, vmax=1)
    for (r, c), v in pd.DataFrame(table.values).stack().items():
        if v >= 0.01:
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.5 else "black")
    ax.set_xticks(range(len(columns)), columns, rotation=40, ha="right")
    ax.set_yticks(range(len(table.index)), [f"{c} (n={(gt['class'] == c).sum()})" for c in table.index])
    ax.set_xlabel("RoI head prediction on the exact GT box")
    ax.set_ylabel("true class")
    ax.set_title("Second stage alone: classification of perfect proposals")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def tsne(objects, features, per_group: int, seed: int = 0):
    """2-D t-SNE of the RoI features on a class-balanced sample (<= per_group per kind x class)."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    sample = (objects.reset_index().groupby(["kind", "class"], group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_group), random_state=seed)))
    x = features[sample["index"].to_numpy()].astype(np.float32)
    x = PCA(n_components=min(50, x.shape[1], len(x) - 1), random_state=seed).fit_transform(x)
    xy = TSNE(n_components=2, init="pca", perplexity=min(30, (len(x) - 1) / 3), random_state=seed).fit_transform(x)
    return sample.drop(columns="index").assign(tsne_x=xy[:, 0], tsne_y=xy[:, 1])


def plot_tsne(points, path: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    gt, fp = points[points["kind"] == "gt"], points[points["kind"] == "fp"]
    colors = plt.get_cmap("tab10")
    for i, name in enumerate(CLASS_NAMES):
        for part, marker, alpha in ((gt, "o", 0.5), (fp, "x", 0.7)):
            p = part[part["class"] == name]
            axes[0].scatter(p["tsne_x"], p["tsne_y"], s=5 if marker == "o" else 9, marker=marker, alpha=alpha,
                            color=colors(i), linewidths=0.6, label=name if marker == "o" else None)
    axes[0].legend(markerscale=3, fontsize=8)
    axes[0].set_title("by class (o = GT object, x = false positive, colored by predicted class)")
    side = np.sqrt(points["area"].clip(lower=1))
    sc = axes[1].scatter(points["tsne_x"], points["tsne_y"], c=np.log2(side), s=4, cmap="viridis", alpha=0.6,
                         vmin=3, vmax=8)
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_ticks(range(3, 9), labels=[str(2 ** t) for t in range(3, 9)])
    cbar.set_label("sqrt(box area), px")
    axes[1].set_title("by object size")
    for outcome in GT_OUTCOMES + FP_OUTCOMES:
        p = points[points["outcome"] == outcome]
        if len(p):
            axes[2].scatter(p["tsne_x"], p["tsne_y"], s=4, alpha=0.6, color=OUTCOME_COLORS[outcome],
                            label=f"{outcome} ({len(p)})")
    axes[2].legend(markerscale=3, fontsize=8)
    axes[2].set_title("by outcome")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("t-SNE of RoI-head features (1024-d box_head output), class-balanced sample")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_objects(args) -> None:
    import numpy as np

    from src.models.faster_rcnn import load_trained

    out = args.out_dir or args.weights.parent / "explain"
    out.mkdir(parents=True, exist_ok=True)
    coco, image_ids = load_frames(args.split_root, args.step, args.max_images)
    model, device = load_trained(args.weights)
    objects, features = collect_objects(model, device, coco, image_ids, args.split_root, args.score_thr)
    objects.to_csv(out / "objects.csv", index=False)
    np.save(out / "roi_features.npy", features)
    summary = stage_summary(objects)
    summary.to_csv(out / "stage_summary.csv")
    print(f"Stage summary ({len(image_ids)} frames, detections with score >= {args.score_thr}):")
    print(summary.round(3).to_string())
    points = tsne(objects, features, args.tsne_per_group)
    points.to_csv(out / "tsne.csv", index=False)
    plots = [plot_outcomes(objects, out / "outcomes.png"), plot_head_confusion(objects, out / "head_confusion.png"),
             plot_tsne(points, out / "tsne.png")]
    print(f"Wrote {out}: objects.csv, roi_features.npy, stage_summary.csv, tsne.csv, "
          + ", ".join(p.name for p in plots))
    log_to_wandb(args, {f"explain/{p.stem}": p for p in plots},
                 {"explain/stage_summary": summary.reset_index(), "explain/tsne_points": points})


# --------------------------------------------------------------------------- Grad-CAM


def gradcam_maps(score, tensors, image, images):
    """Grad-CAM of `score` on each (1, C, h, w) tensor, upsampled to the original frame.

    Returns (cams, evidence: both (len(tensors), H, W), contribution of each tensor).
    cams: Grad-CAM, relu(sum_c mean(grad_c) * act_c) - shows where the channels this score relies on fire,
          anywhere in the frame. evidence: relu(sum_c grad_c * act_c) per pixel - only where gradient flows,
          i.e. what the score actually reads. contribution: |sum(grad * act)|, the first-order Taylor term;
          unlike summing a per-pixel map it does not grow with map resolution."""
    import torch
    import torch.nn.functional as F

    grads = torch.autograd.grad(score, tensors, retain_graph=True, allow_unused=True)
    (rh, rw), size = images.image_sizes[0], tuple(image.shape[-2:])

    def to_frame(m):
        m = F.interpolate(m, size=images.tensors.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(m[..., :rh, :rw], size=size, mode="bilinear", align_corners=False)[0, 0]

    cams, evidence, contribution = [], [], []
    for act, grad in zip(tensors, grads):
        if grad is None:  # e.g. an FPN level the RoI head never pooled from for this box
            grad = torch.zeros_like(act)
        contribution.append(abs((grad * act).sum().item()))
        cams.append(to_frame((grad.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True).relu()))
        evidence.append(to_frame((grad * act).sum(1, keepdim=True).relu()))
    return torch.stack(cams).detach(), torch.stack(evidence).detach(), contribution


def expand(box, factor: float, min_side: float, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max((x2 - x1) * factor, min_side), max((y2 - y1) * factor, min_side)
    return [max(int(cx - w / 2), 0), max(int(cy - h / 2), 0), min(math.ceil(cx + w / 2), width),
            min(math.ceil(cy + h / 2), height)]


def collect_gradcam(model, device, coco, image_ids, split_root: Path, score_thr: float, per_image: int,
                    crop: int = 96, seed: int = 0):
    import random

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from src.data_utils import image_path_for
    from src.models.faster_rcnn import _read_images

    model.requires_grad_(True)  # the frozen stem/layer1 must be in the graph for a C2 gradient
    rng = random.Random(seed)
    rows, crops = [], []
    for n, image_id in enumerate(image_ids, 1):
        image = _read_images([image_path_for(split_root, coco.imgs[image_id]["file_name"])], device)[0]
        gt_boxes, gt_labels, crowd = ground_truth(coco, image_id, device)
        images, stages, levels, _, dets = staged_forward(model, image)
        keep = torch.where(dets["scores"] >= score_thr)[0]
        boxes = dets["boxes"][keep].detach()
        _, status = match(boxes, dets["labels"][keep], gt_boxes, gt_labels, crowd)
        tp = [i for i, s in enumerate(status) if s == "TP"]
        fp = [i for i, s in enumerate(status) if s.startswith("FP")]
        picked = rng.sample(tp, min(len(tp), per_image // 2)) + rng.sample(fp, min(len(fp), per_image // 2))
        if not picked:
            continue
        _, _, level = roi_head_on(model, levels, to_resized(boxes[picked], image, images), images)
        tensors = [stages[k] for k in ("0", "1", "2", "3")] + [levels[k] for k in ("0", "1", "2", "3")]
        height, width = image.shape[-2:]
        for j, i in enumerate(picked):
            cams, evidence, contribution = gradcam_maps(dets["scores"][keep[i]], tensors, image, images)
            box = boxes[i].tolist()
            area = (box[2] - box[0]) * (box[3] - box[1])
            x1, y1, x2, y2 = expand(box, 1.0, 4, width, height)
            # share of the evidence read from inside the box itself; the rest is context pulled in by
            # receptive fields (maps the score does not depend on get NaN)
            total = evidence.sum((1, 2))
            in_box = (evidence[:, y1:y2, x1:x2].sum((1, 2)) / total).tolist()
            share = [c / max(sum(contribution[:4]), 1e-12) for c in contribution[:4]] + \
                    [c / max(sum(contribution[4:]), 1e-12) for c in contribution[4:]]
            rows.append({"image_id": image_id, "status": status[i], "class": CLASS_NAMES[int(dets["labels"][keep[i]]) - 1],
                         "score": dets["scores"][keep[i]].item(), "area": area, "size": size_bucket(area),
                         "fpn_level": f"P{int(level[j]) + 2}",
                         **{f"share_{m}": s for m, s in zip(CAM_MAPS, share)},
                         **{f"in_box_{m}": v for m, v in zip(CAM_MAPS, in_box)}})
            # small crop around the box for the gallery; each map scaled by its own max over the frame
            cx1, cy1, cx2, cy2 = expand(box, 3.0, 48, width, height)
            window = torch.cat([image[None, :, cy1:cy2, cx1:cx2],
                                (cams / cams.amax((1, 2), keepdim=True).clamp(min=1e-12))[None, :, cy1:cy2, cx1:cx2]], 1)
            window = F.interpolate(window, size=(crop, crop), mode="bilinear", align_corners=False)[0]
            scale = [crop / (cx2 - cx1), crop / (cy2 - cy1)]
            crops.append({"image": (window[:3].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy(),
                          "cams": window[3:].cpu().numpy().astype(np.float16),
                          "box": [(box[0] - cx1) * scale[0], (box[1] - cy1) * scale[1],
                                  (box[2] - cx1) * scale[0], (box[3] - cy1) * scale[1]]})
        del stages, levels, dets, tensors
        print(f"{n}/{len(image_ids)} frames, {len(rows)} detections", end="\r")
    print()
    return pd.DataFrame(rows), crops


def plot_level_shares(cams, path: Path) -> Path:
    import matplotlib.pyplot as plt

    tp = cams[cams["status"] == "TP"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, maps, title in ((axes[0], CAM_MAPS[:4], "ResNet stages (C2 = stride 4 ... C5 = stride 32)"),
                            (axes[1], CAM_MAPS[4:], "FPN levels (P2 = stride 4 ... P5 = stride 32)")):
        share = tp.groupby("size")[[f"share_{m}" for m in maps]].mean().reindex([s for s in SIZES if s in set(tp["size"])])
        share.columns = maps
        share.plot.bar(ax=ax, rot=0, colormap="viridis")
        counts = tp["size"].value_counts()
        ax.set_xticklabels([f"{s}\n(n={counts.get(s, 0)})" for s in share.index])
        ax.set_ylabel("mean share of grad x activation")
        ax.set_xlabel("")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Which feature maps explain the score of correct detections, by object size")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_focus(cams, path: Path) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd

    groups = {"TP small": (cams["status"] == "TP") & (cams["size"] == "small"),
              "FP small": cams["status"].str.startswith("FP") & (cams["size"] == "small"),
              "TP medium+large": (cams["status"] == "TP") & (cams["size"] != "small"),
              "FP medium+large": cams["status"].str.startswith("FP") & (cams["size"] != "small")}
    table = pd.DataFrame({f"{name} (n={mask.sum()})": cams.loc[mask, [f"in_box_{m}" for m in CAM_MAPS]].mean().values
                          for name, mask in groups.items() if mask.any()}, index=CAM_MAPS)
    fig, ax = plt.subplots(figsize=(11, 4))
    table.plot.bar(ax=ax, rot=0, color=["#2a9d8f", "#e76f51", "#8ab17d", "#f4a261"][: table.shape[1]])
    ax.set_ylabel("share of grad x activation inside the box")
    ax.set_title("How much of the score is read from the object itself vs its context? Correct vs false positives")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def pick_examples(cams) -> list[tuple[str, int]]:
    """A few telling detections: small/medium/large correct ones and the most confident false positives."""
    picks = []
    tp = cams[cams["status"] == "TP"]
    if len(tp):
        side = tp["area"] ** 0.5
        picks += [("smallest TP", side.idxmin()), ("~40 px TP", (side - 40).abs().idxmin()), ("largest TP", side.idxmax())]
    for kind in FP_OUTCOMES:
        fp = cams[cams["status"] == kind]
        if len(fp):
            picks.append((f"top {kind}", fp["score"].idxmax()))
    return picks


def plot_gallery(cams, crops, path: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    picks = pick_examples(cams)
    fig, axes = plt.subplots(len(picks), 1 + len(CAM_MAPS), figsize=(2 * (1 + len(CAM_MAPS)), 2.1 * len(picks)),
                             squeeze=False)
    for row, (name, index) in zip(axes, picks):
        item, info = crops[index], cams.loc[index]
        for col, ax in enumerate(row):
            ax.imshow(item["image"].mean(-1) if col else item["image"], cmap="gray" if col else None)
            if col:
                ax.imshow(item["cams"][col - 1].astype("float32"), cmap="jet", alpha=0.5, vmin=0, vmax=1)
            x1, y1, x2, y2 = item["box"]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1))
            ax.set_xticks([])
            ax.set_yticks([])
            if col:
                ax.set_title(f"{CAM_MAPS[col - 1]} ({info[f'share_{CAM_MAPS[col - 1]}']:.0%})", fontsize=8)
        row[0].set_title(f"{name}\n{info['class']} {info['score']:.2f}, {info['area'] ** 0.5:.0f}px, "
                         f"{info['fpn_level']}", fontsize=8)
    fig.suptitle("Grad-CAM of single detection scores (each map scaled to its own max over the frame; "
                 "% = share of grad x activation within its group)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_gradcam(args) -> None:
    from src.models.faster_rcnn import load_trained

    out = args.out_dir or args.weights.parent / "explain"
    out.mkdir(parents=True, exist_ok=True)
    coco, image_ids = load_frames(args.split_root, args.step, args.max_images)
    model, device = load_trained(args.weights)
    cams, crops = collect_gradcam(model, device, coco, image_ids, args.split_root, args.score_thr, args.per_image)
    if cams.empty:
        raise RuntimeError(f"No detections with score >= {args.score_thr}")
    cams.to_csv(out / "gradcam.csv", index=False)
    print(cams.groupby(["status", "size"])[[f"share_{m}" for m in CAM_MAPS]].mean().round(2).to_string())
    plots = [plot_level_shares(cams, out / "gradcam_levels.png"), plot_focus(cams, out / "gradcam_focus.png"),
             plot_gallery(cams, crops, out / "gradcam_gallery.png")]
    print(f"Wrote {out}: gradcam.csv, " + ", ".join(p.name for p in plots))
    log_to_wandb(args, {f"explain/{p.stem}": p for p in plots}, {"explain/gradcam": cams})


# --------------------------------------------------------------------------- CLI


def log_to_wandb(args, images: dict[str, Path], tables: dict) -> None:
    from src.models.faster_rcnn import wandb_run

    run = wandb_run(args.wandb_project, args.weights.parent)
    if not run:
        return
    import wandb

    run.log({**{k: wandb.Image(str(p)) for k, p in images.items()},
             **{k: wandb.Table(dataframe=t) for k, t in tables.items()}})
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, step, help_text in (("objects", 5, "stage-wise failures and t-SNE of RoI features"),
                                  ("gradcam", 50, "Grad-CAM over ResNet stages and FPN levels")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--weights", type=Path, required=True)
        sub.add_argument("--split-root", type=Path, required=True, help="Folder with annotations_coco.json")
        sub.add_argument("--step", type=int, default=step, help="Use every k-th frame")
        sub.add_argument("--max-images", type=int, help="Stop after this many frames")
        sub.add_argument("--score-thr", type=float, default=0.3, help="Operating point for TP/FP and 'detected'")
        sub.add_argument("--out-dir", type=Path, help="Default: <weights dir>/explain")
        sub.add_argument("--wandb-project", help="Also log plots and tables to the W&B run of these weights")
        if name == "objects":
            sub.add_argument("--tsne-per-group", type=int, default=600,
                             help="Max points per (GT/FP, class) in the t-SNE sample")
        else:
            sub.add_argument("--per-image", type=int, default=4, help="Detections per frame (half TP, half FP)")
    args = parser.parse_args()
    (run_objects if args.command == "objects" else run_gradcam)(args)


if __name__ == "__main__":
    main()
