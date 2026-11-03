"""Reusable metadata, geometry, sampling, and visualization helpers for VisDrone."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


CLASS_NAMES = [
    "person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"
]


def load_coco(coco_path: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load COCO metadata as image and annotation frames with clip/frame dimensions."""
    with Path(coco_path).open(encoding="utf-8") as file:
        coco = json.load(file)
    images = pd.DataFrame(coco.get("images", []))
    annotations = pd.DataFrame(coco.get("annotations", []))
    if images.empty:
        images = pd.DataFrame(columns=["id", "file_name", "width", "height"])
    if "video" in images:
        images["clip"] = images["video"].fillna(images["file_name"].map(_clip_from_path))
    else:
        images["clip"] = images["file_name"].map(_clip_from_path)
    if "frame_id" not in images:
        images["frame_id"] = images["file_name"].map(_frame_from_path)
    images["split"] = split
    images["frame_id"] = pd.to_numeric(images["frame_id"], errors="coerce")
    if annotations.empty:
        annotations = pd.DataFrame(
            columns=["image_id", "category_id", "bbox", "iscrowd"]
        )
    metadata = images[["id", "clip", "frame_id", "width", "height", "split"]].rename(
        columns={"id": "image_id"}
    )
    annotations = annotations.merge(metadata, on="image_id", how="left")
    if "iscrowd" not in annotations:
        annotations["iscrowd"] = 0
    if "category_id" not in annotations:
        annotations["category_id"] = np.nan
    if "track_id" not in annotations:
        annotations["track_id"] = np.nan
    if "orig_category_id" not in annotations:
        annotations["orig_category_id"] = np.nan
    if "occlusion" not in annotations:
        annotations["occlusion"] = np.nan
    if "truncation" not in annotations:
        annotations["truncation"] = np.nan
    return images, annotations


def _clip_from_path(file_name: str) -> str:
    parts = Path(str(file_name).replace("\\", "/")).parts
    return parts[-2] if len(parts) > 1 else ""


def _frame_from_path(file_name: str) -> float:
    try:
        return float(Path(str(file_name)).stem)
    except ValueError:
        return math.nan


def split_annotations(
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split target annotations (iscrowd=0) from ignored regions (iscrowd=1)."""
    crowd = pd.to_numeric(annotations["iscrowd"], errors="coerce").fillna(0)
    return annotations.loc[crowd.eq(0)].copy(), annotations.loc[crowd.eq(1)].copy()


def namespace_image_ids(
    image_frames: Sequence[pd.DataFrame],
    annotation_frames: Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate splits while making image IDs unique across COCO files."""
    images = pd.concat(image_frames, ignore_index=True).copy()
    annotations = pd.concat(annotation_frames, ignore_index=True).copy()
    images["coco_image_id"] = images["id"]
    annotations["coco_image_id"] = annotations["image_id"]
    mapping = {
        (str(row.split), row.id): index
        for index, row in enumerate(images[["split", "id"]].itertuples(index=False))
    }
    images["id"] = [
        mapping[(str(row.split), row.id)]
        for row in images[["split", "id"]].itertuples(index=False)
    ]
    annotations["image_id"] = [
        mapping[(str(row.split), row.image_id)]
        for row in annotations[["split", "image_id"]].itertuples(index=False)
    ]
    return images, annotations


def coco_xywh_to_yolo(
    bbox: Sequence[float], image_width: float, image_height: float
) -> tuple[float, float, float, float]:
    """Convert pixel COCO [x, y, w, h] to normalized YOLO [cx, cy, w, h]."""
    x, y, width, height = map(float, bbox)
    return (
        (x + width / 2) / image_width,
        (y + height / 2) / image_height,
        width / image_width,
        height / image_height,
    )


def check_yolo_consistency(
    coco_path: str | Path,
    dataset_root: str | Path,
    split: str,
    sample_size: int = 20,
    seed: int = 42,
    atol: float = 1e-5,
) -> dict[str, int]:
    """Assert sampled YOLO target/ignore files match their COCO normalized boxes."""
    images, annotations = load_coco(coco_path, split)
    by_image = {int(key): group for key, group in annotations.groupby("image_id")}
    image_lookup = images.set_index("id")
    rng = random.Random(seed)
    candidates = list(by_image)
    selected = rng.sample(candidates, min(sample_size, len(candidates)))
    compared_target = compared_ignore = 0
    root = Path(dataset_root)
    for image_id in selected:
        image = image_lookup.loc[image_id]
        base = _resolve_image_path(root, str(image["file_name"]))
        label_path = _label_path(base, "labels")
        ignore_path = _label_path(base, "ignore")
        rows = by_image[image_id]
        for crowd, path in ((0, label_path), (1, ignore_path)):
            expected = rows.loc[pd.to_numeric(rows["iscrowd"], errors="coerce").fillna(0).eq(crowd)]
            actual_rows = _read_yolo_rows(path, has_class=(crowd == 0))
            expected_rows: list[tuple[float, ...]] = []
            for _, row in expected.iterrows():
                coords = coco_xywh_to_yolo(
                    row["bbox"], float(image["width"]), float(image["height"])
                )
                if crowd == 0:
                    expected_rows.append((int(row["category_id"]) - 1, *coords))
                else:
                    expected_rows.append(coords)
            actual_sorted = sorted(actual_rows)
            expected_sorted = sorted(expected_rows)
            assert len(actual_sorted) == len(expected_sorted), (
                f"Annotation count mismatch for {path}: "
                f"YOLO={len(actual_sorted)}, COCO={len(expected_sorted)}"
            )
            for actual, expected_row in zip(actual_sorted, expected_sorted):
                assert np.allclose(actual, expected_row, atol=atol, rtol=0), (
                    f"Box mismatch in {path}: YOLO={actual}, COCO={expected_row}"
                )
            if crowd == 0:
                compared_target += len(expected_rows)
            else:
                compared_ignore += len(expected_rows)
    return {
        "sampled_frames": len(selected),
        "target_boxes_compared": compared_target,
        "ignored_regions_compared": compared_ignore,
    }


def _resolve_image_path(root: Path, file_name: str) -> Path:
    clean = Path(file_name.replace("\\", "/"))
    candidates = [root / clean, root / clean.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _label_path(image_path: Path, folder: str) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = folder
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / folder / image_path.parent.name / (
        image_path.stem + ".txt"
    )


def _read_yolo_rows(path: Path, has_class: bool) -> list[tuple[float, ...]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        expected_length = 5 if has_class else 4
        if len(values) != expected_length:
            raise ValueError(f"Expected {expected_length} values per row in {path}")
        rows.append(tuple(values))
    return rows


def dataset_overview(
    image_frames: dict[str, pd.DataFrame], test_csv: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return split totals, frames per clip, and image-resolution counts."""
    images = pd.concat(image_frames.values(), ignore_index=True)
    if test_csv is not None and Path(test_csv).exists():
        test = pd.read_csv(test_csv)
        test["split"] = "test"
        test["clip"] = test["clip"].astype(str)
        images = pd.concat([images, test], ignore_index=True, sort=False)
    per_clip = (
        images.groupby(["split", "clip"], dropna=False)
        .agg(frames=("file_name", "nunique"))
        .reset_index()
    )
    per_clip["resolution"] = (
        images["width"].astype("Int64").astype(str)
        + "x"
        + images["height"].astype("Int64").astype(str)
    )
    resolution = (
        images.groupby(["split", "resolution"], dropna=False)
        .size()
        .rename("frames")
        .reset_index()
    )
    totals = (
        images.groupby("split", dropna=False)
        .agg(
            clips=("clip", "nunique"),
            frames=("file_name", "nunique"),
            resolutions=("resolution", "nunique"),
        )
        .reset_index()
    )
    return totals, per_clip, resolution


def image_path_for(dataset_split_root: str | Path, file_name: str) -> Path:
    """Resolve a COCO image filename against a split's dataset root."""
    return _resolve_image_path(Path(dataset_split_root), file_name)


def select_visual_examples(
    images: pd.DataFrame,
    target_annotations: pd.DataFrame,
    ignored_annotations: pd.DataFrame,
    count: int = 6,
) -> pd.DataFrame:
    """Select varied frames: sparse, dense, smallest/largest, and ignore-heavy."""
    density = target_annotations.groupby("image_id").size().rename("objects")
    ignore_count = ignored_annotations.groupby("image_id").size().rename("ignored")
    candidates = images.copy()
    candidates = candidates.merge(density, left_on="id", right_index=True, how="left")
    candidates = candidates.merge(ignore_count, left_on="id", right_index=True, how="left")
    candidates[["objects", "ignored"]] = candidates[["objects", "ignored"]].fillna(0)
    candidates["pixels"] = candidates["width"] * candidates["height"]
    chosen: list[Any] = []
    for column, ascending in (
        ("objects", True),
        ("objects", False),
        ("pixels", True),
        ("pixels", False),
        ("ignored", False),
    ):
        for image_id in candidates.sort_values(column, ascending=ascending)["id"]:
            if image_id not in chosen and (
                len(chosen) == 0
                or candidates.loc[candidates["id"].eq(image_id), "clip"].iloc[0]
                not in candidates.loc[candidates["id"].isin(chosen), "clip"].astype(str).tolist()
            ):
                chosen.append(image_id)
                break
    for image_id in candidates.sort_values(["ignored", "objects"], ascending=False)["id"]:
        if len(chosen) >= count:
            break
        if image_id not in chosen:
            chosen.append(image_id)
    return candidates.loc[candidates["id"].isin(chosen)].head(count).copy()


def add_box_dimensions(
    annotations: pd.DataFrame, target_only: bool = True
) -> pd.DataFrame:
    """Add pixel box dimensions, area, square-root area, aspect ratio, and class."""
    frame = annotations.copy()
    if target_only and "iscrowd" in frame:
        frame, _ = split_annotations(frame)
    if frame.empty:
        return frame.assign(box_w=pd.Series(dtype=float), box_h=pd.Series(dtype=float))
    boxes = np.asarray(frame["bbox"].tolist(), dtype=float)
    frame["box_w"] = boxes[:, 2]
    frame["box_h"] = boxes[:, 3]
    frame["area"] = frame["box_w"] * frame["box_h"]
    frame["sqrt_area"] = np.sqrt(frame["area"].clip(lower=0))
    frame["aspect_ratio"] = frame["box_w"] / frame["box_h"].replace(0, np.nan)
    if "category_id" in frame:
        frame["class_id"] = pd.to_numeric(frame["category_id"], errors="coerce") - 1
        frame["class_name"] = frame["class_id"].map(
            lambda value: CLASS_NAMES[int(value)]
            if pd.notna(value) and 0 <= int(value) < len(CLASS_NAMES)
            else "unknown"
        )
    return frame


def size_bucket_summary(annotations: pd.DataFrame) -> pd.DataFrame:
    """Count COCO small/medium/large boxes overall and by class."""
    frame = add_box_dimensions(annotations)
    frame["size_bucket"] = pd.cut(
        frame["area"],
        bins=[-np.inf, 32**2, 96**2, np.inf],
        labels=["small", "medium", "large"],
        right=False,
    )
    overall = frame.groupby("size_bucket", observed=False).size().rename("count").reset_index()
    per_class = (
        frame.groupby(["class_name", "size_bucket"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    overall.insert(0, "class_name", "all")
    return pd.concat([overall, per_class], ignore_index=True)


def resized_small_side_shares(
    annotations: pd.DataFrame, image_sizes: pd.DataFrame, image_sizes_list: Sequence[int]
) -> pd.DataFrame:
    """Share of boxes below 4/8/16 px on their smaller side after square letterboxing."""
    frame = add_box_dimensions(annotations)
    dims = image_sizes[["id", "width", "height"]].rename(columns={"id": "image_id"})
    frame = frame.merge(dims, on="image_id", how="left", suffixes=("", "_image"))
    records = []
    for imgsz in image_sizes_list:
        smaller = frame[["box_w", "box_h"]].min(axis=1) * (
            imgsz / frame[["width", "height"]].max(axis=1)
        )
        records.append(
            {
                "imgsz": imgsz,
                **{
                    f"below_{threshold}px_pct": 100 * float((smaller < threshold).mean())
                    for threshold in (4, 8, 16)
                },
                "objects": len(frame),
            }
        )
    return pd.DataFrame(records)


def kmeans_anchors(
    annotations: pd.DataFrame,
    image_sizes: pd.DataFrame,
    imgsz: int,
    k: int,
    seed: int = 42,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """YOLOv2-style IoU k-means on letterboxed pixel width/height pairs."""
    frame = add_box_dimensions(annotations)
    sizes = image_sizes[["id", "width", "height"]].rename(columns={"id": "image_id"})
    frame = frame.merge(sizes, on="image_id", how="left")
    scale = imgsz / frame[["width", "height"]].max(axis=1).to_numpy(dtype=float)
    boxes = frame[["box_w", "box_h"]].to_numpy(dtype=float) * scale[:, None]
    boxes = boxes[np.isfinite(boxes).all(axis=1) & (boxes > 0).all(axis=1)]
    if len(boxes) < k:
        raise ValueError(f"Need at least k={k} valid boxes; found {len(boxes)}")
    rng = np.random.default_rng(seed)
    anchors = boxes[rng.choice(len(boxes), size=k, replace=False)].copy()
    for _ in range(max_iterations):
        ious = _box_iou_wh(boxes, anchors)
        assignments = ious.argmax(axis=1)
        updated = anchors.copy()
        for cluster in range(k):
            members = boxes[assignments == cluster]
            if len(members):
                updated[cluster] = np.median(members, axis=0)
            else:
                updated[cluster] = boxes[rng.integers(len(boxes))]
        if np.allclose(updated, anchors):
            anchors = updated
            break
        anchors = updated
    mean_best_iou = float(_box_iou_wh(boxes, anchors).max(axis=1).mean())
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    return {
        "imgsz": imgsz,
        "k": k,
        "anchors": anchors.round(2).tolist(),
        "mean_best_iou": mean_best_iou,
        "boxes": int(len(boxes)),
    }


def _box_iou_wh(boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    intersection = np.minimum(boxes[:, None, 0], anchors[None, :, 0]) * np.minimum(
        boxes[:, None, 1], anchors[None, :, 1]
    )
    area_boxes = boxes[:, 0] * boxes[:, 1]
    area_anchors = anchors[:, 0] * anchors[:, 1]
    return intersection / np.maximum(
        area_boxes[:, None] + area_anchors[None, :] - intersection, 1e-12
    )


def density_summary(
    image_frames: pd.DataFrame, target_annotations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Return target objects per frame, per-clip counts, and key distribution stats."""
    counts = target_annotations.groupby("image_id").size().rename("objects")
    density = image_frames[["id", "clip"]].merge(
        counts, left_on="id", right_index=True, how="left"
    )
    density["objects"] = density["objects"].fillna(0).astype(int)
    per_clip = density.groupby("clip")["objects"].agg(["count", "median", "max"]).reset_index()
    stats = {
        "frames": int(len(density)),
        "p99_objects_per_frame": float(density["objects"].quantile(0.99)),
        "max_objects_per_frame": int(density["objects"].max()) if len(density) else 0,
    }
    return density, per_clip, stats


def ignored_region_summary(
    images: pd.DataFrame,
    targets: pd.DataFrame,
    ignored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize ignored-frame incidence, union area fraction, and target intersections."""
    target_rows = add_box_dimensions(targets, target_only=False)
    ignored_rows = add_box_dimensions(ignored, target_only=False)
    frame_records = []
    by_image = {key: group for key, group in ignored_rows.groupby("image_id")}
    for _, image in images.iterrows():
        group = by_image.get(image["id"], ignored_rows.iloc[0:0])
        rects = []
        for _, row in group.iterrows():
            x, y, width, height = map(float, row["bbox"])
            rects.append(
                (
                    max(0.0, x),
                    max(0.0, y),
                    min(float(image["width"]), x + width),
                    min(float(image["height"]), y + height),
                )
            )
        area = _rectangle_union_area(rects)
        frame_records.append(
            {
                "image_id": image["id"],
                "clip": image["clip"],
                "ignored_regions": len(group),
                "ignored_area_fraction": area / (float(image["width"]) * float(image["height"])),
            }
        )
    frames = pd.DataFrame(frame_records)
    overlaps = 0
    for image_id, igroup in ignored_rows.groupby("image_id"):
        tgroup = target_rows.loc[target_rows["image_id"].eq(image_id)]
        for _, target in tgroup.iterrows():
            x1, y1, w1, h1 = map(float, target["bbox"])
            for _, ignore in igroup.iterrows():
                x2, y2, w2, h2 = map(float, ignore["bbox"])
                if min(x1 + w1, x2 + w2) > max(x1, x2) and min(
                    y1 + h1, y2 + h2
                ) > max(y1, y2):
                    overlaps += 1
                    break
    stats = pd.DataFrame(
        [
            {
                "frames_with_ignored": int(frames["ignored_regions"].gt(0).sum()),
                "frames": int(len(frames)),
                "frames_with_ignored_pct": 100
                * float(frames["ignored_regions"].gt(0).mean()),
                "mean_ignored_area_pct": 100
                * float(frames["ignored_area_fraction"].mean()),
                "target_boxes_intersecting_ignored": overlaps,
                "target_boxes": len(target_rows),
            }
        ]
    )
    return frames.groupby("clip").agg(
        frames=("image_id", "size"),
        frames_with_ignored=("ignored_regions", lambda values: int(values.gt(0).sum())),
        mean_ignored_area_pct=("ignored_area_fraction", lambda values: 100 * values.mean()),
    ).reset_index(), stats


def _rectangle_union_area(rectangles: Iterable[tuple[float, float, float, float]]) -> float:
    """Calculate exact union area for axis-aligned rectangles using a sweep line."""
    rects = [rect for rect in rectangles if rect[2] > rect[0] and rect[3] > rect[1]]
    xs = sorted({x for rect in rects for x in (rect[0], rect[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (y1, y2) for x1, y1, x2, y2 in rects if x1 < right and x2 > left
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered += end - start
        area += (right - left) * covered
    return area


def temporal_redundancy(
    images: pd.DataFrame,
    targets: pd.DataFrame,
    steps: Sequence[int],
) -> pd.DataFrame:
    """Compute same-track IoU at frame lags and unseen-track share for each sampling step."""
    frame = add_box_dimensions(targets, target_only=False)
    if "track_id" not in frame:
        raise ValueError("Annotations do not include track_id")
    frame = frame.loc[frame["track_id"].notna()].copy()
    ordered_frames = images.sort_values(["clip", "frame_id", "file_name"]).copy()
    ordered_frames["frame_rank"] = ordered_frames.groupby("clip").cumcount()
    frame_ranks = ordered_frames.set_index("id")["frame_rank"]
    frame["frame_rank"] = frame["image_id"].map(frame_ranks)
    results = []
    for step in steps:
        ious: list[float] = []
        tracks_seen: set[tuple[str, Any]] = set()
        all_tracks: set[tuple[str, Any]] = set()
        for (clip, track_id), group in frame.groupby(["clip", "track_id"]):
            group = group.sort_values("frame_rank")
            all_tracks.add((str(clip), track_id))
            if (group["frame_rank"].astype(int) % step == 0).any():
                tracks_seen.add((str(clip), track_id))
            indexed = {
                int(row.frame_rank): np.asarray(row.bbox, dtype=float)
                for row in group.itertuples()
                if pd.notna(row.frame_rank)
            }
            for rank, box in indexed.items():
                next_box = indexed.get(rank + step)
                if next_box is not None:
                    ious.append(_xywh_iou(box, next_box))
        results.append(
            {
                "step": int(step),
                "mean_track_iou": float(np.mean(ious)) if ious else np.nan,
                "track_pairs": len(ious),
                "tracks_never_sampled_pct": 100
                * (len(all_tracks) - len(tracks_seen))
                / len(all_tracks)
                if all_tracks
                else np.nan,
                "tracks": len(all_tracks),
            }
        )
    return pd.DataFrame(results)


def _xywh_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    intersection = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(
        0.0, min(y1 + h1, y2 + h2) - max(y1, y2)
    )
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def make_group_folds(
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build deterministic GroupKFold-style folds with clips as indivisible groups."""
    clips = sorted(images["clip"].dropna().astype(str).unique())
    if len(clips) < n_splits:
        raise ValueError(f"Need at least {n_splits} clips; found {len(clips)}")
    grouped = images.copy()
    grouped["clip"] = grouped["clip"].astype(str)
    try:
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        assignments = splitter.split(grouped, groups=grouped["clip"])
        fold_clips = [
            sorted(grouped.iloc[val_indices]["clip"].unique().tolist())
            for _, val_indices in assignments
        ]
    except (ImportError, TypeError):
        rng = np.random.default_rng(seed)
        tie_order = {clip: rank for rank, clip in enumerate(rng.permutation(clips))}
        group_sizes = grouped.groupby("clip").size().to_dict()
        ordered = sorted(clips, key=lambda clip: (-group_sizes[clip], tie_order[clip]))
        fold_clips = [[] for _ in range(n_splits)]
        fold_sizes = [0] * n_splits
        for clip in ordered:
            fold = min(range(n_splits), key=lambda index: (fold_sizes[index], index))
            fold_clips[fold].append(clip)
            fold_sizes[fold] += group_sizes[clip]
    target = annotations.copy()
    if "iscrowd" in target:
        target, _ = split_annotations(target)
    target = add_box_dimensions(target, target_only=False)
    folds = []
    for index, validation_clips in enumerate(fold_clips):
        train_clips = [clip for clip in clips if clip not in validation_clips]
        val_annotations = target.loc[target["clip"].astype(str).isin(validation_clips)]
        folds.append(
            {
                "fold": index,
                "train_clips": sorted(train_clips),
                "val_clips": sorted(validation_clips),
                "train_frames": int(images["clip"].astype(str).isin(train_clips).sum()),
                "val_frames": int(images["clip"].astype(str).isin(validation_clips).sum()),
                "val_class_counts": {
                    str(int(key)): int(value)
                    for key, value in val_annotations.groupby("class_id").size().items()
                    if pd.notna(key)
                },
                "train_class_counts": {
                    str(int(key)): int(value)
                    for key, value in target.loc[
                        target["clip"].astype(str).isin(train_clips)
                    ].groupby("class_id").size().items()
                    if pd.notna(key)
                },
                "val_median_sqrt_area": float(val_annotations["sqrt_area"].median())
                if len(val_annotations)
                else None,
                "train_median_sqrt_area": float(
                    target.loc[target["clip"].astype(str).isin(train_clips), "sqrt_area"].median()
                )
                if target["clip"].astype(str).isin(train_clips).any()
                else None,
            }
        )
    return folds


def subsampled_frame_list(images: pd.DataFrame, step: int) -> list[dict[str, Any]]:
    """List every kth frame independently within each clip, starting at rank zero."""
    if step < 1:
        raise ValueError("step must be at least 1")
    ordered = images.sort_values(["clip", "frame_id", "file_name"]).copy()
    ordered["rank"] = ordered.groupby("clip").cumcount()
    selected = ordered.loc[ordered["rank"].mod(step).eq(0)]
    columns = [column for column in ("split", "clip", "frame_id", "file_name") if column in selected]
    return selected[columns].to_dict("records")


def save_json(data: Any, path: str | Path) -> None:
    """Write JSON after creating its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(data), indent=2, allow_nan=False), encoding="utf-8"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def visualize_frame(
    image_path: str | Path,
    target_annotations: pd.DataFrame,
    ignored_annotations: pd.DataFrame,
    class_names: Sequence[str] = CLASS_NAMES,
    ax: Any = None,
) -> Any:
    """Draw target boxes by class and ignored boxes as hatched overlays."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from PIL import Image

    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    colors = plt.get_cmap("tab10")
    for _, row in target_annotations.iterrows():
        x, y, width, height = map(float, row["bbox"])
        class_id = int(row["category_id"]) - 1
        color = colors(class_id % 10)
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=1.5))
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        ax.text(x, y, label, color="white", fontsize=7, backgroundcolor=color)
    for _, row in ignored_annotations.iterrows():
        x, y, width, height = map(float, row["bbox"])
        ax.add_patch(
            Rectangle(
                (x, y), width, height, facecolor="gray", alpha=0.25,
                edgecolor="black", hatch="////", linewidth=1
            )
        )
    handles = [
        Patch(facecolor=colors(index), label=name)
        for index, name in enumerate(class_names)
    ]
    handles.append(Patch(facecolor="gray", alpha=0.25, hatch="////", label="ignored region"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)
    ax.set_title(image_path.parent.name + " / " + image_path.name)
    ax.set_axis_off()
    return ax
