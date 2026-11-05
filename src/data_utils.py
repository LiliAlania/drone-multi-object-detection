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

# Pixel precision used to decide that two ignored regions are the same box.
IGNORE_DEDUP_NDIGITS = 2


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def load_coco(
    coco_path: str | Path,
    split: str,
    dedup_ignored: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load COCO metadata as image and annotation frames with clip/frame dimensions.

    In the VisDrone-VID COCO export every ignored region (iscrowd=1) is repeated
    several times per frame (orig_category_id 0 "ignored region" and 11 "others"
    share the same boxes), while the YOLO ``ignore/*.txt`` files hold each region
    once. With ``dedup_ignored=True`` (default) these duplicates are dropped; the
    number removed is stored in ``annotations.attrs["ignored_duplicates_removed"]``.
    """
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
    for column in (
        "iscrowd", "category_id", "track_id", "orig_category_id", "occlusion", "truncation"
    ):
        if column not in annotations:
            annotations[column] = 0 if column == "iscrowd" else np.nan
    annotations["iscrowd"] = (
        pd.to_numeric(annotations["iscrowd"], errors="coerce").fillna(0).astype(int)
    )

    removed = 0
    if dedup_ignored:
        annotations, removed = dedup_ignored_regions(annotations)

    metadata = images[["id", "clip", "frame_id", "width", "height", "split"]].rename(
        columns={"id": "image_id"}
    )
    annotations = annotations.merge(metadata, on="image_id", how="left")
    annotations.attrs["ignored_duplicates_removed"] = removed
    return images, annotations


def dedup_ignored_regions(
    annotations: pd.DataFrame, ndigits: int = IGNORE_DEDUP_NDIGITS
) -> tuple[pd.DataFrame, int]:
    """Keep one ignored region (iscrowd=1) per (image_id, bbox); targets are untouched.

    Returns the filtered frame (with a fresh index) and the number of rows removed.
    """
    if annotations.empty or "bbox" not in annotations:
        return annotations.reset_index(drop=True), 0
    crowd = _crowd_mask(annotations).to_numpy()
    if not crowd.any():
        return annotations.reset_index(drop=True), 0
    keys = [
        (image_id, tuple(round(float(value), ndigits) for value in bbox))
        for image_id, bbox in zip(
            annotations["image_id"].to_numpy()[crowd],
            annotations["bbox"].to_numpy()[crowd],
        )
    ]
    duplicate = np.zeros(len(annotations), dtype=bool)
    duplicate[np.flatnonzero(crowd)] = pd.Series(keys).duplicated().to_numpy()
    return annotations.loc[~duplicate].reset_index(drop=True), int(duplicate.sum())


def _crowd_mask(annotations: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(annotations["iscrowd"], errors="coerce").fillna(0).eq(1)


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
    crowd = _crowd_mask(annotations)
    return annotations.loc[~crowd].copy(), annotations.loc[crowd].copy()


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


# ---------------------------------------------------------------------------
# COCO <-> YOLO consistency
# ---------------------------------------------------------------------------

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
    """Assert sampled YOLO target/ignore files match their COCO normalized boxes.

    Ignored regions are deduplicated in COCO before comparison (see ``load_coco``).
    Rows are matched one-to-one within ``atol`` instead of relying on sort order,
    so near-equal coordinates cannot be paired with the wrong box.
    """
    images, annotations = load_coco(coco_path, split, dedup_ignored=True)
    by_image = {int(key): group for key, group in annotations.groupby("image_id")}
    image_lookup = images.set_index("id")
    rng = random.Random(seed)
    candidates = sorted(by_image)
    selected = rng.sample(candidates, min(sample_size, len(candidates)))
    compared_target = compared_ignore = 0
    root = Path(dataset_root)
    for image_id in selected:
        image = image_lookup.loc[image_id]
        width, height = float(image["width"]), float(image["height"])
        base = _resolve_image_path(root, str(image["file_name"]))
        rows = by_image[image_id]
        crowd_mask = _crowd_mask(rows)
        for crowd, folder in ((0, "labels"), (1, "ignore")):
            path = _label_path(base, folder)
            expected = rows.loc[crowd_mask.eq(bool(crowd))]
            actual_rows = _read_yolo_rows(path, has_class=(crowd == 0))
            expected_rows: list[tuple[float, ...]] = []
            for row in expected.itertuples(index=False):
                coords = coco_xywh_to_yolo(row.bbox, width, height)
                if crowd == 0:
                    expected_rows.append((float(int(row.category_id) - 1), *coords))
                else:
                    expected_rows.append(coords)
            assert len(actual_rows) == len(expected_rows), (
                f"Annotation count mismatch for {path}: "
                f"YOLO={len(actual_rows)}, COCO={len(expected_rows)}"
            )
            unmatched = _unmatched_rows(actual_rows, expected_rows, atol)
            assert not unmatched, (
                f"Box mismatch in {path}: COCO rows without a YOLO match "
                f"(first 3): {unmatched[:3]}"
            )
            if crowd == 0:
                compared_target += len(expected_rows)
            else:
                compared_ignore += len(expected_rows)
    return {
        "sampled_frames": len(selected),
        "target_boxes_compared": compared_target,
        "ignored_regions_compared": compared_ignore,
        "ignored_duplicates_removed": int(
            annotations.attrs.get("ignored_duplicates_removed", 0)
        ),
    }


def _unmatched_rows(
    actual: Sequence[tuple[float, ...]],
    expected: Sequence[tuple[float, ...]],
    atol: float,
) -> list[tuple[float, ...]]:
    """Greedy one-to-one matching; returns expected rows with no actual counterpart."""
    remaining = [np.asarray(row, dtype=float) for row in actual]
    unmatched = []
    for row in expected:
        target = np.asarray(row, dtype=float)
        for index, candidate in enumerate(remaining):
            if np.allclose(candidate, target, atol=atol, rtol=0):
                remaining.pop(index)
                break
        else:
            unmatched.append(tuple(row))
    return unmatched


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
        # Replace the last "images" component so parent folders named "images" are safe.
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = folder
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / folder / image_path.parent.name / (
        image_path.stem + ".txt"
    )


def _read_yolo_rows(path: Path, has_class: bool) -> list[tuple[float, ...]]:
    if not path.exists():
        return []
    rows = []
    expected_length = 5 if has_class else 4
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != expected_length:
            raise ValueError(f"Expected {expected_length} values per row in {path}")
        rows.append(tuple(values))
    return rows


# ---------------------------------------------------------------------------
# Dataset overview and example selection
# ---------------------------------------------------------------------------

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
    images["resolution"] = (
        pd.to_numeric(images["width"]).round().astype("Int64").astype(str)
        + "x"
        + pd.to_numeric(images["height"]).round().astype("Int64").astype(str)
    )
    per_clip = (
        images.groupby(["split", "clip"], dropna=False)
        .agg(
            frames=("file_name", "nunique"),
            resolution=("resolution", lambda values: ", ".join(sorted(values.unique()))),
        )
        .reset_index()
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
    """Select varied frames: sparse, dense, smallest/largest, and ignore-heavy.

    The first picks come from different clips; remaining slots are filled with the
    most ignore-heavy, densest frames. Rows are returned in selection order.
    """
    density = target_annotations.groupby("image_id").size().rename("objects")
    ignore_count = ignored_annotations.groupby("image_id").size().rename("ignored")
    candidates = images.copy()
    candidates = candidates.merge(density, left_on="id", right_index=True, how="left")
    candidates = candidates.merge(ignore_count, left_on="id", right_index=True, how="left")
    candidates[["objects", "ignored"]] = candidates[["objects", "ignored"]].fillna(0)
    candidates["pixels"] = candidates["width"] * candidates["height"]
    chosen: list[Any] = []
    used_clips: set[str] = set()
    for column, ascending in (
        ("objects", True),
        ("objects", False),
        ("pixels", True),
        ("pixels", False),
        ("ignored", False),
    ):
        ordered = candidates.sort_values(column, ascending=ascending, kind="mergesort")
        for image_id, clip in zip(ordered["id"], ordered["clip"].astype(str)):
            if image_id not in chosen and clip not in used_clips:
                chosen.append(image_id)
                used_clips.add(clip)
                break
    ordered = candidates.sort_values(["ignored", "objects"], ascending=False, kind="mergesort")
    for image_id in ordered["id"]:
        if len(chosen) >= count:
            break
        if image_id not in chosen:
            chosen.append(image_id)
    return candidates.set_index("id", drop=False).loc[chosen[:count]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Box geometry, sizes, anchors
# ---------------------------------------------------------------------------

_BOX_COLUMNS = ("box_w", "box_h", "area", "sqrt_area", "aspect_ratio")


def add_box_dimensions(
    annotations: pd.DataFrame, target_only: bool = True
) -> pd.DataFrame:
    """Add pixel box dimensions, area, square-root area, aspect ratio, and class."""
    frame = annotations.copy()
    if target_only and "iscrowd" in frame:
        frame, _ = split_annotations(frame)
    if frame.empty:
        return frame.assign(
            **{column: pd.Series(dtype=float) for column in _BOX_COLUMNS},
            class_id=pd.Series(dtype=float),
            class_name=pd.Series(dtype=object),
        )
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


def _attach_image_size(frame: pd.DataFrame, image_sizes: pd.DataFrame) -> pd.DataFrame:
    """Attach image width/height by image_id, replacing any existing columns.

    ``load_coco`` already merges width/height into annotations; merging again
    without dropping them would create width_x/width_y columns.
    """
    dims = image_sizes[["id", "width", "height"]].rename(columns={"id": "image_id"})
    frame = frame.drop(columns=[column for column in ("width", "height") if column in frame])
    return frame.merge(dims, on="image_id", how="left")


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
    frame = _attach_image_size(add_box_dimensions(annotations), image_sizes)
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
    frame = _attach_image_size(add_box_dimensions(annotations), image_sizes)
    scale = imgsz / frame[["width", "height"]].max(axis=1).to_numpy(dtype=float)
    boxes = frame[["box_w", "box_h"]].to_numpy(dtype=float) * scale[:, None]
    boxes = boxes[np.isfinite(boxes).all(axis=1) & (boxes > 0).all(axis=1)]
    if len(boxes) < k:
        raise ValueError(f"Need at least k={k} valid boxes; found {len(boxes)}")
    rng = np.random.default_rng(seed)
    anchors = boxes[rng.choice(len(boxes), size=k, replace=False)].copy()
    for _ in range(max_iterations):
        assignments = _box_iou_wh(boxes, anchors).argmax(axis=1)
        updated = anchors.copy()
        for cluster in range(k):
            members = boxes[assignments == cluster]
            if len(members):
                updated[cluster] = np.median(members, axis=0)
            else:
                updated[cluster] = boxes[rng.integers(len(boxes))]
        converged = np.allclose(updated, anchors)
        anchors = updated
        if converged:
            break
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


# ---------------------------------------------------------------------------
# Density and ignored regions
# ---------------------------------------------------------------------------

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
        "p99_objects_per_frame": float(density["objects"].quantile(0.99)) if len(density) else 0.0,
        "max_objects_per_frame": int(density["objects"].max()) if len(density) else 0,
    }
    return density, per_clip, stats


def ignored_region_summary(
    images: pd.DataFrame,
    targets: pd.DataFrame,
    ignored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize ignored-frame incidence, union area fraction, and target intersections.

    Returns ``(per_clip, stats)``. Pass deduplicated ignored regions (the default
    from ``load_coco``), otherwise region counts are inflated.
    """
    ignored_boxes = {
        image_id: [tuple(map(float, bbox)) for bbox in group["bbox"]]
        for image_id, group in ignored.groupby("image_id")
    }
    frame_records = []
    for image_id, clip, width, height in images[["id", "clip", "width", "height"]].itertuples(
        index=False
    ):
        width, height = float(width), float(height)
        boxes = ignored_boxes.get(image_id, [])
        rects = [
            (max(0.0, x), max(0.0, y), min(width, x + w), min(height, y + h))
            for x, y, w, h in boxes
        ]
        frame_records.append(
            {
                "image_id": image_id,
                "clip": clip,
                "ignored_regions": len(boxes),
                "ignored_area_fraction": _rectangle_union_area(rects) / (width * height),
            }
        )
    frames = pd.DataFrame(
        frame_records,
        columns=["image_id", "clip", "ignored_regions", "ignored_area_fraction"],
    )

    overlaps = 0
    for image_id, tgroup in targets.groupby("image_id"):
        iboxes = ignored_boxes.get(image_id)
        if not iboxes:
            continue
        for x1, y1, w1, h1 in (tuple(map(float, bbox)) for bbox in tgroup["bbox"]):
            if any(
                min(x1 + w1, x2 + w2) > max(x1, x2) and min(y1 + h1, y2 + h2) > max(y1, y2)
                for x2, y2, w2, h2 in iboxes
            ):
                overlaps += 1

    stats = pd.DataFrame(
        [
            {
                "frames_with_ignored": int(frames["ignored_regions"].gt(0).sum()),
                "frames": int(len(frames)),
                "frames_with_ignored_pct": 100 * float(frames["ignored_regions"].gt(0).mean()),
                "ignored_regions": int(frames["ignored_regions"].sum()),
                "mean_ignored_area_pct": 100 * float(frames["ignored_area_fraction"].mean()),
                "target_boxes_intersecting_ignored": overlaps,
                "target_boxes": int(len(targets)),
            }
        ]
    )
    per_clip = frames.groupby("clip").agg(
        frames=("image_id", "size"),
        frames_with_ignored=("ignored_regions", lambda values: int(values.gt(0).sum())),
        mean_ignored_area_pct=("ignored_area_fraction", lambda values: 100 * values.mean()),
    ).reset_index()
    return per_clip, stats


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


# ---------------------------------------------------------------------------
# Temporal structure, folds, subsampling
# ---------------------------------------------------------------------------

def temporal_redundancy(
    images: pd.DataFrame,
    targets: pd.DataFrame,
    steps: Sequence[int],
) -> pd.DataFrame:
    """Compute same-track IoU at frame lags and unseen-track share for each sampling step."""
    if "track_id" not in targets:
        raise ValueError("Annotations do not include track_id")
    frame = targets.loc[targets["track_id"].notna()].copy()
    ordered_frames = images.sort_values(["clip", "frame_id", "file_name"]).copy()
    ordered_frames["frame_rank"] = ordered_frames.groupby("clip").cumcount()
    frame_ranks = ordered_frames.set_index("id")["frame_rank"]
    frame_clips = ordered_frames.set_index("id")["clip"].astype(str)
    frame["frame_rank"] = frame["image_id"].map(frame_ranks)
    frame["clip"] = frame["image_id"].map(frame_clips)
    frame = frame.loc[frame["frame_rank"].notna()]

    tracks = []
    for (clip, track_id), group in frame.groupby(["clip", "track_id"]):
        indexed = {
            int(rank): np.asarray(bbox, dtype=float)
            for rank, bbox in zip(group["frame_rank"], group["bbox"])
        }
        tracks.append(((clip, track_id), indexed))

    results = []
    for step in steps:
        ious: list[float] = []
        never_sampled = 0
        for _, indexed in tracks:
            if not any(rank % step == 0 for rank in indexed):
                never_sampled += 1
            for rank, box in indexed.items():
                next_box = indexed.get(rank + step)
                if next_box is not None:
                    ious.append(_xywh_iou(box, next_box))
        results.append(
            {
                "step": int(step),
                "mean_track_iou": float(np.mean(ious)) if ious else np.nan,
                "track_pairs": len(ious),
                "tracks_never_sampled_pct": 100 * never_sampled / len(tracks)
                if tracks
                else np.nan,
                "tracks": len(tracks),
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
    grouped = images.loc[images["clip"].notna()].copy()
    grouped["clip"] = grouped["clip"].astype(str)
    clips = sorted(grouped["clip"].unique())
    if len(clips) < n_splits:
        raise ValueError(f"Need at least {n_splits} clips; found {len(clips)}")
    try:
        from sklearn.model_selection import GroupKFold

        # shuffle/random_state exist only in scikit-learn >= 1.6.
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_clips = [
            sorted(grouped.iloc[val_indices]["clip"].unique().tolist())
            for _, val_indices in splitter.split(grouped, groups=grouped["clip"])
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
        fold_clips = [sorted(clip_list) for clip_list in fold_clips]

    target = annotations.copy()
    if "iscrowd" in target:
        target, _ = split_annotations(target)
    target = add_box_dimensions(target, target_only=False)
    target_clip = target["clip"].astype(str)
    image_clip = images["clip"].astype(str)

    def class_counts(frame: pd.DataFrame) -> dict[str, int]:
        return {
            str(int(key)): int(value)
            for key, value in frame.groupby("class_id").size().items()
            if pd.notna(key)
        }

    def median_sqrt_area(frame: pd.DataFrame) -> float | None:
        return float(frame["sqrt_area"].median()) if len(frame) else None

    folds = []
    for index, validation_clips in enumerate(fold_clips):
        val_set = set(validation_clips)
        train_clips = [clip for clip in clips if clip not in val_set]
        train_annotations = target.loc[target_clip.isin(train_clips)]
        val_annotations = target.loc[target_clip.isin(val_set)]
        folds.append(
            {
                "fold": index,
                "train_clips": train_clips,
                "val_clips": sorted(validation_clips),
                "train_frames": int(image_clip.isin(train_clips).sum()),
                "val_frames": int(image_clip.isin(val_set).sum()),
                "val_class_counts": class_counts(val_annotations),
                "train_class_counts": class_counts(train_annotations),
                "val_median_sqrt_area": median_sqrt_area(val_annotations),
                "train_median_sqrt_area": median_sqrt_area(train_annotations),
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


# ---------------------------------------------------------------------------
# I/O and visualization
# ---------------------------------------------------------------------------

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
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict("records"))
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA or value is pd.NaT:
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
    for bbox, category_id in zip(target_annotations["bbox"], target_annotations["category_id"]):
        x, y, width, height = map(float, bbox)
        class_id = int(category_id) - 1
        color = colors(class_id % 10)
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=1.5))
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        ax.text(x, y, label, color="white", fontsize=7, backgroundcolor=color)
    for bbox in ignored_annotations["bbox"]:
        x, y, width, height = map(float, bbox)
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