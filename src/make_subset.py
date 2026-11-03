"""Create a frame-subsampled YOLO dataset and zip it for Colab reuse."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Sequence

from src.data_utils import CLASS_NAMES, image_path_for, load_coco, subsampled_frame_list


def build_subset(
    data_root: str | Path,
    output_dir: str | Path,
    step: int,
    splits: Sequence[str] = ("train", "val"),
    zip_path: str | Path | None = None,
    drive_copy: str | Path | None = None,
) -> Path:
    """Copy every kth frame in each clip to YOLO folders and create a zip archive."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    if step < 1:
        raise ValueError("step must be at least 1")
    invalid = set(splits) - {"train", "val"}
    if invalid:
        raise ValueError(f"Only labeled splits are supported: {sorted(invalid)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_total = 0
    for split in splits:
        split_root = data_root / split
        coco_path = split_root / "annotations_coco.json"
        if not coco_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotations: {coco_path}")
        images, _ = load_coco(coco_path, split)
        selected = subsampled_frame_list(images, step)
        for item in selected:
            source_image = image_path_for(split_root, item["file_name"])
            if not source_image.is_file():
                raise FileNotFoundError(f"Missing source image: {source_image}")
            clip = str(item["clip"])
            image_dest = output_dir / "images" / split / clip / source_image.name
            image_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, image_dest)
            for source_folder, destination_folder, required in (
                ("labels", "labels", True),
                ("ignore", "ignore", False),
            ):
                source_label = _annotation_path(split_root, item["file_name"], source_folder)
                destination = (
                    output_dir
                    / destination_folder
                    / split
                    / clip
                    / f"{source_image.stem}.txt"
                )
                if source_label.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_label, destination)
                elif required:
                    raise FileNotFoundError(f"Missing YOLO labels: {source_label}")
            frame_total += 1
    dataset_yaml = {
        "path": ".",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    if "train" in splits:
        dataset_yaml["train"] = "images/train"
    if "val" in splits:
        dataset_yaml["val"] = "images/val"
    yaml_lines = ["path: ."]
    yaml_lines.extend(
        f"{key}: {value}" for key, value in dataset_yaml.items() if key != "path" and key != "names"
    )
    yaml_lines.append("names:")
    yaml_lines.extend(f"  {index}: {name}" for index, name in dataset_yaml["names"].items())
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    archive_path = Path(zip_path) if zip_path else output_dir.with_suffix(".zip")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent))
    if drive_copy:
        drive_path = Path(drive_copy)
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive_path, drive_path)
    print(f"Created {frame_total} frames; archive: {archive_path}")
    return archive_path


def _annotation_path(split_root: Path, file_name: str, folder: str) -> Path:
    parts = list(Path(file_name.replace("\\", "/")).parts)
    if "images" in parts:
        parts[parts.index("images")] = folder
        return split_root.joinpath(*parts).with_suffix(".txt")
    return split_root / folder / Path(file_name).parent / (Path(file_name).stem + ".txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--drive-copy", type=Path)
    args = parser.parse_args()
    build_subset(
        args.data_root,
        args.output_dir,
        args.step,
        args.splits,
        args.zip_path,
        args.drive_copy,
    )


if __name__ == "__main__":
    main()
