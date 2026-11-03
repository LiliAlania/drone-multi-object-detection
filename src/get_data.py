"""Download and extract the Kaggle competition dataset using the Kaggle CLI."""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename}")
        source.extractall(destination)


def download_data(
    output_dir: str | Path,
    competition: str = "multiple-object-video-detection-hw-2-ucu-ai-2026",
    download_dir: str | Path | None = None,
) -> None:
    """Download via Kaggle CLI and recursively unpack nested competition archives."""
    output_dir = Path(output_dir)
    download_dir = Path(download_dir) if download_dir else output_dir.parent / "kaggle_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition, "-p", str(download_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "Kaggle CLI is missing. Install it with `pip install kaggle`, configure "
            "credentials, and accept the competition rules."
        ) from error
    if result.returncode:
        raise RuntimeError(
            "Kaggle download failed. Install/configure Kaggle credentials and accept the "
            f"competition rules first.\n{(result.stderr or result.stdout).strip()}"
        )
    archives = list(download_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"Kaggle produced no zip archive in {download_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = list(archives)
    extracted: set[Path] = set()
    while pending:
        archive = pending.pop(0)
        if archive in extracted:
            continue
        _safe_extract(archive, output_dir)
        extracted.add(archive)
        nested = [path for path in output_dir.rglob("*.zip") if path not in extracted]
        pending.extend(nested)
    for archive in extracted:
        if archive.is_relative_to(download_dir):
            continue
        archive.unlink()
    print(f"Dataset archives extracted under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/data"))
    parser.add_argument(
        "--competition", default="multiple-object-video-detection-hw-2-ucu-ai-2026"
    )
    parser.add_argument("--download-dir", type=Path)
    args = parser.parse_args()
    download_data(args.output_dir, args.competition, args.download_dir)


if __name__ == "__main__":
    main()
