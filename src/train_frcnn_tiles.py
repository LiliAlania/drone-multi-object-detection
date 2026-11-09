"""Train Faster R-CNN (ResNet-50 FPN v2, COCO pretrain) on the same tiles as YOLO.

Model from src.faster_rcnn.build_model: anchors 8..128 px for tiny objects, 2000 RPN proposals,
500 detections per image, new 8-class (+ background) head. Tiles are fed at their own 640 px.
AMP, gradient accumulation, linear warm-up + step decay (x0.1 at 2/3 and 8/9 of epochs), val mAP on the val tiles every N epochs.

    python -m src.train_frcnn_tiles [--debug] [--resume]
Checkpoints: <runs_dir>/frcnn/{last,best}.pt (loadable by src.faster_rcnn.load_trained); losses in log.csv.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.pipeline_common import DEFAULT_CONFIG, load_config, seed_everything, tiles_dir

TILE_SIZE = 640


def validate(model, device, image_dir: Path, batch: int) -> dict[str, float]:
    """COCO metrics of the model on the val tiles (ground truth from their YOLO labels)."""
    import contextlib
    import io

    import torch

    from src.evaluate import coco_from_yolo_tiles, evaluate_detections
    from src.faster_rcnn import _read_images

    coco, paths = coco_from_yolo_tiles(image_dir)
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            for image_id, out in zip(range(start + 1, start + 1 + len(chunk)), model(_read_images(chunk, device))):
                rows += [(str(image_id), int(c) - 1, float(s), *b)
                         for b, s, c in zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist())]
    model.train()
    import pandas as pd

    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools is verbose
        return evaluate_detections(coco, pd.DataFrame(rows, columns=["image_id", "cls", "score", "x1", "y1", "x2", "y2"]))


def train(cfg: dict, resume: bool = False) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from src.faster_rcnn import YoloDetectionDataset, _collate, build_model

    f, p = cfg["frcnn"], cfg["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tiles = tiles_dir(cfg)
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(YoloDetectionDataset(tiles / "images" / "train", train=True), batch_size=p["frcnn_batch"],
                        shuffle=True, num_workers=p["workers"], collate_fn=_collate, generator=generator,
                        persistent_workers=p["workers"] > 0)
    model = build_model(TILE_SIZE, TILE_SIZE).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.SGD(params, lr=f["lr"], momentum=0.9, weight_decay=1e-4)
    epochs = f["epochs"]
    milestones = [int(epochs * 2 / 3), int(epochs * 8 / 9)]  # "1x" shape: x0.1 at 2/3 and 8/9 of training
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    out = p["runs_dir"] / "frcnn"
    out.mkdir(parents=True, exist_ok=True)
    last, best = out / "last.pt", out / "best.pt"
    start_epoch, step, best_map = 0, 0, -1.0
    if resume:
        if not last.is_file():
            raise FileNotFoundError(f"--resume given but no checkpoint at {last}")
        state = torch.load(last, map_location=device, weights_only=False)  # own file; holds optimizer state
        if state["epochs"] != epochs:
            raise ValueError(f"Checkpoint was started with {state['epochs']} epochs; config has {epochs}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step, best_map = state["epoch"], state["step"], state["best_map"]
        print(f"Resuming after epoch {start_epoch}")

    log = (out / "log.csv").open("a", newline="", encoding="utf-8")
    writer = csv.writer(log)
    if not resume:
        writer.writerow(["epoch", "iter", "lr", "loss", "loss_classifier", "loss_box_reg", "loss_objectness",
                         "loss_rpn_box_reg", "val_mAP", "val_AP50", "val_AP_small"])
    accumulate = f["accumulate"]
    for epoch in range(start_epoch, epochs):
        tick = time.perf_counter()
        sums: dict[str, float] = {}
        optimizer.zero_grad()
        for i, (images, targets) in enumerate(loader, 1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1} iter {i}; lower frcnn.lr")
            scaler.scale(loss / accumulate).backward()
            for k, v in {"loss": loss, **losses}.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            if i % accumulate == 0 or i == len(loader):
                warmup = min(1.0, (step + 1) / f["warmup_iters"])  # linear warm-up over optimizer steps
                for group in optimizer.param_groups:
                    group["lr"] = f["lr"] * 0.1 ** sum(0 < m <= epoch for m in milestones) * warmup  # short runs: no decay
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
            if i % 50 == 0 or i == len(loader):
                print(f"epoch {epoch + 1}/{epochs} iter {i}/{len(loader)} loss {sums['loss'] / i:.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.5f}", end="\r")
        minutes = (time.perf_counter() - tick) / 60
        metrics = {}
        if (epoch + 1) % f["val_every"] == 0 or epoch + 1 == epochs:
            metrics = validate(model, device, tiles / "images" / "val", p["frcnn_batch"])
            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
        n = len(loader)
        writer.writerow([epoch + 1, n, optimizer.param_groups[0]["lr"], *(round(sums[k] / n, 5) for k in (
            "loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")),
            *(round(metrics.get(k, float("nan")), 5) for k in ("mAP", "AP50", "AP_small"))])
        log.flush()
        state = {"model": model.state_dict(), "min_size": TILE_SIZE, "max_size": TILE_SIZE,
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch + 1, "epochs": epochs, "step": step,
                 "best_map": best_map}
        torch.save(state, last.with_suffix(".tmp"))  # write then rename: a disconnect can't corrupt last.pt
        last.with_suffix(".tmp").replace(last)
        if metrics and metrics["mAP"] >= best_map:
            torch.save({"model": state["model"], "min_size": TILE_SIZE, "max_size": TILE_SIZE}, best)
        memory = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(f"epoch {epoch + 1}/{epochs}: loss {sums['loss'] / n:.4f}, {minutes:.1f} min train, "
              f"{n * p['frcnn_batch'] / (minutes * 60):.1f} tiles/s, peak GPU {memory:.1f} GB"
              + (f", val mAP {metrics['mAP']:.4f} AP50 {metrics['AP50']:.4f} AP_small {metrics['AP_small']:.4f}"
                 if metrics else ""))
    log.close()
    return best if best.is_file() else last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue from <runs_dir>/frcnn/last.pt")
    args = parser.parse_args()
    cfg = load_config(args.config, args.debug)
    seed_everything(cfg["seed"])
    print(f"Best weights: {train(cfg, args.resume)}")


if __name__ == "__main__":
    main()
