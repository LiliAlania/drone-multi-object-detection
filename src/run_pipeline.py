"""Run the whole ensemble pipeline in order: tiles -> train both -> SAHI val/test -> WBF -> submission.

    python -m src.run_pipeline --debug          # smoke test (minutes)
    python -m src.run_pipeline --from infer     # e.g. skip tiling/training after a full training run
Each step is the module's own CLI, so any step can also be rerun alone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS = {
    "tiles": [["src.prepare_tiles", "--force"]],
    "train": [["src.train_yolo_p2"], ["src.train_frcnn_tiles"]],
    "infer": [["src.infer_sahi", "--model", m, "--split", s] for m in ("yolo", "frcnn") for s in ("val", "test")],
    "ensemble": [["src.ensemble_wbf", "search"], ["src.ensemble_wbf", "apply", "--split", "test"]],
    "submit": [["src.make_submission"]],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--from", dest="start", choices=list(STEPS), default="tiles")
    args = parser.parse_args()
    extra = ["--config", args.config] + (["--debug"] if args.debug else [])
    names = list(STEPS)
    for name in names[names.index(args.start):]:
        for module, *rest in STEPS[name]:
            tick = time.perf_counter()
            print(f"\n=== {module} {' '.join(rest)} ===", flush=True)
            subprocess.run([sys.executable, "-m", module, *rest, *extra], check=True)
            print(f"=== done in {(time.perf_counter() - tick) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
