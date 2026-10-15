"""Run all generated Walk-Forward experiment configs sequentially."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="config/winrate_experiments")
    parser.add_argument("--output-root", default="results/walk_forward")
    parser.add_argument("--optimizer", choices=("numba", "sweep"), default="numba")
    parser.add_argument("--with-reports", action="store_true")
    args = parser.parse_args()

    config_paths = sorted(Path(args.config_dir).glob("config_walk_forward_*.yaml"))
    if not config_paths:
        raise SystemExit(f"no experiment configs found in {args.config_dir}")

    script = Path(__file__).with_name("run_walk_forward.py")
    for index, config_path in enumerate(config_paths, start=1):
        run_name = config_path.stem.replace("config_walk_forward_", "")
        command = [
            sys.executable,
            str(script),
            str(config_path),
            "--optimizer",
            args.optimizer,
            "--output-root",
            args.output_root,
            "--run-name",
            run_name,
        ]
        if not args.with_reports:
            command.extend(["--no-trade-review", "--no-curve-charts"])
        print(f"[{index}/{len(config_paths)}] starting {run_name}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"experiment failed: {run_name} (exit={completed.returncode})")
        print(f"[{index}/{len(config_paths)}] finished {run_name}", flush=True)


if __name__ == "__main__":
    main()
