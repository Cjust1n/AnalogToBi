#!/usr/bin/env python3
"""Evaluate MCTS batch outputs with the op-amp SPICE pipeline.

The MCTS batch folders store a single sequence as `CIRCUIT_Opamp.txt`, while
`opamp_spice_pipeline.py` expects `run*.txt`. This adapter bridges the two
formats by staging each batch output into a temporary `run0.txt` input folder
and then invoking the existing SPICE pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def evaluate_batch(batch_dir: Path, output_dir: Path, timeout: float, ngspice_bin: str) -> dict:
    batch_dir = batch_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    staged_input = output_dir / "_staged_input"
    staged_input.mkdir(parents=True, exist_ok=True)
    for child in staged_input.iterdir():
        if child.is_file():
            child.unlink()

    candidates = sorted(batch_dir.glob("*.txt"))
    if not candidates:
        return {"batch_dir": str(batch_dir), "status": "no_txt_files"}

    source = candidates[0]
    shutil.copy2(source, staged_input / "run0.txt")

    pipeline = Path(__file__).resolve().parent.parent / "graph-to-netlist" / "opamp_spice_pipeline.py"
    cmd = [
        "python3",
        str(pipeline),
        "--input-dir",
        str(staged_input),
        "--output-dir",
        str(output_dir),
        "--limit",
        "1",
        "--timeout",
        str(timeout),
        "--ngspice-bin",
        ngspice_bin,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    summary = {
        "batch_dir": str(batch_dir),
        "source": str(source),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "output_dir": str(output_dir),
        "metrics_csv": str(output_dir / "opamp_metrics.csv"),
    }
    summary_path = output_dir / "mcts_spice_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MCTS batch outputs with SPICE.")
    parser.add_argument("batch_dirs", nargs="+", type=Path, help="Batch output directories from MCTS.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ngspice-bin", default="ngspice")
    parser.add_argument("--suffix", default="_spice")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    results = []
    for batch_dir in args.batch_dirs:
        output_dir = batch_dir.with_name(batch_dir.name + args.suffix)
        result = evaluate_batch(batch_dir, output_dir, args.timeout, args.ngspice_bin)
        results.append(result)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
