#!/usr/bin/env python3
"""
EXPLORE Benchmark & Comparative Evaluation for Op-Amp Topology Generation.
Compares four decoding methods as outlined in Table 1 & Figure 4 of EXPLORE:
1. Greedy (One-shot Baseline AnalogToBi)
2. Sampling + Filtering (S+F, k=3, temperature=1.0)
3. MCTS-Base (MCTS with uniform child selection, no LM prior guidance)
4. EXPLORE (P-UCB with LM prior + p-filtering p=0.99 + constrained rollout)

Outputs:
- Metric summary table (ERC validity %, SPICE success %, Average Av0, UGB, PM)
- JSON results saved in benchmark output directory
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

import MCTS_Inference_Grammar as explore_mod
from ERC import run_rule_validation


def evaluate_generated_circuit(seq_tokens: List[str], metrics_cache: Dict[str, Dict[str, float]], cache_path: Optional[Path], workdir: Path) -> Dict[str, Any]:
    clean_tokens = [t for t in seq_tokens if t != "TRUNCATE"]
    is_clean, v1, v2, v3, v4 = run_rule_validation(clean_tokens, verbose=False, debug=False)
    ports_ok = explore_mod.check_opamp_ports_present(clean_tokens)

    spice_metrics: Dict[str, float] = {}
    sim_ok = False

    if is_clean and ports_ok:
        spice_metrics = explore_mod.evaluate_spice_sequence(
            clean_tokens,
            metrics_cache,
            cache_path=cache_path,
            workdir=workdir,
            timeout=300,
        )
        sim_ok = bool(spice_metrics)

    return {
        "sequence": clean_tokens,
        "is_erc_clean": is_clean,
        "ports_ok": ports_ok,
        "sim_ok": sim_ok,
        "spice_metrics": spice_metrics,
    }


def run_greedy(lm_model: Any, circuit_type: str, max_len: int, device_str: str, progress: Optional[tqdm] = None) -> List[str]:
    seq = [circuit_type, "VSS"]
    if progress is not None:
        progress.update(len(seq))
        progress.set_postfix_str("last=VSS")
    while len(seq) < max_len:
        allowed = explore_mod.get_allowed_tokens_6state(seq, circuit_type, max_len)
        if not allowed:
            break
        if "TRUNCATE" in allowed:
            seq.append("TRUNCATE")
            break

        probs = explore_mod.lm_next_token_distribution(lm_model, seq, device_str)
        allowed_with_prob = [(tok, probs.get(tok, 1e-8)) for tok in allowed]
        allowed_with_prob.sort(key=lambda x: x[1], reverse=True)
        choice = allowed_with_prob[0][0]
        seq.append(choice)
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str(f"len={len(seq)} last={choice}")
        if choice == "TRUNCATE":
            break
    return seq


def run_sampling_filtering(
    lm_model: Any,
    circuit_type: str,
    max_len: int,
    device_str: str,
    k: int = 3,
    temperature: float = 1.0,
    progress: Optional[tqdm] = None,
) -> List[str]:
    seq = [circuit_type, "VSS"]
    if progress is not None:
        progress.update(len(seq))
        progress.set_postfix_str("last=VSS")
    while len(seq) < max_len:
        allowed = explore_mod.get_allowed_tokens_6state(seq, circuit_type, max_len)
        if not allowed:
            break
        if "TRUNCATE" in allowed:
            seq.append("TRUNCATE")
            break

        probs = explore_mod.lm_next_token_distribution(lm_model, seq, device_str)
        allowed_with_prob = [(tok, probs.get(tok, 1e-8) ** (1.0 / temperature)) for tok in allowed]
        allowed_with_prob.sort(key=lambda x: x[1], reverse=True)
        top_candidates = allowed_with_prob[: min(k, len(allowed_with_prob))]
        tokens_k = [x[0] for x in top_candidates]
        weights_k = [x[1] for x in top_candidates]
        choice = random.choices(tokens_k, weights=weights_k, k=1)[0]
        seq.append(choice)
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str(f"len={len(seq)} last={choice}")
        if choice == "TRUNCATE":
            break
    return seq


def make_mcts_progress(progress: tqdm):
    best_valid_reward = 0.0

    def update(best_len: int, best_reward: float, rolled_seq: List[str]) -> None:
        nonlocal best_valid_reward
        progress.update(1)
        clean_tokens = [t for t in rolled_seq if t != "TRUNCATE"]
        is_clean, *_ = run_rule_validation(clean_tokens, verbose=False, debug=False)
        ports_ok = explore_mod.check_opamp_ports_present(clean_tokens)
        
        # Valid circuit requires both ERC clean alternation AND all op-amp ports present
        if is_clean and ports_ok:
            best_valid_reward = max(best_valid_reward, best_reward)
            progress.set_postfix_str(f"best_len={best_len} best_r={best_valid_reward:.3f} (ERC:PASS, Ports:OK) rollout_len={len(rolled_seq)}")
        elif best_valid_reward > 0.0:
            status = "Ports:Missing" if is_clean else "ERC:FAIL"
            progress.set_postfix_str(f"best_len={best_len} best_r={best_valid_reward:.3f} ({status}) rollout_len={len(rolled_seq)}")
        else:
            status = "Ports:Missing" if is_clean else "ERC:FAIL"
            progress.set_postfix_str(f"best_len={best_len} best_r=None ({status}) rollout_len={len(rolled_seq)}")

    return update


def main() -> int:
    parser = argparse.ArgumentParser(description="EXPLORE Op-Amp Benchmark")
    parser.add_argument("--samples", type=int, default=5, help="Number of benchmark samples per method")
    parser.add_argument("--mcts-iters", type=int, default=32, help="Iterations for MCTS methods")
    parser.add_argument("--max-len", type=int, default=96, help="Maximum sequence length")
    parser.add_argument("--output", type=Path, default=Path("Benchmark_Results"), help="Output directory")
    parser.add_argument("--metrics-cache", type=Path, default=Path("spice_metrics_cache.csv"), help="Persistent cache file")
    parser.add_argument("--device", type=str, default="cuda" if (torch and torch.cuda.is_available()) else "cpu")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cache_path = args.metrics_cache
    metrics_cache = explore_mod.load_metrics_cache(cache_path)

    lm_model = explore_mod.load_lm(device_str=args.device)
    if lm_model is None:
        sys.exit("Error: LM model Pretrain.pth is required for the benchmark.")

    methods = ["Greedy", "Sampling+Filtering", "MCTS-Base", "EXPLORE"]
    results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}

    print("=" * 70)
    print(f"Starting EXPLORE Benchmark on Op-Amp Generation ({args.samples} runs per method)")
    print("=" * 70)

    for run_idx in range(1, args.samples + 1):
        tqdm.write(f"\n--- Run {run_idx}/{args.samples} ---")

        # 1. Greedy
        t0 = time.time()
        with tqdm(total=args.max_len, desc=f"Run {run_idx} Greedy", unit="tok", leave=True) as progress:
            seq_greedy = run_greedy(lm_model, "CIRCUIT_Opamp", args.max_len, args.device, progress=progress)
        res_greedy = evaluate_generated_circuit(seq_greedy, metrics_cache, cache_path, args.output / "greedy_tmp")
        res_greedy["elapsed_sec"] = time.time() - t0
        results["Greedy"].append(res_greedy)
        tqdm.write(f"  [Greedy] Len: {len(seq_greedy)} | ERC: {res_greedy['is_erc_clean']} | SPICE: {res_greedy['sim_ok']}")

        # 2. Sampling + Filtering
        t0 = time.time()
        with tqdm(total=args.max_len, desc=f"Run {run_idx} S+F", unit="tok", leave=True) as progress:
            seq_sf = run_sampling_filtering(lm_model, "CIRCUIT_Opamp", args.max_len, args.device, k=3, temperature=1.0, progress=progress)
        res_sf = evaluate_generated_circuit(seq_sf, metrics_cache, cache_path, args.output / "sf_tmp")
        res_sf["elapsed_sec"] = time.time() - t0
        results["Sampling+Filtering"].append(res_sf)
        tqdm.write(f"  [S+F]    Len: {len(seq_sf)} | ERC: {res_sf['is_erc_clean']} | SPICE: {res_sf['sim_ok']}")

        # 3. MCTS-Base (No LM prior, uniform selection)
        t0 = time.time()
        with tqdm(total=args.mcts_iters, desc=f"Run {run_idx} MCTS-Base", unit="iter", leave=True) as progress:
            root_base, best_base_seq, best_base_r = explore_mod.mcts_explore(
                circuit_type="CIRCUIT_Opamp",
                iterations=args.mcts_iters,
                max_len=args.max_len,
                c_explore=4.0,
                metrics_cache=metrics_cache,
                cache_path=cache_path,
                workdir=args.output / f"mcts_base_run{run_idx}",
                lm_model=None,  # No LM prior
                device_str=args.device,
                p_filter_threshold=1.1,  # Disabled p-filtering
                progress_callback=make_mcts_progress(progress),
            )
        final_base_seq, _ = explore_mod.extract_best_sequence(root_base)
        if best_base_r > 0 and explore_mod.grammar_complete(best_base_seq):
            final_base_seq = best_base_seq
        res_base = evaluate_generated_circuit(final_base_seq, metrics_cache, cache_path, args.output / f"mcts_base_run{run_idx}")
        res_base["elapsed_sec"] = time.time() - t0
        results["MCTS-Base"].append(res_base)
        tqdm.write(f"  [MCTS-B] Len: {len(final_base_seq)} | ERC: {res_base['is_erc_clean']} | SPICE: {res_base['sim_ok']}")

        # 4. EXPLORE (LM-guided P-UCB + p-filtering p=0.99 + constrained rollout)
        t0 = time.time()
        with tqdm(total=args.mcts_iters, desc=f"Run {run_idx} EXPLORE", unit="iter", leave=True) as progress:
            root_exp, best_exp_seq, best_exp_r = explore_mod.mcts_explore(
                circuit_type="CIRCUIT_Opamp",
                iterations=args.mcts_iters,
                max_len=args.max_len,
                c_explore=4.0,
                metrics_cache=metrics_cache,
                cache_path=cache_path,
                workdir=args.output / f"explore_run{run_idx}",
                lm_model=lm_model,
                device_str=args.device,
                p_filter_threshold=0.99,
                progress_callback=make_mcts_progress(progress),
            )
        final_exp_seq, _ = explore_mod.extract_best_sequence(root_exp)
        if best_exp_r > 0 and explore_mod.grammar_complete(best_exp_seq):
            final_exp_seq = best_exp_seq
        res_exp = evaluate_generated_circuit(final_exp_seq, metrics_cache, cache_path, args.output / f"explore_run{run_idx}")
        res_exp["elapsed_sec"] = time.time() - t0
        results["EXPLORE"].append(res_exp)
        tqdm.write(f"  [EXPLORE] Len: {len(final_exp_seq)} | ERC: {res_exp['is_erc_clean']} | SPICE: {res_exp['sim_ok']}")

    # Save summary report
    summary_path = args.output / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 70)
    print(f"{'Method':<20} | {'ERC Pass':<10} | {'SPICE OK':<10} | {'Avg Time (s)':<12}")
    print("-" * 70)

    for m in methods:
        runs = results[m]
        erc_rate = sum(1 for r in runs if r["is_erc_clean"]) / len(runs) * 100.0
        spice_rate = sum(1 for r in runs if r["sim_ok"]) / len(runs) * 100.0
        avg_time = sum(r["elapsed_sec"] for r in runs) / len(runs)
        print(f"{m:<20} | {erc_rate:6.1f}%    | {spice_rate:6.1f}%    | {avg_time:8.2f}s")
    print("=" * 70)
    print(f"Full benchmark details written to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
