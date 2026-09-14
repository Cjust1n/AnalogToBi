#!/usr/bin/env python3
"""
UCT/MCTS search over AnalogToBi-style grammar sequences.

This is a lightweight search prototype that treats topology generation as a
sequential decision process over the bipartite grammar used by AnalogToBi.

It does not modify the existing GPT inference pipeline. Instead, it reuses the
ERC validator as a terminal reward signal and enforces a simple grammar mask
for bipartite alternation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ERC import CIRCUIT_TYPE_TOKENS, ITOS, STOI, VOCAB, run_rule_validation

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from Models.GPT import GPTLanguageModel
else:
    GPTLanguageModel = object  # type: ignore[misc,assignment]


DEVICE_TOKENS = [
    *(f"NM{i}" for i in range(1, 36)),
    *(f"PM{i}" for i in range(1, 36)),
    *(f"NPN{i}" for i in range(1, 28)),
    *(f"PNP{i}" for i in range(1, 28)),
    *(f"R{i}" for i in range(1, 29)),
    *(f"C{i}" for i in range(1, 17)),
    *(f"L{i}" for i in range(1, 25)),
    *(f"DIO{i}" for i in range(1, 9)),
]

EDGE_TOKENS = [
    "M_B", "M_D", "M_G", "M_S", "M_BD", "M_BG", "M_BS", "M_DG", "M_DS", "M_GS", "M_BDG", "M_BDS", "M_BGS", "M_DGS", "M_BDGS",
    "B_B", "B_C", "B_E", "B_BC", "B_BE", "B_CE", "B_BCE",
    "R_C", "C_C", "L_C",
    "D_P", "D_N", "D_NP", "D_PN",
]

PORT_TOKENS = [
    *(f"VIN{i}" for i in range(1, 21)),
    "VOUT",
    *(f"VOUT{i}" for i in range(1, 8)),
    *(f"IIN{i}" for i in range(1, 4)),
    *(f"IOUT{i}" for i in range(1, 6)),
    *(f"VB{i}" for i in range(1, 12)),
    *(f"IB{i}" for i in range(1, 8)),
    *(f"VCONT{i}" for i in range(1, 22)),
    *(token for i in range(1, 4) for token in (f"VCM{i}", f"VREF{i}", f"IREF{i}", f"VRF{i}", f"VIF{i}")),
    *(token for i in range(1, 6) for token in (f"VLO{i}", f"VBB{i}")),
]
NET_TOKENS = ["VSS", "VDD", *(f"NET{i}" for i in range(1, 51)), *PORT_TOKENS]
TRUNCATE = "TRUNCATE"
ROLL_OUT_PRIORITY = [
    "VSS", "VDD", "NET1", "NET2", "VIN1", "VIN2", "VOUT", "VOUT1",
    "NM1", "PM1", "R1", "C1", "L1", "DIO1",
]
OPAMP_EDGE_PRIORITY = [
    "M_DG", "M_G", "M_S", "M_B", "M_BD", "M_BS", "M_BG", "M_DS", "M_GS",
    "B_CE", "B_BE", "B_BC",
    "R_C", "C_C", "L_C",
]
OPAMP_DEVICE_PRIORITY = [
    "PM1", "NM1", "PM2", "NM2", "R1", "C1", "NPN1", "PNP1", "DIO1", "L1",
]


def is_device(token: str) -> bool:
    return token in DEVICE_TOKENS


def is_edge(token: str) -> bool:
    return token in EDGE_TOKENS


def is_net(token: str) -> bool:
    return token in NET_TOKENS


def allowed_tokens(sequence: Sequence[str], circuit_type: str, max_len: int) -> List[str]:
    if len(sequence) >= max_len:
        return [TRUNCATE] if _can_truncate(sequence, circuit_type) else []
    if not sequence:
        return [circuit_type]
    if len(sequence) == 1:
        return ["VSS"]
    if sequence[-2] in CIRCUIT_TYPE_TOKENS and sequence[-1] in {"VSS", "VDD"}:
        if circuit_type == "CIRCUIT_Opamp":
            return OPAMP_EDGE_PRIORITY + [edge for edge in EDGE_TOKENS if edge not in OPAMP_EDGE_PRIORITY]
        return EDGE_TOKENS
    if is_net(sequence[-2]) and is_edge(sequence[-1]):
        if circuit_type == "CIRCUIT_Opamp":
            return OPAMP_DEVICE_PRIORITY + [dev for dev in DEVICE_TOKENS if dev not in OPAMP_DEVICE_PRIORITY]
        return DEVICE_TOKENS
    if is_edge(sequence[-2]) and is_device(sequence[-1]):
        if circuit_type == "CIRCUIT_Opamp":
            return OPAMP_EDGE_PRIORITY + [edge for edge in EDGE_TOKENS if edge not in OPAMP_EDGE_PRIORITY]
        return EDGE_TOKENS
    if is_device(sequence[-2]) and is_edge(sequence[-1]):
        return NET_TOKENS
    if is_edge(sequence[-2]) and is_net(sequence[-1]):
        options = EDGE_TOKENS[:]
        if _can_truncate(sequence, circuit_type):
            options.append(TRUNCATE)
        return options
    return []


def prioritized_tokens(tokens: Sequence[str]) -> List[str]:
    order = {token: idx for idx, token in enumerate(ROLL_OUT_PRIORITY)}
    return sorted(tokens, key=lambda t: (order.get(t, len(ROLL_OUT_PRIORITY)), t))


def prioritize_opamp_tokens(tokens: Sequence[str]) -> List[str]:
    edge_order = {token: idx for idx, token in enumerate(OPAMP_EDGE_PRIORITY)}
    device_order = {token: idx for idx, token in enumerate(OPAMP_DEVICE_PRIORITY)}
    net_order = {token: idx for idx, token in enumerate(ROLL_OUT_PRIORITY)}

    def score(token: str) -> Tuple[int, int, str]:
        if token in OPAMP_EDGE_PRIORITY:
            return (0, edge_order[token], token)
        if token in OPAMP_DEVICE_PRIORITY:
            return (1, device_order[token], token)
        if token in NET_TOKENS:
            return (2, net_order.get(token, len(ROLL_OUT_PRIORITY) + NET_TOKENS.index(token)), token)
        if token == TRUNCATE:
            return (3, 0, token)
        return (4, len(tokens), token)

    return sorted(tokens, key=score)


def load_lm(
    model_path: Path = Path(__file__).resolve().parent / "Pretrain.pth",
    device_str: str = "cpu",
) -> Optional[GPTLanguageModel]:
    if torch is None:
        return None
    if not model_path.exists():
        return None
    model = GPTLanguageModel(len(VOCAB), 256, 1024, 4, 4, 0.2)
    state = torch.load(model_path, map_location=device_str)
    model.load_state_dict(state, strict=False)
    model.to(device_str)
    model.eval()
    return model


def lm_next_token_distribution(
    model: Optional[GPTLanguageModel],
    sequence: Sequence[str],
    device_str: str = "cpu",
) -> Dict[str, float]:
    if torch is None or model is None or not sequence:
        return {}
    idx = torch.tensor(
        [[STOI.get(token, STOI[TRUNCATE]) for token in sequence]],
        dtype=torch.long,
        device=device_str,
    )
    with torch.no_grad():
        logits, _ = model(idx[:, -1024:])
        probs = torch.softmax(logits[0, -1], dim=-1)
    return {ITOS[i]: float(prob) for i, prob in enumerate(probs.detach().cpu().tolist())}


def get_allowed_tokens_6state(sequence: Sequence[str], circuit_type: str, max_len: int) -> List[str]:
    return allowed_tokens(sequence, circuit_type, max_len)


def check_opamp_ports_present(sequence: Sequence[str]) -> bool:
    tokens = set(sequence)
    has_vin_pair = "VIN1" in tokens and "VIN2" in tokens
    has_vout = "VOUT" in tokens or "VOUT1" in tokens
    return has_vin_pair and has_vout and "VDD" in tokens and "VSS" in tokens


def _can_truncate(sequence: Sequence[str], circuit_type: str) -> bool:
    if not sequence or sequence[-1] != "VSS" or len(sequence) < 10:
        return False
    if circuit_type == "CIRCUIT_Opamp":
        return check_opamp_ports_present(sequence)
    return True


def grammar_complete(sequence: Sequence[str]) -> bool:
    return len(sequence) >= 3 and sequence[-1] == TRUNCATE


@dataclass
class MCTSNode:
    sequence: Tuple[str, ...]
    parent: Optional["MCTSNode"] = None
    action: Optional[str] = None
    children: Dict[str, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    untried: List[str] = field(default_factory=list)
    prior: float = 1.0

    def q(self) -> float:
        return self.value / self.visits if self.visits else 0.0


def sequence_key(sequence: Sequence[str]) -> str:
    payload = "->".join(sequence)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def sequence_to_netlist(sequence: Sequence[str], workdir: Optional[Path] = None) -> Optional[Path]:
    tokens = [t for t in sequence if t != TRUNCATE]
    if len(tokens) < 2:
        return None
    workdir = workdir or Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(workdir)) as tmpdir:
        tmp_path = Path(tmpdir)
        seq_file = tmp_path / "sequence.txt"
        seq_file.write_text("->".join(sequence), encoding="utf-8")
        cmd = [
            "python3",
            str(Path(__file__).resolve().parent.parent / "graph-to-netlist" / "main.py"),
            "--single",
            str(seq_file),
        ]
        result = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return None
        netlist = result.stdout.strip()
        if not netlist:
            return None
        cir_path = workdir / f"{sequence_key(sequence)}.cir"
        cir_path.write_text(netlist + "\n", encoding="utf-8")
        return cir_path


def _spice_pipeline_path() -> Path:
    return Path(__file__).resolve().parent.parent / "graph-to-netlist" / "opamp_spice_pipeline.py"


def load_metrics_cache(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None or not path.exists():
        return {}
    cache: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seq = row.get("sequence")
            if not seq:
                continue
            key = row.get("sequence_key") or hashlib.sha1(seq.encode("utf-8")).hexdigest()
            metrics: Dict[str, float] = {}
            for name in ("P_DISS", "UGB", "AV0", "PM", "SETTLING_TIME", "OFFSET", "PSRR", "CMRR"):
                raw = row.get(name)
                if raw is None or raw == "":
                    continue
                try:
                    metrics[name] = float(raw)
                except ValueError:
                    continue
            cache[key] = metrics
    return cache


def evaluate_spice_sequence(
    sequence: Sequence[str],
    metrics_cache: Dict[str, Dict[str, float]],
    evaluator_cmd: Optional[List[str]] = None,
    workdir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    timeout: float = 120,
) -> Dict[str, float]:
    key = sequence_key(sequence)
    if key in metrics_cache:
        return metrics_cache[key]

    workdir = workdir or Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    input_dir = workdir / f"{key}_input"
    output_dir = workdir / f"{key}_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in input_dir.iterdir():
        if child.is_file():
            child.unlink()
    (input_dir / "run0.txt").write_text("->".join(sequence), encoding="utf-8")

    cmd = [
        "python3",
        str(_spice_pipeline_path()),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--limit",
        "1",
        "--timeout",
        str(timeout),
    ]
    subprocess.run(cmd, cwd=str(workdir), check=False, capture_output=True, text=True)

    csv_path = output_dir / "opamp_metrics.csv"
    if not csv_path.exists():
        metrics_cache[key] = {}
        return {}

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        metrics_cache[key] = {}
        return {}
    row = rows[0]
    metrics: Dict[str, float] = {}
    for name in ("P_DISS", "UGB", "AV0", "PM", "SETTLING_TIME", "OFFSET", "PSRR", "CMRR"):
        raw = row.get(name)
        if raw is None or raw == "":
            continue
        try:
            metrics[name] = float(raw)
        except ValueError:
            continue
    metrics_cache[key] = metrics
    if cache_path is not None:
        cache_exists = cache_path.exists()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8", newline="") as fh:
            fieldnames = ["sequence_key", "sequence", "P_DISS", "UGB", "AV0", "PM", "SETTLING_TIME", "OFFSET", "PSRR", "CMRR"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not cache_exists:
                writer.writeheader()
            writer.writerow({"sequence_key": key, "sequence": "->".join(sequence), **metrics})
    return metrics


def terminal_reward(
    sequence: Sequence[str],
    metrics_cache: Dict[str, Dict[str, float]],
    evaluator_cmd: Optional[List[str]] = None,
    workdir: Optional[Path] = None,
) -> float:
    tokens = [t for t in sequence if t != TRUNCATE]
    is_clean, v1, v2, v3, v4 = run_rule_validation(tokens, verbose=False, debug=False)
    reward = 0.1
    if is_clean:
        reward += 0.2
    reward += max(0.0, 0.15 - 0.015 * (len(v1) + len(v2) + len(v3) + len(v4)))

    spice = evaluate_spice_sequence(tokens, metrics_cache, evaluator_cmd=evaluator_cmd, workdir=workdir)
    if spice:
        metric_scores = []
        p_diss = spice.get("P_DISS")
        if p_diss is not None:
            metric_scores.append(max(0.0, 1.0 - min(abs(p_diss) / 1e-2, 1.0)))
        ugb = spice.get("UGB")
        if ugb is not None:
            metric_scores.append(min(max(math.log10(abs(ugb) + 1.0) / 6.0, 0.0), 1.0))
        av0 = spice.get("AV0")
        if av0 is not None:
            metric_scores.append(min(max(abs(av0) / 120.0, 0.0), 1.0))
        pm = spice.get("PM")
        if pm is not None:
            metric_scores.append(min(max(abs(pm) / 90.0, 0.0), 1.0))
        settling = spice.get("SETTLING_TIME")
        if settling is not None:
            metric_scores.append(max(0.0, 1.0 - min(abs(settling) / 1e-6, 1.0)))
        offset = spice.get("OFFSET")
        if offset is not None:
            metric_scores.append(max(0.0, 1.0 - min(abs(offset) / 5e-3, 1.0)))
        psrr = spice.get("PSRR")
        if psrr is not None:
            metric_scores.append(min(max(abs(psrr) / 100.0, 0.0), 1.0))
        cmrr = spice.get("CMRR")
        if cmrr is not None:
            metric_scores.append(min(max(abs(cmrr) / 100.0, 0.0), 1.0))
        if metric_scores:
            reward += sum(metric_scores) / len(metric_scores)
    return max(0.1, reward)


def rollout(
    sequence: List[str],
    circuit_type: str,
    max_len: int,
    metrics_cache: Dict[str, Dict[str, float]],
    evaluator_cmd: Optional[List[str]] = None,
    workdir: Optional[Path] = None,
    lm_model: Optional[GPTLanguageModel] = None,
    device_str: str = "cpu",
    p_filter_threshold: float = 1.1,
) -> Tuple[List[str], float]:
    seq = sequence[:]
    for _ in range(max_len - len(seq)):
        options = prioritized_tokens(allowed_tokens(seq, circuit_type, max_len))
        if circuit_type == "CIRCUIT_Opamp":
            options = prioritize_opamp_tokens(options)
        if not options:
            break
        if lm_model is not None:
            probs = lm_next_token_distribution(lm_model, seq, device_str)
            ranked = sorted(options, key=lambda token: probs.get(token, 0.0), reverse=True)
            if ranked and probs.get(ranked[0], 0.0) >= p_filter_threshold:
                choice = ranked[0]
            else:
                candidates = ranked[: min(10, len(ranked))]
                weights = [max(probs.get(token, 1e-8), 1e-8) for token in candidates]
                choice = random.choices(candidates, weights=weights, k=1)[0]
        else:
            choice = options[0] if len(options) <= 3 else random.choice(options[: min(10, len(options))])
        seq.append(choice)
        if choice == TRUNCATE:
            break
    if seq and seq[-1] != TRUNCATE:
        terminal_options = allowed_tokens(seq, circuit_type, max_len)
        if TRUNCATE in terminal_options:
            seq.append(TRUNCATE)
    if not grammar_complete(seq):
        return seq, 0.0
    return seq, terminal_reward(seq, metrics_cache, evaluator_cmd=evaluator_cmd, workdir=workdir)


def uct_select(node: MCTSNode, exploration: float) -> MCTSNode:
    best_child = None
    best_score = -1e18
    for child in node.children.values():
        exploit = child.q()
        explore = exploration * child.prior * math.sqrt(math.log(node.visits + 1) / (child.visits + 1))
        score = exploit + explore
        if score > best_score:
            best_score = score
            best_child = child
    assert best_child is not None
    return best_child


def mcts(
    circuit_type: str,
    iterations: int,
    max_len: int,
    exploration: float,
    metrics_cache: Dict[str, Dict[str, float]],
    evaluator_cmd: Optional[List[str]] = None,
    workdir: Optional[Path] = None,
    tree_log: Optional[Path] = None,
    lm_model: Optional[GPTLanguageModel] = None,
    device_str: str = "cpu",
    p_filter_threshold: float = 1.1,
    progress_callback: Optional[Callable[[int, float, Sequence[str]], None]] = None,
) -> Tuple[MCTSNode, List[str], float]:
    root = MCTSNode(sequence=())
    root.untried = [circuit_type]
    if tree_log:
        tree_log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = tree_log.open("a", encoding="utf-8") if tree_log else None
    best_terminal_seq: List[str] = [circuit_type]
    best_terminal_reward = -1.0

    for _ in range(iterations):
        node = root
        seq = list(node.sequence)

        while node.untried == [] and node.children:
            node = uct_select(node, exploration)
            seq = list(node.sequence)

        if node.untried:
            action = node.untried.pop(0)
            seq.append(action)
            prior = 1.0
            if lm_model is not None:
                probs = lm_next_token_distribution(lm_model, seq[:-1], device_str)
                prior = max(probs.get(action, 1e-8), 1e-8)
            child = MCTSNode(sequence=tuple(seq), parent=node, action=action, prior=prior)
            child.untried = allowed_tokens(seq, circuit_type, max_len)
            node.children[action] = child
            node = child

        rolled_seq, reward = rollout(
            list(node.sequence),
            circuit_type,
            max_len,
            metrics_cache,
            evaluator_cmd=evaluator_cmd,
            workdir=workdir,
            lm_model=lm_model,
            device_str=device_str,
            p_filter_threshold=p_filter_threshold,
        )
        if reward > best_terminal_reward:
            best_terminal_reward = reward
            best_terminal_seq = rolled_seq[:]
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
        if log_fh is not None:
            log_fh.write(json.dumps({"sequence": list(seq), "reward": reward}) + "\n")
            log_fh.flush()
        if progress_callback is not None:
            progress_callback(len(best_terminal_seq), best_terminal_reward, rolled_seq)

    if log_fh is not None:
        log_fh.close()
    return root, best_terminal_seq, best_terminal_reward


def mcts_explore(
    circuit_type: str,
    iterations: int,
    max_len: int,
    c_explore: float,
    metrics_cache: Dict[str, Dict[str, float]],
    cache_path: Optional[Path] = None,
    workdir: Optional[Path] = None,
    lm_model: Optional[GPTLanguageModel] = None,
    device_str: str = "cpu",
    p_filter_threshold: float = 1.1,
    progress_callback: Optional[Callable[[int, float, Sequence[str]], None]] = None,
) -> Tuple[MCTSNode, List[str], float]:
    return mcts(
        circuit_type=circuit_type,
        iterations=iterations,
        max_len=max_len,
        exploration=c_explore,
        metrics_cache=metrics_cache,
        workdir=workdir,
        lm_model=lm_model,
        device_str=device_str,
        p_filter_threshold=p_filter_threshold,
        progress_callback=progress_callback,
    )


def best_sequence(root: MCTSNode) -> Tuple[List[str], float]:
    best = None
    best_score = -1.0
    stack = [root]
    while stack:
        node = stack.pop()
        score = node.q()
        if score > best_score and node.sequence and len(node.sequence) > 1:
            best_score = score
            best = list(node.sequence)
        if grammar_complete(node.sequence) and score >= best_score:
            best = list(node.sequence)
        stack.extend(node.children.values())
    return best or list(root.sequence), best_score


def extract_best_sequence(root: MCTSNode) -> Tuple[List[str], float]:
    return best_sequence(root)


def main() -> int:
    parser = argparse.ArgumentParser(description="MCTS search over AnalogToBi grammar")
    parser.add_argument("circuit_type", help="Example: Opamp or CIRCUIT_Opamp")
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--exploration", type=float, default=1.4)
    parser.add_argument("--output", type=Path, default=Path("MCTS_Inference"))
    parser.add_argument("--metrics-cache", type=Path, default=None)
    parser.add_argument("--spice-evaluator", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--tree-log", type=Path, default=None)
    args = parser.parse_args()

    circuit_type = args.circuit_type
    if not circuit_type.startswith("CIRCUIT_"):
        circuit_type = f"CIRCUIT_{circuit_type}"
    if circuit_type not in CIRCUIT_TYPE_TOKENS:
        raise SystemExit(f"Unknown circuit type: {circuit_type}")

    random.seed(1337)
    metrics_cache = load_metrics_cache(args.metrics_cache)
    root, terminal_seq, terminal_reward_value = mcts(
        circuit_type,
        args.iterations,
        args.max_len,
        args.exploration,
        metrics_cache=metrics_cache,
        evaluator_cmd=args.spice_evaluator,
        workdir=args.output,
        tree_log=args.tree_log,
    )
    seq, score = best_sequence(root)
    if terminal_reward_value > score or len(terminal_seq) > len(seq):
        seq = terminal_seq
        score = terminal_reward_value

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"{circuit_type}.txt"
    out_path.write_text("->".join(seq) + ("->" if seq else ""), encoding="utf-8")
    summary_path = args.output / f"{circuit_type}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "circuit_type": circuit_type,
                "iterations": args.iterations,
                "max_len": args.max_len,
                "score": score,
                "sequence": seq,
                "sequence_key": sequence_key(seq),
                "netlist_path": str((args.output / f"{sequence_key(seq)}.cir")) if seq else None,
                "terminal_reward": terminal_reward_value,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved best sequence to {out_path}")
    print(f"Score: {score:.4f}")
    print("Sequence:", "->".join(seq))
    


if __name__ == "__main__":
    raise SystemExit(main())
