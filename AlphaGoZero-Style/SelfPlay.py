#!/usr/bin/env python3
"""AlphaGo Zero-style self-learning loop for analog circuit topology search.

This module adapts AlphaGo Zero policy iteration to AnalogToBi/EXPLORE-style
analog topology generation:

1. A unified transformer predicts policy p(a | s) and value v(s).
2. P-UCB MCTS combines neural priors with grammar masks.
3. Self-play generates (state, search policy pi, final reward z) tuples.
4. Training optimises value MSE + policy cross-entropy + AdamW weight decay.
5. Evaluation promotes a candidate checkpoint when it beats the current best.

Cold-start mitigation: pass --pretrained-path Pretrain.pth to initialise
token/position embeddings and the policy head from the supervised language model
before any self-play begins.

Usage examples
--------------
# ERC-only reward, warm start from Pretrain.pth
python AlphaGoZero-Style/SelfPlay.py \\
    --pretrained-path Pretrain.pth \\
    --iterations 5 --episodes 4 --mcts-simulations 32 --max-len 64

# SPICE/ERC reward
python AlphaGoZero-Style/SelfPlay.py \\
    --pretrained-path Pretrain.pth \\
    --use-spice-reward \\
    --iterations 5 --episodes 4 --mcts-simulations 32 --max-len 64
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ERC import ITOS, STOI, VOCAB, run_rule_validation  # noqa: E402
import MCTS_Inference_Grammar as grammar  # noqa: E402


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RewardFn = Callable[[Sequence[str]], float]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelfPlayConfig:
    """Immutable configuration for self-play, MCTS, training, and promotion."""

    circuit_type: str = "CIRCUIT_Opamp"
    max_len: int = 96
    mcts_simulations: int = 128
    c_puct: float = 2.0
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    replay_capacity: int = 20_000
    batch_size: int = 32
    train_steps: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    eval_episodes: int = 16
    promotion_threshold: float = 0.55
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("AlphaGoZero-Style/checkpoints")
    pretrained_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    """One AlphaGo Zero training target collected during self-play."""

    state_tokens: List[str]
    target_policy: List[float]
    final_reward: float


@dataclass
class EpisodeResult:
    """Summary returned by one self-play episode."""

    sequence: List[str]
    reward: float
    examples: List[TrainingExample]
    # ERC info
    is_erc_clean: bool = False
    ports_ok: bool = False
    spice_metrics: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.spice_metrics is None:
            self.spice_metrics = {}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PolicyValueTransformer(nn.Module):
    """Small GPT-decoder style model with policy and value heads.

    Inputs a partial token sequence; outputs:
    - ``policy_logits``: unnormalized logits over the full vocabulary.
    - ``value``: scalar in [0, 1] estimating circuit quality.
    """

    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        n_embd: int = 256,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.final_norm = nn.LayerNorm(n_embd)
        self.policy_head = nn.Linear(n_embd, vocab_size)
        self.value_head = nn.Sequential(
            nn.Linear(n_embd, n_embd), nn.GELU(), nn.Linear(n_embd, 1)
        )
        self.block_size = block_size

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next-action policy logits and state value for a batch."""
        token_ids = token_ids[:, -self.block_size :]
        batch_size, seq_len = token_ids.shape
        positions = (
            torch.arange(seq_len, device=token_ids.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.transformer(hidden, mask=causal_mask)
        last_hidden = self.final_norm(hidden[:, -1, :])
        policy_logits = self.policy_head(last_hidden)
        value = torch.sigmoid(self.value_head(last_hidden)).squeeze(-1)
        return policy_logits, value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokens_to_ids(tokens: Sequence[str], device: str) -> torch.Tensor:
    """Convert token strings to a single-row tensor of vocabulary ids."""
    ids = [STOI.get(token, STOI["TRUNCATE"]) for token in tokens]
    if not ids:
        ids = [STOI["TRUNCATE"]]
    return torch.tensor([ids], dtype=torch.long, device=device)


def mask_and_normalize_policy(
    logits: torch.Tensor, legal_actions: Sequence[str]
) -> Dict[str, float]:
    """Apply grammar mask to logits and return probabilities for legal actions."""
    if not legal_actions:
        return {}
    legal_ids = torch.tensor(
        [STOI[a] for a in legal_actions if a in STOI], device=logits.device
    )
    if legal_ids.numel() == 0:
        return {}
    legal_logits = logits.index_select(dim=-1, index=legal_ids)
    legal_probs = torch.softmax(legal_logits, dim=-1).detach().cpu().tolist()
    legal_tokens = [ITOS[int(idx)] for idx in legal_ids.detach().cpu().tolist()]
    return dict(zip(legal_tokens, legal_probs))


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

@dataclass
class SearchEdge:
    """Statistics stored for one directed tree edge (s, a)."""

    prior: float
    child: "SearchNode"
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchNode:
    """One MCTS node containing a grammar state and outgoing action edges."""

    state: Tuple[str, ...]
    edges: Dict[str, SearchEdge]
    expanded: bool = False
    terminal_value: Optional[float] = None


class AlphaZeroMCTS:
    """P-UCB MCTS guided by a policy-value network and a grammar mask."""

    def __init__(
        self,
        model: PolicyValueTransformer,
        config: SelfPlayConfig,
        reward_fn: RewardFn,
        add_root_noise: bool = True,
    ) -> None:
        self.model = model
        self.config = config
        self.reward_fn = reward_fn
        self.add_root_noise = add_root_noise

    def run(self, state: Sequence[str], pbar: Optional[tqdm] = None) -> SearchNode:
        """Run P-UCB simulations from root state and return the root node."""
        root = SearchNode(tuple(state), {})
        self._expand(root, is_root=True)
        for _ in range(self.config.mcts_simulations):
            self._simulate(root)
            if pbar is not None:
                pbar.update(1)
        return root

    def search_policy(
        self, root: SearchNode, temperature: Optional[float] = None
    ) -> List[float]:
        """Convert root visit counts N(s,a) into AlphaGo Zero target policy π."""
        temperature = self.config.temperature if temperature is None else temperature
        policy = [0.0 for _ in VOCAB]
        if not root.edges:
            return policy
        visits = {action: edge.visits for action, edge in root.edges.items()}
        if temperature <= 1e-8:
            best_action = max(visits, key=visits.get)  # type: ignore[arg-type]
            policy[STOI[best_action]] = 1.0
            return policy
        scaled = {action: count ** (1.0 / temperature) for action, count in visits.items()}
        total = sum(scaled.values())
        if total <= 0:
            uniform = 1.0 / len(scaled)
            for action in scaled:
                policy[STOI[action]] = uniform
            return policy
        for action, count in scaled.items():
            policy[STOI[action]] = count / total
        return policy

    def select_action(
        self, root: SearchNode, temperature: Optional[float] = None
    ) -> str:
        """Sample or greedily choose an action from root visit distribution."""
        pi = self.search_policy(root, temperature)
        legal = [action for action in root.edges if pi[STOI[action]] > 0]
        if not legal:
            return "TRUNCATE"
        weights = [pi[STOI[action]] for action in legal]
        return random.choices(legal, weights=weights, k=1)[0]

    def _simulate(self, node: SearchNode) -> float:
        if self._is_terminal(node.state):
            if node.terminal_value is None:
                node.terminal_value = self.reward_fn(list(node.state))
            return node.terminal_value
        if not node.expanded:
            return self._expand(node, is_root=False)
        action, edge = self._select_edge(node)
        value = self._simulate(edge.child)
        edge.visits += 1
        edge.value_sum += value
        return value

    def _expand(self, node: SearchNode, is_root: bool) -> float:
        legal_actions = grammar.get_allowed_tokens_6state(
            list(node.state), self.config.circuit_type, self.config.max_len
        )
        if not legal_actions:
            node.terminal_value = self.reward_fn(list(node.state))
            node.expanded = True
            return node.terminal_value
        with torch.no_grad():
            logits, value = self.model(tokens_to_ids(node.state, self.config.device))
        priors = mask_and_normalize_policy(logits[0], legal_actions)
        if is_root and self.add_root_noise and priors:
            priors = self._mix_dirichlet_noise(priors)
        node.edges = {
            action: SearchEdge(
                prior=prior,
                child=SearchNode(tuple([*node.state, action]), {}),
            )
            for action, prior in priors.items()
        }
        node.expanded = True
        return float(value.item())

    def _select_edge(self, node: SearchNode) -> Tuple[str, SearchEdge]:
        parent_visits = sum(edge.visits for edge in node.edges.values())
        best_action = ""
        best_edge: Optional[SearchEdge] = None
        best_score = -float("inf")
        for action, edge in node.edges.items():
            exploration = (
                self.config.c_puct
                * edge.prior
                * math.sqrt(parent_visits + 1)
                / (1 + edge.visits)
            )
            score = edge.q + exploration
            if score > best_score:
                best_action = action
                best_edge = edge
                best_score = score
        if best_edge is None:
            raise RuntimeError("Cannot select from an empty MCTS node")
        return best_action, best_edge

    def _mix_dirichlet_noise(self, priors: Dict[str, float]) -> Dict[str, float]:
        actions = list(priors)
        noise = torch.distributions.Dirichlet(
            torch.full((len(actions),), self.config.dirichlet_alpha)
        ).sample()
        return {
            action: (1.0 - self.config.dirichlet_epsilon) * priors[action]
            + self.config.dirichlet_epsilon * float(noise[idx])
            for idx, action in enumerate(actions)
        }

    def _is_terminal(self, state: Sequence[str]) -> bool:
        return bool(state and state[-1] == "TRUNCATE") or len(state) >= self.config.max_len


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity FIFO buffer for self-play training examples."""

    def __init__(self, capacity: int) -> None:
        self._data: Deque[TrainingExample] = deque(maxlen=capacity)

    def extend(self, examples: Iterable[TrainingExample]) -> None:
        self._data.extend(examples)

    def sample(self, batch_size: int) -> List[TrainingExample]:
        return random.sample(list(self._data), k=min(batch_size, len(self._data)))

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------

class SelfPlayDataset(Dataset):  # type: ignore[type-arg]
    """Torch Dataset wrapper around a list of TrainingExamples."""

    def __init__(self, examples: Sequence[TrainingExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]


def collate_examples(
    batch: Sequence[TrainingExample],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length states and stack policy/value targets into tensors."""
    max_len = max(len(item.state_tokens) for item in batch)
    pad_id = STOI["TRUNCATE"]
    state_ids = []
    for item in batch:
        ids = [STOI.get(token, pad_id) for token in item.state_tokens]
        state_ids.append(ids + [pad_id] * (max_len - len(ids)))
    states = torch.tensor(state_ids, dtype=torch.long)
    policies = torch.tensor([item.target_policy for item in batch], dtype=torch.float32)
    rewards = torch.tensor([item.final_reward for item in batch], dtype=torch.float32)
    return states, policies, rewards


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def default_reward(sequence: Sequence[str]) -> float:
    """ERC + port-completeness fallback reward in [0, 1].

    Use this as a fast smoke-test reward when no SPICE simulator is available.
    """
    clean_tokens = [t for t in sequence if t != "TRUNCATE"]
    is_clean, v1, v2, v3, v4 = run_rule_validation(clean_tokens, verbose=False, debug=False)
    violation_count = len(v1) + len(v2) + len(v3) + len(v4)
    score = 0.2 if is_clean else max(0.0, 0.2 - 0.02 * violation_count)
    if grammar.check_opamp_ports_present(clean_tokens):
        score += 0.3
    if sequence and sequence[-1] == "TRUNCATE":
        score += 0.1
    return min(score, 1.0)


def make_spice_reward(
    metrics_cache: Dict[str, Dict[str, float]],
    workdir: Path,
    cache_path: Optional[Path] = None,
) -> RewardFn:
    """Return a reward function that runs the EXPLORE SPICE pipeline.

    Falls back to ``default_reward`` when the sequence does not pass ERC or
    the simulator fails (e.g. ngspice not installed).
    """

    def reward(sequence: Sequence[str]) -> float:
        clean_tokens = [t for t in sequence if t != "TRUNCATE"]
        if not grammar.check_opamp_ports_present(clean_tokens):
            return default_reward(sequence)
        is_clean, *_ = run_rule_validation(clean_tokens, verbose=False, debug=False)
        if not is_clean:
            return default_reward(sequence)
        metrics = grammar.evaluate_spice_sequence(
            clean_tokens, metrics_cache, workdir=workdir, cache_path=cache_path
        )
        if not metrics:
            return default_reward(sequence)
        metric_scores: List[float] = []
        if metrics.get("AV0") is not None:
            metric_scores.append(min(abs(float(metrics["AV0"])) / 120.0, 1.0))
        if metrics.get("UGB") is not None:
            metric_scores.append(
                min(math.log10(abs(float(metrics["UGB"])) + 1.0) / 6.0, 1.0)
            )
        if metrics.get("PM") is not None:
            metric_scores.append(min(abs(float(metrics["PM"])) / 90.0, 1.0))
        if metrics.get("P_DISS") is not None:
            metric_scores.append(
                max(0.0, 1.0 - min(abs(float(metrics["P_DISS"])) / 1e-2, 1.0))
            )
        if not metric_scores:
            return 0.3
        return min(1.0, 0.3 + 0.7 * (sum(metric_scores) / len(metric_scores)))

    return reward


# ---------------------------------------------------------------------------
# Self-play episode
# ---------------------------------------------------------------------------

def _erc_info(sequence: Sequence[str]) -> Tuple[bool, bool]:
    """Return (is_erc_clean, ports_ok) for a finished sequence."""
    clean_tokens = [t for t in sequence if t != "TRUNCATE"]
    is_clean, *_ = run_rule_validation(clean_tokens, verbose=False, debug=False)
    ports_ok = grammar.check_opamp_ports_present(clean_tokens)
    return bool(is_clean), bool(ports_ok)


def run_self_play_episode(
    model: PolicyValueTransformer,
    config: SelfPlayConfig,
    reward_fn: RewardFn,
    outer_pbar: Optional[tqdm] = None,
) -> EpisodeResult:
    """Generate one complete topology episode and AlphaGo Zero training tuples.

    Args:
        model: Policy-value network in eval mode.
        config: Self-play configuration.
        reward_fn: Terminal reward callable.
        outer_pbar: tqdm bar in the parent scope updated with MCTS progress.

    Returns:
        EpisodeResult with sequence, reward, training examples, and ERC info.
    """
    model.eval()
    state: List[str] = [config.circuit_type, "VSS"]
    examples: List[TrainingExample] = []

    while state[-1] != "TRUNCATE" and len(state) < config.max_len:
        mcts_obj = AlphaZeroMCTS(model, config, reward_fn, add_root_noise=True)
        # Inner MCTS progress bar
        with tqdm(
            total=config.mcts_simulations,
            desc=f"  MCTS (len={len(state)})",
            leave=False,
            unit="sim",
        ) as mcts_pbar:
            root = mcts_obj.run(state, pbar=mcts_pbar)
        pi = mcts_obj.search_policy(root, config.temperature)
        examples.append(
            TrainingExample(state_tokens=state[:], target_policy=pi, final_reward=0.0)
        )
        action = mcts_obj.select_action(root, config.temperature)
        state.append(action)
        if outer_pbar is not None:
            outer_pbar.update(1)
            outer_pbar.set_postfix_str(f"last={action}")

    final_reward = reward_fn(state)
    for item in examples:
        item.final_reward = final_reward

    is_erc_clean, ports_ok = _erc_info(state)
    return EpisodeResult(
        sequence=state,
        reward=final_reward,
        examples=examples,
        is_erc_clean=is_erc_clean,
        ports_ok=ports_ok,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_network(
    model: PolicyValueTransformer,
    replay: ReplayBuffer,
    config: SelfPlayConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    """Optimise the policy-value network on recent self-play data.

    Loss = MSE(v, z) + CrossEntropy(p, π)
    """
    if len(replay) == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    optimizer = optimizer or torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    losses: List[float] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []

    with tqdm(
        total=config.train_steps, desc="  Training", leave=False, unit="step"
    ) as pbar:
        for step_idx in range(config.train_steps):
            batch = replay.sample(config.batch_size)
            states, target_policy, target_value = collate_examples(batch)
            states = states.to(config.device)
            target_policy = target_policy.to(config.device)
            target_value = target_value.to(config.device)
            logits, value = model(states)
            log_probs = F.log_softmax(logits, dim=-1)
            policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(value, target_value)
            loss = value_loss + policy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                pi=f"{policy_loss.item():.4f}",
                v=f"{value_loss.item():.4f}",
            )

    return {
        "loss": sum(losses) / len(losses),
        "policy_loss": sum(policy_losses) / len(policy_losses),
        "value_loss": sum(value_losses) / len(value_losses),
    }


# ---------------------------------------------------------------------------
# Evaluation / promotion
# ---------------------------------------------------------------------------

def evaluate_model(
    candidate: PolicyValueTransformer,
    best: PolicyValueTransformer,
    config: SelfPlayConfig,
    reward_fn: RewardFn,
) -> Dict[str, float]:
    """Compare candidate and best models over deterministic evaluation episodes."""
    candidate_rewards: List[float] = []
    best_rewards: List[float] = []
    eval_config = replace(config, temperature=0.0)

    with tqdm(
        total=config.eval_episodes,
        desc="  Evaluation",
        leave=False,
        unit="ep",
    ) as pbar:
        for _ in range(config.eval_episodes):
            candidate_rewards.append(
                run_self_play_episode(candidate, eval_config, reward_fn).reward
            )
            best_rewards.append(
                run_self_play_episode(best, eval_config, reward_fn).reward
            )
            pbar.update(1)
            pbar.set_postfix(
                cand=f"{candidate_rewards[-1]:.3f}",
                best=f"{best_rewards[-1]:.3f}",
            )

    wins = sum(1 for c, b in zip(candidate_rewards, best_rewards) if c > b)
    ties = sum(1 for c, b in zip(candidate_rewards, best_rewards) if c == b)
    win_rate = (wins + 0.5 * ties) / max(1, config.eval_episodes)
    return {
        "candidate_mean": sum(candidate_rewards) / len(candidate_rewards),
        "best_mean": sum(best_rewards) / len(best_rewards),
        "win_rate": win_rate,
        "promote": float(win_rate >= config.promotion_threshold),
    }


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: PolicyValueTransformer,
    config: SelfPlayConfig,
    path: Path,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """Save model weights plus config metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    model: PolicyValueTransformer, path: Path, device: str
) -> None:
    """Load model weights from a SelfPlay checkpoint file."""
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = (
        payload["model_state_dict"]
        if isinstance(payload, dict) and "model_state_dict" in payload
        else payload
    )
    model.load_state_dict(state_dict)


def load_pretrained_weights(
    model: PolicyValueTransformer,
    pretrained_path: Path,
    device: str,
) -> Dict[str, List[str]]:
    """Safely load weights from an external pretrained checkpoint (e.g. Pretrain.pth).

    Handles two cases gracefully:
    * **Direct match**: any state-dict key whose name and shape match a target key
      is loaded verbatim (handles DataParallel ``module.`` prefix automatically).
    * **GPTLanguageModel mapping**: if the checkpoint comes from ``GPT_Pretrain.py``,
      the following key aliases are applied so the shared transformer body is warm-
      started while the new ``value_head`` remains randomly initialised::

          token_embedding_table  →  token_embedding
          position_embedding_table  →  position_embedding
          ln_f                   →  final_norm
          lm_head                →  policy_head

    Returns a dict with keys ``'loaded_keys'``, ``'missing_keys'``,
    ``'unexpected_keys'`` for logging.
    """
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at: {pretrained_path}"
        )

    raw = torch.load(pretrained_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        src_state = raw["model_state_dict"]
    elif isinstance(raw, dict):
        src_state = raw
    elif hasattr(raw, "state_dict"):
        src_state = raw.state_dict()
    else:
        raise ValueError(f"Unrecognised checkpoint format in {pretrained_path}")

    target_state = model.state_dict()
    loaded_keys: List[str] = []
    direct_matched: Dict[str, torch.Tensor] = {}

    # Pass 1 – direct key / shape match (strips module. prefix)
    for k, v in src_state.items():
        clean_k = k.replace("module.", "")
        if clean_k in target_state and target_state[clean_k].shape == v.shape:
            direct_matched[clean_k] = v
            loaded_keys.append(clean_k)

    # Pass 2 – GPTLanguageModel → PolicyValueTransformer key aliases
    gpt_mapping: Dict[str, str] = {
        "token_embedding_table.weight": "token_embedding.weight",
        "position_embedding_table.weight": "position_embedding.weight",
        "ln_f.weight": "final_norm.weight",
        "ln_f.bias": "final_norm.bias",
        "lm_head.weight": "policy_head.weight",
        "lm_head.bias": "policy_head.bias",
    }
    for src_k, tgt_k in gpt_mapping.items():
        if tgt_k in direct_matched:
            continue  # already loaded via direct match
        val = src_state.get(src_k)
        if val is None:
            val = src_state.get(f"module.{src_k}")
        if (
            val is not None
            and tgt_k in target_state
            and target_state[tgt_k].shape == val.shape
        ):
            direct_matched[tgt_k] = val
            loaded_keys.append(f"{src_k} → {tgt_k}")

    # Apply all matched weights
    target_state.update(direct_matched)
    model.load_state_dict(target_state, strict=False)

    all_target_keys = set(model.state_dict().keys())
    matched_target_keys = {k.split(" → ")[-1] for k in loaded_keys}
    missing_keys = sorted(all_target_keys - matched_target_keys)
    unexpected_keys = sorted(
        k for k in src_state if k.replace("module.", "") not in matched_target_keys
    )
    return {
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _format_sequence(seq: List[str], max_tokens: int = 12) -> str:
    """Human-readable token snippet of a sequence."""
    display = seq[:max_tokens]
    suffix = f" … ({len(seq)} tokens)" if len(seq) > max_tokens else ""
    return " → ".join(display) + suffix


def _print_best_episode(
    iteration: int,
    episode: EpisodeResult,
    spice_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Print a formatted summary of the best episode found so far."""
    tqdm.write("")
    tqdm.write("┌" + "─" * 68 + "┐")
    tqdm.write(f"│  ★  Best Episode  (iteration {iteration:>3d})                            │")
    tqdm.write("├" + "─" * 68 + "┤")
    tqdm.write(f"│  Reward : {episode.reward:.4f}                                          │")
    tqdm.write(f"│  ERC    : {'PASS ✓' if episode.is_erc_clean else 'FAIL ✗'}  │  Ports : {'OK ✓' if episode.ports_ok else 'missing ✗'}                        │")
    tqdm.write(f"│  Length : {len(episode.sequence)} tokens                                     │")
    tqdm.write(f"│  Seq    : {_format_sequence(episode.sequence):<54}│")
    if spice_metrics:
        tqdm.write("├" + "─" * 68 + "┤")
        tqdm.write("│  SPICE Metrics:                                                    │")
        for metric, val in spice_metrics.items():
            tqdm.write(f"│    {metric:<12}: {val:>12.4g}                                   │")
    tqdm.write("└" + "─" * 68 + "┘")
    tqdm.write("")


# ---------------------------------------------------------------------------
# Core policy iteration loop
# ---------------------------------------------------------------------------

def policy_iteration(
    model: PolicyValueTransformer,
    config: SelfPlayConfig,
    reward_fn: RewardFn = default_reward,
    iterations: int = 10,
    episodes_per_iteration: int = 8,
) -> PolicyValueTransformer:
    """Run AlphaGo Zero-style policy iteration with tqdm progress display.

    Warm-start: if ``config.pretrained_path`` is set, loads weights before
    starting. The warm-started model becomes the initial ``best_model`` so
    MCTS priors are meaningful from episode 1 (cold-start mitigation).

    After each iteration prints the iteration's best episode summary and the
    running all-time best sequence.

    Returns the best-performing model found across all iterations.
    """
    model.to(config.device)

    # ── Warm-start from pretrained weights ──────────────────────────────────
    if config.pretrained_path is not None and config.pretrained_path.exists():
        tqdm.write(
            f"[★] Loading pretrained weights from: {config.pretrained_path}"
        )
        stats = load_pretrained_weights(
            model, config.pretrained_path, device=config.device
        )
        tqdm.write(f"    ↳ Loaded {len(stats['loaded_keys'])} tensors directly.")
        if stats["missing_keys"]:
            tqdm.write(
                f"    ↳ New (randomly init.) keys: {stats['missing_keys']}"
            )
        tqdm.write("")

    best_model = copy.deepcopy(model).to(config.device)
    replay = ReplayBuffer(config.replay_capacity)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # All-time best tracking
    global_best_episode: Optional[EpisodeResult] = None
    global_best_reward: float = -float("inf")

    # ── Iteration loop ───────────────────────────────────────────────────────
    iter_pbar = tqdm(
        range(1, iterations + 1),
        desc="Policy Iteration",
        unit="iter",
    )
    for iteration in iter_pbar:
        iter_pbar.set_postfix(
            best_r=f"{global_best_reward:.3f}" if global_best_reward > -float("inf") else "N/A"
        )
        tqdm.write(f"\n{'═' * 70}")
        tqdm.write(f"  Iteration {iteration}/{iterations}")
        tqdm.write(f"{'═' * 70}")

        # ── Self-play ────────────────────────────────────────────────────────
        iter_best_episode: Optional[EpisodeResult] = None
        iter_best_reward: float = -float("inf")

        ep_pbar = tqdm(
            range(1, episodes_per_iteration + 1),
            desc=f"  [Iter {iteration}] Self-Play",
            leave=True,
            unit="ep",
        )
        for ep_idx in ep_pbar:
            episode = run_self_play_episode(best_model, config, reward_fn)
            replay.extend(episode.examples)
            ep_pbar.set_postfix(
                r=f"{episode.reward:.3f}",
                erc="✓" if episode.is_erc_clean else "✗",
                ports="✓" if episode.ports_ok else "✗",
                len=len(episode.sequence),
            )
            if episode.reward > iter_best_reward:
                iter_best_reward = episode.reward
                iter_best_episode = episode
            if episode.reward > global_best_reward:
                global_best_reward = episode.reward
                global_best_episode = episode

        # ── Network training ─────────────────────────────────────────────────
        tqdm.write(f"\n  [Iter {iteration}] Training network on {len(replay)} replay examples …")
        train_metrics = train_network(model, replay, config, optimizer)
        tqdm.write(
            f"  Loss: {train_metrics['loss']:.4f}  "
            f"(policy: {train_metrics['policy_loss']:.4f}, "
            f"value: {train_metrics['value_loss']:.4f})"
        )

        # ── Model evaluation & promotion ─────────────────────────────────────
        tqdm.write(f"\n  [Iter {iteration}] Evaluating candidate vs best model …")
        eval_metrics = evaluate_model(model, best_model, config, reward_fn)
        win_rate = eval_metrics["win_rate"]
        promoted = bool(eval_metrics["promote"])
        tqdm.write(
            f"  Win rate: {win_rate:.2%}  "
            f"(candidate: {eval_metrics['candidate_mean']:.3f} "
            f"vs best: {eval_metrics['best_mean']:.3f})  "
            + ("→ PROMOTED ✓" if promoted else "→ not promoted")
        )

        # ── Save checkpoints ─────────────────────────────────────────────────
        candidate_path = config.checkpoint_dir / f"candidate_iter_{iteration:04d}.pth"
        save_checkpoint(
            model,
            config,
            candidate_path,
            {"train": train_metrics, "eval": eval_metrics, "iteration": iteration},
        )
        if promoted:
            best_model = copy.deepcopy(model).to(config.device)
            save_checkpoint(
                best_model,
                config,
                config.checkpoint_dir / "best_model.pth",
                {"iteration": iteration, "eval": eval_metrics},
            )
            tqdm.write(f"  Saved new best model → {config.checkpoint_dir / 'best_model.pth'}")

        # ── Print iteration best episode ─────────────────────────────────────
        if iter_best_episode is not None:
            _print_best_episode(iteration, iter_best_episode)

    # ── Print all-time best ───────────────────────────────────────────────────
    tqdm.write("\n" + "═" * 70)
    tqdm.write("  ALL-TIME BEST EPISODE")
    tqdm.write("═" * 70)
    if global_best_episode is not None:
        _print_best_episode(0, global_best_episode)
    else:
        tqdm.write("  (no successful episode recorded)")

    return best_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlphaGo Zero-style self-play for AnalogToBi topology search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=5, help="Policy iteration rounds")
    parser.add_argument("--episodes", type=int, default=4, help="Self-play episodes per iteration")
    parser.add_argument("--mcts-simulations", type=int, default=32, help="MCTS simulations per step")
    parser.add_argument("--max-len", type=int, default=64, help="Max token sequence length")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("AlphaGoZero-Style/checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default=None,
        help="Path to pretrained .pth file for warm-start (e.g. Pretrain.pth)",
    )
    parser.add_argument(
        "--use-spice-reward",
        action="store_true",
        help="Use ngspice SPICE simulation reward instead of ERC-only fallback",
    )
    parser.add_argument(
        "--cache-csv",
        type=Path,
        default=Path("spice_metrics_cache.csv"),
        help="Persistent SPICE metrics cache CSV",
    )
    parser.add_argument(
        "--spice-workdir",
        type=Path,
        default=Path("AlphaGoZero-Style/spice_runs"),
        help="Working directory for SPICE netlist runs",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=8,
        help="Evaluation episodes for promotion decision",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=100,
        help="Gradient steps per training phase",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    config = SelfPlayConfig(
        max_len=args.max_len,
        mcts_simulations=args.mcts_simulations,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        pretrained_path=args.pretrained_path,
        eval_episodes=args.eval_episodes,
        train_steps=args.train_steps,
    )

    # ── Reward function ───────────────────────────────────────────────────────
    if args.use_spice_reward:
        tqdm.write("[★] Using SPICE/ERC reward function")
        metrics_cache = (
            grammar.load_metrics_cache(args.cache_csv)
            if hasattr(grammar, "load_metrics_cache")
            else {}
        )
        reward_fn = make_spice_reward(
            metrics_cache=metrics_cache,
            workdir=args.spice_workdir,
            cache_path=args.cache_csv,
        )
    else:
        tqdm.write("[★] Using ERC fallback reward function")
        reward_fn = default_reward

    # ── Build model ───────────────────────────────────────────────────────────
    model = PolicyValueTransformer().to(config.device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    tqdm.write(f"[★] PolicyValueTransformer: {total_params:.2f}M parameters on {config.device}")
    tqdm.write("")

    # ── Run policy iteration ──────────────────────────────────────────────────
    t_start = time.time()
    best = policy_iteration(
        model,
        config,
        reward_fn=reward_fn,
        iterations=args.iterations,
        episodes_per_iteration=args.episodes,
    )
    elapsed = time.time() - t_start

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = config.checkpoint_dir / "latest_best_model.pth"
    save_checkpoint(best, config, final_path)
    tqdm.write(f"\n[★] Training complete in {elapsed:.1f}s")
    tqdm.write(f"[★] Best model saved → {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
