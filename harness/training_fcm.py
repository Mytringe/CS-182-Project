"""Training and evaluation with Forgetful Causal Masking (FCM).

FCM preserves the causal next-token objective. During training, sampled input
positions are hidden as attention keys/values, so later positions cannot use
them as context. Evaluation uses ordinary, unmasked causal attention.
"""

from __future__ import annotations

import json
import math
import threading
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, LlamaForCausalLM
from tqdm.auto import tqdm

from data import (DATA_PIPELINE_REV, DATA_REV, MODEL_CONFIG_ID, MODEL_CONFIG_REV,
                  START, fingerprint)
from masking import (
    DEFAULT_BASE_MASK_RATE, DEFAULT_CONTENT_WEIGHT, DEFAULT_OTHER_WEIGHT,
    mask_rate, word_mask,
)


SEED_LOCK = threading.Lock()
MASK_REGIMES = frozenset({"uniform", "uniform_mrd", "pos", "pos_mrd"})
DEFAULT_SEED = 182
DEFAULT_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 128
DEFAULT_TEST_BLOCKS = 512
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_WARMUP_STEPS = 100
DEFAULT_CHECKPOINT_EVERY = 500
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class TrainConfig:
    name: str
    seed: int
    regime: str
    train_targets: int = 110_000_000
    # Unique token budget. None means use a fresh block for every training example.
    unique_train_tokens: int | None = None
    test_blocks: int = DEFAULT_TEST_BLOCKS
    batch_size: int = DEFAULT_BATCH_SIZE  # paper batch size
    micro_batch_size: int = 64  # gradient accumulation retains effective batch size
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    base_mask_rate: float = DEFAULT_BASE_MASK_RATE
    content_weight: float = DEFAULT_CONTENT_WEIGHT
    other_weight: float = DEFAULT_OTHER_WEIGHT
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY

    @property
    def steps(self):
        return math.ceil(self.train_targets / (self.batch_size * self.sequence_length))

    @property
    def train_blocks(self):
        return self.steps * self.batch_size

    @property
    def unique_train_blocks(self):
        if self.unique_train_tokens is None:
            return self.train_blocks
        if self.unique_train_tokens % self.sequence_length:
            raise ValueError("unique_train_tokens must be divisible by sequence_length")
        blocks = self.unique_train_tokens // self.sequence_length
        if blocks < self.batch_size or blocks % self.batch_size:
            raise ValueError("unique_train_tokens must contain a whole number of batches")
        return blocks

    @property
    def effective_epochs(self):
        return self.train_blocks / self.unique_train_blocks


def _new_model(config, seed, device):
    # Construct from config, never from a pretrained checkpoint. This gives each
    # run fresh random weights, with the paper's residual-scaled attention outputs.
    with SEED_LOCK:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = LlamaForCausalLM(config)
        expected_std = 0.02
        if not math.isclose(config.initializer_range, expected_std):
            raise ValueError(f"Expected SimpleStories initializer_range={expected_std}")
        output_std = expected_std / math.sqrt(2 * config.num_hidden_layers)
        for layer in model.model.layers:
            torch.nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=output_std)
    return model.to(device)


def _lr(step, cfg):
    warm = max(1, cfg.warmup_steps)
    if step < warm:
        return cfg.learning_rate * (step + 1) / warm
    progress = (step - warm) / max(1, cfg.steps - warm - 1)
    return cfg.learning_rate * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


def _batch(ids, blocks, length, device, codes=None):
    ix = np.asarray(blocks)[:, None] * length + np.arange(length)
    x = torch.as_tensor(ids[ix].astype(np.int64), device=device)
    y = torch.as_tensor(ids[ix + 1].astype(np.int64), device=device)
    pos = None if codes is None else torch.as_tensor(codes[ix], device=device)
    return x, y, pos


def run_spec(cfg: TrainConfig, model_rev: str = MODEL_CONFIG_REV,
             data_rev: str = DATA_REV) -> dict:
    spec = asdict(cfg)
    spec["model"] = model_rev
    spec["dataset"] = data_rev
    spec["data_pipeline"] = DATA_PIPELINE_REV
    return spec


def run_directory(cfg: TrainConfig, run_root: str | Path,
                  model_rev: str = MODEL_CONFIG_REV, data_rev: str = DATA_REV) -> Path:
    spec = run_spec(cfg, model_rev, data_rev)
    return Path(run_root) / f"{cfg.name}-{fingerprint(spec)}"


def sample_fcm_attention_mask(codes, regime, base_rate, step, steps, generator,
                              content_weight=DEFAULT_CONTENT_WEIGHT,
                              other_weight=DEFAULT_OTHER_WEIGHT):
    """Sample old whole-word distributions, then hide selected keys via FCM attention."""
    if regime == "clean":
        return None, 0.0, 0.0
    if regime not in MASK_REGIMES:
        raise ValueError(f"regime must be clean or one of {sorted(MASK_REGIMES)}")
    rate = mask_rate(regime, step, steps, base_rate)
    pos_weights = ({regime: (content_weight, other_weight)}
                   if regime.startswith("pos") else None)
    hidden = word_mask(codes, regime, rate, generator, pos_weights,
                       protect_first=True)
    eligible_count = (codes % START != 0).sum().clamp_min(1)
    realized_rate = (hidden.sum() / eligible_count).item()
    return ~hidden, rate, realized_rate


def _forward(model, x, key_keep_mask=None):
    if key_keep_mask is None:
        return model(input_ids=x, use_cache=False).logits
    # A 2D attention mask marks keys that remain visible. The Llama model
    # combines it with its built-in causal mask; masked positions remain in
    # the sequence and are still prediction targets, but cannot be attended to.
    return model(input_ids=x, attention_mask=key_keep_mask.to(torch.long),
                 use_cache=False).logits


@torch.inference_mode()
def _evaluate(model, ids, blocks, cfg, device, autocast_dtype):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for i in range(0, blocks, cfg.micro_batch_size):
        x, y, _ = _batch(ids, np.arange(i, min(i + cfg.micro_batch_size, blocks)),
                         cfg.sequence_length, device)
        with torch.autocast("cuda", dtype=autocast_dtype) if device.type == "cuda" else nullcontext():
            logits = _forward(model, x)
        loss_sum += F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()
    model.train()
    return {"test_loss": loss_sum / count, "test_accuracy": correct / count}


def _runtime(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(MODEL_CONFIG_ID, revision=MODEL_CONFIG_REV)
    model_config.use_cache = False
    assert cfg.sequence_length <= model_config.max_position_embeddings
    return device, model_config


def profile_peak_memory(cfg: TrainConfig, train_ids, train_pos) -> dict:
    """Measure one paper-sized forward/backward/update on the current GPU."""
    device, model_config = _runtime(cfg)
    if device.type != "cuda":
        raise RuntimeError("GPU profiling requires a Colab GPU runtime")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = _new_model(model_config, cfg.seed, device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))
    x, y, codes = _batch(train_ids, np.arange(cfg.micro_batch_size), cfg.sequence_length,
                          device, train_pos)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    keep_mask, _, _ = sample_fcm_attention_mask(
        codes, cfg.regime, cfg.base_mask_rate, 0, cfg.steps, generator,
        cfg.content_weight, cfg.other_weight)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=dtype):
        logits = _forward(model, x, keep_mask)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    del model, opt, x, y, keep_mask, logits, loss
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
            "free_bytes_after_profile": free, "total_bytes": total, "dtype": str(dtype)}


def recommend_workers(profile: dict, reserve_gib: float = 5.0, max_workers: int = 4) -> int:
    """Conservative memory cap: leave reserve for CUDA/runtime and cap at four jobs."""
    budget = profile["free_bytes_after_profile"] - int(reserve_gib * 1024**3)
    one_job = max(profile["peak_reserved_bytes"], profile["peak_allocated_bytes"] * 1.25)
    return max(1, min(max_workers, budget // max(1, one_job)))


def fit(cfg: TrainConfig, train_ids, test_ids, train_pos, run_root: str | Path):
    if cfg.regime != "clean" and cfg.regime not in MASK_REGIMES:
        raise ValueError(f"regime must be clean or one of {sorted(MASK_REGIMES)}")
    if not 0.0 <= cfg.base_mask_rate <= 0.5:
        raise ValueError("base_mask_rate must be in [0, 0.5]")
    if len(train_pos) != len(train_ids):
        raise ValueError("train_pos must contain one POS code per training token")
    device, model_config = _runtime(cfg)
    length, batch, steps = cfg.sequence_length, cfg.batch_size, cfg.steps
    micro = cfg.micro_batch_size
    if batch < 1 or micro < 1 or cfg.sequence_length < 1:
        raise ValueError("batch size, micro batch size, and sequence length must be positive")
    if batch % micro:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    if cfg.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    accumulation = batch // micro
    spec = run_spec(cfg)
    run_id = fingerprint(spec)
    run = run_directory(cfg, run_root)
    run.mkdir(parents=True, exist_ok=True)
    settings = run / "settings.json"
    settings.write_text(json.dumps({"run_id": run_id, **spec}, indent=2))
    final = run / "metrics.json"
    if final.exists():
        return json.loads(final.read_text())
    model = _new_model(model_config, cfg.seed, device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay)
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    unique_blocks = cfg.unique_train_blocks
    if len(train_ids) < unique_blocks * length + 1:
        raise ValueError("Prepared training corpus is smaller than unique_train_tokens")
    rng = torch.Generator(device=device)
    records, start_step = [], 0
    checkpoints = sorted(run.glob("step-*.pt"))
    if checkpoints:
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        records = state["history"]
    steps_per_epoch = unique_blocks // batch
    epoch = start_step // steps_per_epoch
    order = np.random.default_rng(cfg.seed + epoch).permutation(unique_blocks)
    milestones = {max(1, round(steps * f)) for f in CHECKPOINT_FRACTIONS}
    for t in tqdm(range(start_step, steps), initial=start_step, total=steps, desc=cfg.name):
        # Each pass is a deterministic shuffle of the full corpus. Deriving the
        # permutation from its epoch also makes interrupted runs resume correctly.
        if t > start_step and t % steps_per_epoch == 0:
            epoch += 1
            order = np.random.default_rng(cfg.seed + epoch).permutation(unique_blocks)
        batch_in_epoch = t % steps_per_epoch
        lo = batch_in_epoch * batch
        batch_blocks = order[lo:lo + batch]
        x, y, codes = _batch(train_ids, batch_blocks, length, device, train_pos)
        rng.manual_seed(cfg.seed * 1000000007 + t)
        keep_mask, scheduled_rate, realized_rate = sample_fcm_attention_mask(
            codes, cfg.regime, cfg.base_mask_rate, t, steps, rng,
            cfg.content_weight, cfg.other_weight)
        opt.param_groups[0]["lr"] = _lr(t, cfg)
        opt.zero_grad(set_to_none=True)
        for lo in range(0, batch, micro):
            hi = lo + micro
            with torch.autocast("cuda", dtype=dtype) if use_amp else nullcontext():
                logits = _forward(model, x[lo:hi], keep_mask[lo:hi] if keep_mask is not None else None)
                loss = F.cross_entropy(logits.float().flatten(0, 1), y[lo:hi].flatten())
                loss = loss / accumulation
            scaler.scale(loss).backward()
            del logits, loss
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step = t + 1
        if step in milestones:
            row = {"name": cfg.name, "seed": cfg.seed, "regime": cfg.regime,
                   "step": step, "targets": step * batch * length,
                   "unique_train_tokens": unique_blocks * length,
                   "effective_epochs": step * batch / unique_blocks,
                   "scheduled_rate": scheduled_rate,
                   "realized_rate": realized_rate,
                   **_evaluate(model, test_ids, cfg.test_blocks, cfg, device, dtype)}
            records.append(row)
        if step in milestones or step % cfg.checkpoint_every == 0:
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "scaler": scaler.state_dict(), "step": step, "seed": cfg.seed,
                     "regime": cfg.regime, "history": records}
            tmp = run / f"step-{step:08d}.pt.tmp"
            torch.save(state, tmp)
            current = run / f"step-{step:08d}.pt"
            tmp.replace(current)
            for old in run.glob("step-*.pt"):
                if old != current and int(old.stem.split("-")[1]) not in milestones:
                    old.unlink(missing_ok=True)
    tmp = final.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2))
    tmp.replace(final)
    return records


def run_many(configs, train_ids, test_ids, train_pos, run_root, max_workers=1):
    """Run listed experiments concurrently on one GPU, bounded by the profile recommendation."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fit, cfg, train_ids, test_ids, train_pos, run_root)
                   for cfg in configs]
        completed = tqdm(as_completed(futures), total=len(futures), desc="Runs complete", unit="run")
        for future in completed:
            results.append(future.result())
    return results
