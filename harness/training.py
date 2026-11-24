"""Legacy learned-vector input-masking trainer.

Retained for reproducibility of the earlier experiments. The current notebook
uses training_fcm.py for Forgetful Causal Masking.
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

from data import DATA_REV, MODEL_CONFIG_ID, MODEL_CONFIG_REV, START, fingerprint
from masking import (
    DEFAULT_BASE_MASK_RATE, DEFAULT_CONTENT_WEIGHT, DEFAULT_OTHER_WEIGHT,
    mask_rate, word_mask,
)


SEED_LOCK = threading.Lock()


@dataclass(frozen=True)
class TrainConfig:
    name: str
    seed: int
    regime: str
    train_targets: int = 110_000_000
    # Unique token budget. None means use a fresh block for every training example.
    unique_train_tokens: int | None = None
    val_blocks: int = 512
    batch_size: int = 128  # paper batch size
    micro_batch_size: int = 64  # gradient accumulation retains effective batch size
    sequence_length: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    base_mask_rate: float = DEFAULT_BASE_MASK_RATE
    content_weight: float = DEFAULT_CONTENT_WEIGHT
    other_weight: float = DEFAULT_OTHER_WEIGHT
    checkpoint_every: int = 500

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
    # Lock protects global RNG while independent models receive deterministic init seeds.
    with SEED_LOCK:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = LlamaForCausalLM(config)
        extra = torch.empty(config.hidden_size).normal_(std=config.initializer_range)
    model.register_parameter("context_mask", torch.nn.Parameter(extra))
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


def _forward(model, x, mask=None):
    if mask is None:
        return model(input_ids=x, use_cache=False).logits
    emb = model.get_input_embeddings()(x)
    return model(inputs_embeds=torch.where(mask.unsqueeze(-1), model.context_mask, emb),
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
    return {"val_loss": loss_sum / count, "val_accuracy": correct / count}


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
        raise RuntimeError("GPU profiling requires the oclab CUDA runtime")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = _new_model(model_config, cfg.seed, device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))
    x, y, codes = _batch(train_ids, np.arange(cfg.micro_batch_size), cfg.sequence_length, device, train_pos)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    mask = word_mask(codes, cfg.regime, cfg.base_mask_rate, generator)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=dtype):
        logits = _forward(model, x, mask)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    del model, opt, x, y, codes, mask, logits, loss
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
            "free_bytes_after_profile": free, "total_bytes": total, "dtype": str(dtype)}


def recommend_workers(profile: dict, reserve_gib: float = 5.0, max_workers: int = 4) -> int:
    """Conservative memory cap: leave reserve for CUDA/runtime and cap at four jobs."""
    budget = profile["free_bytes_after_profile"] - int(reserve_gib * 1024**3)
    one_job = max(profile["peak_reserved_bytes"], profile["peak_allocated_bytes"] * 1.25)
    return max(1, min(max_workers, budget // max(1, one_job)))


def fit(cfg: TrainConfig, train_ids, val_ids, train_pos, run_root: str | Path):
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
    spec = asdict(cfg)
    spec["model"] = MODEL_CONFIG_REV
    spec["dataset"] = DATA_REV
    run_id = fingerprint(spec)
    run = Path(run_root) / f"{cfg.name}-{run_id}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "settings.json").write_text(json.dumps({"run_id": run_id, **spec}, indent=2))
    final = run / "metrics.json"
    if final.exists():
        return json.loads(final.read_text())
    model = _new_model(model_config, cfg.seed, device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay)
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    totals = torch.zeros(4, dtype=torch.int64, device=device)
    unique_blocks = cfg.unique_train_blocks
    if len(train_ids) < unique_blocks * length + 1 or len(train_pos) < unique_blocks * length:
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
        if "totals" in state:
            totals.copy_(state["totals"])
    steps_per_epoch = unique_blocks // batch
    epoch = start_step // steps_per_epoch
    order = np.random.default_rng(cfg.seed + epoch).permutation(unique_blocks)
    pos_weights = {"pos": (cfg.content_weight, cfg.other_weight),
                   "pos_mrd": (cfg.content_weight, cfg.other_weight),
                   "pos_reverse": (cfg.other_weight, cfg.content_weight),
                   "pos_reverse_mrd": (cfg.other_weight, cfg.content_weight)}
    milestones = {max(1, round(steps * f)) for f in (0.25, 0.5, 0.75, 1.0)}
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
        rate = mask_rate(cfg.regime, t, steps, cfg.base_mask_rate)
        rng.manual_seed(cfg.seed * 1000000007 + t)
        mask = word_mask(codes, cfg.regime, rate, rng, pos_weights)
        kind = codes % START
        totals += torch.stack([(kind == 1).sum(), (mask & (kind == 1)).sum(),
                               (kind == 2).sum(), (mask & (kind == 2)).sum()])
        opt.param_groups[0]["lr"] = _lr(t, cfg)
        opt.zero_grad(set_to_none=True)
        for lo in range(0, batch, micro):
            hi = lo + micro
            with torch.autocast("cuda", dtype=dtype) if use_amp else nullcontext():
                logits = _forward(model, x[lo:hi], mask[lo:hi] if rate else None)
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
            n_content, masked_content, n_other, masked_other = totals.tolist()
            row = {"name": cfg.name, "seed": cfg.seed, "regime": cfg.regime,
                   "step": step, "targets": step * batch * length,
                   "unique_train_tokens": unique_blocks * length,
                   "effective_epochs": step * batch / unique_blocks,
                   "scheduled_rate": rate,
                   "realized_rate": (masked_content + masked_other) / max(1, n_content + n_other),
                   "masked_content": masked_content / max(1, n_content),
                   "masked_other": masked_other / max(1, n_other),
                   **_evaluate(model, val_ids, cfg.val_blocks, cfg, device, dtype)}
            records.append(row)
        if step in milestones or step % cfg.checkpoint_every == 0:
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "scaler": scaler.state_dict(), "step": step, "seed": cfg.seed,
                     "regime": cfg.regime, "history": records, "totals": totals.cpu()}
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


def run_many(configs, train_ids, val_ids, train_pos, run_root, max_workers=1):
    """Run listed experiments concurrently on one GPU, bounded by the profile recommendation."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fit, cfg, train_ids, val_ids, train_pos, run_root) for cfg in configs]
        completed = tqdm(as_completed(futures), total=len(futures), desc="Runs complete", unit="run")
        for future in completed:
            results.append(future.result())
    return results
