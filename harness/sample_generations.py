"""Generate comparable continuations from saved SimpleStories training runs.

Example:
    python sample_generations.py \
        --run /path/to/runs_fcm/<run-name>-<run-id> \
        --run /path/to/runs_fcm/<another-run-name>-<run-id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from data import MODEL_CONFIG_ID, MODEL_CONFIG_REV
from training_fcm import DEFAULT_SEED


DEFAULT_PROMPT = (
    "Eagerly, a girl named Kim went hiking with her friends. "
    "They found a secret passageway under a big rock."
)


def find_checkpoint(run_or_checkpoint: str | Path) -> Path:
    path = Path(run_or_checkpoint)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Run/checkpoint path does not exist: {path}")
    checkpoints = sorted(path.glob("step-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No step-*.pt checkpoints found in {path}")
    return checkpoints[-1]


def load_trained_model(checkpoint: Path, config, device):
    model = LlamaForCausalLM(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    label = checkpoint.parent.name
    settings = checkpoint.parent / "settings.json"
    if settings.exists():
        label = json.loads(settings.read_text()).get("name", label)
    return model, state, label


def generate_runs(
    run_paths: list[str | Path],
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int = DEFAULT_SEED,
) -> list[tuple[str, str]]:
    """Load each run's latest saved checkpoint and return its continuation."""
    if not run_paths:
        raise ValueError("Pass at least one run directory or checkpoint file.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG_ID, revision=MODEL_CONFIG_REV, use_fast=True)
    config = AutoConfig.from_pretrained(MODEL_CONFIG_ID, revision=MODEL_CONFIG_REV)
    config.use_cache = True
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]
    results = []

    for run_path in run_paths:
        checkpoint = find_checkpoint(run_path)
        model, state, label = load_trained_model(checkpoint, config, device)
        # Reset the RNG for each model so output differences are easier to compare.
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        continuation = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
        results.append((label, continuation))
        print(f"\n{label} — {checkpoint.name} (step {state['step']})")
        print(prompt + continuation)
        del model, state, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="Run directory or checkpoint file; repeat to compare runs.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate_runs(args.run, args.prompt, args.max_new_tokens,
                  args.temperature, args.top_p, args.seed)


if __name__ == "__main__":
    main()
