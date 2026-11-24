"""SimpleStories corpus preparation and disk cache."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import datasets
import nltk
import numpy as np
from datasets import load_dataset
from nltk.tag import PerceptronTagger
from nltk.tokenize import TreebankWordTokenizer
from tqdm.auto import tqdm


START = 4  # high bit marks the start of a word; low bits: 0 protected, 1 content, 2 other
# Public Hugging Face IDs. MODEL_CONFIG_ID/REV fetch only config/tokenizer assets;
# the training code constructs fresh model weights from the config.
MODEL_CONFIG_ID = "SimpleStories/SimpleStories-5M"
DATA_ID = "SimpleStories/SimpleStories"
# Revisions pin public repository snapshots for reproducibility; these are not credentials.
DATA_REV = "e63b8adc3b1a1bdc7cac5b500d150b71346b0628"
MODEL_CONFIG_REV = "47201cd194f932403b44313fa9c4ac267ad05ebf"
PREPARE_BATCH_SIZE = 256  # Lower to 128 if preprocessing memory becomes tight.
DEFAULT_SHUFFLE_SEED = 416
DATA_PIPELINE_REV = "train-test-v2"


def fingerprint(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def pos_codes(story, spans, tags, ids, offsets, special_ids):
    codes = np.full(len(ids), START, dtype=np.uint8)
    cursor, previous = 0, -1
    for i, (token, (left, right)) in enumerate(zip(ids, offsets)):
        while cursor < len(spans) and spans[cursor][1] <= left:
            cursor += 1
        if token in special_ids or left >= right:
            previous = -1
            continue
        best, overlap, j = -1, 0, cursor
        while j < len(spans) and spans[j][0] < right:
            shared = min(right, spans[j][1]) - max(left, spans[j][0])
            if shared > overlap:
                best, overlap = j, shared
            j += 1
        if best >= 0 and any(c.isalnum() for c in story[slice(*spans[best])]):
            cls = 1 if tags[best].startswith(("NN", "VB", "JJ", "RB")) else 2
            codes[i] = cls + (START if best != previous else 0)
            previous = best
        else:
            previous = -1
    return codes


def prepare_corpus(tokenizer, cache_root: str | Path, train_tokens: int,
                   test_tokens: int,
                   shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
                   include_pos: bool = True):
    """Tokenize the training and test splits into reusable flat token arrays."""
    cache_root = Path(cache_root)
    spec = dict(model=MODEL_CONFIG_REV, dataset=DATA_REV,
                train_split="train", test_split="test",
                train=train_tokens, test=test_tokens,
                shuffle=shuffle_seed,
                pipeline=DATA_PIPELINE_REV + ("-pos" if include_pos else "-token-only"),
                nltk=nltk.__version__ if include_pos else None,
                datasets=datasets.__version__)
    cache = cache_root / ("corpus-" + fingerprint(spec))
    cache.mkdir(parents=True, exist_ok=True)
    complete = cache / "complete.json"
    paths = {k: cache / f"{k}.bin" for k in ("train", "test")}
    if include_pos:
        paths["pos"] = cache / "pos.bin"
    # Save exactly budget + one lookahead token for shifted causal targets.
    expected = {"train": 2 * (train_tokens + 1),
                "test": 2 * (test_tokens + 1)}
    if include_pos:
        expected["pos"] = train_tokens + 1
    if complete.exists() and all(paths[k].exists() and paths[k].stat().st_size == expected[k]
                                 for k in paths):
        return cache, spec
    complete.unlink(missing_ok=True)
    if include_pos:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True, raise_on_error=True)
        tagger = PerceptronTagger()
        word_tokenizer = TreebankWordTokenizer()
    train_rows = load_dataset(DATA_ID, split="train", revision=DATA_REV, streaming=True)
    train_rows = train_rows.select_columns(["story"]).shuffle(seed=shuffle_seed, buffer_size=2048)
    test_rows = load_dataset(DATA_ID, split="test", revision=DATA_REV, streaming=True)
    test_rows = test_rows.select_columns(["story"]).shuffle(seed=shuffle_seed, buffer_size=2048)
    seen_train, seen_test = set(), set()
    used = {"train": 0, "test": 0}
    counts = {"train": 0, "test": 0}
    budgets = {"train": train_tokens + 1, "test": test_tokens + 1}
    with ExitStack() as stack:
        out = {k: stack.enter_context(paths[k].open("wb", buffering=2**20)) for k in paths}
        train_bar = stack.enter_context(tqdm(total=budgets["train"], desc="Tokenizing official train split", unit="tok", unit_scale=True))
        for rows_batch in train_rows.iter(batch_size=PREPARE_BATCH_SIZE):
            batch = []
            for story in rows_batch["story"]:
                normalized = " ".join(story.split())
                if not normalized:
                    continue
                key = hashlib.blake2b(normalized.encode(), digest_size=16).digest()
                if key not in seen_train and used["train"] < budgets["train"]:
                    seen_train.add(key)
                    batch.append(story)
            if not batch:
                if used["train"] >= budgets["train"]:
                    break
                continue
            encoded = tokenizer(batch, return_offsets_mapping=True,
                                add_special_tokens=False, truncation=False, verbose=False)
            for story, ids, offsets in zip(batch, encoded.input_ids, encoded.offset_mapping):
                n = min(budgets["train"] - used["train"], len(ids) + 1)
                if not n:
                    continue
                if include_pos:
                    spans = list(word_tokenizer.span_tokenize(story))
                    tags = [t for _, t in tagger.tag([story[a:b] for a, b in spans])]
                    codes = pos_codes(story, spans, tags, ids, offsets, set(tokenizer.all_special_ids))
                    out["pos"].write(np.append(codes, np.uint8(START))[:n].tobytes())
                out["train"].write(np.asarray(ids + [tokenizer.eos_token_id], dtype="<u2")[:n].tobytes())
                counts["train"] += 1
                used["train"] += n
                train_bar.update(n)
            if used["train"] >= budgets["train"]:
                break
        if used["train"] != budgets["train"]:
            raise RuntimeError(f"Official train split ended after {used['train']:,}/{budgets['train']:,} tokens")

        test_bar = stack.enter_context(tqdm(total=budgets["test"],
                                            desc="Tokenizing test split",
                                            unit="tok", unit_scale=True))
        for rows_batch in test_rows.iter(batch_size=PREPARE_BATCH_SIZE):
            batch = []
            for story in rows_batch["story"]:
                normalized = " ".join(story.split())
                if not normalized:
                    continue
                key = hashlib.blake2b(normalized.encode(), digest_size=16).digest()
                if key in seen_test:
                    continue
                seen_test.add(key)
                if used["test"] < budgets["test"]:
                    batch.append(story)
            if not batch:
                if used["test"] >= budgets["test"]:
                    break
                continue
            encoded = tokenizer(batch, add_special_tokens=False,
                                truncation=False, verbose=False)
            for story, ids in zip(batch, encoded.input_ids):
                n = min(budgets["test"] - used["test"], len(ids) + 1)
                if n:
                    out["test"].write(np.asarray(ids + [tokenizer.eos_token_id], dtype="<u2")[:n].tobytes())
                    counts["test"] += 1
                    used["test"] += n
                    test_bar.update(n)
            if used["test"] >= budgets["test"]:
                break
    if any(used[k] != budgets[k] for k in budgets):
        raise RuntimeError(f"Official splits ended before requested token budgets: {used} / {budgets}")
    info = {"spec": spec, "stories": counts, "tokens": used}
    tmp = complete.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=2))
    tmp.replace(complete)
    return cache, spec


def read_corpus(cache: str | Path, train_tokens: int, test_tokens: int,
                include_pos: bool = True):
    cache = Path(cache)
    train = np.fromfile(cache / "train.bin", dtype="<u2")
    test = np.fromfile(cache / "test.bin", dtype="<u2")
    pos = np.fromfile(cache / "pos.bin", dtype=np.uint8) if include_pos else None
    return train, test, pos


def load_or_prepare_corpus(tokenizer, cache_root: str | Path, train_tokens: int,
                           test_tokens: int,
                           shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
                           include_pos: bool = True):
    """Reuse a compatible train/test cache, preparing it only when needed."""
    cache, spec = prepare_corpus(
        tokenizer, cache_root, train_tokens, test_tokens,
        shuffle_seed=shuffle_seed,
        include_pos=include_pos,
    )
    train, test, pos = read_corpus(
        cache, train_tokens, test_tokens, include_pos=include_pos,
    )
    return cache, spec, train, test, pos
