# SimpleStories Context Masking Experiments

This harness compares a small causal language model trained with ordinary causal attention against Forgetful Causal Masking (FCM) variants. It is designed to run from a Colab GPU notebook, keep run files in Google Drive, and make the experiment settings and outputs easy to find.

## Quickstart

1. Open `masking_experiments.ipynb` in Colab and connect to a GPU runtime.
2. Make this `harness` folder available at `/content/harness` and run the first setup cell. It mounts Google Drive, installs the versions in `requirements.txt`, and loads the tokenizer and model configuration.
3. In the configuration cell, choose a preset, masking regimes, and seeds. The default is `data_constrained` with `clean`, `uniform`, and `uniform_mrd`.
4. Keep `WORKFLOW = 'train'` to start or resume runs. Run the notebook cells from the top. Use `WORKFLOW = 'restore'` to load saved results without training.
5. Find the run folders and reports in `MyDrive/CS182_masking_harness/runs`. The prepared corpus is saved as `MyDrive/CS182_masking_harness/corpus_cache.zip` after tokenizing completes.

The notebook writes checkpoints directly to Drive as they are produced. It does not package them into a second ZIP, and it does not extract ZIP archives. A compatible extracted cache already in `harness/cache` is reused; an incompatible or incomplete cache is regenerated.

## Experiment setup

### Data

- Dataset: [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories), pinned to the revision in `data.py`.
- Training uses the dataset's `train` split. Evaluation uses a deterministic shuffled sample from its `test` split; the notebook evaluates that sample at each checkpoint and plots those scores.
- The default evaluation sample contains 512 blocks of 512 tokens (262,144 prediction targets). The cache stores one extra token for next-token shifting.
- Stories are deduplicated within each split before tokenization. POS annotations are produced for training tokens and cached alongside them.
- Cache identity includes the dataset and tokenizer revisions, token budgets, preprocessing versions, and package versions. A changed setup gets a different cache directory.

### Model and training

- Architecture and tokenizer configuration: SimpleStories 5M Llama configuration. The model is initialized from that configuration for every new run; pretrained model weights are not loaded.
- Training objective: causal next-token prediction. FCM hides sampled input positions as attention keys during training; those positions remain prediction targets. Evaluation and generation use clean causal attention.
- Default focused plan: clean, uniform FCM, and uniform FCM with a decaying rate, each at 96, 24, and 6 Mi unique training tokens. All runs target about 100.7 Mi tokens, so the smaller datasets are revisited more often.
- Effective batch size is 128, split into micro-batches of 32. The notebook profiles memory and recommends a concurrent run count, capped by `MAX_CONCURRENT_RUNS`.
- The default fixed uniform rate is 15%. The decaying schedule starts at 30% and reaches zero at the final step. POS-weighted modes are available in the masking code and can be added to `REGIMES`.
- The notebook's test curves are used to compare and select configurations. Since this same test sample is inspected throughout training, it should be treated as the experiment's evaluation/validation set, not as an untouched final benchmark.

### Outputs

Drive contains one directory per run under `CS182_masking_harness/runs`. Each run directory is fingerprinted from its configuration and contains settings, resumable step checkpoints, and saved test metrics. The root also receives the combined `test_history.csv`, summary table, and plots. The cache ZIP contains token IDs, POS codes, and cache metadata.

## Files

| File | Purpose |
| --- | --- |
| `masking_experiments.ipynb` | Colab entry point: environment setup, run configuration, preprocessing, GPU scheduling, training, and result display. |
| `data.py` | SimpleStories loading, tokenization, POS alignment, cache fingerprints, and corpus reads. |
| `training_fcm.py` | Fresh model initialization, FCM attention-mask application, training, test evaluation, checkpointing, and GPU memory profiling. |
| `masking.py` | Whole-word uniform and POS-weighted mask sampling, including the MRD schedule. |
| `analysis.py` | Test-set summary table and learning-curve plots. |
| `sample_generations.py` | Loads saved run checkpoints and compares generated stories from a prompt. |
| `requirements.txt` | Python package versions used by the notebook. |
| `training.py` | Earlier legacy trainer; the notebook uses `training_fcm.py`. |

## References

- Finke et al., [Parameterized Synthetic Text Generation with SimpleStories](https://arxiv.org/abs/2504.09184), 2025. Dataset and small-model context.
- Liu et al., [Towards Better Few-Shot and Finetuning Performance with Forgetful Causal Language Models](https://arxiv.org/abs/2210.13432), 2022. FCM method: randomly masking past context while retaining causal next-token prediction.
- Yang, Zhang, and Zhao, [Learning Better Masking for Better Language Model Pre-training](https://aclanthology.org/2023.acl-long.400/), ACL 2023. Source for masking-rate decay and POS-weighted masking ideas. This harness adapts these schedules to FCM; it does not reproduce the paper's bidirectional masked-language-model objective.

## Reproducibility notes

Dataset and model-config revisions, random seeds, training settings, and the data-pipeline version are recorded in each run's settings. Keep those files and the corresponding checkpoints together when moving results. Changing the data preprocessing or token budget creates a new cache identity. The Drive cache ZIP is refreshed only after a complete cache has been prepared.
