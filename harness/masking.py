"""Whole-word context masking schedules."""

import torch

from data import START

DEFAULT_BASE_MASK_RATE = 0.15
DEFAULT_CONTENT_WEIGHT = 0.75
DEFAULT_OTHER_WEIGHT = 1.5


def mask_rate(regime: str, step: int, steps: int,
              base_rate: float = DEFAULT_BASE_MASK_RATE) -> float:
    if regime == "clean":
        return 0.0
    if regime.endswith("_mrd"):
        return 2 * base_rate * (1 - step / max(1, steps - 1))
    return base_rate


def word_mask(codes: torch.Tensor, regime: str, rate: float, generator: torch.Generator,
              pos_weights: dict[str, tuple[float, float]] | None = None,
              protect_first: bool = False) -> torch.Tensor:
    kind = codes % START
    eligible = kind != 0
    if rate <= 0:
        return torch.zeros_like(eligible)
    starts = codes >= START
    positions = torch.arange(codes.size(1), device=codes.device).expand_as(codes)
    first_piece = torch.where(starts, positions, 0).cummax(dim=1).values
    draws = torch.rand(codes.shape, generator=generator, device=codes.device).gather(1, first_piece)
    if pos_weights and regime in pos_weights:
        content, other = pos_weights[regime]
        weights = torch.where(kind == 1, content, other) * eligible
        probability = rate * weights * (eligible.sum() / weights.sum().clamp_min(1e-12))
    else:
        probability = rate
    masked = eligible & (draws < probability)
    if protect_first and masked.size(1):
        masked[:, 0] = False
    return masked
