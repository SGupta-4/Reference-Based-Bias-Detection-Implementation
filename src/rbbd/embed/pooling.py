"""Token-to-sentence pooling of final-layer hidden states (§3.1, App. E.4; D-018).

Called by `embed.extract.encode` once per batch. Assumes right padding (checked):
row i has ones in `mask[i, :len_i]` and zeros after. Pooling accumulates in fp32, so an
fp16 forward pass only rounds once, when the pooled vector is cast for the cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

POOLINGS = ("mean", "max", "last")


def check_right_padded(mask: Any) -> None:
    """Raise ValueError unless every row of the [B, T] 0/1 mask is ones then zeros."""
    import torch

    m = mask.to(torch.int64)
    lengths = m.sum(dim=1)
    if bool((lengths == 0).any()):
        raise ValueError("empty sequence in batch")
    positions = torch.arange(m.shape[1], device=m.device)[None, :]
    if not bool(((positions < lengths[:, None]).to(torch.int64) == m).all()):
        raise ValueError("attention mask is not right-padded (D-018)")


def pool(hidden: Any, mask: Any, kind: str) -> Any:
    """Pool hidden [B, T, d] with mask [B, T] -> [B, d] float32.

    mean: average over non-pad tokens (BOS included, D-018); max: element-wise max over
    non-pad tokens; last: the final non-pad token.
    """
    import torch

    h = hidden.to(torch.float32)
    m = mask.to(torch.bool)
    if kind == "mean":
        w = m.unsqueeze(-1).to(torch.float32)
        return (h * w).sum(dim=1) / w.sum(dim=1)
    if kind == "max":
        return h.masked_fill(~m.unsqueeze(-1), float("-inf")).max(dim=1).values
    if kind == "last":
        idx = m.sum(dim=1) - 1
        return h[torch.arange(h.shape[0], device=h.device), idx]
    raise ValueError(f"unknown pooling {kind!r}; expected one of {POOLINGS}")


def pool_all(hidden: Any, mask: Any, kinds: Sequence[str]) -> dict[str, Any]:
    """{kind: [B, d] float32} for every requested pooling, after checking right padding."""
    check_right_padded(mask)
    return {k: pool(hidden, mask, k) for k in kinds}
