"""Mask-aware pooling (D-018). DC-06 (CPU part)."""

import pytest
import torch

from rbbd.embed.pooling import check_right_padded, pool, pool_all


def _batch():
    torch.manual_seed(0)
    hidden = torch.randn(3, 5, 4)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
    return hidden, mask


def test_padding_invariance_cpu(tiny):
    """Pooled vector alone == inside a right-padded batch for mean/max/last."""
    from rbbd.embed.extract import ExtractSettings, encode

    model, tok = tiny
    short = "Women attend community events."
    long_ = "Many different countries are inhabited by physically disabled people today."
    settings = ExtractSettings(token_budget=1 << 20)
    alone = encode(model, tok, [short], settings)
    batched = encode(model, tok, [long_, short, long_ + " again"], settings)
    for kind in ("mean", "max", "last"):
        torch.testing.assert_close(
            torch.from_numpy(batched[kind][1]).float(), torch.from_numpy(alone[kind][0]).float(),
            atol=1e-3, rtol=0,
        )  # fmt: skip


def test_last_token_index():
    """`last` selects the final non-pad token; mean and max ignore padding."""
    hidden, mask = _batch()
    hidden[1, 3:] = 1e6  # padding positions must never leak into the pooled vectors
    torch.testing.assert_close(pool(hidden, mask, "last")[1], hidden[1, 2])
    torch.testing.assert_close(pool(hidden, mask, "last")[2], hidden[2, 0])
    torch.testing.assert_close(pool(hidden, mask, "mean")[1], hidden[1, :3].mean(0))
    torch.testing.assert_close(pool(hidden, mask, "max")[1], hidden[1, :3].max(0).values)
    assert pool(hidden.half(), mask, "mean").dtype == torch.float32


def test_left_padding_rejected():
    """A left-padded mask raises instead of silently pooling pad positions (B-002)."""
    hidden, _ = _batch()
    with pytest.raises(ValueError, match="right-padded"):
        pool_all(hidden, torch.tensor([[0, 0, 1, 1, 1]] * 3), ["mean"])
    with pytest.raises(ValueError, match="empty"):
        check_right_padded(torch.zeros(1, 4, dtype=torch.int64))
