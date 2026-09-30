"""GPU extraction checks (Kaggle). Marked gpu."""

import pytest

pytestmark = pytest.mark.gpu


def test_padding_invariance_fp16():
    """PLACEHOLDER(M2) DC-06 in fp16 on T4, atol 1e-3."""


def test_guard_active_on_real_model():
    """PLACEHOLDER(M2) Guard runs on every batch of a real forward pass."""
