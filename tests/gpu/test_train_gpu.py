"""GPU training checks (Kaggle). Marked gpu."""

import pytest

pytestmark = pytest.mark.gpu


def test_real_adapter_endpoints():
    """PLACEHOLDER(M3) DC-07 on real Tier 0 adapters."""


def test_resume_after_kill():
    """PLACEHOLDER(M3) DC-13: killed run resumes to the same final step."""
