"""NaN/Inf guard (D-006). DC-09."""


def test_guard_triggers_on_injected_inf():
    """PLACEHOLDER(M1) Injected Inf raises NonFiniteError naming the batch; no cache file is
    written."""
