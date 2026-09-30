"""Merge exactness (D-007, D-008). DC-07."""


def test_lora_alpha1_equals_unharmful():
    """PLACEHOLDER(M3) alpha=1 combined adapter dW equals the unharmful dW exactly."""


def test_lora_alpha0_equals_harmful():
    """PLACEHOLDER(M3) alpha=0 combined adapter dW equals the harmful dW exactly."""


def test_lora_combined_delta_w_linear():
    """PLACEHOLDER(M3) Combined dW == a*dW_u + (1-a)*dW_h in fp32, rtol 1e-6."""


def test_full_alpha_endpoints_exact():
    """PLACEHOLDER(M3) (1-a)W_h + aW_u reproduces W_u at a=1 and W_h at a=0 (torch.equal)."""
