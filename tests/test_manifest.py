"""Stage manifests (D-030)."""


def test_valid_manifest_skips_stage():
    """PLACEHOLDER(M0) A complete manifest with a matching config hash makes the runner skip the stage."""


def test_incomplete_manifest_resumes():
    """PLACEHOLDER(M0) complete=false triggers resume, not skip."""
