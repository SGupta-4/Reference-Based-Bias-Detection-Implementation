"""Config loading and hashing (config.py; ARCHITECTURE §4)."""

from rbbd import config
from tests.conftest import REPO_ROOT

BASE = """
run_name: base
seed: 0
store: {repo_id: org/repo, repo_type: auto}
alphas: [1.0, 0.5, 0.0]
paths: {artifacts: null}
"""


def test_merge_base_and_tier(config_dir):
    """base.yaml <- tier file merge: mappings merge per key, scalars and lists are replaced."""
    tier = config_dir(BASE, "run_name: smoke\nstore: {repo_type: dataset}\nalphas: [1.0, 0.0]\n")
    cfg = config.load(tier)
    assert cfg["run_name"] == "smoke"
    assert cfg.get("store.repo_type") == "dataset"
    assert cfg.get("store.repo_id") == "org/repo"  # sibling key kept from base
    assert cfg["alphas"] == [1.0, 0.0]  # list replaced, not concatenated
    assert cfg.get("missing.key", "default") == "default"


def test_config_hash_stable_and_order_independent(config_dir, tmp_path):
    """Same content in different key order gives the same hash; content changes it; paths do not."""
    a = config.load(config_dir(BASE, "run_name: x\nseed: 1\nmodel: {id: m, layer: 32}\n", "a.yaml"))
    b = config.load(config_dir(BASE, "model: {layer: 32, id: m}\nseed: 1\nrun_name: x\n", "b.yaml"))
    assert a.hash == b.hash and len(a.hash) == 16
    c = config.load(config_dir(BASE, "run_name: x\nseed: 1\nmodel: {id: m, layer: 34}\n", "c.yaml"))
    assert c.hash != a.hash
    d = config.load(
        config_dir(
            BASE,
            "run_name: x\nseed: 1\nmodel: {id: m, layer: 32}\npaths: {artifacts: /elsewhere}\n",
            "d.yaml",
        )
    )
    assert d.hash == a.hash  # presentation-only section


def test_missing_required_key_raises(config_dir):
    """A merged config without `seed` is rejected."""
    tier = config_dir("run_name: base\n", "run_name: t\n")
    try:
        config.load(tier)
    except config.ConfigError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("ConfigError not raised")


def test_repo_configs_load():
    """The committed base and smoke configs parse, merge and point at the private store (D-041)."""
    smoke = config.load(REPO_ROOT / "configs" / "smoke.yaml")
    assert smoke["run_name"] == "smoke" and smoke["tier"] == 0
    assert smoke.get("store.repo_id") == "SarthakGupta414/rbbd-artifacts"
    assert smoke["alphas"] == [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
    assert len(smoke.get("probe.access_repos")) == 13
