"""M6-T6 NF4 probe plumbing (D-098): the quant field's cache-key behaviour and the probe
config's train keys. CPU only; the extraction itself is a Kaggle GPU run."""

import pytest

from rbbd import config
from rbbd.finetune import sft
from rbbd.models.loading import LoadSpec
from rbbd.utils.cache import make_key
from tests.conftest import REPO_ROOT


def test_quant_none_keeps_every_existing_key():
    spec = LoadSpec("m", "rev", "fp16", "balanced")
    before = {"model_id": "m", "revision": "rev", "precision": "fp16", "placement": "balanced",
              "quant": None}  # the dict key_fields() returned before D-098  # fmt: skip
    assert make_key(spec.key_fields()) == make_key(before)
    nf4 = LoadSpec("m", "rev", "fp16", "balanced", quant="nf4")
    assert make_key(nf4.key_fields()) != make_key(before)
    with pytest.raises(ValueError, match="quant"):
        LoadSpec("m", quant="int8")


def test_probe_config_reuses_the_tier1_llama_endpoints():
    """Only `embed` (and run_name) differ, so every train key matches and no training runs."""
    base = config.load(REPO_ROOT / "configs" / "tier1_llama3.1-8b.yaml")
    probe = config.load(REPO_ROOT / "configs" / "m6_nf4_llama3.1-8b.yaml")
    assert probe["models"] == base["models"] and probe.get("train") == base.get("train")
    assert probe.get("embed.quant") == "nf4"
    layout = sft.batch_layouts(base.get("train"), "qlora")[0]
    for split in sft.SPLITS:
        keys = [sft.spec_from_config(c, c["models"][0], "qlora", split, 0, "dkey", layout).train_key
                for c in (base, probe)]  # fmt: skip
        assert keys[0] == keys[1]
