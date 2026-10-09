"""Shared fixtures: isolated artifact root, a two-file config directory, synthetic embeddings."""

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Point `$RBBD_ARTIFACTS` at an empty per-test directory."""
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setenv("RBBD_ARTIFACTS", str(root))
    return root


@pytest.fixture
def config_dir(tmp_path):
    """Write `base.yaml` + a tier file and return a function tier_text -> tier path."""

    def make(base_text: str, tier_text: str, tier_name: str = "tier.yaml") -> Path:
        d = tmp_path / "configs"
        d.mkdir(exist_ok=True)
        (d / "base.yaml").write_text(base_text)
        (d / tier_name).write_text(tier_text)
        return d / tier_name

    return make


def make_embedding_set(seed=0, d=16, n_per_group=6, groups=("A", "B", "C"), p=8, q=8, m=40):
    """Synthetic EmbeddingSet: Gaussian rows with a shared offset so cosines are non-trivial."""
    from rbbd.metrics.delta_b import EmbeddingSet

    rng = np.random.default_rng(seed)
    shift = rng.normal(size=d)

    def rows(k):
        return rng.normal(size=(k, d)) + shift

    labels = tuple(g for g in groups for _ in range(n_per_group))
    return EmbeddingSet(rows(len(labels)), labels, rows(p), rows(q), rows(m))


@pytest.fixture
def emb():
    return make_embedding_set()


class FakeTokenizer:
    """Word-level stand-in for a HF tokenizer: BOS id 1, pad id 0, right padding.

    Implements only the calls `embed.extract` makes, so CPU tests need no download.
    """

    padding_side = "right"
    pad_token_id = 0
    vocab_size = 128

    def _ids(self, text, add_special_tokens, truncation, max_length):
        ids = [2 + (sum(map(ord, w)) % (self.vocab_size - 2)) for w in text.split()]
        if add_special_tokens:
            ids = [1, *ids]
        return ids[:max_length] if truncation and max_length else ids

    def __call__(self, texts, add_special_tokens=True, truncation=False, max_length=None,
                 padding=False, return_tensors=None):  # fmt: skip
        import torch

        rows = [self._ids(t, add_special_tokens, truncation, max_length) for t in texts]
        if not padding:
            return {"input_ids": rows}
        width = max(len(r) for r in rows)
        ids = [r + [0] * (width - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        return {"input_ids": ids, "attention_mask": mask}


def make_tiny_model(seed=0, layers=2):
    """A 2-layer random Llama (hidden 16) on CPU in fp32, built from a config."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
    )  # fmt: skip
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture
def tiny():
    return make_tiny_model(), FakeTokenizer()


CHAT_TEMPLATE = (
    "{{ bos_token }} {% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} <end> {% endfor %}"
    "{% if add_generation_prompt %}<assistant> {% endif %}"
)


def make_chat_tokenizer():
    """A real (offline) word-level `PreTrainedTokenizerFast` with a chat template.

    Vocabulary: specials, role tags and words w0..w99 (ids < 128, the tiny model's vocab).
    The prompt rendered with a generation prompt is a token prefix of prompt + completion,
    as with the real templates (checked by `sft.build_examples`).
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>", "<end>"]
    vocab = {t: i for i, t in enumerate(specials + [f"w{i}" for i in range(100)])}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<s>", eos_token="</s>",
                                  pad_token="<pad>", unk_token="<unk>")  # fmt: skip
    tok.chat_template = CHAT_TEMPLATE
    tok.padding_side = "right"
    return tok


def make_rows(n, seed=0, prompt_words=(3, 8), response_words=(2, 6)):
    """`n` synthetic WildGuardMix-like rows ({"prompt", "response"}) over words w0..w99."""
    rng = np.random.default_rng(seed)

    def text(lo, hi):
        return " ".join(f"w{int(i)}" for i in rng.integers(0, 100, size=int(rng.integers(lo, hi))))

    return [{"prompt": text(*prompt_words), "response": text(*response_words)} for _ in range(n)]


# --- M4 spectrum fixtures (tests/test_spectrum.py, tests/test_delta_b_table.py) ----------


def random_adapter(tmp_path, name, seed):
    """A real PEFT LoRA (r=4, alpha=8, random A and B) on the tiny model, saved to disk."""
    import torch
    from peft import LoraConfig, get_peft_model

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(
        r=4, lora_alpha=8, target_modules=targets, init_lora_weights=False, task_type="CAUSAL_LM"
    )
    base = make_tiny_model()  # reseeds the global RNG itself, so seed the adapter after it
    torch.manual_seed(seed)
    model = get_peft_model(base, cfg)
    out = tmp_path / name
    model.save_pretrained(str(out))
    return out


def full_endpoint(tmp_path, name, seed):
    """The tiny model with every weight perturbed, saved as safetensors (a full-FT final)."""
    import torch

    model = make_tiny_model()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen))
    out = tmp_path / name
    model.save_pretrained(str(out), safe_serialization=True)
    return out


def spectrum_of(regime, finals, seed=0):
    """A `models.spectrum.Spectrum` of the tiny model with the given `final/` dirs."""
    from rbbd.models import spectrum as spm

    model = {"id": "tiny/llama", "slug": "tiny", "layer": 2}
    keys = {"unharmful": f"{regime}-u", "harmful": f"{regime}-h"}
    return spm.Spectrum(model, regime, seed, finals, keys)


@pytest.fixture
def lora_spectrum(tmp_path):
    finals = {"unharmful": random_adapter(tmp_path, "u", 1),
              "harmful": random_adapter(tmp_path, "h", 2)}  # fmt: skip
    return spectrum_of("lora", finals)


@pytest.fixture
def full_spectrum(tmp_path):
    finals = {"unharmful": full_endpoint(tmp_path, "fu", 3),
              "harmful": full_endpoint(tmp_path, "fh", 4)}  # fmt: skip
    return spectrum_of("full", finals)


SWEEP_BASE = (
    "run_name: base\nseed: 0\nschema_version: {embed: 1}\n"
    "embed: {checkpoints: [ref, a100, a090, a070, a050, a030, a010, a000], poolings: [mean],"
    " layer_sites: [post_norm], max_length: 64, token_budget: 4096, max_batch: 64}\n"
)
SWEEP_TIER = (
    "run_name: t\nmodels:\n"
    "  - {id: tiny/llama, slug: tiny, layer: 2, precision: fp32, placement: cpu}\n"
    "sentences:\n  subset: {groups: [Women, Muslims, Immigrants], n_targets: 5, n_attr: 10,"
    " n_anchors: 50, variants: [base]}\n"
)


def _stub_stage(ctx):
    """Stands in for `ftdata`/`train` so `embed` finds complete upstream manifests."""
    from rbbd.runner import StageResult

    return StageResult(outputs={})


STUBS = {"ftdata": _stub_stage, "train": _stub_stage}


@pytest.fixture
def sweep(config_dir, artifacts, monkeypatch, lora_spectrum, full_spectrum):
    """A config sweeping ref + 7 α over one LoRA and one full spectrum of the tiny model;
    `loads` counts base loads."""
    from rbbd import config
    from rbbd.embed import extract
    from rbbd.models import spectrum as spm
    from rbbd.utils import env as env_mod

    loads = []

    def fake_load(spec):
        loads.append(spec)
        return make_tiny_model(), FakeTokenizer()

    monkeypatch.setattr("rbbd.models.loading.load_for_inference", fake_load)
    monkeypatch.setattr("rbbd.models.loading.download", lambda spec: "/nonexistent")
    monkeypatch.setattr(extract, "resolve_revision", lambda mid, rev: "sha-test")
    monkeypatch.setattr(spm, "spectra", lambda cfg, root, outs: [lora_spectrum, full_spectrum])
    cfg = config.load(config_dir(SWEEP_BASE, SWEEP_TIER))
    return cfg, env_mod.detect(), loads


