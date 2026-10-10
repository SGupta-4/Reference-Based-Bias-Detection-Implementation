"""Generation harness on CPU (M5; D-050, D-086): shards and resume (DC-13), deadline pause,
chat-template fallback, merged-checkpoint exactness, and the generate → score stages on
the tiny model with a stub engine. No vLLM, no network."""

import csv
import hashlib
import json

import pytest
import torch

from rbbd import config, runner
from rbbd.bench import generate as gen
from rbbd.bench import score as sc
from rbbd.models import adapters as ad
from rbbd.models.loading import LoadSpec, base_decoder
from rbbd.utils import env as env_mod
from tests.conftest import STUBS, SWEEP_TIER, make_chat_tokenizer, make_tiny_model

SAMPLING = gen.Sampling(n=2, temperature=0.7, top_p=1.0, max_new_tokens=8)


def _items(n, system=False):
    def msgs(i):
        head = [{"role": "system", "content": "w1 w2"}] if system else []
        return head + [{"role": "user", "content": " ".join(f"w{(i + k) % 90}" for k in range(3))}]

    return [{"id": f"item{i:03d}", "messages": msgs(i)} for i in range(n)]


class StubEngine:
    """generate_fn stand-in: deterministic text from (tag, seed, sample); records calls."""

    def __init__(self, tag="ref", fail_after=None, text=None):
        self.tag, self.calls, self.fail_after, self.text = tag, [], fail_after, text

    def __call__(self, prompts, params):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("killed")
        self.calls.append([p["seed"] for p in params])
        return [
            [
                {
                    "text": self.text or f"{self.tag}-{p['seed']}-{k} I agree.",
                    "finish_reason": "stop",
                    "n_tokens": 4,
                }
                for k in range(p["n"])
            ]
            for p in params
        ]


def test_shards_resume_without_duplicates(tmp_path):
    """DC-13 for generation: a run killed after one shard resumes, regenerates nothing that
    exists, and ends with every item × sample exactly once."""
    tok, items, out = make_chat_tokenizer(), _items(10), tmp_path / "g"
    with pytest.raises(RuntimeError, match="killed"):
        gen.generate_bench(StubEngine(fail_after=1), tok, items, SAMPLING, out, shard_size=4)
    assert gen.shard_path(out, 0).exists() and not gen.shard_path(out, 1).exists()
    assert not (out / "done.json").exists()
    engine = StubEngine()
    done = gen.generate_bench(engine, tok, items, SAMPLING, out, shard_size=4)
    assert len(engine.calls) == 2 and done["n_shards"] == 3 and done["shards_this_process"] == 2
    rows = gen.read_samples(out)
    assert sorted((r["id"], r["sample"]) for r in rows) == [
        (i["id"], k) for i in items for k in range(2)
    ]
    assert not list(out.glob("*.tmp"))


def test_seeds_are_paired_across_checkpoints(tmp_path):
    tok, items = make_chat_tokenizer(), _items(6)
    a, b = StubEngine("ref"), StubEngine("a050")
    gen.generate_bench(a, tok, items, SAMPLING, tmp_path / "a", shard_size=3, base_seed=0)
    gen.generate_bench(b, tok, items, SAMPLING, tmp_path / "b", shard_size=3, base_seed=0)
    assert a.calls == b.calls
    assert a.calls != [[gen.item_seed(1, i["id"]) for i in items[:3]]]


def test_long_prompts_are_skipped_not_truncated(tmp_path):
    tok = make_chat_tokenizer()
    items = _items(3)
    items[1]["messages"][0]["content"] = " ".join(["w5"] * 200)
    done = gen.generate_bench(
        StubEngine(), tok, items, SAMPLING, tmp_path / "g", max_prompt_tokens=40
    )
    assert done["skipped_too_long"] == ["item001"]
    assert {r["id"] for r in gen.read_samples(tmp_path / "g")} == {"item000", "item002"}


def test_deadline_pauses_between_shards(tmp_path, monkeypatch):
    tok, items, out = make_chat_tokenizer(), _items(8), tmp_path / "g"
    ticks = iter([0.0, 0.0, 10.0, 10.0])  # start, before shard 0, before shard 1, ...
    monkeypatch.setenv(gen.DEADLINE_ENV, "5")
    with pytest.raises(gen.GenerationPaused, match="before shard 1/2"):
        gen.generate_bench(
            StubEngine(), tok, items, SAMPLING, out, shard_size=4, clock=lambda: next(ticks)
        )
    assert gen.shard_path(out, 0).exists() and not (out / "done.json").exists()


def test_render_merges_a_rejected_system_turn():
    class NoSystem:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            if any(m["role"] == "system" for m in messages):
                raise ValueError("Conversation roles must alternate user/assistant")
            return "|".join(m["content"] for m in messages)

    text, mode = gen.render(
        NoSystem(), [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    )
    assert (text, mode) == ("S\n\nU", "merged-system")
    assert gen.render(make_chat_tokenizer(), _items(1, system=True)[0]["messages"])[1] == "chat"


def test_merged_lora_checkpoint_equals_the_adapter(tmp_path, monkeypatch, lora_spectrum):
    """D-050: W0 + ΔW(α) written to disk equals the PEFT model at α (the embedding path,
    D-004/D-007), so generations and embeddings see the same weights."""
    from transformers import LlamaForCausalLM

    from rbbd.models import spectrum as spm

    monkeypatch.setattr("rbbd.models.loading.download", lambda spec: "/nonexistent")
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: make_tiny_model()
    )
    monkeypatch.setattr("rbbd.models.loading.load_tokenizer", lambda *a, **k: make_chat_tokenizer())
    spec = LoadSpec("tiny/llama", "sha", "fp32", "cpu")
    out = tmp_path / "eph" / "a050"
    path, temporary = gen.materialize(spec, lora_spectrum, "a050", out, tmp_path / "artifacts")
    assert temporary and path == str(out)
    merged = LlamaForCausalLM.from_pretrained(path).eval()
    act = spm.Activator(lambda: (make_tiny_model(), None))
    peft_model, _ = act.get(lora_spectrum, "a050")
    ids = torch.tensor([[1, 5, 9, 17]])
    with torch.no_grad():
        a = base_decoder(merged)(input_ids=ids).last_hidden_state
        b = base_decoder(peft_model)(input_ids=ids).last_hidden_state
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="must not be written"):
        gen.materialize(
            spec, lora_spectrum, "a050", tmp_path / "artifacts" / "x", tmp_path / "artifacts"
        )
    assert ad.read_adapter(lora_spectrum.finals["unharmful"])  # endpoints untouched


BENCH_BASE = (
    "run_name: base\nseed: 0\nschema_version: {embed: 1, gen: 1, score: 1}\n"
    "embed: {checkpoints: [ref, a100, a050, a000]}\n"
    "bench: {benches: [dt, toxigen], parallel: false, shard_size: 4,\n"
    "        dt: {n: 2, max_new_tokens: 8}, toxigen: {n: 2, max_new_tokens: 8},\n"
    "        bootstrap: {n: 50, seed: 0}}\n"
)


def _fake_items(cfg, out_dir, benches):
    """prepare_items stand-in: 2 groups × 2 topics × 3 sys × 3 variants DT, 2 × 3 ToxiGen."""
    from rbbd.bench.generate import subset_hash

    dt_items = [
        {
            "id": f"dt-{s}-{g}-{t}-{v}",
            "sys_type": s,
            "group": g,
            "topic": t,
            "variant": v,
            "messages": [
                {"role": "system", "content": "w1"},
                {"role": "user", "content": f"w{v} w2"},
            ],
        }
        for s in ("benign", "untargeted", "targeted")
        for g in ("Women", "Muslims")
        for t in ("hiv", "stem")
        for v in range(3)
    ]
    tox = [
        {"id": f"tg-{g}-{i}", "group": g, "messages": [{"role": "user", "content": f"w{i} w3"}]}
        for g in ("Women", "Muslims")
        for i in range(3)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = {}
    for name, items, tags in (
        (
            "dt",
            dt_items,
            {
                i["id"]: {k: i[k] for k in ("sys_type", "group", "topic", "variant")}
                for i in dt_items
            },
        ),
        ("toxigen", tox, {i["id"]: {"group": i["group"]} for i in tox}),
    ):
        (out_dir / f"items_{name}.json").write_text(json.dumps(items))
        metas[name] = {
            "source": {"stub": name},
            "items": subset_hash(items),
            "n_items": len(items),
            "tags": tags,
        }
    return metas


def _engine_factory(path, settings):
    """Agree on harmful merges, disagree elsewhere, so DT Δ is visible and directional."""
    tag = hashlib.sha256(str(path).encode()).hexdigest()[:6]
    harmful = str(path).endswith("a000") or str(path).endswith("a050")
    return StubEngine(tag, text="I agree." if harmful else "I disagree.")


def test_generate_and_score_stages_end_to_end(config_dir, artifacts, monkeypatch, tmp_path, sweep):
    cfg, env, _ = sweep
    monkeypatch.setenv("RBBD_EPHEMERAL", str(tmp_path / "eph"))
    monkeypatch.setattr(gen, "prepare_items", _fake_items)
    monkeypatch.setattr(gen, "ENGINE_FACTORY", _engine_factory)
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: make_tiny_model()
    )
    monkeypatch.setattr("rbbd.models.loading.load_tokenizer", lambda *a, **k: make_chat_tokenizer())
    monkeypatch.setattr(
        sc,
        "label_toxigen",
        lambda by_job, classify_fn=None: {
            job: [
                {
                    "id": s["id"],
                    "sample": s["sample"],
                    "p_toxic": 0.0,
                    "toxic": False,
                    "empty": False,
                }
                for s in ss
            ]
            for job, ss in by_job.items()
        },
    )
    cfg = config.load(
        config_dir(BENCH_BASE + f"paths: {{results: {tmp_path / 'results'}}}\n", SWEEP_TIER)
    )
    out = runner.run(
        cfg, env_mod.detect(), ["ftdata", "train", "generate", "score"], stage_fns=STUBS
    )
    assert out["generate"] == "ran" and out["score"] == "ran"
    assert not list((tmp_path / "eph" / "rbbd_merged").glob("*")), "merged copies must be deleted"
    with open(tmp_path / "results" / "t" / "bench_scores.csv") as fh:
        rows = list(csv.DictReader(fh))
    dt_rows = [r for r in rows if r["bench"] == "dt" and r["regime"] == "lora"]
    by = {(r["ckpt"], r["unit"]): r for r in dt_rows}
    assert (
        float(by[("ref", "Women")]["score"]) == 0.0 and float(by[("a000", "Women")]["score"]) == 1.0
    )
    assert (
        float(by[("a000", "Women")]["delta"]) == 1.0
        and float(by[("a100", "Women")]["delta"]) == 0.0
    )
    assert {r["regime"] for r in rows} == {"lora", "full"} and {r["bench"] for r in rows} == {
        "dt",
        "toxigen",
    }
    # DC-14: seconds per (checkpoint, benchmark), one process each here.
    (timing_path,) = list((artifacts / "generate").glob("*/timing.json"))
    timing = json.loads(timing_path.read_text())
    assert {(t["ckpt"], t["bench"]) for t in timing} >= {("ref", "dt"), ("a000", "toxigen")}
    assert all(t["processes"] == 1 and t["seconds"] >= 0 for t in timing)
    # A second run is a cache hit and launches nothing.
    again = runner.run(cfg, env_mod.detect(), ["generate", "score"], stage_fns=STUBS)
    assert again == {"generate": "cache hit", "score": "cache hit"}
