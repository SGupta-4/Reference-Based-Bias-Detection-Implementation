"""Anchor pools for the F4 ablation (D-015, D-081): filters, sampling, word pool, nesting,
and the `sentences` stage adding the pools to the union. No network: rows are stubs."""

import json

import pytest

from rbbd import config, runner
from rbbd.data import anchors as A
from rbbd.data import sentences as S
from rbbd.data.hashing import normalize
from rbbd.utils import env as env_mod


def _word(i, prefix):
    """A distinct lowercase alphabetic word per i (the word pool counts only those)."""
    letters = ""
    while True:
        letters = chr(97 + i % 26) + letters
        i //= 26
        if not i:
            return prefix + letters


def fake_fetch(n_alpaca=1500, n_tulu=1500):
    """Stub `fetch`: Alpaca/Tulu-shaped rows, a few of which every filter must drop."""

    def fetch(source, spec):
        if source == "alpaca":
            rows = [{"instruction": f"Describe the {w} garden near the river today.", "input": ""}
                    for w in (_word(i, "kel") for i in range(n_alpaca))]  # fmt: skip
            rows += [
                {"instruction": "Translate this sentence please now.", "input": "Bonjour"},
                {"instruction": "Too short.", "input": ""},
                {"instruction": "Compute x = 2 + 3 for the given values.", "input": ""},
                {"instruction": "Write a poem. Then explain it in detail.", "input": ""},
            ]
            return rows, "sha-alpaca"
        rows = [{"source": "ai2-adapt-dev/oasst", "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": f"Suggest a quiet hobby for rainy {w} evenings."}]}
                for w in (_word(i, "mor") for i in range(n_tulu))]  # fmt: skip
        rows.append(
            {
                "source": "ai2-adapt-dev/wildguardmixtrain",
                "messages": [
                    {"role": "user", "content": "Explain how the old harbour bridge was built."}
                ],
            }
        )
        return rows, "sha-tulu"

    return fetch


POOL_CFG = {
    "size": 1000, "seed": 0,
    "alpaca": {"dataset": "tatsu-lab/alpaca"},
    "tulu": {"dataset": "allenai/tulu-3-sft-mixture", "exclude_sources": ["wildguard"]},
    "word": {"from": ["alpaca", "tulu"]},
}  # fmt: skip


@pytest.mark.parametrize(
    "text,ok",
    [
        ("Name three rivers that flow through Europe.", True),
        ("Name rivers.", False),  # < 5 words
        ("Write a story. Then summarise it.", False),  # two sentences
        ("Solve 3 * x + 2 = 11 for x.", False),  # maths characters
        ("Visit https://example.com and summarise the page.", False),
        ("Décris une journée typique dans une grande ville.", False),  # not ASCII
        (
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
            "fifteen sixteen",
            False,
        ),  # fmt: skip
    ],
)
def test_instruction_filter(text, ok):
    assert A.instruction_ok(text) is ok


def test_build_pools_sizes_filters_and_determinism():
    """1,000 per source; filtered rows and excluded sources never appear; same seed ->
    same pools; authored sentences are never sampled."""
    authored = {normalize("Describe the kela garden near the river today.")}
    pools, stats = A.build_pools(POOL_CFG, authored, ["She felt joy."], fetch=fake_fetch())
    assert {k: len(v) for k, v in pools.items()} == {"alpaca": 1000, "tulu": 1000, "word": 1000}
    assert "Describe the kela garden near the river today." not in pools["alpaca"]
    assert not any("harbour" in s for s in pools["tulu"])
    assert stats["tulu"]["candidates_by_source"] == {"ai2-adapt-dev/oasst": 1500}
    assert stats["alpaca"]["revision"] == "sha-alpaca" and stats["alpaca"]["candidates"] == 1500
    again, _ = A.build_pools(POOL_CFG, authored, ["She felt joy."], fetch=fake_fetch())
    assert again == pools
    other, _ = A.build_pools({**POOL_CFG, "seed": 1}, authored, [], fetch=fake_fetch())
    assert other["alpaca"] != pools["alpaca"]


def test_word_pool_excludes_stopwords_groups_and_lexicon():
    texts = ["the garden river garden women joy quiet quiet quiet"] * 3 + ["river"]
    words = A.word_pool(texts, 3, banned=A.lexicon(["She felt joy."]))
    assert words == ["quiet", "garden", "river"]  # by count, then alphabetical
    with pytest.raises(ValueError, match="distinct content words"):
        A.word_pool(texts, 4, banned={"joy"})


def test_subset_rows_nested_and_seeded():
    small, large = A.subset_rows(1000, 50, seed=3), A.subset_rows(1000, 250, seed=3)
    assert len(small) == 50 and set(small) <= set(large)
    assert A.subset_rows(1000, 50, seed=4) != small
    assert A.subset_rows(1000, A.ALL, seed=0) == list(range(1000))


def test_sentences_stage_adds_pools_to_union(config_dir, artifacts, monkeypatch):
    """The stage adds `anchors/<source>` index entries; the smoke subset slices every
    source; the resolved dataset revisions are recorded."""
    monkeypatch.setattr(A, "fetch_rows", fake_fetch())
    base = "run_name: b\nseed: 0\nsentences:\n  anchor_pools: " + json.dumps(POOL_CFG) + "\n"
    tier = ("run_name: t\nsentences:\n  subset: {groups: [Women, Muslims], n_targets: 3,"
            " n_attr: 4, n_anchors: 20, variants: [base]}\n")  # fmt: skip
    cfg = config.load(config_dir(base, tier))
    runner.run(cfg, env_mod.detect(), ["sentences"])
    union = json.loads(next(artifacts.rglob("sentences/*/union.json")).read_text())
    for src in ("neutral", "alpaca", "tulu", "word"):
        assert len(union["index"][f"anchors/{src}"]) == 20
    assert union["stats"]["anchor_pools"]["tulu"]["revision"] == "sha-tulu"
    assert "anchors/word" in union["set_hashes"]


def test_add_anchor_pools_keeps_authored_sets():
    sets = S.load_sets()
    out, _ = S.add_anchor_pools(sets, POOL_CFG, fetch=fake_fetch())
    assert out.anchors["neutral"] == sets.anchors["neutral"] and out.targets == sets.targets
    assert S.validate(out)  # pools do not enter the authored-set validation
