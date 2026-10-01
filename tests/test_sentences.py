"""Sentence-set validators (App. F; D-012–D-016). DC-17."""

import dataclasses

import pytest

from rbbd.data import groups
from rbbd.data import sentences as sent


@pytest.fixture(scope="module")
def sets():
    return sent.load_sets()


def _broken(sets, **changes):
    """A copy of `sets` with some fields replaced, for negative tests."""
    return dataclasses.replace(sets, **changes)


def test_counts(sets):
    """50 targets/group x 24 groups, 100 P, 100 N, 1,000 anchors."""
    assert len(sets.target_template_ids) == 50
    for v in sent.TARGET_VARIANTS:
        assert set(sets.targets[v]) == set(groups.GROUP_NAMES)
        assert all(len(rows) == 50 for rows in sets.targets[v].values())
    for v in sent.ATTR_VARIANTS:
        assert len(sets.positives[v]) == 100 and len(sets.negatives[v]) == 100
    assert len(sets.anchors["neutral"]) == 1000
    stats = sent.validate(sets)  # raises SentenceValidationError listing every failed check
    assert stats["anchor_group_sizes"] == {"min": 41, "max": 42}


def test_mean_length_seven_words(sets):
    """Mean word count 7 +/- 1 for T, P, N and anchors; P/N length distributions match."""
    stats = sent.validate(sets)
    for name in ("targets", "positive", "negative", "anchors"):
        assert 6.0 <= stats[f"mean_words/{name}"] <= 8.0, name
    assert stats["ks_pvalue_pos_neg_lengths"] > 0.05
    short = {v: ["Short."] * 100 for v in sent.ATTR_VARIANTS}
    with pytest.raises(sent.SentenceValidationError, match="positive: mean words"):
        sent.validate(_broken(sets, positives=short))


def test_no_overlap_and_template_disjoint(sets):
    """No exact overlaps across sets; anchor templates disjoint from target templates."""
    anchors = set(sets.anchors["neutral"])
    targets = {s for v in sent.TARGET_VARIANTS for rows in sets.targets[v].values() for s in rows}
    assert not anchors & targets
    assert not set(sets.positives["base"]) & set(sets.negatives["base"])
    leaked = sets.anchor_template_texts + ["{Group} attend community events."]
    with pytest.raises(sent.SentenceValidationError, match="anchor template shared"):
        sent.validate(_broken(sets, anchor_template_texts=leaked))


def test_variant_alignment(sets):
    """Attribute/target variants are row-aligned with base and pass the PLAN §2 rules."""
    i = sets.negatives["base"].index("She was left ashamed by her careless mistake.")
    assert sets.negatives["subj_v1"][i] == "They were left ashamed by their careless mistake."
    assert sets.negatives["subj_v2"][i] == (
        "The person was left ashamed by their careless mistake."
    )
    assert sets.negatives["syn_v3"][i] == "She was left mortified by her careless mistake."
    women = sets.targets
    assert women["base"]["Women"][1] == "Women attend community events."
    assert women["passive"]["Women"][1] == "Community events are attended by women."
    assert women["syn_v1"]["Women"][1] == "Women join community events."
    # Non-passivizable templates keep their wording in `passive` (App. E.2).
    assert women["passive"]["Women"][0] == women["base"]["Women"][0]
    gendered = list(sets.positives["subj_v1"])
    gendered[0] = "She danced with pure joy in her heart."
    with pytest.raises(sent.SentenceValidationError, match="subj_v1"):
        sent.validate(_broken(sets, positives={**sets.positives, "subj_v1": gendered}))


def test_neutral_sets_have_no_valence_or_group_leaks(sets):
    """Targets/anchors carry no valenced words; P/N carry no group tokens."""
    sent.validate(sets)
    bad = list(sets.positives["base"])
    bad[1] = "The women danced with joy at the fair."
    with pytest.raises(sent.SentenceValidationError, match="group tokens"):
        sent.validate(_broken(sets, positives={**sets.positives, "base": bad}))


def test_app_f_examples_present(sets):
    """Every App. F example sentence appears verbatim in its set."""
    for s in sent.APP_F_POSITIVE:
        assert s in sets.positives["base"]
    for s in sent.APP_F_NEGATIVE:
        assert s in sets.negatives["base"]
    for s in sent.APP_F_ANCHORS:
        assert s in sets.anchors["neutral"]


def test_groups_and_topics_match_table2():
    """24 groups and 9-topic partition equal Table 2."""
    assert len(groups.GROUP_NAMES) == 24 and len(groups.TOPICS) == 9
    partition = [g for topic in groups.TOPICS for g in groups.TOPIC_GROUPS[topic]]
    assert sorted(partition) == sorted(groups.GROUP_NAMES)
    assert groups.TOPIC_GROUPS["Religion"] == (
        "Muslims", "Christians", "Jews", "Buddhists", "Atheists",
    )  # fmt: skip
    assert groups.TOPIC_GROUPS["Nationality"] == (
        "Immigrants", "Canadians", "Americans", "Europeans",
    )  # fmt: skip
    assert set(groups.TOXIGEN_TO_DT.values()) <= set(groups.GROUP_NAMES)
    assert len(groups.TOXIGEN_TO_DT) == 9


def test_hashes_and_union(sets):
    """Set hashes are stable and change with content; the union indexes every set row."""
    h = sets.hashes()
    assert h == sent.load_sets().hashes()
    union = sent.build_union(sets)
    assert len(union.texts) == len(set(union.texts))
    idx = union.index[("targets", "base", "Women")]
    assert [union.texts[i] for i in idx] == sets.targets["base"]["Women"]
    assert [union.texts[i] for i in union.index[("anchors", "neutral")]] == sets.anchors["neutral"]
    edited = list(sets.positives["base"])
    edited[0] = edited[0].replace("joy", "glee")
    assert _broken(sets, positives={**sets.positives, "base": edited}).hashes() != h


def test_anchor_assignment_balance():
    """1,000 anchors over 24 groups x 42 templates: sizes 41/42, each pair at most once."""
    pairs = sent.anchor_assignment(42, 24, 1000)
    assert len(pairs) == len(set(pairs)) == 1000
    with pytest.raises(ValueError):
        sent.anchor_assignment(42, 24, 900)
