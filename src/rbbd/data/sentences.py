"""Sentence sets: load the frozen sources, derive every variant, validate, hash, union.

Sources live in `rbbd/resources/sentences/` (D-012–D-016):
- `targets.tsv`: 50 target templates with `{Group}`/`{group}` placeholders, synonym
  slots `[a|b|c|d]`, and authored passive / rephrased-passive forms;
- `anchors.tsv`: 42 neutral anchor templates, rendered to 1,000 anchors;
- `positive.src`, `negative.src`: 100 attribute sentences each, with synonym slots.

Variants (App. E, T4):
- attributes: base, subj_v1 (they/their), subj_v2 (the person/people), syn_v1..v3;
- targets: base, passive, passive_rephr, syn_v1..v3.
Synonym variant k takes option k of every slot, so all variants stay row-aligned with
base by construction. Subject variants are derived from base by `subject_variant`.

Called by the `sentences` stage (`stage`), by `embed` (M2) through `build_union`, and
by `metrics.delta_b` (M4) through the index maps of `SentenceUnion`.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rbbd.data.groups import BY_NAME, GROUP_NAMES, GROUP_TOKENS
from rbbd.data.hashing import normalize, set_hash, text_hash

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources" / "sentences"

ATTR_VARIANTS = ("base", "subj_v1", "subj_v2", "syn_v1", "syn_v2", "syn_v3")
TARGET_VARIANTS = ("base", "passive", "passive_rephr", "syn_v1", "syn_v2", "syn_v3")
N_TEMPLATES = 50
N_ATTR = 100
N_ANCHORS = 1000
SLOT_OPTIONS = 4

# App. F examples that must appear verbatim (D-012–D-014).
APP_F_TARGET_TEMPLATES = (
    "{Group} live in many different countries.",
    "{Group} attend community events.",
    "{Group} are part of the local population.",
    "{Group} go to the park.",
)
APP_F_POSITIVE = (
    "She danced with pure joy in her heart.",
    "Their love grew stronger with every passing day.",
    "The morning light filled the room with warmth.",
)
APP_F_NEGATIVE = (
    "He felt shame after being caught in a lie.",
    "She began to cry when the pain was unbearable.",
    "They lived under oppression for decades.",
)
APP_F_ANCHORS = (
    "Homosexuals speak multiple languages and dialects.",
    "Many Asians participate in online discussions.",
    "Women use technology in their daily routines.",
)

# Words with clear valence. Targets and anchors are neutral by design, so neither may
# contain one of these or any word used in a P/N synonym slot (D-012, D-014).
VALENCE_WORDS = frozenset(
    {
        "good", "bad", "great", "best", "worst", "happy", "sad", "love", "hate", "enjoy",
        "like", "dislike", "nice", "terrible", "wonderful", "awful", "beautiful", "ugly",
        "kind", "cruel", "angry", "proud", "shame", "fear", "joy", "pain", "success",
        "failure", "excellent", "lazy", "smart", "stupid", "honest", "dishonest",
        "friendly", "hostile", "peaceful", "violent", "dangerous", "safe", "warm",
        "trust", "help", "brave", "evil", "criminal", "dirty", "clever", "loyal",
    }
)  # fmt: skip

# Function words ignored when collecting P/N slot words (multi-word options such as
# "relied on" or "worn out" would otherwise ban "on" and "out" from neutral sentences).
STOPWORDS = frozenset(
    {"a", "an", "the", "on", "at", "in", "up", "out", "of", "to", "by", "for", "with", "and"}
)

GENDERED = frozenset({"he", "she", "him", "her", "his", "hers", "himself", "herself"})
SUBJECT_PRONOUNS = frozenset({"he", "she", "they"})
PASSIVE_RE = re.compile(r"\b(is|are|was|were)\b.*\bby \{group\}")
_SLOT_RE = re.compile(r"\[([^\]]+)\]")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


class SentenceValidationError(ValueError):
    """Raised by `validate` with every failed check listed."""


def words(text: str) -> list[str]:
    """Lower-cased word tokens (letters, apostrophes, hyphens) of `text`."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def slot_options(src: str) -> list[list[str]]:
    """The options of every `[a|b|c|d]` slot in `src`, in order."""
    return [opt.split("|") for opt in _SLOT_RE.findall(src)]


def render_slots(src: str, k: int) -> str:
    """Replace every slot by its k-th option (k=0: base wording, k=1..3: Synonyms v1-v3)."""
    return _SLOT_RE.sub(lambda m: m.group(1).split("|")[k], src)


def render_template(template: str, group: str) -> str:
    """Fill `{Group}` (sentence-initial, T2 spelling) and `{group}` (mid-sentence form)."""
    g = BY_NAME[group]
    return template.replace("{Group}", g.name).replace("{group}", g.mid)


def subject_variant(sentence: str, which: str) -> str:
    """Derive Subject v1 or v2 (App. E.1) from a base attribute sentence.

    v1: gendered pronouns -> they/their/them/themselves; "was" right after a replaced
        subject pronoun -> "were".
    v2: a sentence-initial subject pronoun -> "The person" (He/She) or "People" (They);
        remaining gendered pronouns -> their/them/themselves.
    Inanimate subjects ("The garden ...") are left unchanged in both variants. The P/N
    sources are written in the past tense with only possessive "her", so these rules
    stay grammatical (validated by `validate`).
    """
    tokens = sentence.split(" ")
    out: list[str] = []
    replaced_subject = False
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if i == 0 and low in SUBJECT_PRONOUNS:
            if which == "subj_v1":
                out.append("They")
            else:
                out.append("People" if low == "they" else "The person")
            replaced_subject = low in ("he", "she") or which == "subj_v2"
            continue
        if which == "subj_v1" and i == 1 and replaced_subject and low == "was":
            out.append("were")
            continue
        mapping = {"his": "their", "her": "their", "him": "them", "hers": "theirs",
                   "himself": "themselves", "herself": "themselves"}  # fmt: skip
        if low in mapping:
            rep = mapping[low]
            out.append(rep.capitalize() if tok[0].isupper() else rep)
        else:
            out.append(tok)
    return " ".join(out)


def _read_lines(path: Path) -> list[str]:
    return [
        ln.rstrip("\n") for ln in path.read_text().splitlines() if ln and not ln.startswith("#")
    ]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    lines = _read_lines(path)
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"), strict=True)) for ln in lines[1:]]


def anchor_assignment(n_templates: int, n_groups: int, total: int) -> list[tuple[int, int]]:
    """(group index, template index) pairs giving `total` anchors balanced over groups.

    Every group uses every template, except that the first `n_groups*n_templates - total`
    groups each drop one template (group i drops template (5*i + 7) mod n_templates), so
    group sizes differ by at most one (42 vs 41 for the paper's 1,000; D-014). The offset
    keeps the App. F anchor "Homosexuals speak multiple languages and dialects." (group
    0, template 0) in the set.
    """
    surplus = n_groups * n_templates - total
    if not 0 <= surplus <= n_groups:
        raise ValueError(f"cannot balance {total} anchors over {n_groups}x{n_templates}")
    pairs = []
    for gi in range(n_groups):
        drop = (5 * gi + 7) % n_templates if gi < surplus else None
        pairs.extend((gi, ti) for ti in range(n_templates) if ti != drop)
    return pairs


@dataclass(frozen=True)
class SentenceSets:
    """All sentence sets in their rendered form.

    targets[variant][group] -> 50 sentences (template order, aligned across variants)
    positives[variant], negatives[variant] -> 100 sentences (row-aligned)
    anchors["neutral"] -> 1,000 sentences; anchor_groups / anchor_templates align with it
    """

    targets: dict[str, dict[str, list[str]]]
    target_template_ids: list[str]
    target_sources: list[dict[str, str]]
    positives: dict[str, list[str]]
    negatives: dict[str, list[str]]
    positive_sources: list[str]
    negative_sources: list[str]
    anchors: dict[str, list[str]]
    anchor_groups: list[str]
    anchor_templates: list[str]
    anchor_template_texts: list[str]

    def hashes(self) -> dict[str, str]:
        """Content hash of every (set, variant) list, used in cache keys (D-019)."""
        out = {}
        for v, by_group in self.targets.items():
            out[f"targets/{v}"] = set_hash(s for g in by_group for s in by_group[g])
        for v, rows in self.positives.items():
            out[f"positive/{v}"] = set_hash(rows)
        for v, rows in self.negatives.items():
            out[f"negative/{v}"] = set_hash(rows)
        for src, rows in self.anchors.items():
            out[f"anchors/{src}"] = set_hash(rows)
        return out


def load_sets(root: Path = RESOURCE_DIR) -> SentenceSets:
    """Read the sources under `root` and render every variant (no validation)."""
    target_rows = _read_tsv(root / "targets.tsv")
    targets: dict[str, dict[str, list[str]]] = {v: {} for v in TARGET_VARIANTS}
    for group in GROUP_NAMES:
        for v in TARGET_VARIANTS:
            rendered = []
            for row in target_rows:
                if v == "base":
                    tpl = render_slots(row["base"], 0)
                elif v.startswith("syn_v"):
                    tpl = render_slots(row["base"], int(v[-1]))
                else:
                    tpl = row[v]
                rendered.append(render_template(tpl, group))
            targets[v][group] = rendered

    def attrs(path: Path) -> tuple[dict[str, list[str]], list[str]]:
        src = _read_lines(path)
        base = [render_slots(s, 0) for s in src]
        out = {"base": base}
        out["subj_v1"] = [subject_variant(s, "subj_v1") for s in base]
        out["subj_v2"] = [subject_variant(s, "subj_v2") for s in base]
        for k in (1, 2, 3):
            out[f"syn_v{k}"] = [render_slots(s, k) for s in src]
        return out, src

    positives, pos_src = attrs(root / "positive.src")
    negatives, neg_src = attrs(root / "negative.src")

    anchor_rows = _read_tsv(root / "anchors.tsv")
    pairs = anchor_assignment(len(anchor_rows), len(GROUP_NAMES), N_ANCHORS)
    anchors = [render_template(anchor_rows[ti]["template"], GROUP_NAMES[gi]) for gi, ti in pairs]
    return SentenceSets(
        targets=targets,
        target_template_ids=[r["id"] for r in target_rows],
        target_sources=target_rows,
        positives=positives,
        negatives=negatives,
        positive_sources=pos_src,
        negative_sources=neg_src,
        anchors={"neutral": anchors},
        anchor_groups=[GROUP_NAMES[gi] for gi, _ in pairs],
        anchor_templates=[anchor_rows[ti]["id"] for _, ti in pairs],
        anchor_template_texts=[r["template"] for r in anchor_rows],
    )


def _mean_words(rows: list[str]) -> float:
    return sum(len(words(r)) for r in rows) / len(rows)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    return len(sa & sb) / len(sa | sb)


def validate(sets: SentenceSets) -> dict[str, Any]:
    """Check the sets against PLAN §2 / App. F; raise `SentenceValidationError` on failure.

    Returns summary statistics (counts, mean lengths, subject mix) for the stage manifest.
    """
    from scipy.stats import ks_2samp

    errors: list[str] = []
    stats: dict[str, Any] = {}

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    # Counts and alignment.
    check(len(sets.target_template_ids) == N_TEMPLATES, "need 50 target templates")
    check(len(set(sets.target_template_ids)) == N_TEMPLATES, "target template ids not unique")
    for v in TARGET_VARIANTS:
        check(set(sets.targets[v]) == set(GROUP_NAMES), f"targets/{v}: groups != T2 groups")
        for g, rows in sets.targets[v].items():
            check(len(rows) == N_TEMPLATES, f"targets/{v}/{g}: {len(rows)} != 50")
            check(len(set(rows)) == len(rows), f"targets/{v}/{g}: duplicate sentences")
    for name, by_v in (("positive", sets.positives), ("negative", sets.negatives)):
        for v in ATTR_VARIANTS:
            check(len(by_v[v]) == N_ATTR, f"{name}/{v}: {len(by_v[v])} != 100")
            check(len(set(by_v[v])) == N_ATTR, f"{name}/{v}: duplicate sentences")
    anchors = sets.anchors["neutral"]
    check(len(anchors) == N_ANCHORS, f"anchors: {len(anchors)} != 1000")
    check(len(set(anchors)) == len(anchors), "anchors: duplicate sentences")
    sizes = Counter(sets.anchor_groups)
    check(max(sizes.values()) / min(sizes.values()) <= 1.1, f"anchor group balance {sizes}")

    # Slot integrity: 4 distinct options per slot, 1-2 slots per row.
    for name, src_rows in (
        ("positive", sets.positive_sources),
        ("negative", sets.negative_sources),
        ("targets", [r["base"] for r in sets.target_sources]),
    ):
        for i, src in enumerate(src_rows):
            opts = slot_options(src)
            check(1 <= len(opts) <= 2, f"{name}[{i}]: {len(opts)} slots")
            for o in opts:
                check(len(o) == SLOT_OPTIONS, f"{name}[{i}]: slot {o} needs 4 options")
                check(len(set(o)) == len(o), f"{name}[{i}]: duplicate options {o}")

    # Lengths (App. F: average 7 words) and P/N length matching.
    base_targets = [s for g in GROUP_NAMES for s in sets.targets["base"][g]]
    for name, rows in (
        ("targets", base_targets),
        ("positive", sets.positives["base"]),
        ("negative", sets.negatives["base"]),
        ("anchors", anchors),
    ):
        m = _mean_words(rows)
        stats[f"mean_words/{name}"] = round(m, 2)
        check(6.0 <= m <= 8.0, f"{name}: mean words {m:.2f} outside 7 +/- 1")
    ks = ks_2samp(
        [len(words(s)) for s in sets.positives["base"]],
        [len(words(s)) for s in sets.negatives["base"]],
    )
    stats["ks_pvalue_pos_neg_lengths"] = round(float(ks.pvalue), 4)
    check(ks.pvalue > 0.05, f"P/N word-count distributions differ (KS p={ks.pvalue:.3f})")

    # Overlaps and template disjointness.
    all_targets = {s for v in TARGET_VARIANTS for g in GROUP_NAMES for s in sets.targets[v][g]}
    attr_all = {
        s for by_v in (sets.positives, sets.negatives) for v in ATTR_VARIANTS for s in by_v[v]
    }
    check(not (set(anchors) & all_targets), "anchors overlap targets")
    check(not (set(anchors) & attr_all), "anchors overlap attribute sentences")
    check(not (set(sets.positives["base"]) & set(sets.negatives["base"])), "P and N overlap")
    target_tpls = {normalize(render_slots(r["base"], 0)) for r in sets.target_sources}
    for tpl in sets.anchor_template_texts:
        check(normalize(tpl) not in target_tpls, f"anchor template shared with targets: {tpl}")

    # Group tokens never appear in attribute sentences (any variant).
    for s in attr_all:
        hit = GROUP_TOKENS & set(words(s))
        check(not hit, f"attribute sentence contains group tokens {hit}: {s}")

    # Targets and anchors are neutral: no valence words, no P/N slot words.
    slot_words = {
        w
        for src in sets.positive_sources + sets.negative_sources
        for opts in slot_options(src)
        for o in opts
        for w in words(o)
        if w not in STOPWORDS
    }
    banned = VALENCE_WORDS | slot_words
    group_words = {w for g in GROUP_NAMES for w in words(g)}
    for s in all_targets | set(anchors):
        hit = (banned & set(words(s))) - group_words
        check(not hit, f"neutral sentence contains valenced words {hit}: {s}")

    # App. F examples present verbatim.
    base_tpls = {render_slots(r["base"], 0) for r in sets.target_sources}
    for t in APP_F_TARGET_TEMPLATES:
        check(t in base_tpls, f"App. F target template missing: {t}")
    for s in APP_F_POSITIVE:
        check(s in sets.positives["base"], f"App. F positive missing: {s}")
    for s in APP_F_NEGATIVE:
        check(s in sets.negatives["base"], f"App. F negative missing: {s}")
    for s in APP_F_ANCHORS:
        check(s in anchors, f"App. F anchor missing: {s}")

    # Attribute variants (App. E.1).
    subject_mix: Counter[str] = Counter()
    for name, by_v in (("positive", sets.positives), ("negative", sets.negatives)):
        for i, base in enumerate(by_v["base"]):
            first = base.split(" ")[0].lower()
            subject_mix[first if first in SUBJECT_PRONOUNS else "other"] += 1
            v1, v2 = by_v["subj_v1"][i], by_v["subj_v2"][i]
            check(not (GENDERED & set(words(v1))), f"{name}/subj_v1[{i}] gendered: {v1}")
            check(not (GENDERED & set(words(v2))), f"{name}/subj_v2[{i}] gendered: {v2}")
            if GENDERED & set(words(base)):
                they = {"they", "their", "them", "themselves", "theirs"}
                check(bool(they & set(words(v1))), f"{name}/subj_v1[{i}] lost the subject: {v1}")
            check(first_word(v2) not in SUBJECT_PRONOUNS, f"{name}/subj_v2[{i}] pronoun: {v2}")
            for k in (1, 2, 3):
                syn = by_v[f"syn_v{k}"][i]
                check(syn != base, f"{name}/syn_v{k}[{i}] equals base")
                check(_jaccard(syn, base) >= 0.5, f"{name}/syn_v{k}[{i}] too far from base")
    stats["attribute_subject_mix"] = dict(subject_mix)

    # Target variants (App. E.2).
    for row in sets.target_sources:
        base_tpl = render_slots(row["base"], 0)
        if row["passivizable"] == "1":
            check(bool(PASSIVE_RE.search(row["passive"])), f"targets {row['id']}: passive form")
        else:
            check(
                row["passive"] == base_tpl, f"targets {row['id']}: non-passivizable must keep base"
            )
        check(bool(PASSIVE_RE.search(row["passive_rephr"])), f"targets {row['id']}: passive_rephr")
        for k in (1, 2, 3):
            syn = render_slots(row["base"], k)
            check(_jaccard(syn, base_tpl) >= 0.5, f"targets {row['id']}: syn_v{k} too far")
    stats["n_passivizable_templates"] = sum(r["passivizable"] == "1" for r in sets.target_sources)
    stats["anchor_group_sizes"] = {"min": min(sizes.values()), "max": max(sizes.values())}

    if errors:
        raise SentenceValidationError(
            f"{len(errors)} sentence-set checks failed:\n" + "\n".join(errors)
        )
    return stats


def first_word(sentence: str) -> str:
    """Lower-cased first token of `sentence` ('' if empty)."""
    toks = words(sentence)
    return toks[0] if toks else ""


@dataclass
class SentenceUnion:
    """Deduplicated list of every sentence to embed, plus where each set's rows live in it.

    texts[i] is embedded once; index[(kind, variant)] (attributes, anchors) or
    index[("targets", variant, group)] gives the row positions (int list, set order).
    The embed stage (M2) sorts by token length itself; this order is only canonical.
    """

    texts: list[str] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)
    index: dict[tuple[str, ...], list[int]] = field(default_factory=dict)

    def _add(self, key: tuple[str, ...], rows: list[str], pos: dict[str, int]) -> None:
        idx = []
        for s in rows:
            if s not in pos:
                pos[s] = len(self.texts)
                self.texts.append(s)
                self.hashes.append(text_hash(s))
            idx.append(pos[s])
        self.index[key] = idx

    @property
    def union_hash(self) -> str:
        """Hash of the ordered union (part of the embedding cache key, D-019)."""
        return set_hash(self.texts)


def build_union(sets: SentenceSets) -> SentenceUnion:
    """Union of every target/attribute variant and every anchor source (E1, D-020)."""
    union = SentenceUnion()
    pos: dict[str, int] = {}
    for src, rows in sets.anchors.items():
        union._add(("anchors", src), rows, pos)
    for v in sets.positives:
        union._add(("positive", v), sets.positives[v], pos)
        union._add(("negative", v), sets.negatives[v], pos)
    for v, by_group in sets.targets.items():
        for g, rows in by_group.items():
            union._add(("targets", v, g), rows, pos)
    return union


def subset_sets(sets: SentenceSets, spec: Mapping[str, Any]) -> SentenceSets:
    """A smaller copy of validated `sets` for the Tier 0 smoke run (PLAN §6, D-038).

    spec = {groups: [...], n_targets, n_attr, n_anchors, variants (default ["base"])};
    the first rows of each set are kept, so a subset is a prefix of the full sets and
    its sentences are ones the full validation already passed.
    """
    groups = list(spec["groups"])
    variants = list(spec.get("variants", ["base"]))
    unknown = set(groups) - set(GROUP_NAMES)
    if unknown:
        raise ValueError(f"unknown groups in subset: {sorted(unknown)}")
    nt, na, nn = int(spec["n_targets"]), int(spec["n_attr"]), int(spec["n_anchors"])
    return dataclasses.replace(
        sets,
        targets={
            v: {g: sets.targets[v][g][:nt] for g in groups} for v in variants if v in sets.targets
        },
        target_template_ids=sets.target_template_ids[:nt],
        positives={v: sets.positives[v][:na] for v in variants if v in sets.positives},
        negatives={v: sets.negatives[v][:na] for v in variants if v in sets.negatives},
        anchors={"neutral": sets.anchors["neutral"][:nn]},
        anchor_groups=sets.anchor_groups[:nn],
        anchor_templates=sets.anchor_templates[:nn],
    )


def stage(ctx: Any) -> Any:
    """`sentences` stage: load, validate and write the union plus its index (FLOW.md).

    Outputs (relative to the artifact root): `sentences/<run_key>/union.json` holding
    texts, row hashes, index map, per-set hashes and validation statistics.
    """
    from rbbd.runner import StageResult

    sets = load_sets()
    stats = validate(sets)  # the full sets are always validated, also for a smoke subset
    subset = ctx.cfg.get("sentences.subset")
    if subset:
        sets = subset_sets(sets, subset)
        stats["subset"] = dict(subset)
    union = build_union(sets)
    payload = {
        "union_hash": union.union_hash,
        "set_hashes": sets.hashes(),
        "stats": stats,
        "texts": union.texts,
        "hashes": union.hashes,
        "index": {"/".join(k): v for k, v in union.index.items()},
    }
    out = ctx.stage_dir / "union.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return StageResult(outputs={"union": str(out.relative_to(ctx.env.artifacts_root))})
