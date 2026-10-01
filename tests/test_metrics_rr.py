"""RR and ΔB invariants (Eq. 4–7). DC-03, DC-04, DC-05."""

import dataclasses

import numpy as np

from rbbd.metrics import rr
from rbbd.metrics.delta_b import delta_b_all, group_bias


def _transform(es, fn):
    """Apply `fn` to every embedding matrix of an EmbeddingSet."""
    return dataclasses.replace(
        es,
        targets=fn(es.targets),
        positives=fn(es.positives),
        negatives=fn(es.negatives),
        anchors=fn(es.anchors),
    )


def test_identity_delta_b_zero(emb):
    """ΔB(M, M) == 0 for every group, atol 1e-6, for RR and SEAT (and Procrustes, CKA drift)."""
    out = delta_b_all(emb, emb)
    for method in ("rr", "seat", "procrustes", "cka"):
        assert set(out[method]) == {"A", "B", "C"}
        np.testing.assert_allclose(list(out[method].values()), 0.0, atol=1e-6)


def test_rotation_scale_invariance(emb):
    """Random orthogonal Q and scale s>0 on one model leave r(x) and ΔB unchanged, atol 1e-5."""
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(emb.targets.shape[1],) * 2))
    aud = _transform(emb, lambda x: 3.7 * x @ q)
    np.testing.assert_allclose(
        rr.relative(aud.targets, aud.anchors), rr.relative(emb.targets, emb.anchors), atol=1e-5
    )
    out = delta_b_all(emb, aud)
    np.testing.assert_allclose(list(out["rr"].values()), 0.0, atol=1e-5)
    # A rotated copy is exactly what Procrustes undoes and CKA ignores (§4.5).
    np.testing.assert_allclose(list(out["procrustes"].values()), 0.0, atol=1e-5)
    np.testing.assert_allclose(list(out["cka"].values()), 0.0, atol=1e-5)


def test_anchor_permutation_invariance(emb):
    """Permuting the anchors (the same sentences in both models) leaves ΔB unchanged."""
    aud = _transform(emb, lambda x: x + 0.3 * np.sin(x))
    perm = np.random.default_rng(2).permutation(emb.anchors.shape[0])
    base = delta_b_all(emb, aud)["rr"]
    permuted = delta_b_all(
        dataclasses.replace(emb, anchors=emb.anchors[perm]),
        dataclasses.replace(aud, anchors=aud.anchors[perm]),
    )["rr"]
    np.testing.assert_allclose(list(permuted.values()), list(base.values()), atol=1e-6)


def test_sign_convention():
    """Moving targets toward N makes ΔB negative and toward P positive (§3.4)."""
    from rbbd.metrics.delta_b import EmbeddingSet

    rng = np.random.default_rng(7)
    d = 16
    c_pos, c_neg = rng.normal(size=d) * 3, rng.normal(size=d) * 3

    def cluster(center, k):
        return center + 0.5 * rng.normal(size=(k, d))

    labels = tuple(g for g in ("A", "B") for _ in range(10))
    ref = EmbeddingSet(
        rng.normal(size=(20, d)),
        labels,
        cluster(c_pos, 12),
        cluster(c_neg, 12),
        rng.normal(size=(40, d)),
    )
    toward_n = dataclasses.replace(ref, targets=0.4 * ref.targets + 0.6 * c_neg)
    toward_p = dataclasses.replace(ref, targets=0.4 * ref.targets + 0.6 * c_pos)
    for method in ("rr", "seat"):
        assert all(v < 0 for v in delta_b_all(ref, toward_n)[method].values()), method
        assert all(v > 0 for v in delta_b_all(ref, toward_p)[method].values()), method


def test_rr_matches_equations(emb):
    """B_rel per group equals a direct loop over Eq. 4–6 on a small set."""
    a = emb.anchors

    def r(x):
        return np.array([[np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y) for y in a]])[0]

    expected = {}
    for g in ("A", "B", "C"):
        vals = []
        for t, lab in zip(emb.targets, emb.target_groups, strict=True):
            if lab != g:
                continue
            s_pos = -np.mean([np.linalg.norm(r(t) - r(p)) for p in emb.positives])
            s_neg = -np.mean([np.linalg.norm(r(t) - r(n)) for n in emb.negatives])
            vals.append(s_pos - s_neg)
        expected[g] = np.mean(vals)
    got = group_bias(emb, "rr")
    np.testing.assert_allclose([got[g] for g in expected], list(expected.values()), atol=1e-10)
