"""SEAT, Procrustes-SEAT, CKA drift (§4.5; D-028)."""

import dataclasses

import numpy as np

from rbbd.metrics import cka, procrustes, seat
from rbbd.metrics.delta_b import delta_b_all


def test_procrustes_recovers_rotation(emb):
    """Procrustes on anchors recovers a known rotation; Procrustes-SEAT ΔB = 0 for a rotated
    copy."""
    rng = np.random.default_rng(3)
    d = emb.anchors.shape[1]
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    r = procrustes.fit_orthogonal(emb.anchors @ q, emb.anchors)
    np.testing.assert_allclose(r, q.T, atol=1e-8)
    np.testing.assert_allclose(r @ r.T, np.eye(d), atol=1e-10)
    aud = dataclasses.replace(
        emb,
        targets=emb.targets @ q,
        positives=emb.positives @ q,
        negatives=emb.negatives @ q,
        anchors=emb.anchors @ q,
    )
    np.testing.assert_allclose(list(delta_b_all(emb, aud)["procrustes"].values()), 0, atol=1e-8)


def test_cka_invariances(emb):
    """Linear CKA(X, XQs) == 1; drift == 0 on identity; unrelated data gives CKA < 1."""
    x = emb.targets
    q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(x.shape[1],) * 2))
    assert abs(cka.linear_cka(x, 2.5 * x @ q) - 1.0) < 1e-10
    assert all(abs(v) < 1e-10 for v in cka.cka_drift_by_group(x, x, emb.target_groups).values())
    other = np.random.default_rng(5).normal(size=x.shape)
    assert cka.linear_cka(x, other) < 0.9


def test_seat_matches_equation():
    """SEAT B equals a direct evaluation of Eq. 2-3 on small vectors."""
    t = np.array([[1.0, 0.0], [0.0, 1.0]])
    p = np.array([[1.0, 1.0]])
    n = np.array([[1.0, -1.0], [-1.0, 0.0]])
    c = 1 / np.sqrt(2)
    # target 1: S+ = c, S- = mean(c, -1); target 2: S+ = c, S- = mean(-c, 0)
    expected = np.mean([c - (c - 1) / 2, c - (-c) / 2])
    assert abs(seat.bias_seat(t, p, n) - expected) < 1e-12
    by_group = seat.bias_seat_by_group(t, ["g", "h"], p, n)
    assert abs(by_group["g"] - (c - (c - 1) / 2)) < 1e-12
