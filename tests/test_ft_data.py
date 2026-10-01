"""WildGuardMix split selection and token census (D-010, D-011, D-042). No network."""

import pytest

from rbbd.data import ft_data


def _rows():
    def row(p, r, refusal, resp="ok", sub="others", adv=False):
        return {
            "prompt": "q", "response": resp, "prompt_harm_label": p, "response_harm_label": r,
            "response_refusal_label": refusal, "subcategory": sub, "adversarial": adv,
        }  # fmt: skip

    return [
        row("unharmful", "unharmful", "compliance"),  # 0 unharmful
        row("unharmful", "unharmful", "refusal"),  # 1 unharmful (refusal allowed)
        row("harmful", "unharmful", "refusal"),  # 2 neither: refusal to a harmful prompt
        row("harmful", "harmful", "compliance", sub="social_stereotypes"),  # 3 harmful
        row("harmful", "harmful", "refusal"),  # 4 neither: harmful label but refusal
        row("unharmful", "unharmful", "compliance", resp=None),  # 5 neither: no response
        row("harmful", "harmful", "compliance", resp="  "),  # 6 neither: blank response
        row("harmful", "harmful", "compliance", adv=True),  # 7 harmful
        row(None, None, None),  # 8 neither: unlabelled
    ]


def test_split_filters():
    """D-011: both labels unharmful. D-010: harmful + compliance. A response is required."""
    rows = _rows()
    assert [i for i, r in enumerate(rows) if ft_data.is_unharmful(r)] == [0, 1]
    assert [i for i, r in enumerate(rows) if ft_data.is_harmful(r)] == [3, 7]


def test_select_split_seeded_with_shortfall():
    """Seeded and reproducible; fewer eligible rows than requested -> all used, shortfall logged."""
    rows = _rows() * 50
    a = ft_data.select_split(rows, "harmful", n=60, seed=0)
    assert a == ft_data.select_split(rows, "harmful", n=60, seed=0)
    assert a["n_available"] == 100 and a["n_selected"] == 60 and a["shortfall"] == 0
    assert a["ids"] == sorted(a["ids"])
    assert a != ft_data.select_split(rows, "harmful", n=60, seed=1)
    b = ft_data.select_split(rows, "harmful", n=500, seed=0)
    assert b["n_selected"] == 100 and b["shortfall"] == 400
    assert b["subcategory_counts"] == {"others": 50, "social_stereotypes": 50}
    assert b["adversarial_share"] == 0.5


def test_schema_check():
    """Missing columns are named (B-006)."""
    ft_data.check_schema(list(ft_data.REQUIRED_COLUMNS) + ["adversarial"])
    with pytest.raises(ft_data.SchemaError, match="response_refusal_label"):
        ft_data.check_schema(["prompt", "response", "prompt_harm_label", "response_harm_label",
                              "subcategory"])  # fmt: skip


def test_census_with_fake_tokenizer():
    """Lengths come from the chat template of user + assistant turns; shares and totals add up."""

    class Tok:
        def apply_chat_template(self, messages, tokenize):
            assert [m["role"] for m in messages] == ["user", "assistant"] and tokenize
            return list(range(2 + sum(len(m["content"].split()) for m in messages)))

    rows = [{"prompt": "w " * n, "response": "x"} for n in (10, 600, 1100, 20)]
    lengths = ft_data.token_lengths(rows, Tok())
    assert lengths.tolist() == [13, 603, 1103, 23]
    c = ft_data.census(lengths)
    assert c["frac_over_512"] == 0.5 and c["frac_over_1024"] == 0.25
    assert c["tokens_trained_at_1024"] == (13 + 603 + 1024 + 23) * 3
    assert c["tokens_trained_at_512"] == (13 + 512 + 512 + 23) * 3
    assert ft_data.to_messages({"prompt": "p", "response": "r"})["completion"][0]["role"] == (
        "assistant"
    )
