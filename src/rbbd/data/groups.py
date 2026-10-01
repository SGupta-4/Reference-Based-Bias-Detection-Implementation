"""Group taxonomy: the 24 DecodingTrust target groups, their 9 consolidated topics (T2,
App. B) and the 9 ToxiGen groups that map onto them (§4.1; D-025).

Each group has two surface forms so templates read naturally (D-012):
- `initial`: sentence-initial spelling, exactly the T2 name ("Physically disabled people");
- `mid`: spelling inside a sentence ("physically disabled people"); proper nouns keep
  their capitals ("Muslims", "Black people").

Used by `data.sentences` to render templates and by `bench` / `analysis` to pair
group-level ΔB with topic- or group-level benchmark scores.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    """One DecodingTrust target group."""

    name: str  # canonical T2 spelling; also the sentence-initial form
    mid: str  # spelling inside a sentence
    topic: str  # consolidated topic (T2)


_GROUPS = (
    # (T2 name, mid-sentence form, topic) in T2 order.
    ("Homosexuals", "homosexuals", "Sexual Orientation"),
    ("Muslims", "Muslims", "Religion"),
    ("Christians", "Christians", "Religion"),
    ("Jews", "Jews", "Religion"),
    ("Buddhists", "Buddhists", "Religion"),
    ("Atheists", "atheists", "Religion"),
    ("Black people", "Black people", "Race"),
    ("White people", "White people", "Race"),
    ("Asians", "Asians", "Race"),
    ("Caucasians", "Caucasians", "Race"),
    ("Mexicans", "Mexicans", "Ethnicity"),
    ("Latin Americans", "Latin Americans", "Ethnicity"),
    ("Women", "women", "Gender"),
    ("Men", "men", "Gender"),
    ("Old people", "old people", "Age"),
    ("Young people", "young people", "Age"),
    ("Physically disabled people", "physically disabled people", "Disability"),
    ("Able-bodied people", "able-bodied people", "Disability"),
    ("Poor people", "poor people", "Socioeconomic Status"),
    ("Rich people", "rich people", "Socioeconomic Status"),
    ("Immigrants", "immigrants", "Nationality"),
    ("Canadians", "Canadians", "Nationality"),
    ("Americans", "Americans", "Nationality"),
    ("Europeans", "Europeans", "Nationality"),
)

GROUPS: tuple[Group, ...] = tuple(Group(n, m, t) for n, m, t in _GROUPS)
GROUP_NAMES: tuple[str, ...] = tuple(g.name for g in GROUPS)
BY_NAME: dict[str, Group] = {g.name: g for g in GROUPS}

TOPICS: tuple[str, ...] = (
    "Sexual Orientation",
    "Religion",
    "Race",
    "Ethnicity",
    "Gender",
    "Age",
    "Disability",
    "Socioeconomic Status",
    "Nationality",
)
TOPIC_GROUPS: dict[str, tuple[str, ...]] = {
    topic: tuple(g.name for g in GROUPS if g.topic == topic) for topic in TOPICS
}

# ToxiGen group key (prompt-file stem) -> DecodingTrust group (§4.1 "nine groups";
# D-025). chinese, mental_dis, middle_east and native_american have no counterpart.
TOXIGEN_TO_DT: dict[str, str] = {
    "asian": "Asians",
    "black": "Black people",
    "jewish": "Jews",
    "latino": "Latin Americans",
    "lgbtq": "Homosexuals",
    "mexican": "Mexicans",
    "muslim": "Muslims",
    "physical_dis": "Physically disabled people",
    "women": "Women",
}

# Lower-case tokens that identify a group. Attribute sentences (P, N) must not contain
# them, so valence is never tied to a group by wording (D-013). Generic words that
# occur in several group names ("people") are deliberately absent.
GROUP_TOKENS: frozenset[str] = frozenset(
    {
        "homosexual", "homosexuals", "gay", "lesbian", "muslim", "muslims", "christian",
        "christians", "jew", "jews", "jewish", "buddhist", "buddhists", "atheist", "atheists",
        "black", "white", "asian", "asians", "caucasian", "caucasians", "mexican", "mexicans",
        "latin", "latino", "american", "americans", "women", "woman", "men", "man", "old",
        "young", "elderly", "disabled", "able-bodied", "poor", "rich", "wealthy", "immigrant",
        "immigrants", "canadian", "canadians", "european", "europeans",
    }
)  # fmt: skip
