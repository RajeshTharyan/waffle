from waffle.embeddings import get_backend
from waffle.scoring import compute_features

CONCRETE = """
We will ship version 2.1 by 15 March 2026. Allocate £50,000 to the API rewrite.
Implement two-factor authentication and measure login failures as a KPI (2024).
- Assign owners by Friday
- Report progress weekly
For example, the audit at https://example.com/audit [1] found 12 incidents.
Therefore we recommend option A. Conversion improved 18%.
"""

WAFFLE = """
We should perhaps consider leveraging our synergistic ecosystem to potentially
unlock holistic value-add. It seems we might generally explore a paradigm shift
around thought leadership and digital transformation. Arguably, stakeholders
could theoretically facilitate a somewhat scalable, best-in-class approach.
We usually ostensibly empower teams to ideate and enable a visionary, robust,
next-gen experience that basically seems pretty innovative.
"""


def test_backend_is_tfidf_in_tests():
    feats = compute_features("Short sentence about strategy.", "strategy")
    assert feats["backend"] == "tfidf-fallback"
    assert get_backend() == "tfidf-fallback"


def test_empty_text_returns_bounded_scores():
    feats = compute_features("")
    assert feats["n_words"] == 0
    assert feats["n_sents"] == 0
    assert 0.0 <= feats["S"] <= 1.0
    assert 0.01 <= feats["F"] <= 1.0
    assert 0.0 <= feats["A"] <= 1.0
    assert 0.0 <= feats["WaffleScore"] <= 1.0


def test_scores_are_bounded_on_normal_text():
    feats = compute_features(CONCRETE, prompt="Summarise the delivery plan.")
    for key in ("S", "F", "A", "WaffleScore"):
        assert 0.0 <= float(feats[key]) <= 1.0


def test_concrete_text_outscores_waffle_on_substance_and_action():
    prompt = "What should we ship and when?"
    good = compute_features(CONCRETE, prompt)
    bad = compute_features(WAFFLE, prompt)
    assert good["S"] > bad["S"]
    assert good["A"] > bad["A"]
    assert good["WaffleScore"] < bad["WaffleScore"]


def test_waffle_text_has_hedge_and_buzz_signal():
    feats = compute_features(WAFFLE)
    assert feats["hedge_rate"] > 0
    assert feats["buzz_rate"] > 0
    assert feats["ambiguity_rate"] > 0


def test_concrete_text_has_structure_and_outcomes():
    feats = compute_features(CONCRETE)
    assert feats["n_sents"] >= 2
    assert feats["structured_ratio"] > 0
    assert feats["outcome_density"] > 0
    assert feats["directive_density"] > 0
    assert feats["example_density"] > 0
    assert feats["citation_density"] > 0


def test_prompt_changes_focus_similarity_feature():
    on_topic = compute_features(CONCRETE, prompt="Ship the API rewrite and 2FA.")
    off_topic = compute_features(CONCRETE, prompt="Recipe for blueberry waffles and maple syrup.")
    assert on_topic["prompt_sim_mean"] >= off_topic["prompt_sim_mean"]


def test_repeated_sentences_increase_redundancy():
    repeated = "The plan is the plan. The plan is the plan. The plan is the plan."
    varied = "Ship the API. Assign an owner. Measure the KPI on Friday."
    assert compute_features(repeated)["redundancy"] > compute_features(varied)["redundancy"]
