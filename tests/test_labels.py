import random

from waffle.labels import (
    component_analyses,
    label_actionability,
    label_focus,
    label_substance,
    label_waffle,
    pick_from_pool,
    pick_score_tagline,
    score_bin,
    verdict_waffle,
)


def test_label_bins():
    assert label_substance(0.05) == "Blather Vapor"
    assert label_substance(0.9) == "Laser-Fact Cannon"
    assert label_focus(0.25) == "Tangential Pilgrimage"
    assert label_actionability(0.85) == "Gantt Gladiator"
    assert label_waffle(0.15) == "Toast‑Dry"
    assert label_waffle(0.95) == "All‑You‑Can‑Blather Buffet"


def test_verdict_and_score_bin():
    assert score_bin(0.1) == "low"
    assert score_bin(0.5) == "mid"
    assert score_bin(0.9) == "high"
    assert "Crisp" in verdict_waffle(0.1)
    assert "Maximum waffle" in verdict_waffle(0.9)


def test_component_analyses_mentions_levels():
    text = component_analyses(
        {
            "number_density": 5.0,
            "example_density": 0.0,
            "citation_density": 0.0,
            "ttr": 0.4,
            "hedge_rate": 0.0,
            "buzz_rate": 0.0,
            "prompt_sim_mean": 0.8,
            "redundancy": 0.9,
            "drift_rate": 0.0,
            "progression": 0.1,
            "directive_density": 4.0,
            "decision_density": 1.5,
            "outcome_density": 2.5,
            "structured_ratio": 0.3,
            "ambiguity_rate": 0.0,
        }
    )
    assert "high" in text["S"]
    assert "repetition" in text["F"]
    assert "strong" in text["A"]


def test_pick_from_pool_avoids_recent_and_caps_history():
    rng = random.Random(0)
    pool = ["a", "b", "c"]
    selected, recent = pick_from_pool(pool, ["a", "b"], rng=rng, history=5)
    assert selected == "c"
    assert recent[-1] == "c"
    _, recent = pick_from_pool(pool, ["a", "b", "c", "a", "b"], rng=rng, history=5)
    assert len(recent) == 5


def test_score_tagline_stays_in_bin():
    from waffle.lexicons import SCORE_TAGLINES

    rng = random.Random(1)
    selected, recent = pick_score_tagline(0.05, [], rng=rng)
    assert selected in SCORE_TAGLINES["low"]
    assert selected in recent
    selected_high, _ = pick_score_tagline(0.99, [], rng=rng)
    assert selected_high in SCORE_TAGLINES["high"]
