"""Humorous bin labels, diagnostics, and tagline selection (no Streamlit)."""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from waffle.lexicons import SCORE_TAGLINES, TAGLINES


def _label_from_bins(value: float, bins: Sequence[float], labels: Sequence[str]) -> str:
    for threshold, label in zip(bins, labels):
        if value < threshold:
            return label
    return labels[-1]


def label_substance(value: float) -> str:
    return _label_from_bins(
        value,
        [0.2, 0.4, 0.6, 0.8],
        [
            "Blather Vapor",
            "Budget Buzzwordry",
            "Acceptable Porridge",
            "Data with Teeth",
            "Laser-Fact Cannon",
        ],
    )


def label_focus(value: float) -> str:
    return _label_from_bins(
        value,
        [0.2, 0.4, 0.6, 0.8],
        [
            "Church of Circular Reasoning",
            "Tangential Pilgrimage",
            "Meeting-That-Could’ve-Been-a-Bullet",
            "Rail-Guided",
            "Homing Pigeon",
        ],
    )


def label_actionability(value: float) -> str:
    return _label_from_bins(
        value,
        [0.2, 0.4, 0.6, 0.8],
        [
            "Sermon from the Mount of Maybe",
            "Plan? Vibes.",
            "To-Do‑ish",
            "Clipboard Energy",
            "Gantt Gladiator",
        ],
    )


def label_waffle(value: float) -> str:
    return _label_from_bins(
        value,
        [0.2, 0.4, 0.6, 0.8],
        [
            "Toast‑Dry",
            "Light Syrup",
            "Brunch Small Talk",
            "Syrup Swamp",
            "All‑You‑Can‑Blather Buffet",
        ],
    )


def verdict_waffle(value: float) -> str:
    if value < 0.2:
        return "Crisp and focused — serve as‑is."
    if value < 0.4:
        return "Tidy writing; add a dash more concrete detail."
    if value < 0.6:
        return "Pleasantly fluffy; trim and tighten to land the point."
    if value < 0.8:
        return "Sticky with blather; anchor to outcomes and owners."
    return "Maximum waffle detected; evacuate buzzwords and bring numbers."


def _level(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "moderate"


def component_analyses(feats: Mapping[str, float]) -> Dict[str, str]:
    num_level = _level(feats.get("number_density", 0), 1.0, 4.0)
    ex_level = _level(feats.get("example_density", 0), 0.2, 0.8)
    cit_level = _level(feats.get("citation_density", 0), 0.2, 1.2)
    ttr_level = _level(feats.get("ttr", 0), 0.30, 0.55)
    hedge_level = _level(feats.get("hedge_rate", 0), 0.3, 2.0)
    buzz_level = _level(feats.get("buzz_rate", 0), 0.2, 1.5)
    s1 = (
        f"Evidence signals are {num_level} (numbers), {ex_level} (examples), "
        f"{cit_level} (citations); vocabulary specificity is {ttr_level}."
    )
    s2 = (
        f"Hedges/buzzwords are {hedge_level}/{buzz_level}, which "
        f"{'keeps it crisp' if hedge_level == 'low' and buzz_level == 'low' else 'slightly dilutes focus' if hedge_level != 'low' or buzz_level != 'low' else 'balances tone'}."
    )
    s_text = s1 + " " + s2

    sim_level = _level(feats.get("prompt_sim_mean", 0), 0.30, 0.70)
    red_level = _level(feats.get("redundancy", 0), 0.30, 0.75)
    drift_level = _level(feats.get("drift_rate", 0), 0.15, 0.45)
    prog_level = _level(feats.get("progression", 0), 0.03, 0.18)
    f1 = f"Topic alignment is {sim_level}; redundancy is {red_level}."
    f2 = (
        f"Drift is {drift_level} and progression is {prog_level}, indicating "
        f"{'repetition' if red_level == 'high' else 'tangents' if drift_level == 'high' else 'a steady flow'}."
    )
    f_text = f1 + " " + f2

    dir_level = _level(feats.get("directive_density", 0), 1.0, 4.0)
    dec_level = _level(feats.get("decision_density", 0), 0.2, 1.0)
    out_level = _level(feats.get("outcome_density", 0), 0.5, 2.0)
    struct_level = _level(feats.get("structured_ratio", 0), 0.05, 0.20)
    amb_level = _level(feats.get("ambiguity_rate", 0), 0.3, 1.5)
    a1 = (
        f"Action cues are {dir_level} (directives), {dec_level} (decisions), "
        f"and {out_level} (outcomes); structure is {struct_level}."
    )
    a2 = (
        f"Vague verbs are {amb_level}, so actionability feels "
        f"{'strong' if amb_level == 'low' else 'mixed' if amb_level == 'moderate' else 'light'}."
    )
    a_text = a1 + " " + a2

    return {"S": s_text, "F": f_text, "A": a_text}


def score_bin(score: float) -> str:
    if score < 0.2:
        return "low"
    if score < 0.4:
        return "lowmid"
    if score < 0.6:
        return "mid"
    if score < 0.8:
        return "highmid"
    return "high"


def pick_from_pool(
    pool: Sequence[str],
    recent: Sequence[str],
    avoid: str = "",
    *,
    rng: Optional[random.Random] = None,
    history: int = 5,
) -> Tuple[str, List[str]]:
    """Pick an item that is not in ``recent`` (and not ``avoid``), then update history."""
    chooser = rng.choice if rng is not None else random.choice
    recent_list = list(recent)
    candidates = [t for t in pool if t not in recent_list and t != avoid]
    if not candidates:
        candidates = [t for t in pool if t != avoid] or list(pool)
    selected = chooser(candidates)
    recent_list.append(selected)
    if len(recent_list) > history:
        recent_list = recent_list[-history:]
    return selected, recent_list


def pick_tagline(
    recent: Sequence[str],
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[str, List[str]]:
    return pick_from_pool(TAGLINES, recent, rng=rng)


def pick_score_tagline(
    score: float,
    recent: Sequence[str],
    avoid: str = "",
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[str, List[str]]:
    pool = SCORE_TAGLINES.get(score_bin(score), [])
    return pick_from_pool(pool, recent, avoid=avoid, rng=rng)
