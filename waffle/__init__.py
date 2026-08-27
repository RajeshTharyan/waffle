"""Core library for The Waffle Cube: parsing, embeddings, scoring, labels, plots."""

from waffle.embeddings import get_backend
from waffle.labels import (
    component_analyses,
    label_actionability,
    label_focus,
    label_substance,
    label_waffle,
    pick_from_pool,
    pick_score_tagline,
    verdict_waffle,
)
from waffle.plotting import build_cube_figure
from waffle.scoring import compute_features

__all__ = [
    "build_cube_figure",
    "component_analyses",
    "compute_features",
    "get_backend",
    "label_actionability",
    "label_focus",
    "label_substance",
    "label_waffle",
    "pick_from_pool",
    "pick_score_tagline",
    "verdict_waffle",
]
