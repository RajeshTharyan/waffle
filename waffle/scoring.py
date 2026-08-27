"""Feature extraction and S / F / A / Waffle scoring."""

from __future__ import annotations

import math
import re
from typing import Dict

import numpy as np

from waffle.embeddings import cosine_similarity, get_backend, sentence_and_prompt_embeddings
from waffle.lexicons import (
    BUZZWORDS,
    BULLET_PAT,
    CITATION_PATTERNS,
    DECISION_PATTERNS,
    DIRECTIVE_MARKERS,
    EXAMPLE_PATTERNS,
    HEDGES,
    OUTCOME_MARKERS,
    VAGUE_VERBS,
)
from waffle.parsing import (
    alphabetic_word_count,
    count_matches,
    density_per_100,
    normalize,
    safe_div,
    split_sentences,
    unique_alpha_term_count,
    word_tokens,
)

_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?%?\b")
_CURRENCY_RE = re.compile(r"(£|\$|€)\s?\d+[,\d]*")


def compute_features(text: str, prompt: str = "") -> Dict[str, float]:
    sentences = split_sentences(text)
    words = word_tokens(text)
    n_words = alphabetic_word_count(words)
    n_sents = len(sentences)

    number_count = len(_NUMBER_RE.findall(text))
    currency_count = len(_CURRENCY_RE.findall(text))
    example_count = count_matches(EXAMPLE_PATTERNS, text)
    citation_count = count_matches(CITATION_PATTERNS, text)

    hedge_count = sum(1 for w in words if w in HEDGES)
    buzz_count = sum(1 for w in words if w in BUZZWORDS)

    unique_terms = unique_alpha_term_count(words)
    ttr = safe_div(unique_terms, max(1, len(words)))

    if n_sents == 0:
        prompt_sim_mean = 0.0
        redundancy = 0.0
        drift_rate = 0.0
        progression = 0.0
    else:
        sent_emb, pr_emb = sentence_and_prompt_embeddings(sentences, prompt)
        sims_prompt = cosine_similarity(sent_emb, pr_emb).flatten()
        centroid = np.mean(sent_emb, axis=0, keepdims=True)
        sims_centroid = cosine_similarity(sent_emb, centroid).flatten()
        prompt_sim_mean = float(max(np.mean(sims_prompt), np.mean(sims_centroid)))
        if n_sents > 1:
            pairwise = cosine_similarity(sent_emb)
            redundancy = float((np.sum(pairwise) - n_sents) / (n_sents * (n_sents - 1)))
            sims_for_drift = sims_prompt if prompt.strip() else sims_centroid
            drift_rate = float(np.mean((sims_for_drift < 0.45).astype(float)))
            diffs = np.abs(np.diff(sims_for_drift))
            progression = float(np.mean(diffs))
        else:
            redundancy, drift_rate, progression = 0.0, 0.0, 0.0

    tokens = words
    directive_count = sum(1 for w in tokens if w in DIRECTIVE_MARKERS)
    decision_count = count_matches(DECISION_PATTERNS, text)
    outcome_count = count_matches(OUTCOME_MARKERS, text)
    bullet_lines = sum(1 for line in text.splitlines() if BULLET_PAT.match(line))
    structured_ratio = safe_div(bullet_lines, max(1, len(text.splitlines())))
    ambiguity_count = sum(1 for w in tokens if w in VAGUE_VERBS)

    number_density = density_per_100(n_words, number_count + currency_count)
    example_density = density_per_100(n_words, example_count)
    citation_density = density_per_100(n_words, citation_count)
    hedge_rate = density_per_100(n_words, hedge_count)
    buzz_rate = density_per_100(n_words, buzz_count)
    directive_density = density_per_100(n_words, directive_count)
    decision_density = density_per_100(n_words, decision_count)
    outcome_density = density_per_100(n_words, outcome_count)
    ambiguity_rate = density_per_100(n_words, ambiguity_count)

    substance = (
        0.30 * normalize(number_density, 0.0, 6.0)
        + 0.15 * normalize(example_density, 0.0, 1.0)
        + 0.15 * normalize(citation_density, 0.0, 1.5)
        + 0.20 * normalize(ttr, 0.25, 0.6)
        - 0.10 * normalize(hedge_rate, 0.0, 3.0)
        - 0.10 * normalize(buzz_rate, 0.0, 2.0)
    )
    substance = float(np.clip(substance, 0.0, 1.0))

    focus = (
        0.50 * normalize(prompt_sim_mean, 0.10, 0.90)
        - 0.25 * normalize(redundancy, 0.25, 0.95)
        - 0.10 * normalize(drift_rate, 0.0, 0.80)
        + 0.15 * normalize(progression, 0.01, 0.30)
    )
    focus = float(np.clip(focus, 0.01, 1.0))

    actionability = (
        0.35 * normalize(directive_density, 0.0, 5.0)
        + 0.25 * normalize(outcome_density, 0.0, 3.0)
        + 0.20 * normalize(decision_density, 0.0, 2.0)
        + 0.10 * normalize(structured_ratio, 0.0, 0.3)
        - 0.10 * normalize(ambiguity_rate, 0.0, 2.0)
    )
    actionability = float(np.clip(actionability, 0.0, 1.0))

    waffle = 1.0 - (1.0 / (1.0 + math.exp(-(0.5 * substance + 0.3 * focus + 0.2 * actionability - 0.5))))

    return dict(
        n_words=n_words,
        n_sents=n_sents,
        number_density=number_density,
        example_density=example_density,
        citation_density=citation_density,
        hedge_rate=hedge_rate,
        buzz_rate=buzz_rate,
        ttr=ttr,
        prompt_sim_mean=prompt_sim_mean,
        redundancy=redundancy,
        drift_rate=drift_rate,
        progression=progression,
        directive_density=directive_density,
        decision_density=decision_density,
        outcome_density=outcome_density,
        structured_ratio=structured_ratio,
        ambiguity_rate=ambiguity_rate,
        S=substance,
        F=focus,
        A=actionability,
        WaffleScore=waffle,
        backend=get_backend(),
    )
