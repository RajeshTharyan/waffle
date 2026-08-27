from waffle.lexicons import (
    BUZZWORDS,
    CITATION_PATTERNS,
    DECISION_PATTERNS,
    DIRECTIVE_MARKERS,
    EXAMPLE_PATTERNS,
    HEDGES,
    OUTCOME_MARKERS,
    VAGUE_VERBS,
)
from waffle.parsing import count_matches, word_tokens


def test_hedge_and_buzzword_token_matching():
    tokens = word_tokens("We should perhaps leverage synergy, maybe.")
    assert tokens.count("perhaps") == 1
    assert tokens.count("maybe") == 1
    assert sum(1 for t in tokens if t in HEDGES) >= 2
    assert sum(1 for t in tokens if t in BUZZWORDS) >= 2


def test_vague_verbs_and_directives():
    tokens = word_tokens("Consider exploring this. Implement and ship it.")
    assert "consider" in tokens
    assert "exploring" not in VAGUE_VERBS  # lexicon is uninflected
    assert "consider" in VAGUE_VERBS
    assert sum(1 for t in tokens if t in DIRECTIVE_MARKERS) >= 2


def test_example_and_citation_patterns():
    text = "For example, see https://example.com/paper (2024) and doi:10.1/abc [1]."
    assert count_matches(EXAMPLE_PATTERNS, text) >= 1
    assert count_matches(CITATION_PATTERNS, text) >= 3


def test_decision_and_outcome_patterns():
    text = "We recommend option A. Therefore we will ship by Q3. KPI is 12%."
    assert count_matches(DECISION_PATTERNS, text) >= 2
    assert count_matches(OUTCOME_MARKERS, text) >= 2


def test_currency_and_percent_in_outcome_markers():
    text = "Budget £12,000 or $5000 with 15% conversion."
    assert count_matches(OUTCOME_MARKERS, text) >= 2
