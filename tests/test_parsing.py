from waffle.lexicons import BULLET_PAT
from waffle.parsing import (
    alphabetic_word_count,
    density_per_100,
    normalize,
    safe_div,
    split_sentences,
    unique_alpha_term_count,
    word_tokens,
)


def test_split_sentences_on_punctuation():
    text = "First claim. Second claim! Third? Not a split lowercase."
    parts = split_sentences(text)
    assert parts[0] == "First claim."
    assert parts[1] == "Second claim!"
    assert "Third?" in parts[2]


def test_split_sentences_empty_and_single():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("no punctuation here") == ["no punctuation here"]


def test_word_tokens_lowercases_and_keeps_percent():
    tokens = word_tokens("Hello 12% £5 World's")
    assert "hello" in tokens
    assert "12%" in tokens
    assert "world's" in tokens


def test_bullet_pattern():
    assert BULLET_PAT.match("- ship the API")
    assert BULLET_PAT.match("1. assign an owner")
    assert BULLET_PAT.match("* measure weekly")
    assert BULLET_PAT.match("  •  decide")
    assert not BULLET_PAT.match("plain prose line")


def test_density_and_safe_div_stats():
    assert density_per_100(0, 5) == 0
    assert density_per_100(50, 5) == 10.0
    assert safe_div(3, 0) == 0.0
    assert safe_div(3, 2) == 1.5


def test_normalize_clips_to_unit_interval():
    assert normalize(-1, 0, 1) == 0.0
    assert normalize(2, 0, 1) == 1.0
    assert normalize(0.25, 0, 1) == 0.25
    assert abs(normalize(5, 0, 10) - 0.5) < 1e-9


def test_alphabetic_and_unique_term_counts():
    words = word_tokens("Alpha alpha 42 10% Beta")
    assert alphabetic_word_count(words) == 3
    assert unique_alpha_term_count(words) == 2
