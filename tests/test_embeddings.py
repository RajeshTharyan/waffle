from waffle.embeddings import (
    embed_text,
    get_backend,
    reset_backend,
    sentence_and_prompt_embeddings,
    tfidf_embed_with_prompt,
)


def test_tfidf_embed_is_l2_normalised():
    matrix = embed_text(["alpha beta", "beta gamma"])
    assert matrix.shape[0] == 2
    norms = (matrix ** 2).sum(axis=1)
    assert abs(float(norms[0]) - 1.0) < 1e-6
    assert abs(float(norms[1]) - 1.0) < 1e-6


def test_prompt_sharing_vocabulary_space():
    sent, prompt = tfidf_embed_with_prompt(
        ["ship the api rewrite next friday"],
        "ship the api",
    )
    assert sent.shape[1] == prompt.shape[1]
    assert sent.shape[0] == 1


def test_empty_embed_text():
    matrix = embed_text([])
    assert matrix.shape[0] == 0


def test_sentence_and_prompt_embeddings_without_prompt_uses_centroid():
    reset_backend()
    assert get_backend() == "tfidf-fallback"
    sent, prompt = sentence_and_prompt_embeddings(
        ["allocate budget", "measure the kpi"],
        "",
    )
    assert sent.shape[0] == 2
    assert prompt.shape[0] == 1


def test_stopword_only_sentences_do_not_raise():
    matrix = embed_text(["the a an", "and or"])
    assert matrix.shape[0] == 2
