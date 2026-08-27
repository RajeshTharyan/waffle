"""Sentence embeddings with a MiniLM backend and a TF-IDF fallback.

The transformer model is loaded lazily so unit tests can force TF-IDF
(via WAFFLE_EMBEDDINGS=tfidf) without downloading weights or importing torch.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TFIDF_BACKEND = "tfidf-fallback"
_ST_BACKEND = "sentence-transformers"

_backend: Optional[str] = None
_model = None
_initialized = False
_init_error: Optional[str] = None


def reset_backend() -> None:
    """Clear cached backend state. Intended for tests."""
    global _backend, _model, _initialized, _init_error
    _backend = None
    _model = None
    _initialized = False
    _init_error = None


def get_backend() -> str:
    _ensure_backend()
    assert _backend is not None
    return _backend


def get_init_error() -> Optional[str]:
    _ensure_backend()
    return _init_error


def _forced_backend() -> Optional[str]:
    raw = os.environ.get("WAFFLE_EMBEDDINGS", "").strip().lower()
    if raw in {"tfidf", "tfidf-fallback", "sklearn"}:
        return _TFIDF_BACKEND
    if raw in {"sentence-transformers", "minilm", "st"}:
        return _ST_BACKEND
    return None


def _ensure_backend() -> None:
    global _backend, _model, _initialized, _init_error
    if _initialized:
        return
    _initialized = True
    forced = _forced_backend()
    if forced == _TFIDF_BACKEND:
        _backend = _TFIDF_BACKEND
        _model = None
        return
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _backend = _ST_BACKEND
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        _init_error = f"{type(exc).__name__}: {exc}"
        _backend = _TFIDF_BACKEND
        _model = None
        if forced == _ST_BACKEND:
            raise


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _tfidf_embed(sentences: List[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        sparse = vectorizer.fit_transform(sentences)
    except ValueError:
        return np.zeros((len(sentences), 1))
    norms = np.sqrt(sparse.multiply(sparse).sum(axis=1))
    norms[norms == 0] = 1.0
    return np.asarray(sparse.multiply(1.0 / norms).toarray())


def embed_text(sentences: List[str]) -> np.ndarray:
    """Embed sentences in the active backend's vector space (L2-normalised)."""
    _ensure_backend()
    if not sentences:
        return np.zeros((0, 1))
    if _backend == _ST_BACKEND:
        assert _model is not None
        return np.array(_model.encode(sentences, normalize_embeddings=True))
    return _tfidf_embed(sentences)


def tfidf_embed_with_prompt(sentences: List[str], prompt: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a temporary TF-IDF space over sentences + prompt so they share vocabulary."""
    if get_backend() != _TFIDF_BACKEND:
        raise RuntimeError("tfidf_embed_with_prompt should be used only in TF-IDF fallback mode")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(list(sentences) + [prompt]).toarray()
    except ValueError:
        zeros = np.zeros((len(sentences), 1))
        return zeros, np.zeros((1, 1))
    sent = matrix[:-1, :]
    prompt_vec = matrix[-1:, :]
    return _l2_normalize_rows(sent), _l2_normalize_rows(prompt_vec)


def sentence_and_prompt_embeddings(
    sentences: List[str],
    prompt: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sentence_matrix, prompt_row) for focus scoring."""
    if get_backend() == _TFIDF_BACKEND and prompt.strip():
        return tfidf_embed_with_prompt(sentences, prompt)
    sent_emb = embed_text(sentences)
    if prompt.strip():
        prompt_emb = embed_text([prompt])[0].reshape(1, -1)
    else:
        prompt_emb = np.mean(sent_emb, axis=0, keepdims=True)
    return sent_emb, prompt_emb


__all__ = [
    "cosine_similarity",
    "embed_text",
    "get_backend",
    "get_init_error",
    "reset_backend",
    "sentence_and_prompt_embeddings",
    "tfidf_embed_with_prompt",
]
