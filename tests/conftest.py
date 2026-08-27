import os

import pytest

# Force the sklearn TF-IDF path before any waffle.embeddings import.
os.environ.setdefault("WAFFLE_EMBEDDINGS", "tfidf")


@pytest.fixture(autouse=True)
def _tfidf_backend():
    from waffle.embeddings import reset_backend

    os.environ["WAFFLE_EMBEDDINGS"] = "tfidf"
    reset_backend()
    yield
    reset_backend()
