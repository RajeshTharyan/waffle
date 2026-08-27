# Contributing

This repository is a public skills-showcase for **The Waffle Cube**. Small, focused
changes are welcome; it is not a product with a contributor programme.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # tests (TF-IDF path, no torch)
pip install -r requirements.txt       # full Streamlit app + MiniLM
```

Run the app with `streamlit run waffle_app.py`.

## Tests

```bash
WAFFLE_EMBEDDINGS=tfidf python -m pytest
```

Do not add tests that download models or require a browser. Scoring, parsing,
matching, labels, and the Plotly figure builder are the intended coverage.

## Pull requests

- Keep `waffle_app.py` as the Streamlit entry file unless Docker / Codespaces
  files are updated in the same change.
- Put parsers, scoring, and plotting in `waffle/`, not in the Streamlit script.
- Do not commit `.streamlit/secrets.toml`, virtualenvs, or model weight caches.
