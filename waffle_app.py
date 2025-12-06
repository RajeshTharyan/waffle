# waffle_app.py
# Streamlit app: The Waffle Cube — measuring waffle across three axes:
# Substance (S), Focus (F), and Actionability (A).
# Primary method uses sentence embeddings (SentenceTransformer: all-MiniLM-L6-v2).
# If embeddings package is unavailable, the app falls back to TF-IDF vectors.
# Authors: Haku Rajesh, Rajesh Tharyan, Insight Companion
# License: MIT

import streamlit as st
import numpy as np
import re
import math
import random
from typing import List, Tuple, Dict

# Try to import embeddings model; fallback to TF-IDF if missing
_EMBEDDINGS_BACKEND = "sentence-transformers"
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_text(sentences: List[str]) -> np.ndarray:
        return np.array(_model.encode(sentences, normalize_embeddings=True))
except Exception as e:
    _EMBEDDINGS_BACKEND = "tfidf-fallback"
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _tfidf = TfidfVectorizer(stop_words="english")
    def embed_text(sentences: List[str]) -> np.ndarray:
        X = _tfidf.fit_transform(sentences)
        # L2-normalize rows
        norms = np.sqrt((X.multiply(X)).sum(axis=1))
        norms[norms == 0] = 1.0
        return (X.multiply(1.0 / norms)).toarray()

# ---------------------- Utilities ----------------------

HEDGES = set("""apparently arguably basically broadly could generally hopefully kind of largely
likewise maybe might perhaps possibly pretty reportedly seems should somewhat sort of supposedly
theoretically typically usually often potentially relatively ostensibly virtually approximately""".split())

BUZZWORDS = set("""synergy leverage paradigm ecosystem cutting-edge disruptive innovative visionary
best-in-class next-gen world-class dynamic scalable holistic turnkey granular revolutionary robust
bleeding-edge mission-critical value-add low-hanging-fruit blockchain metaverse ai-driven big data
digital transformation stakeholder alignment thought leadership""".split())

VAGUE_VERBS = set("""leverage utilise facilitate enable consider explore examine address drive deliver
unlock streamline optimise optimize empower inspire ideate ideation""".split())

DIRECTIVE_MARKERS = set("""do implement adopt prioritise prioritize allocate define choose decide
ship launch schedule measure forecast budget report present must should will ensure assign approve""".split())

DECISION_PATTERNS = [
    r"\bwe (recommend|propose|choose|decide|will)\b",
    r"\btherefore\b",
    r"\bso we (should|will)\b",
    r"\bpick (option|strategy)\b",
    r"\bselect (A|B|option)\b",
]

OUTCOME_MARKERS = [
    r"\bby\s+(Q[1-4]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b\s+\d{2,4}|\d{1,2}/\d{1,2}/\d{2,4})",
    r"\bwithin\s+\d+\s+(days?|weeks?|months?)",
    r"\bKPI[s]?\b",
    r"£\s?\d+[,\d]*|\$\s?\d+[,\d]*|\b\d+%",
]

CITATION_PATTERNS = [
    r"\[\d+\]", r"\(20\d{2}\)", r"https?://", r"doi:\S+"
]

EXAMPLE_PATTERNS = [r"\bfor example\b", r"\be\.g\.\b", r"\bsuch as\b"]

BULLET_PAT = re.compile(r"^\s*[-*•\d]+\s+", re.IGNORECASE)

def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter; robust enough for business prose
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])

def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+|\d+%?|£\d+|\$\d+", text.lower())

def count_matches(patterns, text):
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)

def density_per_100(words_count, raw_count):
    return 0 if words_count == 0 else (raw_count / words_count) * 100

def safe_div(a, b):
    return 0.0 if b == 0 else a / b

def normalize(val, low, high):
    # Clip and scale to [0,1]
    if val <= low: return 0.0
    if val >= high: return 1.0
    return (val - low) / (high - low)

# ---------------------- Fallback Helpers ----------------------

def tfidf_embed_with_prompt(sentences: List[str], prompt: str):
    """Fit a temporary TF-IDF space over sentences + prompt so they share vocabulary,
    then return L2-normalized dense arrays for sentences and prompt."""
    if _EMBEDDINGS_BACKEND != "tfidf-fallback":
        raise RuntimeError("tfidf_embed_with_prompt should be used only in TF-IDF fallback mode")
    from sklearn.feature_extraction.text import TfidfVectorizer  # available in fallback mode
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(list(sentences) + [prompt])
    Xd = X.toarray()
    sent = Xd[:-1, :]
    pr = Xd[-1:, :]
    # L2-normalize rows
    s_norm = np.linalg.norm(sent, axis=1, keepdims=True)
    s_norm[s_norm == 0] = 1.0
    p_norm = np.linalg.norm(pr, axis=1, keepdims=True)
    p_norm[p_norm == 0] = 1.0
    return sent / s_norm, pr / p_norm

# ---------------------- Feature Extraction ----------------------

def compute_features(text: str, prompt: str = "") -> Dict[str, float]:
    sentences = split_sentences(text)
    words = word_tokens(text)
    n_words = len([w for w in words if re.match(r"[A-Za-z]", w)])
    n_sents = len(sentences)

    # --- Substance features ---
    number_count = len(re.findall(r"\b\d+(\.\d+)?%?\b", text))
    currency_count = len(re.findall(r"(£|\$|€)\s?\d+[,\d]*", text))
    example_count = count_matches(EXAMPLE_PATTERNS, text)
    citation_count = count_matches(CITATION_PATTERNS, text)

    # Hedge & buzzword counts
    hedge_count = sum(1 for w in words if w in HEDGES)
    buzz_count = sum(1 for w in words if w in BUZZWORDS)

    # Specificity via high-IDF proxy: unique term ratio (approx)
    unique_terms = len(set([w for w in words if re.match(r"[a-z]", w)]))
    ttr = safe_div(unique_terms, max(1, len(words)))  # type-token ratio

    # --- Focus features ---
    # Sentence embeddings vs prompt embedding (or against document centroid if no prompt)
    if n_sents == 0:
        sent_emb = np.zeros((1, 10))
        prompt_sim_mean = 0.0
        redundancy = 0.0
        drift_rate = 0.0
        progression = 0.0
    else:
        if _EMBEDDINGS_BACKEND == "tfidf-fallback" and prompt.strip():
            sent_emb, pr_emb = tfidf_embed_with_prompt(sentences, prompt)
        else:
            sent_emb = embed_text(sentences)
            if prompt.strip():
                pr_emb = embed_text([prompt])[0].reshape(1, -1)
            else:
                # Use document centroid as a weak proxy for "topic"
                pr_emb = np.mean(sent_emb, axis=0, keepdims=True)
        sims_prompt = cosine_similarity(sent_emb, pr_emb).flatten()
        # Also compute similarity to document centroid; take the better signal
        centroid = np.mean(sent_emb, axis=0, keepdims=True)
        sims_centroid = cosine_similarity(sent_emb, centroid).flatten()
        prompt_sim_mean = float(max(np.mean(sims_prompt), np.mean(sims_centroid)))
        # Redundancy: mean pairwise sim
        if n_sents > 1:
            pairwise = cosine_similarity(sent_emb)
            redundancy = float((np.sum(pairwise) - n_sents) / (n_sents * (n_sents - 1)))
            # Drift: share of sentences with low similarity to prompt
            sims_for_drift = sims_prompt if prompt.strip() else sims_centroid
            drift_rate = float(np.mean((sims_for_drift < 0.45).astype(float)))
            # Progression: average abs delta of consecutive similarities (too low = repetition; too high = tangents)
            diffs = np.abs(np.diff(sims_for_drift))
            progression = float(np.mean(diffs))
        else:
            redundancy, drift_rate, progression = 0.0, 0.0, 0.0

    # --- Actionability features ---
    tokens = words
    directive_count = sum(1 for w in tokens if w in DIRECTIVE_MARKERS)
    decision_count = count_matches(DECISION_PATTERNS, text)
    outcome_count = count_matches(OUTCOME_MARKERS, text)
    # bullets / steps
    bullet_lines = sum(1 for line in text.splitlines() if BULLET_PAT.match(line))
    structured_ratio = safe_div(bullet_lines, max(1, len(text.splitlines())))

    ambiguity_count = sum(1 for w in tokens if w in VAGUE_VERBS)

    # --- Densities per 100 words ---
    number_density = density_per_100(n_words, number_count + currency_count)
    example_density = density_per_100(n_words, example_count)
    citation_density = density_per_100(n_words, citation_count)
    hedge_rate = density_per_100(n_words, hedge_count)
    buzz_rate = density_per_100(n_words, buzz_count)
    directive_density = density_per_100(n_words, directive_count)
    decision_density = density_per_100(n_words, decision_count)
    outcome_density = density_per_100(n_words, outcome_count)
    ambiguity_rate = density_per_100(n_words, ambiguity_count)

    # ----------------- Build Indices (0–1) -----------------
    # Substance
    S = (
        0.30 * normalize(number_density, 0.0, 6.0) +
        0.15 * normalize(example_density, 0.0, 1.0) +
        0.15 * normalize(citation_density, 0.0, 1.5) +
        0.20 * normalize(ttr, 0.25, 0.6) -
        0.10 * normalize(hedge_rate, 0.0, 3.0) -
        0.10 * normalize(buzz_rate, 0.0, 2.0)
    )
    S = float(np.clip(S, 0.0, 1.0))

    # Focus
    # Loosen focus thresholds and slightly rebalance weights to avoid collapse to 0
    F = (
        0.50 * normalize(prompt_sim_mean, 0.10, 0.90) -
        0.25 * normalize(redundancy, 0.25, 0.95) -
        0.10 * normalize(drift_rate, 0.0, 0.80) +
        0.15 * normalize(progression, 0.01, 0.30)
    )
    F = float(np.clip(F, 0.01, 1.0))

    # Actionability
    A = (
        0.35 * normalize(directive_density, 0.0, 5.0) +
        0.25 * normalize(outcome_density, 0.0, 3.0) +
        0.20 * normalize(decision_density, 0.0, 2.0) +
        0.10 * normalize(structured_ratio, 0.0, 0.3) -
        0.10 * normalize(ambiguity_rate, 0.0, 2.0)
    )
    A = float(np.clip(A, 0.0, 1.0))

    # Waffle Score: higher = more waffle
    # Invert the "good" signals (S, F, A) using a sigmoid
    waffle = 1.0 - (1.0 / (1.0 + math.exp(- (0.5*S + 0.3*F + 0.2*A - 0.5))))

    return dict(
        n_words=n_words, n_sents=n_sents,
        number_density=number_density,
        example_density=example_density,
        citation_density=citation_density,
        hedge_rate=hedge_rate, buzz_rate=buzz_rate, ttr=ttr,
        prompt_sim_mean=prompt_sim_mean, redundancy=redundancy,
        drift_rate=drift_rate, progression=progression,
        directive_density=directive_density, decision_density=decision_density,
        outcome_density=outcome_density, structured_ratio=structured_ratio,
        ambiguity_rate=ambiguity_rate,
        S=S, F=F, A=A, WaffleScore=waffle,
        backend=_EMBEDDINGS_BACKEND
    )

# ---------------------- Humorous Labels ----------------------

def _label_from_bins(value: float, bins: list, labels: list) -> str:
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
    # Short verdict sentences matching waffle score bins
    if value < 0.2:
        return "Crisp and focused — serve as‑is."
    if value < 0.4:
        return "Tidy writing; add a dash more concrete detail."
    if value < 0.6:
        return "Pleasantly fluffy; trim and tighten to land the point."
    if value < 0.8:
        return "Sticky with blather; anchor to outcomes and owners."
    return "Maximum waffle detected; evacuate buzzwords and bring numbers."

# ---------------------- Explanatory Analyses ----------------------

def _level(value: float, low: float, high: float) -> str:
    if value < low: return "low"
    if value > high: return "high"
    return "moderate"

def component_analyses(feats: Dict[str, float]) -> Dict[str, str]:
    # Substance
    num_level = _level(feats.get('number_density', 0), 1.0, 4.0)
    ex_level = _level(feats.get('example_density', 0), 0.2, 0.8)
    cit_level = _level(feats.get('citation_density', 0), 0.2, 1.2)
    ttr_level = _level(feats.get('ttr', 0), 0.30, 0.55)
    hedge_level = _level(feats.get('hedge_rate', 0), 0.3, 2.0)
    buzz_level = _level(feats.get('buzz_rate', 0), 0.2, 1.5)
    s1 = f"Evidence signals are {num_level} (numbers), {ex_level} (examples), {cit_level} (citations); vocabulary specificity is {ttr_level}."
    s2 = f"Hedges/buzzwords are {hedge_level}/{buzz_level}, which {'keeps it crisp' if hedge_level=='low' and buzz_level=='low' else 'slightly dilutes focus' if hedge_level!='low' or buzz_level!='low' else 'balances tone'}."
    s_text = s1 + " " + s2

    # Focus
    sim_level = _level(feats.get('prompt_sim_mean', 0), 0.30, 0.70)
    red_level = _level(feats.get('redundancy', 0), 0.30, 0.75)
    drift_level = _level(feats.get('drift_rate', 0), 0.15, 0.45)
    prog_level = _level(feats.get('progression', 0), 0.03, 0.18)
    f1 = f"Topic alignment is {sim_level}; redundancy is {red_level}."
    f2 = f"Drift is {drift_level} and progression is {prog_level}, indicating {'repetition' if red_level=='high' else 'tangents' if drift_level=='high' else 'a steady flow'}."
    f_text = f1 + " " + f2

    # Actionability
    dir_level = _level(feats.get('directive_density', 0), 1.0, 4.0)
    dec_level = _level(feats.get('decision_density', 0), 0.2, 1.0)
    out_level = _level(feats.get('outcome_density', 0), 0.5, 2.0)
    struct_level = _level(feats.get('structured_ratio', 0), 0.05, 0.20)
    amb_level = _level(feats.get('ambiguity_rate', 0), 0.3, 1.5)
    a1 = f"Action cues are {dir_level} (directives), {dec_level} (decisions), and {out_level} (outcomes); structure is {struct_level}."
    a2 = f"Vague verbs are {amb_level}, so actionability feels {'strong' if amb_level=='low' else 'mixed' if amb_level=='moderate' else 'light'}."
    a_text = a1 + " " + a2

    return { 'S': s_text, 'F': f_text, 'A': a_text }

# ---------------------- Taglines ----------------------

TAGLINES: List[str] = [
    "Behold, the Waffleometer has spoken.",
    "Hot off the griddle: your waffle reading.",
    "Sermon from the Mount of Maybe: transcript attached.",
    "The Church of Circular Reasoning is now in session.",
    "Buzzword barometer: beeps detected.",
    "We regret to inform you: synergy is not a KPI.",
    "Fresh data, fewer carbs.",
    "This just in: actionable items spotted in the wild.",
    "Your deck called; it wants fewer clouds, more rocks.",
    "The Blather Index has opinions.",
    "Forecast: scattered insights with a chance of decisions.",
    "We measured the vibe. The vibe asked for numbers.",
    "Mission accomplished? Let’s check.",
    "The KPI gods demand tributes of integers.",
    "Circular logic detected. Please exit the roundabout.",
    "Now screening: Return of the Metrics.",
    "Plot twist: specifics matter.",
    "Breaking: ‘Consider’ considers retiring.",
    "Detour avoided. We’re on the main road now.",
    "Granularity located. Bring a sieve.",
    "Your paragraph tried to pivot. We pivoted back.",
    "Today’s forecast: 0% chance of synergy showers.",
    "New from R&D: fact-flavored sentences.",
    "Our sensors detect a whiff of ‘perhaps’.",
    "Attention passengers: we are approaching Action Station.",
    "Good news: the buzzword budget is down 30%.",
    "The fluff filter caught a big one.",
    "Introducing: the Accountability Accelerator.",
    "We checked — your nouns can lift more.",
    "Repetition loop broken. You’re welcome.",
    "Surprise audit: verbs found idle.",
    "Your writing asked for a gym membership.",
    "We found the point. It was hiding in plain sight.",
    "The idea arrived. The details took the stairs.",
    "Breaking: ‘leverage’ leverages nothing.",
    "We pinged the plan. The plan pinged back.",
    "Spreadsheets are ready. Words will comply.",
    "We carbonated your claims with facts.",
    "Less vibe, more live data.",
    "Your waffle cone is leaking. Apply outcomes.",
    "Talked to the roadmap. It wants dates.",
    "The synergy siren has been silenced.",
    "We poked the jargon. It deflated.",
    "Breaking: nouns promoted to proper nouns.",
    "Congratulations, you have unlocked: Bullet Points.",
    "We adjusted the focus. It stopped daydreaming.",
    "We added teeth to your data. It can bite now.",
    "Non-actionable vibes escorted off the premises.",
    "Your text took a lap; we set a finish line.",
    "Idea density upgraded from mist to drizzle.",
    "We drained the Syrup Swamp (most of it).",
    "New achievement: Decisions Made On Purpose.",
    "We de-buzzed your buzzwords.",
    "Archaeology report: found artifacts of meaning.",
    "We put the ‘why’ back on speaking terms with the ‘how’.",
    "Your claims got IDs. Security approves.",
    "Narrative GPS acquired. Recalculating route.",
    "We replaced maybe with Monday.",
    "We replaced ambition with owners.",
    "The prose did a stand-up. It has blockers.",
    "We set your thoughts to release mode.",
    "Proof of work delivered. Buzzwords on vacation.",
    "Your strategy stopped networking and started working.",
    "We tuned the signal. Static reduced.",
    "Loose ends tied. Bow optional.",
    "Your text now ships with instructions.",
    "We removed three loops and a detour.",
    "Focus engaged. Side quests postponed.",
    "Benchmarks updated. Hype downgraded.",
    "We brought a ruler to your ambition.",
    "Meet your nouns: now with payloads.",
    "We took your plan to task.",
    "Goodbye fluff, hello stuff.",
    "We added gravity to your ideas.",
    "We swapped glitter for glue.",
    "The committee of caveats has been adjourned.",
    "Clarity called. We answered.",
    "The plot found its spine.",
    "Your verbs now come with verbs.",
    "We replaced whispers with numbers.",
    "The roadmap put on shoes.",
    "The exec summary learned to summarize.",
    "We cut the parade and kept the marching orders.",
    "The thought leadership found a map.",
    "Your plan checked into reality.",
    "Ideas grounded. Taxi to runway.",
    "We turned circular into forward.",
    "Your deck caught a deadline.",
    "We placed targets where the arrows land.",
    "Congratulations: your writing can lift a metric.",
    "From vibes to deliverables in under 60 seconds.",
    "We added dates so time can find you.",
    "The vision put on reading glasses.",
    "Your pitch learned basic carpentry.",
    "We brought receipts to the meeting.",
    "We swapped adjectives for evidence.",
    "Your memo discovered ground truth.",
    "The plan found its calendar.",
    "We upgraded ‘soon’ to ‘by Friday’.",
    "Your story learned plot armor.",
    "We trained your nouns to carry weight.",
    "The waffle maker is unplugged (for now).",
    "Your bullets became bullseyes.",
    "The elevator pitch now fits in an elevator.",
    "We de-echoed the echo chamber.",
    "We rebooted your clarity settings.",
    "We added handles to your ideas.",
]

def choose_tagline() -> str:
    """Pick a random tagline while avoiding repeats from the last 5 selections."""
    history_key = "recent_taglines"
    if history_key not in st.session_state:
        st.session_state[history_key] = []  # list of recent strings
    recent: List[str] = st.session_state[history_key]
    # Build candidate pool excluding recent items
    candidates = [t for t in TAGLINES if t not in recent]
    if not candidates:
        candidates = TAGLINES[:]
    selected = random.choice(candidates)
    recent.append(selected)
    # Keep only the last 5
    if len(recent) > 5:
        recent = recent[-5:]
    st.session_state[history_key] = recent
    return selected

# Score-bucketed verdict taglines
SCORE_TAGLINES: Dict[str, List[str]] = {
    "low": [
        "Zero fluff, all stuff.",
        "Sharper than a budget review.",
        "Audit‑ready and meeting‑friendly.",
        "Signal so clean it squeaks.",
        "Facts doing cartwheels.",
        "Executive‑safe since paragraph one.",
        "Precision served hot.",
        "Clarity with extra crunch.",
    ],
    "lowmid": [
        "Mostly meat, light garnish.",
        "Almost crisp — a sprinkle more specifics.",
        "Focused with minor scenic views.",
        "Hints of waffle; nothing a metric can’t fix.",
        "Good spine, add a few ribs.",
        "Nearly airtight — add timestamps.",
        "Roadmap visible, zoom in slightly.",
        "Solid draft; bolt on outcomes.",
    ],
    "mid": [
        "Balanced breakfast: half waffle, half plan.",
        "Pleasantly fluffy — trim for takeoff.",
        "Two edits from ruthless clarity.",
        "The compass works; pick a trail.",
        "Narrative cruising; tighten landing gear.",
        "Promising shape, soft edges.",
        "Middle‑manager energy; promote with proof.",
        "Add owners, watch it sprint.",
    ],
    "highmid": [
        "Sticky in parts — deploy numbers and names.",
        "Roundabout detected; take the first exit to ‘Action’.",
        "Vibes are winning — bench them for verbs.",
        "Too much sermon, not enough schedule.",
        "Ideas floating; add gravity and Gantt.",
        "Trim the tour, keep the destination.",
        "Syrup levels high — introduce receipts.",
        "Buzzwords circling — switch to plain speak.",
    ],
    "high": [
        "Maximum waffle — declare a metrics emergency.",
        "Sermon from the Mount of Maybe — bring dates.",
        "Cathedral of caveats — open book the KPIs.",
        "Blatherquake — stabilize with outcomes and owners.",
        "Lost in the Church of Circular Reasoning.",
        "Synergy storm — evacuate to specifics.",
        "Fog advisory — lights on, numbers out.",
        "Buzzword bonfire — stop, drop, and measure.",
    ],
}

def _score_bin(score: float) -> str:
    if score < 0.2:
        return "low"
    if score < 0.4:
        return "lowmid"
    if score < 0.6:
        return "mid"
    if score < 0.8:
        return "highmid"
    return "high"

def choose_tagline_for_score(score: float, avoid: str = "") -> str:
    key = _score_bin(score)
    pool = SCORE_TAGLINES.get(key, [])
    hist_key = "recent_verdict_taglines"
    recent: List[str] = st.session_state.get(hist_key, [])
    candidates = [t for t in pool if t not in recent and t != avoid]
    if not candidates:
        candidates = [t for t in pool if t != avoid] or pool[:]
    selected = random.choice(candidates)
    recent.append(selected)
    if len(recent) > 5:
        recent = recent[-5:]
    st.session_state[hist_key] = recent
    return selected

# ---------------------- UI ----------------------

st.set_page_config(page_title="The Waffle Cube", page_icon="🧇", layout="wide")
st.title("🧇 The Waffle Cube")
st.markdown(
    '<div class="subtitle">Operationalising Obfuscation: An Empirical Approach to Waffle Intensity.</div>',
    unsafe_allow_html=True,
)

# Smaller fonts for description and metric scores
st.markdown(
    """
    <style>
    /* Metric value and label sizes */
    div[data-testid="stMetricValue"] > div { font-size: 1.1rem; line-height: 1.2; color: #ffffff; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; line-height: 1.1; color: #ffffff; }
    /* Description paragraph */
    .desc { font-size: 0.95rem; color: #ffffff; }
    /* Subtitle under title */
    .subtitle { font-size: 1.4rem; color: #ffffff; font-style: italic; margin: -0.2rem 0 0.4rem 0; }
    /* Results headline */
    .tagline { font-size: 1.4rem; color: #ffffff; font-weight: 700; margin: 0.25rem 0 0.4rem 0; }
    .headline-score { font-size: 1.35rem; font-weight: 700; margin: 0.1rem 0; }
    .headline { font-size: 1.2rem; font-weight: 700; margin: 0 0 0.6rem 0; color: #ffffff; }
    /* Full-bleed divider */
    hr.full-bleed { border: none; border-top: 2px solid #e0e0e0; margin: 0.4rem 0 0.9rem 0; width: 100vw; position: relative; left: 50%; right: 50%; margin-left: -50vw; margin-right: -50vw; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Full-width divider separating header from app content
st.markdown('<hr class="full-bleed" />', unsafe_allow_html=True)

st.markdown(
    '<p class="desc">Measure <em>waffle</em> across <strong>Substance</strong>, <strong>Focus</strong>, and <strong>Actionability</strong>. Upload a document or paste text.</p>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload a .txt or .md document", type=["txt", "md"])
    prompt = st.text_input("Prompt / Question (improves the Focus metric)", value="Summarise the strategy in three points.")

with col2:
    sample = st.text_area("Or paste text here", height=300, help="You can combine this with an uploaded file; both will be analysed.")

text = ""
if uploaded is not None:
    try:
        text += uploaded.read().decode("utf-8", errors="ignore")
    except:
        st.warning("Could not decode file as UTF-8.")

text += "\n" + sample if sample else ""

text = text.strip()

st.caption(f"Embeddings backend: **{_EMBEDDINGS_BACKEND}**")

if st.button("Analyse") or (text and len(text) > 10):
    if not text:
        st.error("Please upload or paste some text.")
    else:
        feats = compute_features(text, prompt)
        s_label = label_substance(feats['S'])
        f_label = label_focus(feats['F'])
        a_label = label_actionability(feats['A'])
        w_label = label_waffle(feats['WaffleScore'])
        # Results banner: tagline (random, no repeat within last 5), headline score, verdict
        tagline_text = choose_tagline()
        st.markdown(f'<div class="tagline">Congratulations — {tagline_text}. Waffle score: {w_label}.</div>', unsafe_allow_html=True)
        verdict_tag = choose_tagline_for_score(feats["WaffleScore"], avoid=tagline_text)
        st.markdown(f'<div class="headline">Verdict: {verdict_tag}</div>', unsafe_allow_html=True)
        left_col, right_col = st.columns([1,1])
        with left_col:
            m1, m2 = st.columns(2)
            m1.metric("Meatiness Quotient (Substance, S)", f"{feats['S']:.2f} — {s_label}")
            m2.metric("Laser Aim (Focus, F)", f"{feats['F']:.2f} — {f_label}")
            m3, m4 = st.columns(2)
            m3.metric("Get‑Stuff‑Done Quotient (Actionability, A)", f"{feats['A']:.2f} — {a_label}")
            m4.metric("Waffle Score (↑ = more waffle)", f"{feats['WaffleScore']:.2f} — {w_label}")
            # Component analyses (two sentences each)
            expl = component_analyses(feats)
            st.markdown("<hr style='margin: 0.6rem 0' />", unsafe_allow_html=True)
            st.markdown(f"<div class='desc'><strong>Substance:</strong> {expl['S']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='desc'><strong>Focus:</strong> {expl['F']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='desc'><strong>Actionability:</strong> {expl['A']}</div>", unsafe_allow_html=True)
            st.subheader("Feature Diagnostics")
            with st.expander("Show raw feature values"):
                st.json({k: v for k, v in feats.items() if k not in ['backend']})

        with right_col:
            # 3D scatter: one point; strong grid on white background
            try:
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Scatter3d(
                    x=[feats['S']], y=[feats['F']], z=[feats['A']],
                    mode='markers+text', text=["Your Text"], textposition="top right",
                    marker=dict(size=6, color='crimson'),
                    textfont=dict(color='black')
                )])
                fig.update_layout(
                    scene=dict(
                        bgcolor='white',
                        domain=dict(x=[0.02, 0.92], y=[0.02, 0.92]),
                        xaxis=dict(
                            range=[0,1], backgroundcolor='white',
                            gridcolor='#9a9a9a', gridwidth=2,
                            zerolinecolor='#777777', zerolinewidth=2,
                            title=dict(text='Meatiness Quotient', font=dict(color='black')),
                            tickfont=dict(color='black'),
                            showbackground=True
                        ),
                        yaxis=dict(
                            range=[0,1], backgroundcolor='white',
                            gridcolor='#9a9a9a', gridwidth=2,
                            zerolinecolor='#777777', zerolinewidth=2,
                            title=dict(text='Laser Aim', font=dict(color='black')),
                            tickfont=dict(color='black'),
                            showbackground=True
                        ),
                        zaxis=dict(
                            range=[0,1], backgroundcolor='white',
                            gridcolor='#9a9a9a', gridwidth=2,
                            zerolinecolor='#777777', zerolinewidth=2,
                            title=dict(text='Get‑Stuff‑Done Quotient', font=dict(color='black')),
                            tickfont=dict(color='black'),
                            showbackground=True
                        )
                    ),
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    height=520,
                    margin=dict(l=90, r=90, t=70, b=90)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Install plotly for a 3D scatter: `pip install plotly`.")

        

        
else:
    st.info("Upload or paste text, optionally add the prompt, then click **Analyse**.")

