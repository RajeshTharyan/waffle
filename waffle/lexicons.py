"""Lexical sets, regex patterns, and tagline pools used by scoring and the UI."""

from __future__ import annotations

import re
from typing import Dict, List

HEDGES = set(
    """apparently arguably basically broadly could generally hopefully kind of largely
likewise maybe might perhaps possibly pretty reportedly seems should somewhat sort of supposedly
theoretically typically usually often potentially relatively ostensibly virtually approximately""".split()
)

BUZZWORDS = set(
    """synergy leverage paradigm ecosystem cutting-edge disruptive innovative visionary
best-in-class next-gen world-class dynamic scalable holistic turnkey granular revolutionary robust
bleeding-edge mission-critical value-add low-hanging-fruit blockchain metaverse ai-driven big data
digital transformation stakeholder alignment thought leadership""".split()
)

VAGUE_VERBS = set(
    """leverage utilise facilitate enable consider explore examine address drive deliver
unlock streamline optimise optimize empower inspire ideate ideation""".split()
)

DIRECTIVE_MARKERS = set(
    """do implement adopt prioritise prioritize allocate define choose decide
ship launch schedule measure forecast budget report present must should will ensure assign approve""".split()
)

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
    r"\[\d+\]",
    r"\(20\d{2}\)",
    r"https?://",
    r"doi:\S+",
]

EXAMPLE_PATTERNS = [r"\bfor example\b", r"\be\.g\.\b", r"\bsuch as\b"]

BULLET_PAT = re.compile(r"^\s*[-*•\d]+\s+", re.IGNORECASE)

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
