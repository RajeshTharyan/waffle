# 🧇 The Waffle Cube  
## A Three-Dimensional Embeddings Framework for Quantifying Verbosity

**Authors:** Haku Rajesh, Rajesh Tharyan and Insight Companion  
**Date:** November 27, 2025  

---

## Abstract

This project introduces **The Waffle Cube**, an operationalised metric system for quantifying verbosity, topical drift, and lack of decisional content in managerial or academic prose. The framework evaluates text along three interpretable dimensions: **Substance (S)**, **Focus (F)**, and **Actionability (A)**. Each dimension is derived from linguistically meaningful surface and semantic features, computed using sentence embeddings (MiniLM) or a TF-IDF fallback. A sigmoid-inverted composite index - the **Waffle Score** - represents overall communicative inefficiency. While presented humorously, the model provides serious analytic and pedagogical utility for writing instruction, clarity audits, and professional development. 

---

## 1. Introduction

The English language, glorious and over-caffeinated, has long suffered under the weight of unnecessary words. From committee minutes to annual reports, humans appear evolutionarily predisposed to add three more adjectives where one would do. *Waffle*, in this context, is not breakfast but behaviour: a caloric surplus of syntax, a syrupy excess of semi-relevant clauses.

In academic circles, waffle manifests as the *hedge spiral*, wherein authors construct entire ecosystems of caveats before daring to state a claim. In corporate communications, it takes the form of *PowerPoint bloat*, in which bullet points procreate inorganically without a grand design - or any design at all,  until the original insight has fled the slide deck in despair.

Despite centuries of stylistic advice, no quantitative framework has managed to measure waffle scientifically. Until now.

This work therefore proposes a rigorously unserious but methodologically sound approach: to model waffle as a measurable artifact in three orthogonal dimensions: Substance, Focus, and Actionability and to compress this space into a single interpretable measure, the **Waffle Score**.

Waffle is treated not as random noise but as a structured linguistic phenomenon, detectable through embeddings and lexical statistics. In the same way a spectrometer reveals the chemical composition of stars, the Waffle Cube reveals the informational composition of sentences. Beneath the surface of every overwrought paragraph lies a turbulent ecology of half-formed notions and speculative verbs desperately searching for an object. This is treated not merely as stylistic clutter but as evidence of the cognitive compost heap from which managerial language blooms. 

In essence, waffle is the observable residue of the human brain’s attempt to disguise uncertainty as strategy - a form of *scrambled and coagulated mind matter*, rich in semantic calories but low in nutritional truth. By applying embedding models to this verbal soup, the framework aims to separate protein (meaning) from froth (presentation), yielding what is described as the first reproducible taxonomy of linguistic entropy. 

---

## 2. Related Work

Traditional readability indices (Flesch, Gunning Fog) measure difficulty rather than density.  They cannot distinguish between “complex ideas clearly stated” and “simple ideas stretched beyond reason.” Recent NLP developments for example Gurevych (2019) enable fine-grained semantic comparison using embeddings, allowing us to estimate how \textit{on-topic} or \textit{repetitive} a text may be. Parallel work in requirements engineering (Briand et al., 2016) and text summarisation  provides inspiration for measuring focus, progression, and outcome orientation (Zhang et al., 2020). 

---

## 3. Methodology

### 3.1 Conceptual Model

The Waffle Cube operationalises verbosity as the inverse of linguistic utility across three measurable axes:

1. **Substance (S)** - Are we saying anything that could survive contact with a spreadsheet?  
2. **Focus (F)** - Does the argument remain on-topic, or has it drifted into a scenic detour about “paradigm shifts”?  
3. **Actionability (A)** - Could a rational person execute something based on this paragraph, or merely nod thoughtfully and forget?

Each axis is normalised to the range [0, 1] based on empirical thresholds. The cube structure provides a geometric metaphor: an ideal text sits near (1, 1, 1) - dense, coherent, and executable - whereas pure waffle collapses toward the origin.

---

### 3.2 Sentence Representation and Similarity

Sentence embeddings `e(s_i)` are computed using the *all-MiniLM-L6-v2* transformer model. If unavailable, a TF-IDF fallback creates a shared vocabulary space between the document and the user’s prompt `p`, ensuring comparable semantic geometry.

Cosine similarity between sentences and `p` estimates topical adherence, while inter-sentence similarities provide redundancy and progression indicators. High pairwise similarity implies repetition (looping waffle); low values suggest drift (aimless waffle).

---

### 3.3 Substance (S)

Substance quantifies evidential density and linguistic specificity:

S = 0.30·n̂ + 0.15·êx + 0.15·ĉi + 0.20·t̂tr − 0.10·ĥ − 0.10·b̂z


The model rewards numbers, examples, citations, and lexical variety, while penalising hedges (“perhaps”, “somewhat”) and buzzwords (“synergy”, “ecosystem”). A low Substance score corresponds to what editors call *word fog*.

**Where:**

 **n̂ – normalised numeric and currency density**  
  Captures the prevalence of quantitative expressions relative to total text length, indicating the degree of empirical or financial specificity. Higher values signal reliance on measurable evidence rather than qualitative description. Persistently low values are characteristic of narrative-heavy or impressionistic prose.

- **êx – normalised example density**  
  Measures the frequency of explicit illustrative phrases such as “for example” or “such as,” reflecting how often abstract claims are grounded in concrete instances. Well-calibrated example density improves interpretability and reader comprehension. Very low values often indicate declarative writing that assumes rather than demonstrates understanding.

- **ĉi – normalised citation density**  
  Reflects the presence of formal references, including citations, URLs, or DOIs, as a proportion of the text. This variable captures the extent to which claims are externally verifiable and evidence-backed. Low citation density frequently corresponds to opinion-driven or internally focused discourse.

- **t̂tr – type–token ratio (lexical diversity proxy)**  
  Represents the ratio of unique tokens to total tokens, serving as a measure of vocabulary variety and linguistic precision. Higher values suggest discriminating word choice and reduced repetition. Lower values often indicate reliance on generic phrasing or circular restatement.

- **ĥ – hedge density**  
  Measures the frequency of epistemic qualifiers such as “perhaps,” “possibly,” or “somewhat,” signalling uncertainty or qualification. Moderate levels may be appropriate in exploratory or academic contexts. Sustained high hedge density, however, weakens commitment and obscures substantive claims.

- **b̂z – buzzword density**  
  Quantifies the prevalence of fashionable or managerial jargon that carries rhetorical weight but limited operational meaning. Such terms can inflate perceived sophistication without adding informational value. Elevated buzzword density is a hallmark of performative or impression-management-oriented prose.


---

### 3.4 Focus (F)

Focus measures coherence and logical progression:

F = 0.50·ŝim − 0.25·r̂ed − 0.10·d̂rift + 0.15·p̂rog

**ŝim – topic adherence**  
  Measures semantic alignment between individual sentences and the central topic or reference prompt. High values indicate focused, on-topic discourse with limited diversion. Low values suggest thematic dilution or weak anchoring to the core argument.

- **r̂ed – redundancy**  
  Captures semantic repetition across sentences, indicating whether content advances or merely restates prior points. Excessive redundancy often reflects looping or circular argumentation. Moderate redundancy may be acceptable for emphasis, but high values typically signal inefficiency.

- **d̂rift – off-topic wanderings**  
  Quantifies the extent to which the text diverges semantically from its stated topic. High drift scores indicate scenic detours that weaken argumentative coherence. Controlled drift may occur in exploratory writing, but sustained drift erodes clarity and focus.

- **p̂rog – narrative progression**  
  Measures whether successive sentences introduce new information or advance the argument in a structured manner. High values reflect purposeful development and logical sequencing. Low values suggest stagnation or thematic stalling.

Weighting was empirically adjusted to avoid collapsing Focus to near zero in legitimate exploratory writing. 

---

### 3.5 Actionability (A)

Actionability evaluates the practical “do-ness” of prose:

A = 0.35·d̂ir + 0.25·ôut + 0.20·d̂ec + 0.10·ŝtruct − 0.10·âmb

High values indicate clear verbs (“implement”, “decide”), measurable outcomes (dates, KPIs, percentages), and structural cues (bullet lists). Low values indicate speculative, vibe-based text (“explore”, “enable”, “consider”).

**Where:**

 **d̂ir – directive density**  
  Measures the frequency of imperative or action-oriented verbs such as “implement,” “prioritise,” or “measure.” High values indicate instructionally clear and execution-ready prose. Low values correspond to descriptive text with limited operational guidance.

- **ôut – outcome density**  
  Captures the prevalence of explicit deliverables, deadlines, KPIs, or measurable end states. This variable reflects the degree to which intentions are translated into observable results. Low outcome density is typical of aspirational or vision-led statements.

- **d̂ec – decision cue density**  
  Measures the presence of explicit commitments or choice statements, such as “we decide” or “we recommend.” High values signal decisional clarity and ownership. Low values indicate deferral, equivocation, or avoidance of commitment.

- **ŝtruct – structural ratio**  
  Represents the proportion of structured elements such as bullet points, numbered steps, or ordered lists. Higher values reflect organised, execution-friendly presentation. Very low values are associated with free-form prose lacking procedural clarity.

- **âmb – ambiguity density**  
  Quantifies the frequency of vague verbs and non-committal language that obscure responsibility or intent. High ambiguity density makes action translation difficult despite apparent positivity. Low ambiguity density corresponds to precise, accountable communication.


---

### 3.6 Composite Waffle Score

The Waffle Score inversely aggregates the three axes via a sigmoid transformation:

W = 1 − σ(0.5S + 0.3F + 0.2A − 0.5)
σ(x) = 1 / (1 + e^(−x))


This maps virtuous clarity to low scores (“Toast-Dry”) and syrupy circumlocution to high scores (“All-You-Can-Blather Buffet”). The resulting index is bounded, smooth, and interpretable. 

---

## 4. Interpretation and Diagnostics

### 4.1 Categorical Mapping

Continuous values (S, F, A, W) are discretised into humorous linguistic bins. For example, `S < 0.2` becomes *Blather Vapor*, whereas `S > 0.8` earns *Laser-Fact Cannon*. Each dimension maps onto a rhetorical spectrum:

- **Substance:** gaseous adjectives → weaponised data  
- **Focus:** tangential sermons → missile-grade precision  
- **Actionability:** “Plan? Vibes.” → “Gantt Gladiator.” 

---

### 4.2 Diagnostic Text Generation

Beyond numeric scores, the system produces short interpretive diagnostics. Each dimension yields a two-sentence analysis combining linguistic metrics with playful commentary.

Examples include statements indicating low evidential density and high hedge rates, moderate topic alignment with high redundancy and drift, or low action cues dominated by vague verbs. While humorous, these diagnostics are grounded in linguistic evidence and serve as accessible feedback. Empirically, users report significant improvements in clarity motivated by avoiding unflattering diagnostic labels. 

---

## 5. Implementation and Visualisation

The application is built in **Streamlit 1.37+**, using Sentence-BERT embeddings with TF-IDF fallback. It provides:

- Interactive 3D visualisation of (S, F, A) via Plotly  
- Randomised taglines to ensure novelty and engagement  
- JSON feature diagnostics and a transparent metric pipeline 

---

## 6. Applications and Ethics

The Waffle Cube is both satire and tool. Used responsibly, it supports concise communication and evidence-driven writing. Used recklessly, it may undermine entire consulting industries. The model assumes English business discourse norms; calibration is advised for other rhetorical traditions. 

---

## 7. Conclusion

The **Waffle Cube** merges humour with NLP precision to create a novel metric of rhetorical efficiency. It treats verbosity not as a vice but as a variable - measurable, improvable, and occasionally delicious. 

---

## References

1. Reimers, N. and Gurevych, I. (2019).Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP) and the 9th International Joint Conference on Natural Language Processing (IJCNLP), pp. 3982–3992. Association for Computational Linguistics. https://arxiv.org/abs/1908.10084  

2. Méndez Fernández, D., Wagner, S., Kalinowski, M., Felderer, M., Mafra, P., Vetrò, A., Conte, T., Christiansson, M.-T., Greer, D., Lassenius, C., Männistö, T., Matulevičius, R., Penzenstadler, B., Rodrigues, G., Sillitti, A., and Briand, L. (2016).Naming the Pain in Requirements Engineering: A Design for a Global Family of Surveys and First Results from Germany.	Information and Software Technology, 57, 616–643.
   
3. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi, Y. (2020).BERTScore: Evaluating Text Generation with BERT. In International Conference on Learning Representations.ICLR. https://arxiv.org/abs/1904.09675 

