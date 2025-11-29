# 🧇 The Waffle Cube  
## A Three-Dimensional Embeddings Framework for Quantifying Managerial Verbosity

**Authors:** Haku Rajesh, Rajesh Tharyan and Insight Companion  
**Date:** November 27, 2025  

---

## Abstract

This project introduces **The Waffle Cube**, an operationalised metric system for quantifying verbosity, topical drift, and lack of decisional content in managerial or academic prose.  
The framework evaluates text along three interpretable dimensions: **Substance (S)**, **Focus (F)**, and **Actionability (A)**. Each dimension is derived from linguistically meaningful surface and semantic features, computed using sentence embeddings (MiniLM) or a TF-IDF fallback.  
A sigmoid-inverted composite index - the **Waffle Score** - represents overall communicative inefficiency.  

While presented humorously, the model provides serious analytic and pedagogical utility for writing instruction, clarity audits, and professional development. :contentReference[oaicite:0]{index=0}

---

## 1. Introduction

The English language, glorious and over-caffeinated, has long suffered under the weight of unnecessary words. From committee minutes to annual reports, humans appear evolutionarily predisposed to add three more adjectives where one would do. *Waffle*, in this context, is not breakfast but behaviour: a caloric surplus of syntax, a syrupy excess of semi-relevant clauses.

In academic circles, waffle manifests as the *hedge spiral*, wherein authors construct entire ecosystems of caveats before daring to state a claim. In corporate communications, it takes the form of *PowerPoint bloat*, in which bullet points reproduce asexually until the original insight has fled the slide deck in despair.

Despite centuries of stylistic advice, no quantitative framework has managed to measure waffle scientifically. Until now.

This work therefore proposes a rigorously unserious but methodologically sound approach: to model waffle as a measurable artifact in three orthogonal dimensions: Substance, Focus, and Actionability and to compress this space into a single interpretable measure, the **Waffle Score**.

Waffle is treated not as random noise but as a structured linguistic phenomenon, detectable through embeddings and lexical statistics. In the same way a spectrometer reveals the chemical composition of stars, the Waffle Cube reveals the informational composition of sentences.

Beneath the surface of every overwrought paragraph lies a turbulent ecology of half-formed notions and speculative verbs desperately searching for an object. This is treated not merely as stylistic clutter but as evidence of the cognitive compost heap from which managerial language blooms. In essence, waffle is the observable residue of the human brain’s attempt to disguise uncertainty as strategy - a form of *scrambled and coagulated mind matter*, rich in semantic calories but low in nutritional truth. By applying embedding models to this verbal soup, the framework aims to separate protein (meaning) from froth (presentation), yielding what is described as the first reproducible taxonomy of linguistic entropy. :contentReference[oaicite:1]{index=1}

---

## 2. Related Work

Traditional readability indices (Flesch, Gunning Fog) measure difficulty rather than density. They cannot distinguish between “complex ideas clearly stated” and “simple ideas stretched beyond reason”.  

Recent NLP developments enable fine-grained semantic comparison using embeddings, allowing estimation of how on-topic or repetitive a text may be. Parallel work in requirements engineering and text summarisation provides inspiration for measuring focus, progression, and outcome orientation. :contentReference[oaicite:2]{index=2}

---

## 3. Methodology

### 3.1 Conceptual Model

The Waffle Cube operationalises verbosity as the inverse of linguistic utility across three measurable axes:

1. **Substance (S)** - Are we saying anything that could survive contact with a spreadsheet?  
2. **Focus (F)** - Does the argument remain on-topic, or has it drifted into a scenic detour about “paradigm shifts”?  
3. **Actionability (A)** - Could a rational person execute something based on this paragraph, or merely nod thoughtfully and forget?

Each axis is normalised to the range [0, 1] based on empirical thresholds. The cube structure provides a geometric metaphor: an ideal text sits near (1, 1, 1) - dense, coherent, and executable - whereas pure waffle collapses toward the origin. :contentReference[oaicite:3]{index=3}

---

### 3.2 Sentence Representation and Similarity

Sentence embeddings `e(s_i)` are computed using the *all-MiniLM-L6-v2* transformer model. If unavailable, a TF-IDF fallback creates a shared vocabulary space between the document and the user’s prompt `p`, ensuring comparable semantic geometry.

Cosine similarity between sentences and `p` estimates topical adherence, while inter-sentence similarities provide redundancy and progression indicators. High pairwise similarity implies repetition (looping waffle); low values suggest drift (aimless waffle). :contentReference[oaicite:4]{index=4}

---

### 3.3 Substance (S)

Substance quantifies evidential density and linguistic specificity:

