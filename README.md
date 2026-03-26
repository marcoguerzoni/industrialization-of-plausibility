# The Industrialization of Plausibility
### Large Language Models and the Transformation of Semiosis

**Author:** Marco Guerzoni — Università degli Studi di Milano-Bicocca (DEMS)

---

## Overview

This repository contains all materials for the empirical component of the article *"The Industrialization of Plausibility: Large Language Models and the Transformation of Semiosis"*, which tests three semiotic theses about LLMs using a five-experiment computational design grounded in Umberto Eco's semiotics.

| Thesis | Claim | Experiments |
|--------|-------|-------------|
| T1 | LLMs instantiate Eco's encyclopedia | E4, E1 |
| T2 | LLMs tend structurally toward overcoding | E2, E5 |
| T3 | LLM use entails an unevenly distributed semiotic cost of context | E3 |

---

## Repository Structure

```
.
├── README.md
├── .gitignore
├── empirics/
│   ├── experimental_protocol.md   # 22-week timeline, budget, expert requirements
│   ├── article/
│   │   └── article_v4.tex         # Current article manuscript
│   ├── stimuli/
│   │   ├── exp4_minimal_pairs.csv         # E4: 50 minimal-pair sentences (template)
│   │   ├── exp1_polysemous_terms.csv      # E1: 30 polysemous terms + 5 prompt variants
│   │   └── exp2_concepts.csv              # E2: 40 concepts (charged + controls)
│   ├── prompts/
│   │   └── prompts_all.json               # All prompt templates, keyed by experiment/condition
│   ├── scripts/
│   │   ├── utils.py                       # Shared: API clients, logging, embedding
│   │   ├── exp4_collect.py                # E4 data collection
│   │   ├── exp1_collect.py                # E1 data collection
│   │   ├── exp2_collect.py                # E2 data collection
│   │   ├── exp5_collect.py                # E5 data collection
│   │   ├── exp3_collect.py                # E3 data collection + LLM judge
│   │   ├── exp4_analyze.py                # E4 analysis (disambiguation accuracy)
│   │   ├── exp1_analyze.py                # E1 analysis (UMAP + silhouette)
│   │   ├── exp2_analyze.py                # E2 analysis (schema conformity + BERTopic)
│   │   ├── exp5_analyze.py                # E5 analysis (diachronic trajectories)
│   │   └── exp3_analyze.py                # E3 analysis (quality + effort-to-quality)
│   ├── data/                              # Raw API outputs (gitignored)
│   └── results/                           # Figures and tables (gitignored by default)
└── requirements.txt
```

---

## Experiment Summary

| Exp | Tests | N outputs | Primary outcome |
|-----|-------|-----------|-----------------|
| E4 | T1 (mechanistic) | 6,000 | Accuracy on critical items vs. co-occurrence baseline |
| E1 | T1 (distributional) | 9,000 | Silhouette(context) > Silhouette(lemma), paraphrase condition |
| E2 | T2 (synchronic) | 48,000 | Schema-conformity score vs. matched controls + corpus baseline |
| E5 | T2 (diachronic) | 4,000 | Generation × concept-type interaction |
| E3 | T3 | 14,400 + user study N=120 | Quality (output side) + effort-to-quality ratio (production side) |

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys before running any collection script.

---

## Pre-registration

Pre-registration will be filed on OSF before data collection begins. DOI will be inserted here upon filing.

---

## Citation

```
Guerzoni, M. (forthcoming). The Industrialization of Plausibility:
Large Language Models and the Transformation of Semiosis.
```
