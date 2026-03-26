# Experimental Protocol and Timeline
**Project:** The Industrialization of Plausibility — Empirical Component
**Author:** Marco Guerzoni
**Last updated:** 2026-03-26 (revised after critical review; reflects article_v4.tex)

**Key changes from previous version:**
- E4 now includes **critical items** (n=24): stimuli where corpus-frequency sense ≠ encyclopedic sense. These are the primary test items; non-critical items are secondary.
- E2 primary outcome changed from **Shannon entropy → schema-conformity score** (cosine similarity to pre-constructed cultural prototype). Entropy retained as secondary. Requires additional prototype-construction task in Week 2.
- E3 user study now operationalizes the **production-side semiotic cost** via effort-to-quality ratio (time + edits + self-reported difficulty ÷ output quality). This is required for the inequality claim; output quality alone is insufficient.
- E4 adds **GPT-2 XL as co-occurrence baseline** (in addition to WSD and dictionary baselines).
- Pre-registration must include **adjudication rules** for conflicting evidence within a thesis.

---

## Overview

Five experiments totalling approximately **78,400 LLM-generated outputs** across 3 models, plus a user study (N=120). The total timeline from preparation to submission-ready results is **22 weeks**, organized in four phases. Work that can run in parallel is marked **[PARALLEL]**.

---

## Phase 1 — Preparation and Pre-Registration (Weeks 1–4)

This phase produces no data. Its outputs are the materials, codebooks, and pre-registration that determine the validity of everything that follows. Do not skip or compress this phase.

### Week 1: Stimulus construction (Exp 4 and Exp 1)

**Tasks:**
- Construct 50 minimal-pair sentences for Exp 4 (dictionary-encyclopedia diagnostic).
  - 15 shallow, 15 medium, 20 deep inferential-depth items.
  - **NEW:** Within each stratum, identify ~8 items as *critical items*: items where the encyclopedic reading is corpus-frequency-non-dominant. Verify using WordNet sense frequency data and sense-tagged corpus statistics (SemCor). Add columns: [is_critical_item, corpus_dominant_sense, encyclopedic_target_sense, verification_source].
  - Draft in a spreadsheet: columns = [sentence_a, sentence_b, target_word, ground_truth_reading, inferential_depth_category, is_critical_item, note].
- Draft 30 polysemous terms for Exp 1 with strata (high/medium/near-monosemous) and 5 prompt variants each.
  - Paraphrase conditions require careful wording: domain implied through narrative, no domain vocabulary.
- Draft conceptual selection for Exp 2: 20 culturally charged + 20 matched controls.
  - Matching criteria: COCA word frequency (±15%), Wikipedia article length (±20%).

**Deliverable:** Three stimulus spreadsheets, version-controlled on OSF or GitHub private repo.

---

### Week 2: Expert annotation, rating protocols, and schema prototype construction

**Tasks:**
- Recruit 5 annotators for Exp 4 minimal pairs (linguists with lexical semantics background).
  - Administer disambiguation task. Target: Fleiss' κ ≥ 0.80. Discard and replace items below threshold.
  - **NEW:** Have annotators also verify the critical-item corpus-frequency classifications (confirm that the target sense is indeed non-dominant for critical items).
- Recruit 5 raters for Exp 2 cultural-charge ratings (Likert 1–7 stereotypy scale).
  - Administer ratings. Compute ICC. Replace items with ICC < 0.70.
- **NEW — Schema prototype construction for Exp 2 (primary outcome):**
  - Recruit 30 culturally representative human raters (separate from stereotypy raters above).
  - Administer feature-elicitation task: list the 5 features most associated with each of the 40 concepts (20 charged + 20 controls). Record free-text responses.
  - Synthesize: for each concept, create a prototype description from top-k most frequently mentioned features. Embed each prototype description using e5-large-v2.
  - Store prototype embeddings as .npy files before any LLM data collection begins. This is the pre-data-collection ground for the schema-conformity primary outcome.
  - For six person-concepts, additionally consult Fiske et al. (2002) stereotype content benchmarks.
- Recruit 10 domain experts (2 per high-complexity domain: law, medicine, pharmacology, finance, regulation) for ground-truth construction (Exp 3).
  - Each expert constructs one benchmark response per domain; a second expert verifies it.

**Deliverable:** Annotated stimulus sets with inter-rater statistics; prototype embedding files (.npy) for all 40 Exp 2 concepts; ground-truth response documents for Exp 3.

**Time budget for experts:**
- Linguistic annotators (Exp 4): ~2.5 hours each (add critical-item verification).
- Cultural-charge raters (Exp 2): ~1 hour each.
- Schema prototype raters (Exp 2, NEW): ~1.5 hours each (feature elicitation for 40 concepts).
- Domain experts (Exp 3 ground truth): ~3–4 hours each (construction + verification round).

---

### Week 3: Codebook finalization and UMAP hyperparameter grid search

**Tasks:**
- Finalize all prompt templates (system prompt, user turn for each condition). Store as JSON.
- Conduct UMAP hyperparameter grid search on a development set of 5 held-out polysemous terms (not used in main analysis).
  - Grid: n_neighbors ∈ {5, 15, 30, 50}, min_dist ∈ {0.0, 0.1, 0.5}.
  - Criterion: silhouette stability across 5 random seeds.
  - Fix hyperparameters; document them.
- Construct the $2^4 = 16$-condition factorial prompt matrix for Exp 3 (three domains only for full factorial; 4-level design for remaining 7 domains).
- Construct domain-complexity rating rubric and administer to 5 raters (confirm high/standard classification of 10 domains; ICC > 0.75 required).

**Deliverable:** Locked prompt JSON library; fixed UMAP parameters document; Exp 3 factorial design matrix.

---

### Week 4: Pre-registration and infrastructure setup

**Tasks:**
- Write and submit pre-registration document on OSF.
  - Must include: all hypotheses stated as directional predictions, analysis plan with model specifications, stopping rules, list of pre-registered interaction terms (especially Exp 3: domain × level), criteria for accepting LLM-as-judge (r ≥ 0.75 with human experts).
  - **NEW — Adjudication rules (must be in pre-registration):**
    - Thesis 1: E4 (mechanistic) takes precedence; E4 confirmation + E1 disconfirmation = restricted confirmation. Both disconfirmed = full disconfirmation.
    - Thesis 2: E2 (synchronic) and E5 (diachronic) are independent; both required for full confirmation.
    - Thesis 3: output-side prediction (factorial) alone supports weak quality-differential claim; production-side (user study effort-to-quality ratio) required for the inequality claim.
  - **NEW:** Specify that for E2, schema conformity is the primary DV and entropy is secondary; specify adjudication rule when they conflict (schema conformity takes precedence).
  - Record OSF pre-registration DOI and insert into Shared Methods section of the article.
- Set up API access and test calls for all three models.
- Set up logging infrastructure: every API call must write to a local log file [timestamp, model, prompt_id, temperature, response, token_count].
- Set up versioned output directory structure.
- Install and test: Python environment with `openai`, `replicate`/`together`, `mistralai`, `sentence_transformers`, `umap-learn`, `bertopic`, `scikit-learn`, `statsmodels`, `pingouin`, `nltk`, `spacy`.

**Deliverable:** Pre-registration link (share with co-author/supervisor for sign-off); working API test scripts; output directory scaffolding.

---

## Phase 2 — Data Collection (Weeks 5–11)

Data collection for experiments can run in parallel where API budgets allow. The ordering below reflects logical priority and risk management (simpler experiments first).

### Week 5: Exp 4 data collection [PARALLEL with Exp 2 setup]

- Run disambiguation probe for all 50 sentence pairs × 20 responses × 3 models × 2 temperatures (T=0.0 and T=0.7).
- **NEW:** Also run GPT-2 XL (co-occurrence baseline) on all 50 sentence pairs; no temperature variation needed (greedy decoding).
- Total calls: 50 × 20 × 3 × 2 = **6,000 API calls** (LLMs) + 50 (GPT-2 XL greedy).
- Estimated cost: ~$30–60 (GPT-4o dominant cost; Llama and Mistral cheaper).
- Run automated classifier on responses to assign [encyclopedia / dictionary / neither]. Validate on 50 held-out manually annotated examples (two independent coders; target κ ≥ 0.80).
- **NEW:** Run separate analysis on critical items (n=24) vs. non-critical items (n=26). The primary test statistic is LLM accuracy on critical items exceeding GPT-2 XL accuracy on the same items.

**Output:** 6,000 classified responses + classifier validation statistics.

---

### Week 6: Exp 1 data collection [PARALLEL with Exp 4 analysis]

- Run 150 prompts × 20 responses × 3 models = **9,000 API calls** at T=0.7.
- Embed all outputs under both embedding models (text-embedding-3-large and e5-large-v2).
- Total embeddings: 9,000 × 2 = 18,000 vectors.

**Output:** 9,000 raw texts + 18,000 embedding vectors stored as numpy arrays.

---

### Week 7: Exp 2 data collection

- Run 40 concepts × 100 responses × 4 temperatures × 3 models = **48,000 API calls**.
- This is the largest single data collection. Batch with rate-limit management.
- Estimated cost: ~$200–350 (GPT-4o is most expensive; batch API or off-peak usage recommended).
- Collect Wikipedia and Common Crawl corpus baseline texts simultaneously (50 Wikipedia paragraphs + 100 CC documents per concept = automated scraping, no API cost).

**Output:** 48,000 raw texts; corpus baseline documents.

---

### Week 8: Exp 5 data collection

- Run 20 concepts × 50 responses × 4 model generations = **4,000 API calls** at T=0.8.
- Note: GPT-3 (text-davinci-001) and GPT-3.5 legacy endpoints must be confirmed available. If unavailable, document and use closest accessible frozen version.
- Estimated cost: ~$40–80.

**Output:** 4,000 raw texts across 4 model generations.

---

### Weeks 9–10: Exp 3 LLM output generation [PARALLEL with Exp 2 and Exp 5 analysis]

- Run 160 prompt conditions × 30 responses × 3 models = **14,400 API calls**.
- LLM-as-judge scoring: 14,400 responses × 3 dimensions × 1 judge model = **43,200 judge calls**.
- Calibration subset: 200 responses × 3 judge models (inter-judge) = 600 additional calls.
- Intra-judge reliability: 200 responses re-scored after 48-hour gap = 200 additional calls.
- Total Exp 3 API calls: ~**58,400**.
- Estimated cost: ~$150–250.

**Output:** 14,400 raw texts + 14,400 structured quality scores + calibration statistics.

---

### Week 11: User study (Exp 3 — semiotic cost user side)

- Recruit N=120 participants via Prolific Academic (stratified: 4 education levels × 2 domain-familiarity levels × 3 domains = 120 cells; 10 per cell is minimum, aim for 12).
- Administer task: participants write a prompt for their assigned domain without instructions on structure. Measure: time-on-task (platform-recorded), prompt text (saved), self-reported difficulty (in-platform Likert item).
- Feed participant-authored prompts to LLM (GPT-4o); score outputs with the validated judge rubric.
- Two coders independently code each prompt for D/R/V/C component presence (target κ ≥ 0.80).

**Time budget for user study:**
- Participant task: ~15–20 minutes per participant.
- Platform: Prolific Academic; estimated participant cost at £8/hour = ~£24–27 per participant = ~£2,900–3,250 total.
- Coder time: 2 coders × 120 prompts × ~3 min = ~6 hours each.

**Output:** 120 participant prompts + coded components + quality scores + self-report data.

---

## Phase 3 — Analysis (Weeks 12–17)

### Week 12: Exp 4 analysis

- Compute disambiguation accuracy by system (LLM × 3, WSD, dictionary, co-occurrence/GPT-2 XL, human, random) and inferential depth.
- **NEW primary analysis:** Accuracy on critical items (n=24) vs. non-critical items (n=26), by system. The pre-registered test is: LLMs > GPT-2 XL co-occurrence baseline on critical items at medium and deep inferential depth (three-way interaction: system × depth × item_type).
- Fit mixed-effects logistic regression: correct ~ system × depth × item_type + (1|sentence) + (1|lemma).
- Produce Figure 2: grouped bar chart (accuracy by system × depth category), with critical-items sub-panel highlighted.

**Deliverable:** Exp 4 results table + Figure 2 (with critical-items panel).

---

### Week 13: Exp 1 analysis

- Run clustering analysis under both embedding models, both grouping schemes (lemma and context), all three ambiguity strata.
- Compute silhouette scores; compare against three baselines (TF-IDF, GloVe, WSD).
- Run UMAP visualizations; produce Figure 1 (side-by-side lemma-colored and context-colored projections, with silhouette violin plot inset and baseline bar chart).
- Run sensitivity analysis across full UMAP hyperparameter grid (supplementary appendix table).

**Deliverable:** Exp 1 results table + Figure 1 + supplementary hyperparameter sensitivity table.

---

### Week 14: Exp 2 analysis [PARALLEL with Exp 5 analysis]

**Primary analysis — schema conformity:**
- For each of 48,000 outputs, compute cosine similarity to the concept's prototype embedding (pre-constructed in Week 2). This is the primary DV.
- Fit mixed-effects model: schema_conformity ~ concept_type (charged vs. control) × temperature + (1|concept).
- Compare LLM schema conformity for charged concepts against corpus baseline (Wikipedia + CC). Model-amplification test: LLM score > corpus score.
- Temperature ablation: piecewise regression to test for conformity ceiling in charged concepts (structural floor prediction).

**Secondary analysis — entropy:**
- Apply BERTopic under three hyperparameter configurations; compute Spearman ρ for rank-order stability.
- Compute Shannon entropy per concept per temperature per configuration.
- Fit mixed-effects model (entropy ~ cultural_charge_rating × temperature + (1|concept)).
- Compute training-corpus entropy baselines; compare against LLM entropy.
- Apply adjudication rule if schema conformity and entropy conflict: schema conformity takes precedence.

**Sub-analysis — demographic markers:** demographic marker extraction for six person-concepts (descriptive only, n=6).

**Produce Figure 3:** Two-panel figure: (a) schema conformity box plots sorted by cultural charge, with corpus baseline overlay; (b) entropy box plots as secondary reference.

**Deliverable:** Exp 2 results table + Figure 3 (two-panel) + demographic sub-analysis table.

---

### Week 14 [PARALLEL]: Exp 5 analysis

- Apply same BERTopic pipeline to diachronic dataset.
- Fit mixed-effects model (entropy ~ generation × concept_type + (1|concept)).
- Produce Figure 4: entropy trajectory lines across model generations (changing vs. stable concepts).

**Deliverable:** Exp 5 results table + Figure 4.

---

### Weeks 15–16: Exp 3 analysis

**Output-side analysis:**
- Validate LLM-as-judge (intra-judge ρ, inter-judge agreement, human correlation ICC(2,1)).
  - Gate: if r < 0.75 with human experts, demote to secondary analysis and flag.
  - Note: GPT-4o must NOT judge its own outputs; use Claude-3.5-Sonnet and Mistral-Large for GPT-4o outputs.
- Fit main mixed-effects model (quality ~ level × domain_complexity + (1|domain) + (1|participant)).
- Fit piecewise regression (linear vs. threshold model; AIC comparison).
- Run $2^4$ factorial regression for three full-factorial domains. Pre-registered primary predictors: V and C.

**Production-side analysis (NEW — inequality claim):**
- Compute effort-to-quality ratio = (time_on_task + n_edits + self_reported_difficulty) / output_quality for each user-study participant.
  - Normalize each component to [0,1] before summing. Pre-register normalization method.
- Compute partial correlation: effort_to_quality_ratio ~ education_level, controlling for domain_familiarity.
- Pre-registered prediction: negative partial correlation (lower education = more effort per unit quality).
- Moderation by domain_complexity tested as exploratory analysis.

**Produce Figure 5:** 10-panel quality curves + user-study scatter plot (effort-to-quality ratio × education level). Table 3: judge calibration; Table 4: multi-model replication summary.

**Deliverable:** Exp 3 full results + Figure 5 + Tables 3–4 + user-study results (including effort-to-quality ratio analysis).

---

### Week 17: Cross-experiment integration

- Assemble Table 4 (multi-model replication): identify which findings replicate across all three LLMs.
  - Claims restricted to replicating patterns; model-specific findings reported separately.
- Write Discussion: Theory-Measurement Bridge revisit; model dependency analysis; social-semiotic implications; limitations.
- Write revised Experimental Design section using `experiment_section_v1.tex` as base.

---

## Phase 4 — Writing and Submission (Weeks 18–22)

| Week | Task |
|------|------|
| 18 | Integrate results into full manuscript; insert figures and tables |
| 19 | First complete draft circulated to co-author(s)/trusted reviewer |
| 20 | Revisions based on internal review |
| 21 | Formatting, reference check, supplementary materials preparation |
| 22 | Submit to target journal |

---

## Resource Requirements

### LLM APIs

| Provider | Model | Use | Estimated calls | Estimated cost |
|----------|-------|-----|-----------------|----------------|
| OpenAI | GPT-4o (`gpt-4o-2024-08-06`) | Generator (Exp 1–5); LLM judge (Exp 3); embedder | ~60,000 generation + ~44,000 judge + embeddings | ~$400–600 |
| OpenAI | text-embedding-3-large | Embedder (Exp 1) | ~18,000 | ~$5 |
| OpenAI | GPT-3 / GPT-3.5 legacy | Diachronic study (Exp 5) | ~2,000 | ~$10–20 |
| Together AI or Replicate | Llama-3-70B-Instruct | Generator (replication) | ~25,000 | ~$30–60 |
| Mistral AI | Mistral-Large-2 | Generator (replication) | ~25,000 | ~$40–80 |
| Hugging Face | e5-large-v2 | Embedder (Exp 1, cross-validation) | ~18,000 | Free (self-hosted) or ~$5 via API |

**Estimated total API budget: €500–800**
*(Significant savings possible via OpenAI Batch API, off-peak usage, and caching repeated prompts)*

---

### Human Expert Requirements

| Role | N needed | Task | Time commitment | When needed |
|------|----------|------|-----------------|-------------|
| Linguistic annotators (lexical semantics) | 5 | Annotate Exp 4 minimal pairs + verify critical-item corpus classifications | ~2.5 hours each | Week 2 |
| Cultural-charge raters | 5 | Rate 40 concepts on stereotypy scale (Exp 2) | ~1 hour each | Week 2 |
| **Schema prototype raters (NEW)** | **30** | **Feature-elicitation task for 40 concepts (Exp 2 primary outcome)** | **~1.5 hours each** | **Week 2** |
| Domain experts — Law (landlord-tenant) | 2 | Construct + verify ground-truth response (Exp 3) | ~3–4 hours each | Week 2–3 |
| Domain experts — Medicine | 2 | Same | ~3–4 hours each | Week 2–3 |
| Domain experts — Pharmacology / adverse events | 2 | Same | ~3–4 hours each | Week 2–3 |
| Domain experts — Finance | 2 | Same | ~3–4 hours each | Week 2–3 |
| Domain experts — Regulatory compliance | 2 | Same | ~3–4 hours each | Week 2–3 |
| Domain complexity raters | 5 | Rate 10 domains on technicality scale | ~1 hour | Week 3 |
| Human expert evaluators (Exp 3 judge validation) | 2 per domain = 10 | Score 200-response stratified subsample against rubric | ~4–6 hours each | Week 15–16 |
| Response coders (Exp 3 user study) | 2 | Code 120 participant prompts for D/R/V/C components | ~6 hours each | Week 11 |

**Total expert-hours: approximately 160–190 hours across 50–60 distinct experts** (increase driven by 30-person schema prototype panel).

The schema prototype raters (n=30) do not require specialist expertise — they should be culturally representative adults (not academics). Recruit via Prolific Academic simultaneously with the user study or via university participant pools. Task takes ~1.5 hours and should be compensated at standard Prolific rates (~£12/hr). **Total prototype rater cost: ~£540–600.**

All other roles as before: PhD students in relevant fields (annotation tasks), academic colleagues in law/medicine/finance faculties (domain expert tasks).

---

### User Study Recruitment

| Item | Specification |
|------|---------------|
| Platform | Prolific Academic |
| N | 120 participants |
| Stratification | 4 education levels × 2 domain-familiarity levels × 3 task domains |
| Task duration | ~15–20 minutes |
| Compensation | £8/hour (≈ £2.20–2.70 per participant) |
| **Total participant cost** | **≈ £280–330** (plus Prolific platform fee ~30%) = **≈ £370–430** |
| Ethical approval | Required before recruitment; submit to IRB/Ethics Committee at end of Phase 1 |

---

### Software Stack

All software is open-source unless noted.

| Category | Tool | Version | Purpose |
|----------|------|---------|---------|
| Language | Python | 3.11+ | All analysis |
| LLM API clients | `openai` | ≥1.30 | GPT-4o generation and embedding |
| | `mistralai` | latest | Mistral generation |
| | `together` or `replicate` | latest | Llama generation |
| Embeddings | `sentence-transformers` | ≥2.6 | e5-large-v2 embedding |
| Dimensionality reduction | `umap-learn` | ≥0.5 | UMAP projections (Exp 1) |
| Topic modeling | `bertopic` | ≥0.16 | Frame extraction (Exp 2, 5) |
| Clustering / evaluation | `scikit-learn` | ≥1.4 | Silhouette scores, k-means |
| Statistical modeling | `statsmodels` | ≥0.14 | Mixed-effects models, piecewise regression |
| | `pingouin` | ≥0.5 | ICC, Cohen's κ, inter-rater statistics |
| | `scipy` | ≥1.11 | Spearman correlation, McNemar's test |
| NLP utilities | `spacy` | ≥3.7 | Tokenization, NER for demographic analysis |
| | `nltk` | ≥3.8 | Flesch-Kincaid, TTR computation (Exp 3 user study) |
| WSD baseline | `EWISER` or `AMuSE-WSD` | — | Word sense disambiguation baseline (Exp 4) |
| Visualization | `matplotlib` | ≥3.8 | Figures |
| | `seaborn` | ≥0.13 | Violin plots, box plots |
| Data management | `pandas` | ≥2.0 | Data frames |
| | `numpy` | ≥1.26 | Vector operations |
| Versioning | `git` + GitHub/OSF | — | Code and data versioning |
| Pre-registration | OSF (osf.io) | — | Pre-registration document |
| Logging | Custom Python logging to JSON | — | Full API call log |
| Environment | `conda` or `venv` + `requirements.txt` | — | Reproducibility |

**Note on compute:** All analyses run on a standard laptop (16GB RAM) except BERTopic on 48,000 documents (Exp 2), which benefits from a GPU or a cloud compute instance (Google Colab Pro or equivalent, ~$10–20 for the run).

---

## Summary Budget

| Item | Estimated cost |
|------|---------------|
| LLM API (generation + judge + embeddings) | €500–800 |
| User study participants (Prolific, Exp 3) | €430–500 |
| Schema prototype raters (Prolific, Exp 2 NEW) | €540–600 |
| Expert annotation / evaluation (if contracted) | €300–600 (or €0 if via academic exchange) |
| Compute (cloud GPU for BERTopic, Exp 2) | €15–30 |
| **Total** | **€1,800–2,530** |

---

## Critical Path

The following items are on the critical path — delays here delay everything:

1. **Expert recruitment (Week 2):** Domain experts for ground-truth construction are the hardest to recruit. Contact them in Week 1, not Week 2.
2. **Pre-registration submission (Week 4):** Data collection cannot begin before this is filed. Do not compress Phase 1 to speed up collection.
3. **Exp 3 judge validation (Week 15–16):** If the LLM judge does not meet the r ≥ 0.75 threshold with human experts, the analysis plan changes substantially. Build a contingency: have human-expert scoring cover all four prompt levels for at least two domains as a fallback primary analysis.
4. **Legacy model access for Exp 5:** Confirm GPT-3 and GPT-3.5 API availability in Week 1. If legacy endpoints are deprecated, identify archival alternatives immediately (e.g., text outputs from published studies, or substitute open-source models with known training cutoffs such as Mistral-7B-v0.1).
