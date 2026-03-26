"""
exp2_analyze.py — Analysis for Experiment E2.
Primary outcome: schema-conformity score.
Secondary outcome: Shannon entropy via BERTopic.

Outputs (saved to results/exp2/):
  schema_conformity.csv        — per-concept per-model per-temperature scores
  entropy_bertopic.csv         — BERTopic entropy across 3 configurations
  mixed_effects_results.txt    — statsmodels MixedLM output
  figure3_conformity.pdf       — Figure 3 (two-panel)
  demographic_markers.csv      — person-concept demographic analysis
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from utils import exp_data_dir, load_jsonl, load_numpy, get_logger, RESULTS_DIR

logger = get_logger("exp2_analyze")
DATA = exp_data_dir("exp2")
OUT  = RESULTS_DIR / "exp2"
OUT.mkdir(exist_ok=True)


# ── Schema conformity (PRIMARY) ───────────────────────────────────────────────

def compute_schema_conformity(stimuli_path: str, embed_key: str = "e5") -> pd.DataFrame:
    """
    For each output, compute cosine similarity to the concept's prototype embedding.
    Returns a DataFrame with columns:
    concept_id, concept, concept_type, model, temperature, response_index, conformity
    """
    df_stim = pd.read_csv(stimuli_path, comment="#")
    proto_path = DATA / f"prototypes_{embed_key}.npy"

    if not proto_path.exists():
        raise FileNotFoundError(
            f"Prototype embeddings not found at {proto_path}. "
            "Run prototype construction first (Week 2 of protocol)."
        )

    # Load prototypes: shape (n_concepts, embed_dim)
    # Assume order matches df_stim row order
    prototypes = load_numpy(proto_path)

    records = []
    for response_file in sorted(DATA.glob(f"responses_*_{embed_key}.npy")):
        # Filename: embeddings_{model}_{temp}_{embed_key}.npy
        # Parse model and temperature from filename
        parts = response_file.stem.split("_")
        # Find corresponding response JSONL
        jsonl_stem = "_".join(parts[1:-1])  # strip "embeddings_" and "_e5"
        jsonl_path = DATA / f"responses_{jsonl_stem}.jsonl"
        if not jsonl_path.exists():
            continue

        resp_records = load_jsonl(jsonl_path)
        embeddings = load_numpy(response_file)  # shape (n_records, embed_dim)

        for i, (rec, emb) in enumerate(zip(resp_records, embeddings)):
            # Find prototype for this concept
            concept_idx = df_stim[df_stim["concept_id"] == rec["concept_id"]].index
            if len(concept_idx) == 0:
                continue
            proto = prototypes[concept_idx[0]]
            similarity = 1.0 - cosine(emb, proto)  # cosine similarity

            records.append({
                "concept_id":   rec["concept_id"],
                "concept":      rec["concept"],
                "concept_type": rec["concept_type"],
                "model":        rec["model"],
                "temperature":  rec["temperature"],
                "response_index": rec.get("response_index", i),
                "conformity":   float(similarity),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT / "schema_conformity_raw.csv", index=False)
    logger.info(f"Schema conformity: {len(df)} records saved.")
    return df


def fit_mixed_effects(df: pd.DataFrame) -> None:
    """
    Mixed-effects model:
    conformity ~ concept_type * temperature + (1|concept)
    """
    import statsmodels.formula.api as smf

    df["concept_type_bin"] = (df["concept_type"] == "charged").astype(int)
    md = smf.mixedlm(
        "conformity ~ concept_type_bin * temperature",
        df,
        groups=df["concept"],
    )
    result = md.fit(reml=False)
    summary = result.summary().as_text()

    out_path = OUT / "mixed_effects_conformity.txt"
    out_path.write_text(summary)
    logger.info(f"Mixed-effects results → {out_path}")
    print(summary)


# ── BERTopic entropy (SECONDARY) ──────────────────────────────────────────────

BERTOPIC_CONFIGS = {
    "conservative": {"min_topic_size": 15},
    "default":      {"min_topic_size": 10},
    "fine_grained": {"min_topic_size": 5},
}


def compute_entropy(probs: list[float]) -> float:
    import math
    return -sum(p * math.log2(p) for p in probs if p > 0)


def run_bertopic_analysis(df_conf: pd.DataFrame) -> pd.DataFrame:
    """
    Apply BERTopic to each concept × model × temperature slice.
    Return DataFrame with entropy per slice per config.
    Check stability: Spearman ρ ≥ 0.85 across configs.
    """
    from bertopic import BERTopic

    entropy_records = []

    for response_file in sorted(DATA.glob("responses_*_T08.jsonl")):
        # Only run on T=0.8 for main entropy comparison
        resp_records = load_jsonl(response_file)
        df_resp = pd.DataFrame(resp_records)

        for concept_id, grp in df_resp.groupby("concept_id"):
            texts = grp["response"].tolist()
            entropies_by_config = {}

            for config_name, kwargs in BERTOPIC_CONFIGS.items():
                try:
                    topic_model = BERTopic(**kwargs, verbose=False)
                    topics, _ = topic_model.fit_transform(texts)
                    topic_counts = pd.Series(topics).value_counts(normalize=True)
                    # Exclude outlier topic (-1)
                    topic_probs = topic_counts[topic_counts.index != -1].tolist()
                    if not topic_probs:
                        entropy = 0.0
                    else:
                        entropy = compute_entropy(topic_probs)
                except Exception as e:
                    logger.warning(f"BERTopic failed for {concept_id}/{config_name}: {e}")
                    entropy = np.nan

                entropies_by_config[config_name] = entropy
                entropy_records.append({
                    "concept_id": concept_id,
                    "concept":    grp["concept"].iloc[0],
                    "concept_type": grp["concept_type"].iloc[0],
                    "model":      grp["model"].iloc[0],
                    "temperature": 0.8,
                    "config":     config_name,
                    "entropy":    entropy,
                })

    df_entropy = pd.DataFrame(entropy_records)

    # Stability check: Spearman ρ across configs per model
    df_pivot = df_entropy.pivot_table(
        index=["concept_id", "model"],
        columns="config",
        values="entropy",
    ).dropna()

    for c1, c2 in [("conservative", "default"), ("default", "fine_grained"),
                   ("conservative", "fine_grained")]:
        rho, p = spearmanr(df_pivot[c1], df_pivot[c2])
        logger.info(f"Spearman ρ ({c1} vs {c2}) = {rho:.3f} (p={p:.4f})")
        if rho < 0.85:
            logger.warning(f"  ρ < 0.85 — reporting entropy ranges, not point estimates.")

    df_entropy.to_csv(OUT / "entropy_bertopic.csv", index=False)
    logger.info(f"BERTopic entropy → {OUT / 'entropy_bertopic.csv'}")
    return df_entropy


# ── Figure 3 ──────────────────────────────────────────────────────────────────

def plot_figure3(df_conf: pd.DataFrame, df_entropy: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: schema conformity
    df_agg = (df_conf[df_conf["temperature"] == 0.8]
              .groupby(["concept", "concept_type"])["conformity"]
              .mean().reset_index()
              .sort_values("conformity", ascending=False))

    sns.boxplot(
        data=df_conf[df_conf["temperature"] == 0.8],
        x="concept", y="conformity", hue="concept_type",
        order=df_agg["concept"].tolist(),
        ax=axes[0], palette={"charged": "#d62728", "control": "#1f77b4"},
    )
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=7)
    axes[0].set_title("(a) Schema conformity (T=0.8)", fontsize=11)
    axes[0].set_ylabel("Cosine similarity to prototype")
    axes[0].set_xlabel("")

    # Panel B: entropy (secondary)
    df_ent_agg = (df_entropy[df_entropy["config"] == "default"]
                  .groupby(["concept", "concept_type"])["entropy"]
                  .mean().reset_index()
                  .sort_values("entropy"))

    sns.boxplot(
        data=df_entropy[df_entropy["config"] == "default"],
        x="concept", y="entropy", hue="concept_type",
        order=df_ent_agg["concept"].tolist(),
        ax=axes[1], palette={"charged": "#d62728", "control": "#1f77b4"},
    )
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=7)
    axes[1].set_title("(b) Shannon entropy over topics (T=0.8) — secondary", fontsize=11)
    axes[1].set_ylabel("Shannon entropy (bits)")
    axes[1].set_xlabel("")

    plt.tight_layout()
    fig.savefig(OUT / "figure3_conformity_entropy.pdf", bbox_inches="tight")
    logger.info(f"Figure 3 → {OUT / 'figure3_conformity_entropy.pdf'}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="../stimuli/exp2_concepts.csv")
    parser.add_argument("--embed-key", default="e5", choices=["e5", "openai"])
    parser.add_argument("--skip-bertopic", action="store_true")
    args = parser.parse_args()

    logger.info("=== E2 Analysis: schema conformity (primary) ===")
    df_conf = compute_schema_conformity(args.stimuli, args.embed_key)

    logger.info("=== E2 Analysis: mixed-effects model ===")
    fit_mixed_effects(df_conf)

    df_entropy = pd.DataFrame()
    if not args.skip_bertopic:
        logger.info("=== E2 Analysis: BERTopic entropy (secondary) ===")
        df_entropy = run_bertopic_analysis(df_conf)

    if not df_entropy.empty:
        logger.info("=== E2 Analysis: Figure 3 ===")
        plot_figure3(df_conf, df_entropy)

    logger.info("E2 analysis complete.")


if __name__ == "__main__":
    main()
