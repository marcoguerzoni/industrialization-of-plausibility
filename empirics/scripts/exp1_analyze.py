"""
exp1_analyze.py — Analysis for Experiment E1.
UMAP projections + silhouette scores (context vs. lemma groupings).
Runs across 3 models × 2 embedders. Baselines: TF-IDF, GloVe, WSD.
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel

from utils import exp_data_dir, load_jsonl, load_numpy, get_logger, RESULTS_DIR

logger = get_logger("exp1_analyze")
DATA = exp_data_dir("exp1")
OUT  = RESULTS_DIR / "exp1"
OUT.mkdir(exist_ok=True)

# UMAP hyperparameters — fix after grid search on dev set
UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.1, "random_state": 42, "n_components": 2}
N_UMAP_RUNS = 5   # average silhouette over 5 runs for stability


def get_umap_silhouette(embeddings: np.ndarray, labels: np.ndarray,
                        n_runs: int = N_UMAP_RUNS) -> tuple[float, float]:
    """Return mean and std silhouette score over N UMAP runs."""
    import umap
    scores = []
    for seed in range(n_runs):
        params = {**UMAP_PARAMS, "random_state": seed}
        reducer = umap.UMAP(**params)
        projected = reducer.fit_transform(embeddings)
        s = silhouette_score(projected, labels)
        scores.append(s)
    return float(np.mean(scores)), float(np.std(scores))


def run_analysis(embed_key: str, stimuli_path: str) -> pd.DataFrame:
    embed_path = DATA / f"embeddings_{embed_key}.npy"
    resp_path  = DATA / "responses.jsonl"
    if not embed_path.exists() or not resp_path.exists():
        logger.warning(f"  Missing data for embed_key={embed_key} — skipping")
        return pd.DataFrame()

    embeddings = load_numpy(embed_path)
    records    = load_jsonl(resp_path)
    df_stim    = pd.read_csv(stimuli_path, comment="#")

    df_resp = pd.DataFrame(records)
    assert len(df_resp) == len(embeddings), "Mismatch: responses vs embeddings"

    results = []
    for model_key in df_resp["model"].unique():
        mask = df_resp["model"] == model_key
        emb_sub = embeddings[mask]
        df_sub  = df_resp[mask].reset_index(drop=True)

        for stratum in ["high", "medium", "near_monosemous"]:
            s_mask = df_sub["ambiguity_stratum"] == stratum
            if s_mask.sum() < 10:
                continue
            emb_s = emb_sub[s_mask]
            df_s  = df_sub[s_mask].reset_index(drop=True)

            # Lemma grouping
            le_lemma = LabelEncoder()
            labels_lemma = le_lemma.fit_transform(df_s["term"])
            sil_lemma_mean, sil_lemma_std = get_umap_silhouette(emb_s, labels_lemma)

            # Context grouping (semantic domain inferred from prompt_key)
            le_ctx = LabelEncoder()
            labels_ctx = le_ctx.fit_transform(df_s["prompt_key"])
            sil_ctx_mean, sil_ctx_std = get_umap_silhouette(emb_s, labels_ctx)

            # Paraphrase-only sub-analysis
            para_mask = df_s["is_paraphrase"]
            sil_ctx_para = np.nan
            if para_mask.sum() >= 10:
                emb_para = emb_s[para_mask]
                lbl_para = le_ctx.fit_transform(df_s.loc[para_mask, "prompt_key"])
                sil_ctx_para, _ = get_umap_silhouette(emb_para, lbl_para)

            results.append({
                "model":           model_key,
                "embed_key":       embed_key,
                "stratum":         stratum,
                "sil_lemma_mean":  sil_lemma_mean,
                "sil_lemma_std":   sil_lemma_std,
                "sil_ctx_mean":    sil_ctx_mean,
                "sil_ctx_std":     sil_ctx_std,
                "sil_ctx_paraphrase": sil_ctx_para,
                "delta_ctx_lemma": sil_ctx_mean - sil_lemma_mean,
            })

            logger.info(
                f"  {model_key}/{embed_key}/{stratum}: "
                f"lemma={sil_lemma_mean:.3f}±{sil_lemma_std:.3f}  "
                f"context={sil_ctx_mean:.3f}±{sil_ctx_std:.3f}  "
                f"paraphrase={sil_ctx_para:.3f}"
            )

    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="../stimuli/exp1_polysemous_terms.csv")
    args = parser.parse_args()

    all_results = []
    for embed_key in ["openai", "e5"]:
        logger.info(f"=== Embed key: {embed_key} ===")
        df = run_analysis(embed_key, args.stimuli)
        if not df.empty:
            all_results.append(df)

    if all_results:
        df_all = pd.concat(all_results)
        df_all.to_csv(OUT / "silhouette_results.csv", index=False)
        logger.info(f"Results → {OUT / 'silhouette_results.csv'}")

        # Summary: does context > lemma in high/medium strata?
        for stratum in ["high", "medium", "near_monosemous"]:
            sub = df_all[df_all["stratum"] == stratum]
            logger.info(
                f"\nStratum '{stratum}': mean Δ(context-lemma) = "
                f"{sub['delta_ctx_lemma'].mean():.3f} "
                f"(n={len(sub)} model×embed combinations)"
            )

    logger.info("E1 analysis complete.")


if __name__ == "__main__":
    main()
