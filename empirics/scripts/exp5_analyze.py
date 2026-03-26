"""
exp5_analyze.py — Analysis for Experiment E5.
Diachronic trajectory of schema conformity across model generations.
Pre-registered test: generation × concept_type interaction.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from utils import exp_data_dir, load_jsonl, load_numpy, get_logger, RESULTS_DIR

logger = get_logger("exp5_analyze")
DATA = exp_data_dir("exp5")
DATA_E2 = exp_data_dir("exp2")  # reuse E2 prototype embeddings
OUT  = RESULTS_DIR / "exp5"
OUT.mkdir(exist_ok=True)

GENERATION_ORDER = {"gen1": 1, "gen2": 2, "gen3": 3, "gen4": 4}


def compute_conformity(embed_key: str = "e5") -> pd.DataFrame:
    proto_path = DATA_E2 / f"prototypes_{embed_key}.npy"
    if not proto_path.exists():
        raise FileNotFoundError(f"Prototype embeddings not found: {proto_path}")

    # Load prototypes — build a concept → vector mapping
    # Assume a concept_order.json was saved during prototype construction
    import json
    order_path = DATA_E2 / "concept_order.json"
    if not order_path.exists():
        raise FileNotFoundError("concept_order.json not found in data/exp2/")
    concept_order = json.loads(order_path.read_text())  # list of concept names
    prototypes = load_numpy(proto_path)
    proto_map = {c: prototypes[i] for i, c in enumerate(concept_order)}

    from utils import embed_e5, embed_openai
    records = []

    for gen_key in GENERATION_ORDER:
        resp_path = DATA / f"responses_{gen_key}.jsonl"
        if not resp_path.exists():
            logger.warning(f"  {resp_path.name} not found — skipping gen {gen_key}")
            continue

        resp_records = load_jsonl(resp_path)
        texts  = [r["response"] for r in resp_records]

        # Embed all texts
        vecs = embed_e5(texts) if embed_key == "e5" else embed_openai(texts)

        for i, (rec, vec) in enumerate(zip(resp_records, vecs)):
            concept = rec["concept"]
            if concept not in proto_map:
                continue
            proto = proto_map[concept]
            sim = 1.0 - cosine(vec, proto)
            records.append({
                "concept":       concept,
                "concept_type":  rec["concept_type"],
                "generation":    gen_key,
                "generation_ord": GENERATION_ORDER[gen_key],
                "model":         rec["model"],
                "conformity":    float(sim),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT / "conformity_diachronic.csv", index=False)
    logger.info(f"Diachronic conformity: {len(df)} records → {OUT / 'conformity_diachronic.csv'}")
    return df


def fit_mixed_effects(df: pd.DataFrame) -> None:
    import statsmodels.formula.api as smf

    df["concept_type_bin"] = (df["concept_type"] == "changing").astype(int)
    md = smf.mixedlm(
        "conformity ~ generation_ord * concept_type_bin",
        df, groups=df["concept"],
    )
    result = md.fit(reml=False)
    summary = result.summary().as_text()
    out_path = OUT / "mixed_effects_diachronic.txt"
    out_path.write_text(summary)
    logger.info(f"Mixed-effects results → {out_path}")
    print(summary)


def plot_trajectories(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_agg = (df.groupby(["concept", "concept_type", "generation_ord"])["conformity"]
              .mean().reset_index())

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {"changing": "#d62728", "stable": "#1f77b4"}

    for concept, grp in df_agg.groupby("concept"):
        ct = grp["concept_type"].iloc[0]
        ax.plot(grp["generation_ord"], grp["conformity"],
                color=palette[ct], alpha=0.4, linewidth=1)

    # Bold mean lines
    for ct in ["changing", "stable"]:
        mean_line = (df_agg[df_agg["concept_type"] == ct]
                     .groupby("generation_ord")["conformity"].mean())
        ax.plot(mean_line.index, mean_line.values,
                color=palette[ct], linewidth=2.5,
                label=f"{ct.capitalize()} (mean)")

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["GPT-3\n(2020)", "GPT-3.5\n(2021)", "GPT-4\n(2023)", "GPT-4o\n(2024)"])
    ax.set_ylabel("Schema conformity (cosine similarity to prototype)")
    ax.set_title("Figure 4: Diachronic overcoding — schema conformity across model generations")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "figure4_diachronic.pdf", bbox_inches="tight")
    logger.info(f"Figure 4 → {OUT / 'figure4_diachronic.pdf'}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-key", default="e5", choices=["e5", "openai"])
    args = parser.parse_args()

    df = compute_conformity(args.embed_key)
    fit_mixed_effects(df)
    plot_trajectories(df)
    logger.info("E5 analysis complete.")


if __name__ == "__main__":
    main()
