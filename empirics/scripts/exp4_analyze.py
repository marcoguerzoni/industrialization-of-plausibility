"""
exp4_analyze.py — Analysis for Experiment E4.
Primary test: LLMs vs co-occurrence baseline on CRITICAL ITEMS.
Mixed-effects logistic regression: correct ~ system × depth × item_type.
"""

import numpy as np
import pandas as pd
import re

from utils import exp_data_dir, load_jsonl, get_logger, RESULTS_DIR

logger = get_logger("exp4_analyze")
DATA = exp_data_dir("exp4")
OUT  = RESULTS_DIR / "exp4"
OUT.mkdir(exist_ok=True)


def classify_response(response: str, ground_truth: str) -> str:
    """
    Classify response as 'encyclopedic', 'dictionary', or 'neither'.
    Placeholder: replace with trained classifier in production.
    """
    resp_lower = response.lower()
    gt_lower   = ground_truth.lower()
    if gt_lower in resp_lower:
        return "encyclopedic"
    return "neither"


def load_all_responses(stimuli_path: str, temperature: float) -> pd.DataFrame:
    df_stim = pd.read_csv(stimuli_path, comment="#")
    stim_map = {row["item_id"]: row for _, row in df_stim.iterrows()}

    tag = "T0" if temperature == 0.0 else "T07"
    records = []

    # LLM responses
    resp_path = DATA / f"responses_{tag}.jsonl"
    if resp_path.exists():
        for rec in load_jsonl(resp_path):
            label = classify_response(rec["response"], rec["ground_truth"])
            records.append({
                "item_id":          rec["item_id"],
                "sentence_col":     rec["sentence_col"],
                "is_critical_item": rec["is_critical_item"],
                "inferential_depth": rec["inferential_depth"],
                "system":           rec["model"],
                "correct":          int(label == "encyclopedic"),
                "temperature":      temperature,
            })

    # GPT-2 XL baseline
    gpt2_path = DATA / "gpt2xl_responses.jsonl"
    if gpt2_path.exists():
        for rec in load_jsonl(gpt2_path):
            label = classify_response(rec["response"], rec["ground_truth"])
            records.append({
                "item_id":          rec["item_id"],
                "sentence_col":     rec["sentence_col"],
                "is_critical_item": rec["is_critical_item"],
                "inferential_depth": rec["inferential_depth"],
                "system":           "gpt2_xl",
                "correct":          int(label == "encyclopedic"),
                "temperature":      0.0,
            })

    return pd.DataFrame(records)


def fit_logistic_regression(df: pd.DataFrame) -> None:
    """
    Mixed-effects logistic regression:
    correct ~ system * inferential_depth * is_critical_item + (1|item_id) + (1|target_word)
    """
    import statsmodels.formula.api as smf

    df["is_critical"] = df["is_critical_item"].astype(int)
    df["depth_num"]   = df["inferential_depth"].map(
        {"shallow": 1, "medium": 2, "deep": 3}
    ).fillna(2)

    # Restrict to sentence_b (the encyclopedic test condition)
    df_b = df[df["sentence_col"] == "sentence_b"].copy()

    md = smf.glm(
        "correct ~ C(system) * depth_num * is_critical",
        data=df_b,
        family=__import__("statsmodels.genmod.families", fromlist=["Binomial"]).Binomial(),
    )
    result = md.fit()
    summary = result.summary().as_text()
    (OUT / "logistic_regression.txt").write_text(summary)
    logger.info("Logistic regression saved.")
    print(summary)


def summarize_accuracy(df: pd.DataFrame) -> None:
    df_b = df[df["sentence_col"] == "sentence_b"]
    tbl = (df_b.groupby(["system", "inferential_depth", "is_critical_item"])["correct"]
           .agg(["mean", "count"]).reset_index())
    tbl.columns = ["system", "depth", "is_critical", "accuracy", "n"]
    tbl.to_csv(OUT / "accuracy_summary.csv", index=False)
    logger.info(f"\n{tbl.to_string(index=False)}")


def plot_figure2(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_b = df[df["sentence_col"] == "sentence_b"].copy()
    df_b["depth"] = pd.Categorical(df_b["inferential_depth"],
                                   categories=["shallow", "medium", "deep"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, is_crit, title in zip(
        axes,
        [False, True],
        ["Non-critical items", "CRITICAL items\n(encyclopedic ≠ corpus-dominant)"]
    ):
        sub = df_b[df_b["is_critical_item"] == is_crit]
        agg = sub.groupby(["system", "depth"])["correct"].mean().reset_index()
        sns.barplot(data=agg, x="depth", y="correct", hue="system", ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Inferential depth")
        ax.set_ylabel("Accuracy (encyclopedic reading)")
        ax.set_ylim(0, 1)

    plt.suptitle("Figure 2: E4 — Disambiguation accuracy by system × depth × item type")
    plt.tight_layout()
    fig.savefig(OUT / "figure2_disambiguation.pdf", bbox_inches="tight")
    logger.info(f"Figure 2 → {OUT / 'figure2_disambiguation.pdf'}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="../stimuli/exp4_minimal_pairs.csv")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    logger.info("=== E4 Analysis ===")
    df = load_all_responses(args.stimuli, args.temperature)
    logger.info(f"Loaded {len(df)} records")

    summarize_accuracy(df)
    fit_logistic_regression(df)
    plot_figure2(df)
    logger.info("E4 analysis complete.")


if __name__ == "__main__":
    main()
