"""
exp3_analyze.py — Analysis for Experiment E3.
Output side: mixed-effects quality model + component attribution.
Production side: effort-to-quality ratio from user study.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from utils import exp_data_dir, load_jsonl, get_logger, RESULTS_DIR

logger = get_logger("exp3_analyze")
DATA = exp_data_dir("exp3")
OUT  = RESULTS_DIR / "exp3"
OUT.mkdir(exist_ok=True)


# ── Output-side analysis ──────────────────────────────────────────────────────

def load_scores() -> pd.DataFrame:
    path = DATA / "judge_scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Judge scores not found: {path}")
    records = load_jsonl(path)
    df = pd.DataFrame(records)
    # Composite quality score = mean of 3 dimensions
    df["quality"] = df[["factual_accuracy", "practical_actionability",
                         "domain_precision"]].mean(axis=1)
    df["complexity_bin"] = (df["complexity"] == "high").astype(int)
    return df


def fit_main_model(df: pd.DataFrame) -> None:
    """Mixed-effects: quality ~ level * complexity + (1|domain) + (1|model)"""
    import statsmodels.formula.api as smf

    md = smf.mixedlm(
        "quality ~ level * complexity_bin",
        df,
        groups=df["domain"],
        re_formula="~1",
    )
    result = md.fit(reml=False)
    summary = result.summary().as_text()
    (OUT / "mixed_effects_quality.txt").write_text(summary)
    logger.info("Main mixed-effects model saved.")
    print(summary)


def test_monotonicity(df: pd.DataFrame) -> None:
    """Compare linear vs. threshold (step-function) model via AIC."""
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    df_agg = df.groupby(["domain", "level"])["quality"].mean().reset_index()

    linear = smf.ols("quality ~ level", df_agg).fit()
    # Threshold: step at level >= 2 (informed → expert transition)
    df_agg["step"] = (df_agg["level"] >= 2).astype(int)
    threshold = smf.ols("quality ~ step", df_agg).fit()

    logger.info(f"Linear model AIC:    {linear.aic:.2f}")
    logger.info(f"Threshold model AIC: {threshold.aic:.2f}")
    winner = "linear" if linear.aic < threshold.aic else "threshold (step)"
    logger.info(f"Lower AIC → {winner} model preferred")

    with open(OUT / "monotonicity_test.txt", "w") as f:
        f.write(f"Linear AIC:    {linear.aic:.4f}\n")
        f.write(f"Threshold AIC: {threshold.aic:.4f}\n")
        f.write(f"Preferred:     {winner}\n")


def component_attribution(df: pd.DataFrame) -> None:
    """Regression on D/R/V/C binary predictors for focal domains."""
    import statsmodels.formula.api as smf

    focal_domains = ["landlord_tenant", "rental_review", "financial_instrument"]
    df_focal = df[df["domain"].isin(focal_domains)].copy()

    md = smf.ols("quality ~ D + R + V + C + D:R + D:V + D:C + R:V + R:C + V:C",
                 data=df_focal).fit()
    summary = md.summary().as_text()
    (OUT / "component_attribution.txt").write_text(summary)
    logger.info("Component attribution (2^4 factorial) saved.")
    print(summary)

    # Pre-registered prediction: V and C largest coefficients
    coefs = md.params[["V", "C", "D", "R"]].abs().sort_values(ascending=False)
    logger.info(f"Coefficient magnitude rank: {coefs.to_dict()}")


# ── Production-side: effort-to-quality ratio ──────────────────────────────────

def load_user_study() -> pd.DataFrame:
    path = DATA / "user_study_responses.csv"
    if not path.exists():
        logger.warning(f"User study data not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_effort_to_quality(df_users: pd.DataFrame) -> pd.DataFrame:
    """
    effort = normalized(time_on_task) + normalized(n_edits) + normalized(self_reported_difficulty)
    ratio  = effort / output_quality
    """
    for col in ["time_on_task", "n_edits", "self_reported_difficulty", "output_quality"]:
        if col not in df_users.columns:
            raise ValueError(f"User study CSV missing column: {col}")

    df = df_users.copy()
    for col in ["time_on_task", "n_edits", "self_reported_difficulty"]:
        mn, mx = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - mn) / (mx - mn + 1e-9)

    df["effort"] = (df["time_on_task_norm"] +
                    df["n_edits_norm"] +
                    df["self_reported_difficulty_norm"])
    df["effort_to_quality"] = df["effort"] / (df["output_quality"] + 1e-9)
    return df


def fit_inequality_model(df: pd.DataFrame) -> None:
    """
    Partial correlation: effort_to_quality ~ education_level,
    controlling for domain_familiarity.
    Pre-registered prediction: negative partial correlation.
    """
    import pingouin as pg

    result = pg.partial_corr(
        data=df,
        x="education_level",
        y="effort_to_quality",
        covar="domain_familiarity",
        method="pearson",
    )
    logger.info(f"\nPartial correlation (effort_to_quality ~ education | domain_familiarity):")
    logger.info(result.to_string())
    result.to_csv(OUT / "inequality_partial_corr.csv")

    r = result["r"].values[0]
    p = result["p-val"].values[0]
    if r < 0 and p < 0.05:
        logger.info("  → Confirmed: negative partial correlation (p < .05)")
    else:
        logger.info(f"  → Not confirmed: r={r:.3f}, p={p:.4f}")


# ── Figure 5 ──────────────────────────────────────────────────────────────────

def plot_figure5(df: pd.DataFrame, df_users: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_domains = df["domain"].nunique()
    ncols = 5
    nrows = (n_domains + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharey=True)
    axes_flat = axes.flatten()

    domains = sorted(df["domain"].unique())
    for ax, domain in zip(axes_flat, domains):
        sub = df[df["domain"] == domain]
        by_level = sub.groupby("level")["quality"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(by_level["level"], by_level["mean"], yerr=by_level["sem"],
                    marker="o", capsize=4)
        ax.set_title(domain.replace("_", "\n"), fontsize=8)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Prompt level")
        if ax == axes_flat[0]:
            ax.set_ylabel("Mean quality score (1–5)")

    for ax in axes_flat[len(domains):]:
        ax.set_visible(False)

    plt.suptitle("Figure 5: Output quality by prompt level and domain (E3)", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT / "figure5_quality_curves.pdf", bbox_inches="tight")
    logger.info(f"Figure 5 → {OUT / 'figure5_quality_curves.pdf'}")
    plt.close()

    # User study scatter
    if not df_users.empty and "effort_to_quality" in df_users.columns:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.regplot(data=df_users, x="education_level", y="effort_to_quality",
                    ax=ax2, scatter_kws={"alpha": 0.5})
        ax2.set_xlabel("Education level (1=low, 4=high)")
        ax2.set_ylabel("Effort-to-quality ratio")
        ax2.set_title("User study: effort-to-quality ratio × education level")
        fig2.savefig(OUT / "figure5b_effort_quality.pdf", bbox_inches="tight")
        logger.info(f"Figure 5b → {OUT / 'figure5b_effort_quality.pdf'}")
        plt.close()


def main():
    logger.info("=== E3 Analysis: output side ===")
    df = load_scores()
    fit_main_model(df)
    test_monotonicity(df)
    component_attribution(df)

    logger.info("=== E3 Analysis: production side (user study) ===")
    df_users = load_user_study()
    if not df_users.empty:
        df_users = compute_effort_to_quality(df_users)
        df_users.to_csv(OUT / "user_study_processed.csv", index=False)
        fit_inequality_model(df_users)
    else:
        logger.info("  User study data not yet available.")

    logger.info("=== E3 Analysis: Figure 5 ===")
    plot_figure5(df, df_users if not df_users.empty else pd.DataFrame())
    logger.info("E3 analysis complete.")


if __name__ == "__main__":
    main()
