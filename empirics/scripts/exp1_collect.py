"""
exp1_collect.py — Data collection for Experiment E1.
Encyclopedic organization at distributional scale (UMAP + silhouette).

Usage:
  python exp1_collect.py --stimuli ../stimuli/exp1_polysemous_terms.csv
  python exp1_collect.py --stimuli ../stimuli/exp1_polysemous_terms.csv --dry-run
"""

import argparse, time
import pandas as pd
from utils import (exp_data_dir, get_logger, load_prompts,
                   generate_all_models, embed_both, save_jsonl, load_jsonl,
                   save_numpy, MODELS)

logger = get_logger("exp1_collect")
OUT = exp_data_dir("exp1")
SYSTEM = "You are a helpful assistant. Respond in English only."
K = 20
TEMPERATURE = 0.7
PROMPT_COLS = ["prompt_neutral", "prompt_explicit_a", "prompt_explicit_b",
               "prompt_paraphrase_a", "prompt_paraphrase_b"]
PROMPT_KEYS = ["neutral", "explicit_a", "explicit_b", "paraphrase_a", "paraphrase_b"]


def collect(df: pd.DataFrame, dry_run: bool) -> None:
    out_path = OUT / "responses.jsonl"
    if out_path.exists():
        logger.info(f"  {out_path.name} exists — skipping")
        return

    records = []
    for _, row in df.iterrows():
        for col, key in zip(PROMPT_COLS, PROMPT_KEYS):
            prompt = row.get(col, "")
            if not isinstance(prompt, str) or not prompt.strip():
                continue

            meta = {
                "term_id":          row["term_id"],
                "term":             row["term"],
                "ambiguity_stratum": row["ambiguity_stratum"],
                "prompt_key":       key,
                "is_paraphrase":    key.startswith("paraphrase"),
                "prompt":           prompt,
                "temperature":      TEMPERATURE,
            }

            if dry_run:
                logger.info(f"  [DRY RUN] {row['term_id']} / {key}")
                continue

            responses = generate_all_models(
                SYSTEM, prompt, temperature=TEMPERATURE, n=K, log_dir=OUT
            )
            for model_key, texts in responses.items():
                for i, text in enumerate(texts):
                    records.append({**meta, "model": model_key,
                                    "response_index": i, "response": text})
            time.sleep(0.4)

    if not dry_run:
        save_jsonl(records, out_path)
        logger.info(f"  Saved {len(records)} records → {out_path}")


def embed(dry_run: bool) -> None:
    resp_path = OUT / "responses.jsonl"
    if not resp_path.exists():
        logger.warning("  No responses.jsonl found — run collection first")
        return

    records = load_jsonl(resp_path)
    texts = [r["response"] for r in records]

    for embed_key in ["openai", "e5"]:
        out_path = OUT / f"embeddings_{embed_key}.npy"
        if out_path.exists():
            logger.info(f"  {out_path.name} exists — skipping")
            continue
        if dry_run:
            logger.info(f"  [DRY RUN] would embed {len(texts)} texts with {embed_key}")
            continue

        from utils import embed_openai, embed_e5
        vecs = embed_openai(texts) if embed_key == "openai" else embed_e5(texts)
        save_numpy(vecs, out_path)
        logger.info(f"  Saved {vecs.shape} → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="../stimuli/exp1_polysemous_terms.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.stimuli, comment="#")
    logger.info(f"Loaded {len(df)} terms")

    logger.info("=== E1: LLM responses ===")
    collect(df, args.dry_run)

    logger.info("=== E1: Embeddings ===")
    embed(args.dry_run)

    logger.info("E1 collection complete.")


if __name__ == "__main__":
    main()
