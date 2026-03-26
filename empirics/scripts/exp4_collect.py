"""
exp4_collect.py — Data collection for Experiment E4.
Dictionary–Encyclopedia Diagnostic.

Outputs (saved to data/exp4/):
  responses_T0.jsonl   — all responses at T=0.0 (deterministic)
  responses_T07.jsonl  — all responses at T=0.7 (distributional)
  api_*.jsonl          — per-model API call logs
  gpt2xl_responses.jsonl — co-occurrence baseline (GPT-2 XL, greedy)

Usage:
  python exp4_collect.py --stimuli ../stimuli/exp4_minimal_pairs.csv
  python exp4_collect.py --stimuli ../stimuli/exp4_minimal_pairs.csv --dry-run
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from utils import (
    DATA_DIR, PROMPTS_FILE, get_logger, load_prompts,
    generate_all_models, save_jsonl, exp_data_dir, MODELS
)

logger = get_logger("exp4_collect")
OUT = exp_data_dir("exp4")
SYSTEM = "You are a helpful assistant. Respond in English only."
K = 20          # responses per prompt per model
TEMPERATURES = [0.0, 0.7]


def collect_llm_responses(df: pd.DataFrame, prompts: dict, dry_run: bool) -> None:
    template = prompts["exp4"]["template"]

    for temp in TEMPERATURES:
        tag = "T0" if temp == 0.0 else "T07"
        out_path = OUT / f"responses_{tag}.jsonl"

        if out_path.exists():
            logger.info(f"  {out_path.name} already exists — skipping")
            continue

        records = []
        for _, row in df.iterrows():
            for sentence_col, sense_col in [("sentence_a", "sense_a"),
                                             ("sentence_b", "sense_b")]:
                user_prompt = template.format(
                    target=row["target_word"],
                    sentence=row[sentence_col],
                )
                meta = {
                    "item_id":         row["item_id"],
                    "target_word":     row["target_word"],
                    "sentence_col":    sentence_col,
                    "ground_truth":    row[sense_col],
                    "inferential_depth": row["inferential_depth"],
                    "is_critical_item": row["is_critical_item"],
                    "temperature":     temp,
                    "prompt":          user_prompt,
                }

                if dry_run:
                    logger.info(f"  [DRY RUN] {row['item_id']} / {sentence_col} / T={temp}")
                    continue

                responses = generate_all_models(
                    SYSTEM, user_prompt, temperature=temp, n=K, log_dir=OUT
                )
                for model_key, texts in responses.items():
                    for i, text in enumerate(texts):
                        records.append({**meta, "model": model_key,
                                        "response_index": i, "response": text})

                time.sleep(0.5)

        if not dry_run:
            save_jsonl(records, out_path)
            logger.info(f"  Saved {len(records)} records → {out_path}")


def collect_gpt2xl_baseline(df: pd.DataFrame, prompts: dict, dry_run: bool) -> None:
    """GPT-2 XL greedy decoding: pure co-occurrence baseline."""
    out_path = OUT / "gpt2xl_responses.jsonl"
    if out_path.exists():
        logger.info(f"  {out_path.name} already exists — skipping")
        return

    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    logger.info("Loading GPT-2 XL...")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2-xl",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    template = prompts["exp4"]["template"]
    records = []

    for _, row in df.iterrows():
        for sentence_col, sense_col in [("sentence_a", "sense_a"),
                                         ("sentence_b", "sense_b")]:
            user_prompt = template.format(
                target=row["target_word"],
                sentence=row[sentence_col],
            )
            if dry_run:
                logger.info(f"  [DRY RUN] GPT2XL {row['item_id']} / {sentence_col}")
                continue

            inputs = tokenizer(user_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,   # greedy = pure co-occurrence
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            records.append({
                "item_id":          row["item_id"],
                "target_word":      row["target_word"],
                "sentence_col":     sentence_col,
                "ground_truth":     row[sense_col],
                "inferential_depth": row["inferential_depth"],
                "is_critical_item": row["is_critical_item"],
                "model":            "gpt2-xl",
                "temperature":      0.0,
                "response":         completion,
            })

    if not dry_run:
        save_jsonl(records, out_path)
        logger.info(f"  Saved {len(records)} GPT-2 XL records → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="../stimuli/exp4_minimal_pairs.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.stimuli, comment="#")
    logger.info(f"Loaded {len(df)} stimulus pairs from {args.stimuli}")

    prompts = load_prompts()

    logger.info("=== E4: LLM responses ===")
    collect_llm_responses(df, prompts, args.dry_run)

    logger.info("=== E4: GPT-2 XL co-occurrence baseline ===")
    collect_gpt2xl_baseline(df, prompts, args.dry_run)

    logger.info("E4 collection complete.")


if __name__ == "__main__":
    main()
