"""
exp2_collect.py — Data collection for Experiment E2.
Schema Conformity and Overcoding (synchronic).

Outputs (saved to data/exp2/):
  responses_{model}_{temp}.jsonl   — LLM outputs per model per temperature
  embeddings_{model}_{embed}.npy   — sentence embeddings
  corpus_wikipedia.jsonl           — Wikipedia baseline texts
  corpus_cc.jsonl                  — Common Crawl baseline texts
  prototypes_{embed}.npy           — prototype embeddings (must exist before running)
  api_*.jsonl                      — API logs

Usage:
  python exp2_collect.py --stimuli ../stimuli/exp2_concepts.csv
  python exp2_collect.py --stimuli ../stimuli/exp2_concepts.csv --dry-run
  python exp2_collect.py --stimuli ../stimuli/exp2_concepts.csv --skip-corpus
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np

from utils import (
    exp_data_dir, get_logger, load_prompts,
    generate_all_models, embed_both,
    save_jsonl, load_jsonl, save_numpy, MODELS
)

logger = get_logger("exp2_collect")
OUT = exp_data_dir("exp2")
SYSTEM = "You are a helpful assistant. Respond in English only."
K = 100
TEMPERATURES = [0.2, 0.5, 0.8, 1.2]


def collect_llm_responses(df: pd.DataFrame, prompts: dict, dry_run: bool) -> None:
    template = prompts["exp2"]["template"]

    for temp in TEMPERATURES:
        for model_key in MODELS:
            tag = f"{model_key}_T{str(temp).replace('.','')}"
            out_path = OUT / f"responses_{tag}.jsonl"
            if out_path.exists():
                logger.info(f"  {out_path.name} exists — skipping")
                continue

            records = []
            for _, row in df.iterrows():
                user_prompt = template.format(concept=row["concept"])
                meta = {
                    "concept_id":   row["concept_id"],
                    "concept":      row["concept"],
                    "concept_type": row["concept_type"],
                    "temperature":  temp,
                    "model":        model_key,
                    "prompt":       user_prompt,
                }

                if dry_run:
                    logger.info(f"  [DRY RUN] {row['concept_id']} T={temp} model={model_key}")
                    continue

                from utils import (get_openai_client, generate_openai,
                                   get_together_client, generate_together,
                                   get_mistral_client, generate_mistral)

                if model_key == "gpt4o":
                    texts = generate_openai(get_openai_client(), MODELS["gpt4o"],
                                            SYSTEM, user_prompt, temp, K,
                                            log_path=OUT / "api_gpt4o.jsonl")
                elif model_key == "llama":
                    texts = generate_together(get_together_client(), MODELS["llama"],
                                              SYSTEM, user_prompt, temp, K,
                                              log_path=OUT / "api_llama.jsonl")
                else:
                    texts = generate_mistral(get_mistral_client(), MODELS["mistral"],
                                             SYSTEM, user_prompt, temp, K,
                                             log_path=OUT / "api_mistral.jsonl")

                for i, text in enumerate(texts):
                    records.append({**meta, "response_index": i, "response": text})

                time.sleep(0.3)

            if not dry_run:
                save_jsonl(records, out_path)
                logger.info(f"  Saved {len(records)} records → {out_path}")


def embed_responses(dry_run: bool) -> None:
    """Compute embeddings for all response files."""
    for response_file in sorted(OUT.glob("responses_*.jsonl")):
        for embed_key in ["openai", "e5"]:
            embed_path = OUT / f"embeddings_{response_file.stem}_{embed_key}.npy"
            if embed_path.exists():
                logger.info(f"  {embed_path.name} exists — skipping")
                continue
            if dry_run:
                logger.info(f"  [DRY RUN] would embed {response_file.name} → {embed_path.name}")
                continue

            records = load_jsonl(response_file)
            texts = [r["response"] for r in records]

            from utils import embed_openai, embed_e5
            if embed_key == "openai":
                vecs = embed_openai(texts)
            else:
                vecs = embed_e5(texts)

            save_numpy(vecs, embed_path)
            logger.info(f"  Saved {vecs.shape} embeddings → {embed_path}")


def collect_corpus_baseline(df: pd.DataFrame, dry_run: bool) -> None:
    """Collect Wikipedia and Common Crawl baseline texts via Wikipedia API."""
    import urllib.request, urllib.parse

    out_wiki = OUT / "corpus_wikipedia.jsonl"
    if out_wiki.exists():
        logger.info("  Wikipedia corpus exists — skipping")
    elif not dry_run:
        records = []
        for _, row in df.iterrows():
            concept = row["concept"]
            url = (
                "https://en.wikipedia.org/w/api.php?"
                + urllib.parse.urlencode({
                    "action": "query", "format": "json",
                    "prop": "extracts", "exintro": True,
                    "titles": concept.replace(" ", "_"), "explaintext": True,
                })
            )
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = resp.read().decode("utf-8")
                import json as _json
                pages = _json.loads(data).get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "")
                    # Take first 50 paragraphs
                    paragraphs = [p.strip() for p in extract.split("\n")
                                  if len(p.strip()) > 80][:50]
                    for i, para in enumerate(paragraphs):
                        records.append({
                            "concept_id": row["concept_id"],
                            "concept": concept,
                            "concept_type": row["concept_type"],
                            "source": "wikipedia",
                            "para_index": i,
                            "text": para,
                        })
            except Exception as e:
                logger.warning(f"  Wikipedia fetch failed for '{concept}': {e}")
            time.sleep(0.5)

        save_jsonl(records, out_wiki)
        logger.info(f"  Saved {len(records)} Wikipedia paragraphs → {out_wiki}")
    else:
        logger.info(f"  [DRY RUN] would fetch Wikipedia for {len(df)} concepts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="../stimuli/exp2_concepts.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-corpus", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.stimuli, comment="#")
    logger.info(f"Loaded {len(df)} concepts ({(df.concept_type=='charged').sum()} charged, "
                f"{(df.concept_type=='control').sum()} controls)")

    prompts = load_prompts()

    logger.info("=== E2: LLM responses ===")
    collect_llm_responses(df, prompts, args.dry_run)

    logger.info("=== E2: Embeddings ===")
    embed_responses(args.dry_run)

    if not args.skip_corpus:
        logger.info("=== E2: Corpus baseline ===")
        collect_corpus_baseline(df, args.dry_run)

    logger.info("E2 collection complete.")


if __name__ == "__main__":
    main()
