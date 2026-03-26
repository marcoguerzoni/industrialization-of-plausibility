"""
exp5_collect.py — Data collection for Experiment E5.
Diachronic stability of schema conformity across model generations.

Model generations:
  gen1 = GPT-3       (text-davinci-001, ~2020 cutoff)
  gen2 = GPT-3.5     (gpt-3.5-turbo-0125, ~2021 cutoff)
  gen3 = GPT-4       (gpt-4-0613, ~2023 cutoff)
  gen4 = GPT-4o      (gpt-4o-2024-08-06, ~2024 cutoff)

Note: legacy model availability must be verified before running.
If text-davinci-001 is unavailable, document and substitute nearest archival.

Usage:
  python exp5_collect.py --stimuli_e2 ../stimuli/exp2_concepts.csv
  python exp5_collect.py --dry-run
"""

import argparse, time
import pandas as pd
from utils import exp_data_dir, get_logger, load_prompts, save_jsonl, get_openai_client

logger = get_logger("exp5_collect")
OUT = exp_data_dir("exp5")
SYSTEM = "You are a helpful assistant. Respond in English only."
K = 50
TEMPERATURE = 0.8

# Pinned generation → model mapping
# Verified 2026-03-26: text-davinci-001 and text-davinci-003 are deprecated (404).
# gen1 substituted with davinci-002 (nearest accessible GPT-3-era checkpoint;
# completions API; training cutoff ~2021-09 — document in pre-registration as deviation).
# gen2/gen3/gen4 confirmed available.
GENERATION_MODELS = {
    "gen1": "davinci-002",          # Fallback: text-davinci-001 deprecated 2024-01
    "gen2": "gpt-3.5-turbo-0125",
    "gen3": "gpt-4-0613",
    "gen4": "gpt-4o-2024-08-06",
}
# PRE-REGISTRATION NOTE: gen1 cutoff is ~2021-09 (not ~2020 as originally planned).
# This compresses the gen1→gen2 gap. Report as limitation. Diachronic claim remains
# testable across gen2→gen4; gen1 interpreted as "early RLHF-era baseline".

# E5 concept IDs — 10 changing-salience + 10 stable (subset of E2 concepts)
CHANGING_CONCEPTS = [
    "pandemic", "remote work", "artificial intelligence", "vaccine",
    "disinformation", "climate anxiety", "supply chain", "NFT",
    "long COVID", "diversity and inclusion"
]
STABLE_CONCEPTS = []  # populate from E2 stable-charge controls


def collect(concepts: list[dict], prompts: dict, dry_run: bool) -> None:
    template = prompts["exp5"]["template"]

    for gen_key, model_id in GENERATION_MODELS.items():
        out_path = OUT / f"responses_{gen_key}.jsonl"
        if out_path.exists():
            logger.info(f"  {out_path.name} exists — skipping")
            continue

        records = []
        client = get_openai_client()

        for c in concepts:
            concept = c["concept"]
            concept_type = c["concept_type"]
            user_prompt = template.format(concept=concept)

            if dry_run:
                logger.info(f"  [DRY RUN] {gen_key} / {concept}")
                continue

            try:
                # davinci-002 uses the legacy completions endpoint, not chat
                if model_id == "davinci-002":
                    resp = client.completions.create(
                        model=model_id,
                        prompt=f"{SYSTEM}\n\n{user_prompt}",
                        temperature=TEMPERATURE,
                        max_tokens=300,
                        n=K,
                    )
                    choices_text = [c.text for c in resp.choices]
                else:
                    resp = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user",   "content": user_prompt}],
                        temperature=TEMPERATURE,
                        n=K,
                    )
                    choices_text = [c.message.content for c in resp.choices]

                for i, text in enumerate(choices_text):
                    records.append({
                        "concept":       concept,
                        "concept_type":  concept_type,
                        "generation":    gen_key,
                        "model":         model_id,
                        "temperature":   TEMPERATURE,
                        "response_index": i,
                        "response":      text,
                    })
            except Exception as e:
                logger.error(f"  Failed {gen_key}/{concept}: {e}")
            time.sleep(0.5)

        if not dry_run and records:
            save_jsonl(records, out_path)
            logger.info(f"  Saved {len(records)} records → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli_e2", default="../stimuli/exp2_concepts.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.stimuli_e2, comment="#")
    prompts = load_prompts()

    # Build E5 concept list from E2 stimuli + hardcoded changing-salience list
    concepts = []
    for name in CHANGING_CONCEPTS:
        concepts.append({"concept": name, "concept_type": "changing"})
    # Add stable controls from E2 (first 10 control concepts)
    controls = df[df["concept_type"] == "control"].head(10)
    for _, row in controls.iterrows():
        concepts.append({"concept": row["concept"], "concept_type": "stable"})

    logger.info(f"E5 concepts: {len(concepts)} "
                f"({sum(c['concept_type']=='changing' for c in concepts)} changing, "
                f"{sum(c['concept_type']=='stable' for c in concepts)} stable)")

    collect(concepts, prompts, args.dry_run)
    logger.info("E5 collection complete.")


if __name__ == "__main__":
    main()
