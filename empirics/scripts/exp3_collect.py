"""
exp3_collect.py — Data collection and LLM-judge scoring for Experiment E3.
Semiotic cost of context (prompt-factorial + user study interface).

Usage:
  python exp3_collect.py --prompts ../prompts/prompts_all.json --dry-run
"""

import argparse, json, time
from pathlib import Path
import pandas as pd
from utils import (exp_data_dir, get_logger, load_prompts,
                   generate_all_models, save_jsonl, load_jsonl,
                   get_openai_client, generate_openai, MODELS)

logger = get_logger("exp3_collect")
OUT = exp_data_dir("exp3")
SYSTEM = "You are a helpful assistant. Respond in English only."
K = 30
TEMPERATURE = 0.7

DOMAINS = {
    "landlord_tenant":      {"complexity": "high"},
    "medical_test":         {"complexity": "high"},
    "pharma_adverse_event": {"complexity": "high"},
    "financial_instrument": {"complexity": "high"},
    "regulatory_complaint": {"complexity": "high"},
    "rental_review":        {"complexity": "standard"},
    "dietary_advice":       {"complexity": "standard"},
    "cover_letter":         {"complexity": "standard"},
    "insurance_claim":      {"complexity": "standard"},
    "academic_appeal":      {"complexity": "standard"},
}

CANONICAL_LEVELS = ["L0", "L1", "L2", "L3"]
# Full 2^4 factorial only for focal domains
FOCAL_DOMAINS = ["landlord_tenant", "rental_review", "financial_instrument"]
COMPONENTS = ["D", "R", "V", "C"]

def factorial_conditions() -> list[dict]:
    """Generate all 2^4 = 16 factorial conditions."""
    conditions = []
    for i in range(16):
        cond = {c: bool((i >> j) & 1) for j, c in enumerate(COMPONENTS)}
        cond["condition_id"] = f"{''.join(c if cond[c] else '~'+c for c in COMPONENTS)}"
        cond["level"] = sum(cond[c] for c in COMPONENTS)
        conditions.append(cond)
    return conditions


def collect_factorial(prompts_data: dict, dry_run: bool) -> None:
    """Collect LLM responses for all prompt conditions."""
    out_path = OUT / "responses_factorial.jsonl"
    if out_path.exists():
        logger.info(f"  {out_path.name} exists — skipping")
        return

    records = []
    exp3_prompts = prompts_data.get("exp3", {})

    for domain, domain_meta in DOMAINS.items():
        is_focal = domain in FOCAL_DOMAINS
        conditions = factorial_conditions() if is_focal else [
            {"D": l >= 1, "R": l >= 2, "V": l >= 2, "C": l >= 3,
             "condition_id": f"L{l}", "level": l}
            for l in range(4)
        ]

        # Load domain-specific prompt texts
        domain_prompts = exp3_prompts.get("example_domain_" + domain, {})
        if not domain_prompts:
            logger.warning(f"  No prompts found for domain '{domain}' in prompts_all.json — skipping")
            continue

        for cond in conditions:
            # For canonical 4-level design, use L0–L3 keys directly
            level_key = f"L{cond['level']}" if not is_focal else cond["condition_id"]
            user_prompt = domain_prompts.get(f"L{cond['level']}", "")
            if not user_prompt:
                continue

            meta = {
                "domain":       domain,
                "complexity":   domain_meta["complexity"],
                "condition_id": cond["condition_id"],
                "level":        cond["level"],
                "D": cond["D"], "R": cond["R"], "V": cond["V"], "C": cond["C"],
                "is_focal":     is_focal,
                "prompt":       user_prompt,
            }

            if dry_run:
                logger.info(f"  [DRY RUN] {domain} / {cond['condition_id']}")
                continue

            responses = generate_all_models(
                SYSTEM, user_prompt, temperature=TEMPERATURE, n=K, log_dir=OUT
            )
            for model_key, texts in responses.items():
                for i, text in enumerate(texts):
                    records.append({**meta, "model": model_key,
                                    "response_index": i, "response": text})
            time.sleep(0.4)

    if not dry_run:
        save_jsonl(records, out_path)
        logger.info(f"  Saved {len(records)} records → {out_path}")


def run_llm_judge(ground_truth_dir: Path, dry_run: bool) -> None:
    """
    Score each response against the domain ground truth.
    GPT-4o is EXCLUDED from judging its own outputs.
    Claude-3.5-Sonnet and Mistral-Large judge GPT-4o outputs.
    """
    resp_path = OUT / "responses_factorial.jsonl"
    out_path  = OUT / "judge_scores.jsonl"
    if out_path.exists():
        logger.info(f"  {out_path.name} exists — skipping judge scoring")
        return
    if not resp_path.exists():
        logger.warning("  No responses_factorial.jsonl — run collection first")
        return

    records   = load_jsonl(resp_path)
    prompts   = load_prompts()
    judge_tmpl = prompts["exp3_judge"]["template"]

    scored = []
    for rec in records:
        gt_path = ground_truth_dir / f"{rec['domain']}_ground_truth.txt"
        if not gt_path.exists():
            logger.warning(f"  No ground truth for {rec['domain']} — skipping")
            continue
        ground_truth = gt_path.read_text(encoding="utf-8")

        judge_prompt = judge_tmpl.format(
            domain=rec["domain"],
            ground_truth=ground_truth,
            response=rec["response"],
        )

        # Choose judge: avoid self-judging
        if rec["model"] == "gpt4o":
            judge_model = "claude-3-5-sonnet-20241022"  # cross-model judge
        else:
            judge_model = MODELS["gpt4o"]

        if dry_run:
            logger.info(f"  [DRY RUN] judge {rec['domain']} / {rec['condition_id']}")
            continue

        try:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=judge_model if "gpt" in judge_model else "gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            score_json = json.loads(resp.choices[0].message.content)
            scored.append({**rec, **score_json, "judge_model": judge_model})
        except Exception as e:
            logger.error(f"  Judge failed for {rec['domain']}: {e}")
        time.sleep(0.2)

    if not dry_run:
        save_jsonl(scored, out_path)
        logger.info(f"  Saved {len(scored)} scored records → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="../prompts/prompts_all.json")
    parser.add_argument("--ground-truth-dir", default="../stimuli/ground_truth",
                        help="Directory with {domain}_ground_truth.txt files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompts_data = load_prompts()
    gt_dir = Path(args.ground_truth_dir)

    logger.info("=== E3: LLM response collection ===")
    collect_factorial(prompts_data, args.dry_run)

    logger.info("=== E3: LLM-as-judge scoring ===")
    run_llm_judge(gt_dir, args.dry_run)

    logger.info("E3 collection complete.")


if __name__ == "__main__":
    main()
