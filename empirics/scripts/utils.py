"""
utils.py — Shared utilities for all experiments.
Handles: API clients, structured logging, embedding, output paths.
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PROMPTS_FILE = ROOT / "prompts" / "prompts_all.json"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def exp_data_dir(exp_id: str) -> Path:
    d = DATA_DIR / exp_id
    d.mkdir(exist_ok=True)
    return d


# ── Logging ────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ── Prompt loading ─────────────────────────────────────────────────────────────

def load_prompts() -> dict:
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ── API log ────────────────────────────────────────────────────────────────────

def log_call(log_path: Path, record: dict) -> None:
    """Append a single API call record to a JSONL log file."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_call_id(prompt: str, model: str, temperature: float, index: int) -> str:
    h = hashlib.md5(f"{prompt}{model}{temperature}{index}".encode()).hexdigest()[:8]
    return f"{model.split('/')[-1]}_{h}_{index}"


# ── OpenAI generation ──────────────────────────────────────────────────────────

def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def generate_openai(
    client,
    model: str,
    system: str,
    user: str,
    temperature: float,
    n: int,
    seed: Optional[int] = 42,
    log_path: Optional[Path] = None,
) -> list[str]:
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=temperature,
        n=n,
    )
    if seed is not None:
        kwargs["seed"] = seed

    resp = client.chat.completions.create(**kwargs)
    texts = [c.message.content for c in resp.choices]

    if log_path:
        log_call(log_path, {
            "ts": datetime.utcnow().isoformat(),
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "prompt_hash": hashlib.md5(user.encode()).hexdigest()[:8],
            "n_responses": n,
            "usage": resp.usage.model_dump() if resp.usage else None,
        })
    return texts


# ── Together AI generation (Llama) ─────────────────────────────────────────────

def get_together_client():
    from together import Together
    return Together(api_key=os.environ["TOGETHER_API_KEY"])


def generate_together(
    client,
    model: str,
    system: str,
    user: str,
    temperature: float,
    n: int,
    log_path: Optional[Path] = None,
) -> list[str]:
    texts = []
    for _ in range(n):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=temperature,
        )
        texts.append(resp.choices[0].message.content)
        time.sleep(0.1)

    if log_path:
        log_call(log_path, {
            "ts": datetime.utcnow().isoformat(),
            "model": model,
            "temperature": temperature,
            "prompt_hash": hashlib.md5(user.encode()).hexdigest()[:8],
            "n_responses": n,
        })
    return texts


# ── Mistral generation ─────────────────────────────────────────────────────────

def get_mistral_client():
    from mistralai import Mistral
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def generate_mistral(
    client,
    model: str,
    system: str,
    user: str,
    temperature: float,
    n: int,
    log_path: Optional[Path] = None,
) -> list[str]:
    texts = []
    for _ in range(n):
        resp = client.chat.complete(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=temperature,
        )
        texts.append(resp.choices[0].message.content)
        time.sleep(0.15)

    if log_path:
        log_call(log_path, {
            "ts": datetime.utcnow().isoformat(),
            "model": model,
            "temperature": temperature,
            "prompt_hash": hashlib.md5(user.encode()).hexdigest()[:8],
            "n_responses": n,
        })
    return texts


# ── All-models generation ──────────────────────────────────────────────────────

MODELS = {
    "gpt4o":   os.getenv("GPT4O_MODEL",   "gpt-4o-2024-08-06"),
    "llama":   os.getenv("LLAMA_MODEL",   "meta-llama/Llama-3-70b-chat-hf"),
    "mistral": os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
}


def generate_all_models(
    system: str,
    user: str,
    temperature: float,
    n: int,
    log_dir: Optional[Path] = None,
) -> dict[str, list[str]]:
    """Run the same prompt on all three models. Returns dict model_key → [responses]."""
    oai  = get_openai_client()
    tog  = get_together_client()
    mist = get_mistral_client()

    results = {}
    log = lambda k: (log_dir / f"{k}_api.jsonl") if log_dir else None

    results["gpt4o"]   = generate_openai(oai,  MODELS["gpt4o"],   system, user, temperature, n, log_path=log("gpt4o"))
    results["llama"]   = generate_together(tog,  MODELS["llama"],   system, user, temperature, n, log_path=log("llama"))
    results["mistral"] = generate_mistral(mist, MODELS["mistral"], system, user, temperature, n, log_path=log("mistral"))
    return results


# ── Embeddings ─────────────────────────────────────────────────────────────────

def embed_openai(texts: list[str], model: str = "text-embedding-3-large") -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(input=texts, model=model)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def embed_e5(texts: list[str], model: str = "intfloat/e5-large-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    # e5 requires "passage: " prefix for documents
    prefixed = [f"passage: {t}" for t in texts]
    return m.encode(prefixed, normalize_embeddings=True, show_progress_bar=True)


def embed_both(texts: list[str]) -> dict[str, np.ndarray]:
    return {
        "openai": embed_openai(texts),
        "e5":     embed_e5(texts),
    }


# ── Save / load helpers ────────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_numpy(arr: np.ndarray, path: Path) -> None:
    np.save(path, arr)


def load_numpy(path: Path) -> np.ndarray:
    return np.load(path)
