"""Day 1 smoke test — run as: python smoke_test.py"""
import os, sys

results = {}

# --- OpenAI chat models ---
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env

for model in ['gpt-4o-2024-08-06', 'gpt-3.5-turbo-0125', 'gpt-4-0613']:
    try:
        r = client.chat.completions.create(
            model=model, messages=[{'role': 'user', 'content': 'Reply OK'}], max_tokens=5)
        results[model] = 'OK'
    except Exception as e:
        results[model] = f'FAIL: {e}'

# --- Completions API (gen1 fallback) ---
try:
    r = client.completions.create(model='davinci-002', prompt='Reply OK', max_tokens=5)
    results['davinci-002 (gen1 fallback)'] = 'OK'
except Exception as e:
    results['davinci-002 (gen1 fallback)'] = f'FAIL: {e}'

# --- Embeddings ---
try:
    e = client.embeddings.create(model='text-embedding-3-large', input=['test'])
    results['text-embedding-3-large'] = f'OK dim={len(e.data[0].embedding)}'
except Exception as e:
    results['text-embedding-3-large'] = f'FAIL: {e}'

# --- wordfreq ---
try:
    from wordfreq import zipf_frequency
    results['wordfreq'] = f'OK (nurse zipf={zipf_frequency("nurse","en"):.2f})'
except Exception as e:
    results['wordfreq'] = f'FAIL: {e}'

# --- NLTK WordNet ---
try:
    from nltk.corpus import wordnet as wn
    results['nltk wordnet'] = f'OK ({len(wn.synsets("bank"))} synsets for bank)'
except Exception as e:
    results['nltk wordnet'] = f'FAIL: {e}'

# --- NLTK SemCor ---
try:
    from nltk.corpus import semcor
    list(semcor.tagged_sents(tag='sem'))[:1]
    results['nltk semcor'] = 'OK'
except Exception as e:
    results['nltk semcor'] = f'FAIL: {e}'

# --- Wikipedia REST API ---
try:
    import requests
    headers = {'User-Agent': 'IndustrializationOfPlausibility/1.0 (research; contact: marco.guerzoni@unimib.it)'}
    r = requests.get('https://en.wikipedia.org/w/api.php',
        params={'action':'query','prop':'extracts','exintro':True,'explaintext':True,
                'titles':'Nurse','format':'json','redirects':1},
        headers=headers, timeout=10)
    page = next(iter(r.json()['query']['pages'].values()))
    n = len(page.get('extract',''))
    results['wikipedia api'] = f'OK ({n} chars for Nurse)'
except Exception as e:
    results['wikipedia api'] = f'FAIL: {e}'

# --- Hugging Face datasets ---
try:
    import datasets
    results['huggingface datasets'] = f'OK (v{datasets.__version__})'
except Exception as e:
    results['huggingface datasets'] = f'FAIL: {e}'

# --- Print report ---
print()
print('=' * 64)
print('  DAY 1 SMOKE TEST RESULTS  —  2026-03-26')
print('=' * 64)
ok = fail = 0
for k, v in results.items():
    s = 'PASS' if v.startswith('OK') else 'FAIL'
    ok += (s == 'PASS'); fail += (s == 'FAIL')
    print(f'  {k:<38s}  [{s}]  {v}')
print('-' * 64)
print(f'  {ok}/{ok+fail} checks passed')
print('=' * 64)

# Write to file for record
with open('smoke_test_results.txt', 'w') as f:
    for k, v in results.items():
        s = 'PASS' if v.startswith('OK') else 'FAIL'
        f.write(f'{k}: [{s}] {v}\n')
print(f'\nResults saved to smoke_test_results.txt')
sys.exit(0 if fail == 0 else 1)
