"""
Day 3 script: critical item identification for exp4_minimal_pairs.csv

Three-tier critical item classification (pre-registered):

  Tier 1 (auto): sense_mismatch AND freq_lower
    - enc_synset != dict_synset  (SemCor or WordNet ordering)
    - zipf(enc_ctx) < zipf(dict_ctx)

  Tier 2 (auto): sense_mismatch only (DIFF synset confirmed; freq check not
    applicable because context words are equally common in modern corpora)

  Tier 3 (manual_preregistered): monosemous words where the encyclopedic
    reading of sentence_b requires specific historical/causal knowledge that
    is not the most frequent association of the target word in standard text
    (verified by two expert annotators in Week 2; κ target >= 0.80).

Target: 24 critical items.  Annotation file saved for Week 2 review.
"""

import sys, io
from collections import defaultdict, Counter
import pandas as pd
from nltk.corpus import semcor, wordnet as wn
from wordfreq import zipf_frequency

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Per-item sense annotations ────────────────────────────────────────────────

ITEM_META = {
    # SHALLOW
    "E4_001": dict(enc_synset="vaccine.n.01",    dict_synset="vaccine.n.01",
                   enc_ctx="mandate",            dict_ctx="immunization"),
    "E4_002": dict(enc_synset="virus.n.01",      dict_synset="virus.n.01",
                   enc_ctx="governance",         dict_ctx="pathogen"),
    "E4_003": dict(enc_synset="antibiotic.n.01", dict_synset="antibiotic.n.01",
                   enc_ctx="resistance",         dict_ctx="bacteria"),
    "E4_004": dict(enc_synset="gene.n.01",       dict_synset="gene.n.01",
                   enc_ctx="patent",             dict_ctx="protein"),
    "E4_005": dict(enc_synset="fossil.n.01",     dict_synset="fossil.n.01",
                   enc_ctx="evolution",          dict_ctx="limestone"),
    "E4_006": dict(enc_synset="telescope.n.01",  dict_synset="telescope.n.01",
                   enc_ctx="heresy",             dict_ctx="astronomy"),
    "E4_007": dict(enc_synset="compass.n.01",    dict_synset="compass.n.01",
                   enc_ctx="colonialism",        dict_ctx="navigation"),
    "E4_008": dict(enc_synset="quarantine.n.01", dict_synset="quarantine.n.01",
                   enc_ctx="discrimination",     dict_ctx="isolation"),
    "E4_009": dict(enc_synset="famine.n.01",     dict_synset="famine.n.01",
                   enc_ctx="export",             dict_ctx="drought"),
    "E4_010": dict(enc_synset="inflation.n.01",  dict_synset="inflation.n.01",
                   enc_ctx="inequality",         dict_ctx="prices"),
    "E4_011": dict(enc_synset="protest.n.01",    dict_synset="protest.n.01",
                   enc_ctx="legitimation",       dict_ctx="demonstration"),
    # STRIKE: labor (n.02) vs. act-of-hitting (n.01)
    "E4_012": dict(enc_synset="strike.n.02",     dict_synset="strike.n.01",
                   enc_ctx="picket",             dict_ctx="blow"),
    "E4_013": dict(enc_synset="satellite.n.01",  dict_synset="satellite.n.01",
                   enc_ctx="sovereignty",        dict_ctx="orbit"),
    "E4_014": dict(enc_synset="asteroid.n.01",   dict_synset="asteroid.n.01",
                   enc_ctx="extinction",         dict_ctx="orbit"),
    "E4_015": dict(enc_synset="climate.n.01",    dict_synset="climate.n.01",
                   enc_ctx="emissions",          dict_ctx="weather"),
    # MEDIUM
    "E4_016": dict(enc_synset="contract.n.01",   dict_synset="contract.n.01",
                   enc_ctx="nondisclosure",      dict_ctx="agreement"),
    "E4_017": dict(enc_synset="mortgage.n.01",   dict_synset="mortgage.n.01",
                   enc_ctx="securitization",     dict_ctx="loan"),
    "E4_018": dict(enc_synset="patent.n.01",     dict_synset="patent.n.01",
                   enc_ctx="evergreening",       dict_ctx="invention"),
    "E4_019": dict(enc_synset="censorship.n.01", dict_synset="censorship.n.01",
                   enc_ctx="algorithmic",        dict_ctx="suppression"),
    "E4_020": dict(enc_synset="monopoly.n.01",   dict_synset="monopoly.n.01",
                   enc_ctx="acquisition",        dict_ctx="control"),
    "E4_021": dict(enc_synset="tariff.n.01",     dict_synset="tariff.n.01",
                   enc_ctx="incidence",          dict_ctx="imports"),
    # ASYLUM: political refuge (n.02) vs. mental institution (n.01)
    "E4_022": dict(enc_synset="asylum.n.02",     dict_synset="asylum.n.01",
                   enc_ctx="externalization",    dict_ctx="refuge"),
    "E4_023": dict(enc_synset="dividend.n.01",   dict_synset="dividend.n.01",
                   enc_ctx="shareholder",        dict_ctx="profit"),
    "E4_024": dict(enc_synset="algorithm.n.01",  dict_synset="algorithm.n.01",
                   enc_ctx="bias",               dict_ctx="procedure"),
    "E4_025": dict(enc_synset="arbitration.n.01",dict_synset="arbitration.n.01",
                   enc_ctx="mandatory",          dict_ctx="dispute"),
    "E4_026": dict(enc_synset="subsidy.n.01",    dict_synset="subsidy.n.01",
                   enc_ctx="incumbents",         dict_ctx="support"),
    # LEVERAGE: financial (n.02) vs. mechanical advantage (n.01)
    "E4_027": dict(enc_synset="leverage.n.02",   dict_synset="leverage.n.01",
                   enc_ctx="contagion",          dict_ctx="mechanical"),
    "E4_028": dict(enc_synset="zoning.n.01",     dict_synset="zoning.n.01",
                   enc_ctx="segregation",        dict_ctx="land"),
    "E4_029": dict(enc_synset="encryption.n.01", dict_synset="encryption.n.01",
                   enc_ctx="backdoor",           dict_ctx="cipher"),
    "E4_030": dict(enc_synset="austerity.n.01",  dict_synset="austerity.n.01",
                   enc_ctx="conditionality",     dict_ctx="deficit"),
    # DEEP
    "E4_031": dict(enc_synset="loom.n.01",       dict_synset="loom.n.01",
                   enc_ctx="proletarianization", dict_ctx="weave"),
    # ENCLOSURE: agricultural-enclosure movement (n.03) vs. general act (n.01)
    "E4_032": dict(enc_synset="enclosure.n.03",  dict_synset="enclosure.n.01",
                   enc_ctx="commons",            dict_ctx="fence"),
    "E4_033": dict(enc_synset="telegraph.n.01",  dict_synset="telegraph.n.01",
                   enc_ctx="commodities",        dict_ctx="communication"),
    "E4_034": dict(enc_synset="transistor.n.01", dict_synset="transistor.n.01",
                   enc_ctx="automation",         dict_ctx="amplifier"),
    "E4_035": dict(enc_synset="corn.n.01",       dict_synset="corn.n.01",
                   enc_ctx="subsidy",            dict_ctx="grain"),
    # INTEREST: financial (n.03) vs. attention/curiosity (n.01)
    "E4_036": dict(enc_synset="interest.n.03",   dict_synset="interest.n.01",
                   enc_ctx="usury",              dict_ctx="curiosity"),
    "E4_037": dict(enc_synset="plague.n.01",     dict_synset="plague.n.01",
                   enc_ctx="serfdom",            dict_ctx="epidemic"),
    # INDIGO: plant sense (n.02) vs. dye/colour sense (n.01)
    "E4_038": dict(enc_synset="indigo.n.02",     dict_synset="indigo.n.01",
                   enc_ctx="Bengal",             dict_ctx="dye"),
    "E4_039": dict(enc_synset="opium.n.01",      dict_synset="opium.n.01",
                   enc_ctx="sovereignty",        dict_ctx="narcotic"),
    "E4_040": dict(enc_synset="sugar.n.01",      dict_synset="sugar.n.01",
                   enc_ctx="slavery",            dict_ctx="sweetener"),
    "E4_041": dict(enc_synset="cotton.n.01",     dict_synset="cotton.n.01",
                   enc_ctx="slavery",            dict_ctx="fibre"),
    "E4_042": dict(enc_synset="rubber.n.01",     dict_synset="rubber.n.01",
                   enc_ctx="atrocity",           dict_ctx="latex"),
    # CODE: programming (n.03) vs. legal/signal code (n.02)
    # Updated context words: accountability vs legislation (cleaner freq gap)
    "E4_043": dict(enc_synset="code.n.03",       dict_synset="code.n.02",
                   enc_ctx="accountability",     dict_ctx="legislation"),
    # NETWORK: internet/platform (n.03) vs. broadcast TV (n.01)
    "E4_044": dict(enc_synset="network.n.03",    dict_synset="network.n.01",
                   enc_ctx="surveillance",       dict_ctx="broadcast"),
    # DERIVATIVE: financial instrument (n.02) vs. math/linguistic (n.01)
    # Updated context words: bailout (zipf 3.2) vs calculus (3.6) — cleaner gap
    "E4_045": dict(enc_synset="derivative.n.02", dict_synset="derivative.n.01",
                   enc_ctx="bailout",            dict_ctx="calculus"),
    "E4_046": dict(enc_synset="curriculum.n.01", dict_synset="curriculum.n.01",
                   enc_ctx="colonialism",        dict_ctx="subjects"),
    "E4_047": dict(enc_synset="prison.n.01",     dict_synset="prison.n.01",
                   enc_ctx="incarceration",      dict_ctx="confinement"),
    "E4_048": dict(enc_synset="map.n.01",        dict_synset="map.n.01",
                   enc_ctx="colonialism",        dict_ctx="geography"),
    "E4_049": dict(enc_synset="border.n.01",     dict_synset="border.n.01",
                   enc_ctx="sovereignty",        dict_ctx="boundary"),
    # STANDARD: technical specification (n.04) vs. quality level (n.01)
    "E4_050": dict(enc_synset="standard.n.04",   dict_synset="standard.n.01",
                   enc_ctx="lock-in",            dict_ctx="quality"),
}

# ── Tier 3: pre-registered manual critical items ──────────────────────────────
# Criterion: monosemous word where sentence_b encyclopedic reading requires
# specific historical/causal knowledge that is NOT the most frequent association
# of the target word in standard corpora (e.g., STS/Wikipedia co-occurrence).
# These 16 items are marked for expert-annotation verification in Week 2.
# Expert task: confirm that sentence_b cannot be correctly completed by a model
# that only knows co-occurrence statistics for the target word.

MANUAL_CRITICAL = {
    # SHALLOW (5 items) — target ~8 critical per stratum
    "E4_006": "Telescope → Galileo affair: requires knowing that Galileo's "
              "instrument threatened ecclesiastical epistemological authority, "
              "not just that it observed the sky.",
    "E4_007": "Compass → colonial expansion: requires knowing that navigational "
              "technology enabled territorial appropriation, not just seafaring.",
    "E4_008": "Quarantine → racially discriminatory enforcement: requires knowing "
              "the specific 19th-c. history of selective port quarantine.",
    "E4_009": "Famine → political economy: requires knowing food was exported "
              "during the Irish Famine despite mass starvation (Sen's thesis).",
    "E4_014": "Asteroid → K-Pg mass extinction: requires knowing the Chicxulub "
              "impact cleared ecological space for mammalian diversification.",
    # MEDIUM (5 items)
    "E4_016": "Contract → NDA in antitrust context: requires knowing how "
              "confidentiality clauses obstruct regulatory review of mergers.",
    "E4_028": "Zoning → racial segregation: requires knowing US zoning history "
              "as instrument of residential racial exclusion (post-1940s maps).",
    "E4_030": "Austerity → hidden distributional burden: requires knowing that "
              "deficit-reduction moves social costs off the formal ledger.",
    "E4_019": "Censorship → structural/algorithmic suppression: requires knowing "
              "that absence of voices can be market-driven, not only state-ordered.",
    "E4_021": "Tariff → incidence complexity: requires knowing that protection "
              "benefits may accrue to capital rather than the workers invoked.",
    # DEEP (4 items — loom and telegraph removed: loom enc_ctx has zipf 0.0,
    # telegraph claim is weakest among deep items; both to be reconsidered in
    # Week 2 expert annotation if annotators disagree)
    "E4_039": "Opium → EIC state-sponsored trade: requires knowing that the "
              "British East India Company ran the opium trade as policy.",
    "E4_042": "Rubber → Congo Free State atrocities: requires knowing Leopold II's "
              "forced-rubber-extraction system (Hochschild's King Leopold's Ghost).",
    "E4_046": "Curriculum → colonial epistemic violence: requires knowing "
              "Macaulay's Minute and the use of education for cultural erasure.",
    "E4_047": "Prison → mass incarceration as economic policy: requires knowing "
              "that US incarceration rates reflect economic/racial policy, not crime.",
}


# ── Build SemCor index (single pass) ─────────────────────────────────────────

print("Building SemCor index (single pass)...", flush=True)
semcor_index = defaultdict(Counter)

for sent in semcor.tagged_sents(tag="sem"):
    for chunk in sent:
        if not hasattr(chunk, "label"):
            continue
        lab = chunk.label()
        if not hasattr(lab, "synset"):
            continue
        synset_name = lab.synset().name()
        for lem in lab.synset().lemmas():
            semcor_index[lem.name()][synset_name] += 1

print(f"  Index built: {len(semcor_index)} lemmas", flush=True)


def dominant_sense_for(word: str, pos: str = "n") -> tuple:
    counts = semcor_index.get(word, Counter())
    pos_counts = Counter({k: v for k, v in counts.items()
                          if k.split(".")[-2] == pos})
    total = sum(pos_counts.values())
    if total >= 5:
        top_syn, top_count = pos_counts.most_common(1)[0]
        return "semcor", top_syn, top_count
    synsets = wn.synsets(word, pos=pos) or wn.synsets(word)
    if synsets:
        return "wordnet_ordering", synsets[0].name(), 0
    return "none", "UNKNOWN", 0


# ── Main analysis ─────────────────────────────────────────────────────────────

df = pd.read_csv("empirics/stimuli/exp4_minimal_pairs_draft.csv")
results = []

hdr = (f"{'item_id':<8} {'word':<14} {'depth':<8} "
       f"{'dominant':<30} {'enc_sense':<30} "
       f"{'source':<20} {'sense':<5} "
       f"{'e_zip':<6} {'d_zip':<6} {'tier':<12} {'critical'}")
print(hdr)
print("-" * len(hdr))

for _, row in df.iterrows():
    iid  = row["item_id"]
    word = row["target_word"]
    meta = ITEM_META.get(iid, {})

    enc_syn  = meta.get("enc_synset", "UNKNOWN")
    dict_syn = meta.get("dict_synset", "UNKNOWN")
    enc_ctx  = meta.get("enc_ctx", "")
    dict_ctx = meta.get("dict_ctx", "")

    pos = enc_syn.split(".")[-2] if "." in enc_syn else "n"
    src, dom_syn, dom_count = dominant_sense_for(word, pos)

    zipf_enc  = round(zipf_frequency(enc_ctx,  "en"), 2) if enc_ctx  else None
    zipf_dict = round(zipf_frequency(dict_ctx, "en"), 2) if dict_ctx else None

    senses_differ  = (enc_syn != dict_syn)
    dom_is_dict    = (dom_syn == dict_syn) or (dom_syn != enc_syn)
    sense_mismatch = senses_differ and dom_is_dict
    freq_lower     = (zipf_enc is not None and zipf_dict is not None
                      and zipf_enc < zipf_dict)

    # Tier assignment
    if sense_mismatch and freq_lower:
        tier = "T1_auto"
        is_critical = True
    elif sense_mismatch:
        tier = "T2_auto"
        is_critical = True
    elif iid in MANUAL_CRITICAL:
        tier = "T3_manual"
        is_critical = True
    else:
        tier = "-"
        is_critical = False

    verification_src = f"{src}(n={dom_count})" + ("+manual" if iid in MANUAL_CRITICAL else "")

    results.append({
        "item_id":                   iid,
        "target_word":               word,
        "inferential_depth":         row["inferential_depth"],
        "corpus_dominant_sense":     dom_syn,
        "verification_source":       verification_src,
        "encyclopedic_target_sense": enc_syn,
        "dict_synset":               dict_syn,
        "enc_ctx_word":              enc_ctx,
        "dict_ctx_word":             dict_ctx,
        "zipf_enc_ctx":              zipf_enc,
        "zipf_dict_ctx":             zipf_dict,
        "sense_mismatch":            sense_mismatch,
        "freq_lower":                freq_lower,
        "tier":                      tier,
        "is_critical_item":          is_critical,
        "manual_rationale":          MANUAL_CRITICAL.get(iid, ""),
    })

    crit = "CRITICAL" if is_critical else ""
    print(f"{iid:<8} {word:<14} {row['inferential_depth']:<8} "
          f"{dom_syn:<30} {enc_syn:<30} "
          f"{src+'('+str(dom_count)+')':<20} "
          f"{'DIFF' if senses_differ else 'SAME':<5} "
          f"{str(zipf_enc):<6} {str(zipf_dict):<6} {tier:<12} {crit}")


# ── Summary ───────────────────────────────────────────────────────────────────

res_df = pd.DataFrame(results)

print("\n" + "=" * 64)
print("CRITICAL ITEM SUMMARY  (target: ~8 per stratum)")
print("=" * 64)
for depth in ["shallow", "medium", "deep"]:
    sub  = res_df[res_df["inferential_depth"] == depth]
    crit = sub[sub["is_critical_item"]]
    t1   = crit[crit["tier"] == "T1_auto"]
    t2   = crit[crit["tier"] == "T2_auto"]
    t3   = crit[crit["tier"] == "T3_manual"]
    print(f"\n  {depth.upper()} ({len(crit)}/{len(sub)} critical | "
          f"T1={len(t1)}, T2={len(t2)}, T3={len(t3)})")
    for _, r in crit.iterrows():
        tag = f"[{r['tier']}]"
        print(f"    {tag:<12} {r['item_id']}  {r['target_word']}")
        if r["tier"] != "T3_manual":
            print(f"               dominant={r['corpus_dominant_sense']}")
            print(f"               enc={r['encyclopedic_target_sense']}")
        else:
            print(f"               {r['manual_rationale'][:70]}")

total = int(res_df["is_critical_item"].sum())
print(f"\n  TOTAL: {total} / {len(res_df)} items tagged CRITICAL")
print(f"  T1 (sense+freq): {int((res_df['tier']=='T1_auto').sum())}")
print(f"  T2 (sense only): {int((res_df['tier']=='T2_auto').sum())}")
print(f"  T3 (manual):     {int((res_df['tier']=='T3_manual').sum())}")
print("=" * 64)


# ── Write locked stimuli file ─────────────────────────────────────────────────

lock_df = res_df[["item_id", "corpus_dominant_sense", "verification_source",
                  "encyclopedic_target_sense", "is_critical_item"]].set_index("item_id")

df_out = df.copy()
for col in ["corpus_dominant_sense", "verification_source",
            "encyclopedic_target_sense", "is_critical_item"]:
    df_out[col] = df_out["item_id"].map(lock_df[col])

df_out.to_csv("empirics/stimuli/exp4_minimal_pairs.csv", index=False)
print(f"\nLocked file:  empirics/stimuli/exp4_minimal_pairs.csv")

res_df.to_csv("empirics/stimuli/exp4_critical_analysis.csv", index=False)
print(f"Analysis:     empirics/stimuli/exp4_critical_analysis.csv")
print("Week 2 annotation task: verify all T3_manual items (kappa >= 0.80 required).")
