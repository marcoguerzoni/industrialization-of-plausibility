"""
Day 4 script: build exp1_polysemous_terms.csv
30 polysemous target words × 5 prompt variants = 150 prompts total.
Ambiguity bands verified via WordNet synset count + wordfreq.

Prompt conditions:
  P1 — Direct: maximal ambiguity, no context
  P2 — Domain-implied: 2-sentence narrative implying a sense without naming it
  P3 — Domain-explicit: same narrative + one domain cue word added
  P4 — Contrastive: forces the less-dominant sense via contrast instruction
  P5 — Definition: control condition; should produce near-dictionary responses
"""

import pandas as pd
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

# ── Stimulus definitions ───────────────────────────────────────────────────────
# Format per entry:
#   word, band, P2_text, P3_text, P4_contrast_word, P2_implied_sense, note

STIMULI = {

    # ── HIGH AMBIGUITY (≥ 4 synsets) ─────────────────────────────────────────

    "bank": dict(
        band="high",
        P2="She walked along the edge of the field where the willows grew thickest "
           "and the current ran fastest. The bank had eroded significantly after last "
           "winter's floods.",
        P3="She walked along the water's edge where the willows grew thickest and the "
           "current ran fastest. The riverbank had eroded significantly after last "
           "winter's floods.",
        P4_contrast="river",
        P2_sense="riverbank (geographical landform)",
        note="financial institution / riverbank / to tilt / depository — classic polysemy",
    ),
    "crane": dict(
        band="high",
        P2="The workers arrived early to the site, and the enormous machine was "
           "already in operation, its long arm swinging cables and steel beams "
           "across the half-finished frame.",
        P3="The workers arrived early to the construction site, and the enormous "
           "machine was already in operation, its arm swinging steel beams "
           "across the half-finished structure.",
        P4_contrast="heavy machinery",
        P2_sense="construction crane (machine)",
        note="construction machine / wading bird / to stretch neck — common polysemy",
    ),
    "spring": dict(
        band="high",
        P2="After months of bare branches and frozen ground, the morning air smelled "
           "different and the birds had begun to sing again.",
        P3="After months of bare branches and frozen ground, the April air smelled "
           "different and the nesting birds had returned.",
        P4_contrast="winter",
        P2_sense="the season spring",
        note="season / coiled mechanism / water source / to jump — highly polysemous",
    ),
    "pitch": dict(
        band="high",
        P2="The musician paused, adjusted the instrument, and played the note again "
           "— this time it sat exactly where the conductor wanted it.",
        P3="The musician adjusted the tuning and played the note again — this time "
           "its musical frequency sat exactly where the conductor wanted it.",
        P4_contrast="music",
        P2_sense="musical pitch (frequency of a note)",
        note="musical pitch / sales pitch / playing field / tar / gradient",
    ),
    "bat": dict(
        band="high",
        P2="At dusk, they emerged from the old barn eaves in dozens, navigating "
           "through the dark by sound rather than sight.",
        P3="At dusk, the nocturnal mammals emerged from the old barn eaves in dozens, "
           "navigating through the dark by echolocation.",
        P4_contrast="any animal",
        P2_sense="bat (flying mammal, Chiroptera)",
        note="flying mammal / cricket bat / baseball bat / to hit",
    ),
    "fair": dict(
        band="high",
        P2="The town had waited all year for it: the rides, the livestock competitions, "
           "the smell of fried food, and the long rows of handmade goods on display.",
        P3="The town had waited all year for the annual fair: the rides, the livestock "
           "competitions, the fried food, and the rows of craft stalls.",
        P4_contrast="a public event",
        P2_sense="fair (outdoor public event, funfair)",
        note="just/equitable / outdoor event / light complexion / satisfactory",
    ),
    "plant": dict(
        band="high",
        P2="The complex employed twelve hundred workers across three shifts, running "
           "the heavy machinery continuously around the clock.",
        P3="The industrial complex employed twelve hundred workers across three shifts, "
           "running the factory machinery continuously around the clock.",
        P4_contrast="a factory",
        P2_sense="industrial plant (manufacturing facility)",
        note="botanical organism / industrial facility / to place/insert / spy",
    ),
    "match": dict(
        band="high",
        P2="The two teams had trained for months for this moment, and the stadium "
           "was full two hours before the referee's whistle.",
        P3="The two teams had trained for months for the football match, and the "
           "stadium was packed two hours before the referee's whistle.",
        P4_contrast="a sporting event",
        P2_sense="sporting match (contest)",
        note="sporting contest / fire-starting stick / correspondence/fit / marriage",
    ),
    "bark": dict(
        band="high",
        P2="The forester ran her hand along the outside of the trunk, reading "
           "its texture and color for signs of disease or old age.",
        P3="The forester ran her hand along the outer layer of the tree trunk, "
           "reading its texture and color for signs of fungal disease.",
        P4_contrast="a tree",
        P2_sense="bark (outer covering of tree trunk)",
        note="tree surface / dog sound / type of sailing vessel",
    ),
    "draft": dict(
        band="high",
        P2="The writer had filled three notebooks with crossed-out lines and "
           "marginal notes before the structure of the argument finally became clear.",
        P3="The writer had filled three notebooks revising the manuscript, crossing "
           "out lines and adding marginal notes, before the argument finally held.",
        P4_contrast="a written document",
        P2_sense="draft (early version of a written text)",
        note="written document early version / air current / military conscription / draught beer",
    ),

    # ── MEDIUM AMBIGUITY (2-3 synsets) ────────────────────────────────────────

    "cell": dict(
        band="medium",
        P2="The researcher looked through the lens and watched as the membrane "
           "began to divide, the nucleus pulling apart into two equal halves.",
        P3="The biologist looked through the microscope lens and watched as the "
           "cell membrane began to divide, the nucleus separating into two halves.",
        P4_contrast="a biological organism",
        P2_sense="biological cell",
        note="biological cell / prison cell / mobile phone cell / battery cell",
    ),
    "chapter": dict(
        band="medium",
        P2="The members gathered every third Tuesday in the lodge hall, reviewed old "
           "business, and voted on the budget for the coming quarter.",
        P3="The members of the local chapter gathered every third Tuesday, reviewed "
           "old business, and voted on the organization's quarterly budget.",
        P4_contrast="an organization",
        P2_sense="chapter (branch of an organization)",
        note="section of a book / branch of an organization / historical period",
    ),
    "case": dict(
        band="medium",
        P2="The attorneys had spent six months gathering depositions, and the judge "
           "had scheduled the main hearing for early spring.",
        P3="The attorneys had spent six months in discovery, and the judge had "
           "scheduled the court hearing for early spring.",
        P4_contrast="a legal proceeding",
        P2_sense="legal case",
        note="legal case / container / medical case / grammatical case",
    ),
    "seal": dict(
        band="medium",
        P2="The colony gathered on the rocky outcrop at low tide, the young ones "
           "barking noisily as the adults returned from deeper water.",
        P3="The marine mammals gathered on the rocky outcrop at low tide, the pups "
           "barking as the adults returned from their Arctic fishing grounds.",
        P4_contrast="any animal",
        P2_sense="seal (pinniped marine mammal)",
        note="pinniped marine mammal / wax/official seal / Navy SEAL",
    ),
    "plate": dict(
        band="medium",
        P2="The geologist pointed to the fault line on the map and explained how "
           "pressure had been building for decades along the boundary.",
        P3="The geologist pointed to the tectonic fault line on the map and explained "
           "how crustal pressure had been building for decades along the plate boundary.",
        P4_contrast="a dish",
        P2_sense="tectonic plate",
        note="dish / tectonic plate / license plate / printing plate / to coat with metal",
    ),
    "lead": dict(
        band="medium",
        P2="They had to take precautions when renovating — the pipes dated from the "
           "nineteen-twenties, and the paint on the walls was even older.",
        P3="They had to take precautions when renovating the Victorian house — the "
           "water pipes were original, and the paint contained the toxic heavy metal.",
        P4_contrast="a metal",
        P2_sense="lead (the heavy metal element, Pb)",
        note="heavy metal Pb / to guide/direct / theatrical lead role / dog lead",
    ),
    "pupil": dict(
        band="medium",
        P2="The ophthalmologist shone the light directly at it and watched carefully "
           "for the response, checking whether both sides reacted symmetrically.",
        P3="The ophthalmologist shone the light at the eye and watched for the "
           "pupillary response, checking whether both pupils reacted symmetrically.",
        P4_contrast="part of the eye",
        P2_sense="pupil (aperture in the iris of the eye)",
        note="aperture in the iris / student/learner",
    ),
    "scale": dict(
        band="medium",
        P2="The musician practiced the ascending and descending patterns every "
           "morning, running through all twelve before moving on to repertoire.",
        P3="The pianist practiced all twelve musical scales every morning, running "
           "through both ascending and descending patterns before touching the repertoire.",
        P4_contrast="music",
        P2_sense="musical scale (ordered sequence of notes)",
        note="musical scale / weighing instrument / fish/reptile scale / to climb",
    ),
    "crown": dict(
        band="medium",
        P2="The dentist took an impression of the tooth and explained that the "
           "procedure would require two appointments and a temporary fitting.",
        P3="The dentist took an impression and explained that fitting the dental "
           "crown would require two appointments, including a temporary placement.",
        P4_contrast="a tooth",
        P2_sense="dental crown (tooth restoration)",
        note="dental crown / monarch's headwear / top of head / crown of a tree",
    ),
    "file": dict(
        band="medium",
        P2="The craftsman worked methodically along the edge, removing tiny amounts "
           "of material with each stroke to achieve exactly the right fit.",
        P3="The metalworker used the hand tool methodically along the edge, removing "
           "tiny filings with each stroke to achieve exactly the right fit.",
        P4_contrast="a hand tool",
        P2_sense="file (abrasive hand tool)",
        note="abrasive hand tool / document / computer file / to file legally",
    ),

    # ── NEAR-MONOSEMOUS (1-2 synsets; second sense archaic/domain-restricted) ─

    "umbrella": dict(
        band="near_monosemous",
        P2="She opened it quickly as the first drops began to fall, and walked "
           "briskly between the bus stop and the building entrance.",
        P3="She opened the rain canopy quickly as the first drops began to fall, "
           "and walked briskly to the office entrance.",
        P4_contrast="weather protection",
        P2_sense="umbrella (portable rain shield)",
        note="portable rain shield (dominant) / organizational umbrella term (metaphorical, secondary)",
    ),
    "mercury": dict(
        band="near_monosemous",
        P2="The old thermometer had been recalled because of the danger of the "
           "silver liquid inside escaping if the glass broke.",
        P3="The old clinical thermometer had been recalled — the toxic silver liquid "
           "element inside posed a contamination risk if the glass broke.",
        P4_contrast="a chemical element",
        P2_sense="mercury (liquid metal element, Hg)",
        note="liquid metal Hg (dominant) / planet Mercury / Roman god Mercury",
    ),
    "bronze": dict(
        band="near_monosemous",
        P2="The sculptor had spent three months on the casting, and when the mould "
           "was broken open the figure stood exactly as she had modelled it in clay.",
        P3="The sculptor had spent three months on the lost-wax casting, and when "
           "the bronze mould was broken open the metal figure stood perfectly.",
        P4_contrast="a metal",
        P2_sense="bronze (copper-tin alloy used in casting)",
        note="copper-tin alloy (dominant) / bronze medal (secondary derivative) / to get a tan (informal)",
    ),
    "amber": dict(
        band="near_monosemous",
        P2="The paleontologist held the specimen up to the light — the insect "
           "inside had been preserved perfectly for forty million years.",
        P3="The paleontologist held the fossilized resin up to the light — the "
           "trapped insect had been preserved perfectly for forty million years.",
        P4_contrast="a fossil material",
        P2_sense="amber (fossilized tree resin)",
        note="fossilized resin (dominant) / amber color / traffic amber (secondary, derivative)",
    ),
    "slate": dict(
        band="near_monosemous",
        P2="The roofer worked along the ridge, replacing the cracked ones carefully "
           "so that the rain would not find its way through before winter.",
        P3="The roofer worked along the ridge, replacing each cracked grey roofing "
           "tile so that the rain could not penetrate before winter.",
        P4_contrast="a roofing material",
        P2_sense="slate (fine-grained grey rock used in roofing)",
        note="roofing rock (dominant) / political electoral slate (secondary) / blank-slate idiom",
    ),
    "mortar": dict(
        band="near_monosemous",
        P2="The bricklayer spread it evenly between each course of bricks, pressing "
           "the layers together firmly and wiping the excess from the face.",
        P3="The bricklayer spread the binding compound between each course of bricks, "
           "pressing the masonry together and wiping the excess clean.",
        P4_contrast="a building material",
        P2_sense="mortar (binding material in masonry)",
        note="masonry binding material (dominant) / pestle-and-mortar bowl (secondary) / artillery mortar (secondary)",
    ),
    "jade": dict(
        band="near_monosemous",
        P2="The carver had worked the block for three months, following the natural "
           "grain, shaping the pendant to be worn at the Lunar New Year.",
        P3="The craftsman had worked the block of precious stone for three months, "
           "following its natural grain, carving the pendant for the New Year ceremony.",
        P4_contrast="a mineral or gemstone",
        P2_sense="jade (nephrite/jadeite gemstone)",
        note="nephrite/jadeite stone (dominant) / jade green color / to exhaust/bore (archaic verb)",
    ),
    "marrow": dict(
        band="near_monosemous",
        P2="The transplant required finding a compatible donor — someone whose "
           "stem cells were close enough in type to rebuild the patient's immune system.",
        P3="The bone marrow transplant required finding a compatible donor whose "
           "stem cells were close enough in type to rebuild the patient's immune system.",
        P4_contrast="bone tissue",
        P2_sense="bone marrow (soft tissue inside bones)",
        note="bone marrow (dominant) / vegetable marrow (British: large squash, secondary)",
    ),
    "sterling": dict(
        band="near_monosemous",
        P2="The hallmark on the back confirmed its purity — it had been assayed "
           "and certified to the standard that had been in use since the thirteenth century.",
        P3="The hallmark on the back confirmed the silver content — it had been "
           "assayed and certified to the sterling standard in use since the thirteenth century.",
        P4_contrast="silver",
        P2_sense="sterling silver (standard silver alloy)",
        note="sterling silver standard (dominant) / excellent-quality adjective (secondary, derivative)",
    ),
    "mast": dict(
        band="near_monosemous",
        P2="The riggers climbed before dawn to check the lines, making sure the "
           "rigging was sound before the vessel left harbor with the tide.",
        P3="The sailors climbed the sailing mast before dawn to check the rigging, "
           "making sure all lines were sound before the ship left harbor.",
        P4_contrast="a sailing vessel",
        P2_sense="mast (vertical spar on a sailing vessel)",
        note="sailing vessel mast (dominant) / telecommunications/TV transmission mast (secondary)",
    ),
}


# ── Build prompts ─────────────────────────────────────────────────────────────

def make_prompts(word: str, meta: dict) -> dict:
    return {
        "prompt_P1": f"Describe {word} in a few sentences.",
        "prompt_P2": meta["P2"],
        "prompt_P3": meta["P3"],
        "prompt_P4": (f"Describe {word}, making sure to distinguish it "
                      f"from {meta['P4_contrast']}."),
        "prompt_P5": f"Give a dictionary definition of {word}.",
    }


# ── Verify synset counts ──────────────────────────────────────────────────────

def synset_count(word: str) -> int:
    return len(wn.synsets(word))


# ── Build dataframe ───────────────────────────────────────────────────────────

BAND_THRESHOLDS = {"high": (4, 99), "medium": (2, 3), "near_monosemous": (1, 2)}

rows = []
warnings = []
item_id = 1

for word, meta in STIMULI.items():
    prompts   = make_prompts(word, meta)
    n_synsets = synset_count(word)
    zipf      = round(zipf_frequency(word, "en"), 2)
    band      = meta["band"]

    # Threshold checks
    low_thr, high_thr = BAND_THRESHOLDS[band]
    if not (low_thr <= n_synsets):
        warnings.append(f"  {word}: n_synsets={n_synsets} (expected >= {low_thr} for {band})")
    if not (3.5 <= zipf <= 6.5):
        warnings.append(f"  {word}: zipf={zipf} (outside 3.5–6.5 range)")

    rows.append({
        "item_id":            f"E1_{item_id:03d}",
        "target_word":        word,
        "ambiguity_band":     band,
        "n_wordnet_synsets":  n_synsets,
        "zipf_frequency":     zipf,
        **prompts,
        "P2_implied_sense":   meta["P2_sense"],
        "P4_contrast_word":   meta["P4_contrast"],
        "note":               meta["note"],
    })
    item_id += 1

df = pd.DataFrame(rows)

# ── Quality checks ────────────────────────────────────────────────────────────

print("=" * 64)
print("EXP 1 STIMULUS QUALITY CHECKS")
print("=" * 64)

if warnings:
    print("\nWARNINGS:")
    for w in warnings:
        print(w)
else:
    print("\nAll items pass band and frequency thresholds.")

print(f"\nBand counts:")
print(df["ambiguity_band"].value_counts().to_string())

print(f"\nTotal items: {len(df)} | Total prompts: {len(df) * 5}")

print(f"\nSynset distribution:")
for band in ["high", "medium", "near_monosemous"]:
    sub = df[df["ambiguity_band"] == band]
    print(f"  {band:<18} n_synsets: "
          f"min={sub['n_wordnet_synsets'].min()}  "
          f"max={sub['n_wordnet_synsets'].max()}  "
          f"mean={sub['n_wordnet_synsets'].mean():.1f}")

print(f"\nZipf frequency distribution (should be 3.5–6.5):")
for band in ["high", "medium", "near_monosemous"]:
    sub = df[df["ambiguity_band"] == band]
    print(f"  {band:<18} zipf: "
          f"min={sub['zipf_frequency'].min()}  "
          f"max={sub['zipf_frequency'].max()}  "
          f"mean={sub['zipf_frequency'].mean():.2f}")

print(f"\nAll words:")
print(df[["item_id","target_word","ambiguity_band","n_wordnet_synsets",
          "zipf_frequency"]].to_string(index=False))

# ── P2 quality check: implied sense ──────────────────────────────────────────
print("\nP2 implied senses (verify domain is NOT named explicitly):")
for _, r in df.iterrows():
    print(f"  [{r['item_id']}] {r['target_word']:<12} => {r['P2_implied_sense']}")

# ── P4 contrast words: frequency check ───────────────────────────────────────
print("\nP4 contrast word frequency check (should have zipf >= 4.0):")
p4_issues = []
for _, r in df.iterrows():
    cw = r["P4_contrast_word"]
    # Take first word of multi-word contrasts
    first_word = cw.split()[0]
    cz = round(zipf_frequency(first_word, "en"), 2)
    flag = "" if cz >= 4.0 else "  <<< LOW"
    print(f"  {r['target_word']:<12} vs '{cw}'  (zipf={cz}){flag}")
    if cz < 4.0:
        p4_issues.append(r["target_word"])

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = "empirics/stimuli/exp1_polysemous_terms.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
