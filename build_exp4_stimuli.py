"""
Day 2 script: build exp4_minimal_pairs_draft.csv
Constructs 50 sentence pairs (sentence_a = dictionary-sufficient,
sentence_b = encyclopedic-required), stratified by inferential depth.
Critical-item columns left blank; filled in Day 3 via SemCor + wordfreq.
"""

import pandas as pd
from wordfreq import zipf_frequency

# ── Stimulus definitions ───────────────────────────────────────────────────────
# Format: (target_word, sentence_a, sentence_b, ground_truth_reading, note)

SHALLOW = [
    (
        "vaccine",
        "The doctor administered the vaccine before the patient's trip abroad.",
        "The vaccine became a flashpoint in debates about bodily autonomy and state authority long before the pandemic made those debates visible.",
        "Public health institution with contested political history; not merely a medical product.",
        "Vaccine hesitancy, mandates, pharmaceutical politics",
    ),
    (
        "virus",
        "The virus spread rapidly through the crowded hospital ward.",
        "The virus that reshaped global governance was not novel in its biology; it was novel in the institutional unpreparedeness it exposed.",
        "Pandemic event as political and institutional crisis, not merely biological phenomenon.",
        "COVID-19 / HIV as institutional mirror",
    ),
    (
        "antibiotic",
        "The patient was prescribed an antibiotic to clear the bacterial infection.",
        "The antibiotic age may be ending — not because the chemistry failed, but because the incentive to develop new ones does not exist inside a market logic.",
        "Antibiotic resistance as consequence of commodification; market failure in drug development.",
        "AMR crisis, pharmaceutical R&D economics",
    ),
    (
        "gene",
        "The gene encodes the instructions for building a specific protein in the cell.",
        "When a gene becomes a patentable asset, the question of who owns the body is no longer metaphorical.",
        "Gene as proprietary commodity; genomic capitalism, biotech patents.",
        "Myriad Genetics v. AMP, BRCA patents",
    ),
    (
        "fossil",
        "The fossil was found preserved in limestone at the base of the cliff.",
        "Every fossil found in that formation tells the same story: that the world before us was not a warm-up act.",
        "Fossil as evidence of deep time, human contingency, evolutionary history.",
        "Deep time, evolution, paleontological narrative",
    ),
    (
        "telescope",
        "The astronomer used a telescope to observe the moons of Jupiter.",
        "When the authorities told him to put away his telescope, the dispute was not about astronomy — it was about who had the right to produce facts.",
        "Telescope as instrument of epistemological conflict; Galileo affair, authority vs. empirical knowledge.",
        "Galileo, science and power",
    ),
    (
        "compass",
        "The sailor used a compass to determine which direction was north.",
        "The compass did not help sailors find their way home; it helped empires find other people's land.",
        "Compass as instrument of colonial navigation and territorial expansion.",
        "Age of Exploration, colonialism",
    ),
    (
        "quarantine",
        "Passengers arriving from the affected region were placed in quarantine for two weeks.",
        "Quarantine orders in nineteenth-century port cities were enforced selectively, and the selection followed a logic that had nothing to do with epidemiology.",
        "Quarantine as historically racialized public health instrument.",
        "Discriminatory quarantine enforcement, Chinese Exclusion Act era",
    ),
    (
        "famine",
        "The region suffered a devastating famine following three years of drought.",
        "It was called a famine, but the grain ships never stopped leaving the port; the food was there, and the people were not allowed to eat it.",
        "Political economy of famine: food exported during starvation; famine as political failure not natural disaster.",
        "Irish Famine 1845-52, Sen's entitlement theory",
    ),
    (
        "inflation",
        "Inflation caused the price of basic goods to rise sharply over the course of the year.",
        "Inflation is never a neutral event: it erodes wages faster than savings, which is why the same rate can be a crisis for one household and a minor inconvenience for another.",
        "Inflation as distributionally unequal phenomenon; real asset holders vs. wage earners.",
        "Distributional effects of inflation",
    ),
    (
        "protest",
        "Hundreds of people gathered outside the parliament building to protest the new law.",
        "The protest was called a threat to public order first, then an understandable expression of frustration, then an inevitable turning point — always in that sequence.",
        "Social movement dynamics: criminalization → normalization → historical legitimation cycle.",
        "Social movement literature; Overton window dynamics",
    ),
    (
        "strike",
        "The workers announced a strike after negotiations with management broke down.",
        "The strike was broken — not by argument, but by importing men who had no other option, and calling it the free market.",
        "Strike-breaking via reserve labor army; labor market as power asymmetry.",
        "Industrial labor history; reserve army of labor",
    ),
    (
        "satellite",
        "The satellite orbits Earth every ninety minutes, transmitting data to ground stations.",
        "The satellite changed the meaning of sovereignty: a border is no longer only a line on the ground but a frequency in the sky, and some nations cannot afford the frequency.",
        "Satellite communications as contested infrastructure; spectrum sovereignty, digital divide.",
        "ITU spectrum allocation, GPS dependence",
    ),
    (
        "asteroid",
        "The asteroid was detected by ground-based telescopes as it passed near Earth.",
        "The last time an object of that mass arrived, it did not just end an era — it cleared the stage for everything that came after, including us.",
        "K-Pg mass extinction event; asteroids as agents of evolutionary history.",
        "Chicxulub impactor, mammalian diversification",
    ),
    (
        "climate",
        "The climate of the coastal region is mild, with warm summers and wet winters.",
        "The climate did not change on its own; it changed because the costs of changing it were distributed to people who had not made the decision.",
        "Climate change as collective action problem with unequal distributional burden; IPCC, fossil fuel lobbying, Paris Agreement.",
        "Climate justice, IPCC AR6, fossil fuel industry denial campaigns",
    ),
]

MEDIUM = [
    (
        "contract",
        "Both parties signed a contract before the project could begin.",
        "The contract bound both parties to silence about the acquisition, which is why the antitrust review was unable to proceed.",
        "Contract as instrument of regulatory opacity; NDAs in merger transactions.",
        "Contractual silence, antitrust evasion",
    ),
    (
        "mortgage",
        "The family took out a mortgage to purchase their first home.",
        "The mortgage was sold twice before the family made their first payment, and by the third year they were sending money to an entity they had never heard of.",
        "Mortgage securitization; MBS markets; 2008 financial crisis mechanics.",
        "Mortgage-backed securities, subprime crisis",
    ),
    (
        "patent",
        "The company was awarded a patent for its new manufacturing process.",
        "The patent expired, but the drug never became affordable, because the original had been replaced by a modified version just different enough to restart the clock.",
        "Evergreening: patent extension strategies in pharmaceuticals to prevent generic competition.",
        "Pharmaceutical evergreening, TRIPS Agreement",
    ),
    (
        "censorship",
        "The book was subjected to censorship before it could be distributed.",
        "By the time the investigation began, most people had already stopped noticing which voices were amplified and which had simply ceased to appear.",
        "Structural censorship via platform algorithms and market concentration; not only state censorship.",
        "Algorithmic suppression, market-driven content exclusion",
    ),
    (
        "monopoly",
        "The company held a monopoly over the distribution of electricity in the region.",
        "The monopoly was not won by outcompeting rivals but by acquiring them and then drafting the regulations that made further acquisition easier than further competition.",
        "Platform monopoly via regulatory capture; monopoly as political rather than purely economic phenomenon.",
        "Big Tech antitrust, regulatory capture",
    ),
    (
        "tariff",
        "The government imposed a tariff on imported steel to protect domestic industry.",
        "The tariff was presented as a measure for workers, but its main legible effect was in the accounts of firms that had already stopped employing them.",
        "Tariff incidence: protection benefits capital, not necessarily labor; industrial policy complexity.",
        "Trade policy incidence, Stolper-Samuelson in reverse",
    ),
    (
        "asylum",
        "She applied for political asylum after fleeing persecution in her home country.",
        "Asylum had been a right, then became a procedure, then became a queue administered at a distance by offices that would never see her face.",
        "Externalization of asylum processing; bureaucratic attrition as deterrence policy.",
        "EU-Turkey deal, offshore processing, asylum deterrence",
    ),
    (
        "dividend",
        "The company paid shareholders a quarterly dividend of fifty cents per share.",
        "The dividend was generous; the workforce was the smallest it had been in thirty years; the two facts were presented in separate sections of the annual report.",
        "Shareholder primacy: dividend maximization via labor cost reduction; financialization.",
        "Shareholder value ideology, Friedman doctrine",
    ),
    (
        "algorithm",
        "The algorithm ranks search results according to a measure of relevance.",
        "The algorithm was neutral in the way a locked door is neutral — it applied the same rule to everyone, once you accepted who had written the rule.",
        "Algorithmic bias: formally neutral procedures encoding structural inequality.",
        "Facial recognition bias, credit scoring discrimination",
    ),
    (
        "arbitration",
        "The employment dispute was referred to arbitration rather than the court system.",
        "The arbitration clause was buried in the terms of service, and it meant that the complaint would be heard by someone selected and paid by the company being complained about.",
        "Mandatory arbitration as barrier to collective action; captured arbitration process.",
        "SCOTUS Epic Systems v. Lewis, consumer arbitration clauses",
    ),
    (
        "subsidy",
        "The government introduced a subsidy to reduce the cost of renewable energy for consumers.",
        "The subsidy was described as support for innovation, but it was larger, and older, and differently distributed than that description implied.",
        "Political economy of subsidy: who receives them, why, and how subsidy allocation encodes political power (fossil fuels, agriculture, banking).",
        "Fossil fuel subsidies, farm bill, tax expenditures as hidden subsidies",
    ),
    (
        "leverage",
        "The investment fund used leverage to amplify returns on its asset portfolio.",
        "When the leverage unwound, it did not unwind slowly; it reached the pension funds before the word 'crisis' had appeared in a single headline.",
        "Systemic risk propagation in leveraged financial system; 2008 crisis mechanics.",
        "Leverage cycles, Lehman collapse, contagion",
    ),
    (
        "zoning",
        "The city's zoning regulations restricted commercial development in residential neighborhoods.",
        "The zoning map drawn in 1948 still predicted, with remarkable accuracy, which neighborhoods had parks and which had highways.",
        "Racially exclusionary zoning; long-term spatial inequality; environmental racism.",
        "Racial zoning, redlining, urban geography of inequality",
    ),
    (
        "encryption",
        "The message was encrypted before transmission to prevent unauthorized access.",
        "Encryption became a political question the moment governments realized that a lock they could not open was a lock they intended to prohibit.",
        "Crypto wars; state demands for backdoors; privacy vs. security in digital policy.",
        "FBI v. Apple, Clipper chip, key escrow debate",
    ),
    (
        "austerity",
        "The government announced a period of austerity to reduce the public deficit.",
        "Austerity balanced the budget on paper by moving the costs to a line that no accountant was required to track.",
        "Social cost of austerity: hidden distributional burden on public-service users; IMF conditionality.",
        "Greek debt crisis, IMF austerity conditionality, Blyth 2013",
    ),
]

DEEP = [
    (
        "loom",
        "The weaver operated the loom to produce lengths of woolen cloth.",
        "When the loom changed the valley's social fabric within a generation, the metaphor was not an accident.",
        "Mechanized weaving → artisan displacement → proletarianization → Industrial Revolution's social transformation.",
        "Luddites, textile mechanization, Thompson's Making of the English Working Class",
    ),
    (
        "enclosure",
        "The farmer built an enclosure to keep the livestock from wandering.",
        "The enclosure of the commons was the first act of the drama that would end, three generations later, with the factory whistle.",
        "English enclosure → privatization of common land → peasant displacement → proletarian labor supply for industry.",
        "Enclosure Acts, primitive accumulation, Polanyi's Great Transformation",
    ),
    (
        "telegraph",
        "The telegraph message was transmitted along the wire and received within seconds.",
        "The telegraph did not change how fast news traveled; it changed who news was for.",
        "Telegraph as financial infrastructure; reordering of information asymmetries; Reuters, AP; military command.",
        "James Carey, communication as culture; telegraph and commodity markets",
    ),
    (
        "transistor",
        "The transistor amplifies the electrical signal passing through the circuit.",
        "The transistor was a physics experiment until it became a question about which kinds of labor were still worth paying for.",
        "Transistor → IC → computer → automation of cognitive labor → de-skilling; digital transformation of work.",
        "Moore's Law, Braverman deskilling thesis, platform labor",
    ),
    (
        "corn",
        "The farmer planted corn in the fields along the river.",
        "The history of corn in America is also the history of why the cheapest calories are the ones most likely to kill you.",
        "Corn subsidies → overproduction → HFCS → obesity epidemic → farm bill politics; commodification of food.",
        "Pollan Omnivore's Dilemma, HFCS, USDA farm policy",
    ),
    (
        "interest",
        "The bank charged a rate of interest on the outstanding loan balance.",
        "The argument about whether interest is a fee for service or a mechanism of extraction is older than capitalism and has never been resolved to everyone's satisfaction.",
        "Usury debates from Aristotle through medieval church law through Islamic finance to contemporary financialization; interest as accumulation vs. exploitation.",
        "Aristotelian chrematistics, usury prohibition, interest-free finance",
    ),
    (
        "plague",
        "The plague killed a large share of the population in the affected regions.",
        "The plague did not only empty the villages; it emptied the argument for serfdom, because a dead serf cannot be compelled to work.",
        "Black Death → labor shortage → collapse of feudal labor relations → peasant bargaining power → end of serfdom.",
        "Black Death 1347-51, labor shortage, Brenner debate, Postan thesis",
    ),
    (
        "indigo",
        "Indigo was used to produce a deep blue dye prized by cloth merchants.",
        "Indigo built fortunes in Bengal and left behind a system of coercion that the British called agriculture and the peasants called something else.",
        "Indigo Revolt 1859-60; forced cultivation under British planters; peasant resistance; colonial agrarian economy.",
        "Nil Darpan, Bengal indigo, Indigo Commission 1860",
    ),
    (
        "opium",
        "Opium was widely used in nineteenth-century medicine to relieve pain.",
        "The opium trade was not a failure of governance; it was governance — conducted by a corporation with shareholders, a charter, and an army.",
        "British East India Company's opium trade; Opium Wars; state-sponsored drug trade as imperial policy.",
        "EIC Bengal opium, First and Second Opium Wars, treaty ports",
    ),
    (
        "sugar",
        "Sugar is extracted from the juice of crushed sugarcane.",
        "Sugar was the sweetness of empire, and the bitterness was not in the cane.",
        "Atlantic slave trade; sugar plantation economy; Caribbean colonies; triangular trade and primitive accumulation.",
        "Sidney Mintz Sweetness and Power, Caribbean slavery, ASIENTO",
    ),
    (
        "cotton",
        "Cotton is a soft fiber harvested from the bolls of the cotton plant.",
        "The cotton economy could not have operated without a prior decision to treat one category of human beings as part of the capital stock.",
        "American slavery as foundation of antebellum capitalism; cotton → financial instruments → Northern banks.",
        "Baptist The Half Has Never Been Told, slave capitalism",
    ),
    (
        "rubber",
        "Rubber is derived from the latex sap of the rubber tree and valued for its elasticity.",
        "The rubber that reached European factories had already cost more than it was ever sold for — but not in any currency that anyone was required to report.",
        "Congo Free State; Leopold II's forced rubber extraction; atrocities; Adam Hochschild King Leopold's Ghost.",
        "Congo Free State, quota system, Henry Morton Stanley",
    ),
    (
        "code",
        "The developer wrote code to automate the data entry process.",
        "When the code began deciding who received loans and who did not, the question of who had written the code became a question of constitutional law.",
        "Algorithmic lending decisions; automated decision-making; accountability gap; ECOA discrimination.",
        "Fair lending algorithms, explainability requirements, CFPB",
    ),
    (
        "network",
        "The social network allowed users to share messages and photographs with friends.",
        "The network had two billion users and understood each of them better than their families did — a knowledge it had monetized before anyone thought to regulate it.",
        "Surveillance capitalism; attention economy; behavioral data extraction; regulatory lag.",
        "Zuboff Surveillance Capitalism, Cambridge Analytica, GDPR",
    ),
    (
        "derivative",
        "A derivative is a financial instrument whose value depends on an underlying asset.",
        "The derivative market was large enough by 2007 that unwinding it was not a financial problem; it was a political one, and the politics were resolved in favor of the market.",
        "Credit default swaps; 2008 financial crisis; too-big-to-fail; political economy of bailouts; socialization of losses.",
        "AIG, TARP, CDO-squared, Tett Fool's Gold",
    ),
    (
        "curriculum",
        "The school's curriculum covered mathematics, history, and three foreign languages.",
        "What was taught in the schools of a colony was never merely a question about education.",
        "Colonial curriculum as cultural erasure; knowledge as instrument of political domination; Macaulay's Minute on Indian Education.",
        "Colonial education, Macaulay Minute 1835, Ngugi decolonize the mind",
    ),
    (
        "prison",
        "The convicted man was sentenced to five years in prison.",
        "The prison population quadrupled in forty years without a corresponding increase in crime, which suggested that the explanation required an account of economics rather than criminology.",
        "Mass incarceration; prison-industrial complex; racial and economic determinants of incarceration; 13th Amendment exception.",
        "Alexander New Jim Crow, prison labor, mandatory minimums",
    ),
    (
        "map",
        "The explorer used a map to plan the route through the mountain pass.",
        "The map arrived before the territory it described, and in several cases the territory was then reorganized to fit the map.",
        "Cartography as colonial instrument; terra nullius; arbitrary boundary-drawing; Sykes-Picot, Berlin Conference.",
        "Harley cartography and power, Berlin Conference 1884, Sykes-Picot",
    ),
    (
        "border",
        "The customs officer checked travelers' documents at the border crossing.",
        "The border had always been there, in the sense that a line drawn seventy years ago by men who had never visited had always been there.",
        "Colonial border-drawing; arbitrariness of post-colonial nation-state borders; African and Middle Eastern cases.",
        "Scramble for Africa, Sykes-Picot, post-colonial conflict",
    ),
    (
        "standard",
        "The engineering team ensured the product met the applicable industry standard.",
        "The standard that won was not the better one; it was the one whose sponsor could afford to wait until the others ran out of money.",
        "Standards wars; path dependency; lock-in; VHS/Betamax, QWERTY, USB — incumbency as competitive advantage.",
        "David 1985 QWERTY, standards battles, technological lock-in",
    ),
]

# ── Build dataframe ────────────────────────────────────────────────────────────

rows = []
item_id = 1

for depth_label, items in [("shallow", SHALLOW), ("medium", MEDIUM), ("deep", DEEP)]:
    for tw, sa, sb, gt, note in items:
        rows.append({
            "item_id":               f"E4_{item_id:03d}",
            "target_word":           tw,
            "sentence_a":            sa,
            "sentence_b":            sb,
            "ground_truth_reading":  gt,
            "inferential_depth":     depth_label,
            "is_critical_item":      "",   # filled Day 3
            "corpus_dominant_sense": "",   # filled Day 3
            "encyclopedic_target_sense": "", # filled Day 3
            "verification_source":   "",   # filled Day 3
            "zipf_frequency":        round(zipf_frequency(tw, "en"), 2),
            "note":                  note,
        })
        item_id += 1

df = pd.DataFrame(rows)

# ── Frequency check (tiered thresholds by depth) ──────────────────────────────
# Shallow >= 3.5 | Medium >= 3.2 | Deep >= 2.8
# Rationale: deep items require specialized historical vocabulary; LLM training
# data covers these well regardless of everyday-speech frequency.

THRESHOLDS = {"shallow": 3.5, "medium": 3.2, "deep": 2.8}

fail = df.apply(lambda r: r["zipf_frequency"] < THRESHOLDS[r["inferential_depth"]], axis=1)
low_freq = df[fail]
if not low_freq.empty:
    print("WARNING — items below tiered threshold:")
    for _, row in low_freq.iterrows():
        thr = THRESHOLDS[row["inferential_depth"]]
        print(f"  {row['item_id']}  {row['target_word']:<15s}  zipf={row['zipf_frequency']}  (threshold={thr}  depth={row['inferential_depth']})")
else:
    print("All items meet tiered zipf thresholds (shallow>=3.5, medium>=3.2, deep>=2.8).")

print(f"\nStrata counts:\n{df['inferential_depth'].value_counts().to_string()}")
print(f"\nTotal items: {len(df)}")

# ── Save ───────────────────────────────────────────────────────────────────────

out_path = "empirics/stimuli/exp4_minimal_pairs_draft.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("\nFirst 3 rows preview:")
print(df[["item_id","target_word","inferential_depth","zipf_frequency"]].head(20).to_string(index=False))
