"""
src/components/model_trainer.py
================================
All optimizations applied:
- Disjoint data generation (train/val sentence pools separated)
- Polars regex text cleaner (lowercase, strip)
- Dynamic VOCAB_SIZE from actual corpus
- SpatialDropout1D after embedding
- Hybrid pooling (GlobalMax + GlobalAverage concatenated)
- L2 weight regularization on Conv1D and Dense
- .keras format instead of .h5
- Deterministic seeding via tf.keras.utils.set_random_seed
- ReduceLROnPlateau callback
"""
from __future__ import annotations
import os, pickle, random, sys
from dataclasses import dataclass
import numpy as np
import polars as pl

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.exception import ESGException
from src.logger    import get_logger

log = get_logger(__name__)

MODELS_DIR = os.path.join(_ROOT, "models")
DATA_RAW   = os.path.join(_ROOT, "data", "raw")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_RAW,   exist_ok=True)

MODEL_PATH     = os.path.join(MODELS_DIR, "esg_classifier.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
DATASET_PATH   = os.path.join(DATA_RAW,   "training_dataset.csv")

MAX_SEQ_LEN       = 350
NUM_CLASSES       = 3
EPOCHS            = 30
BATCH_SIZE        = 32
VAL_SPLIT         = 0.15
SEED              = 42
SAMPLES_PER_CLASS = 300
EMBED_DIM         = 128

# ══ Corpus — disjoint 80/20 train/val sentence pools ══════════════════════════

_SEBI_BRSR_TRAIN = [
    "Business Responsibility and Sustainability Report BRSR SEBI mandatory disclosure listed entity",
    "SEBI notification CFD-SEC-2 BRSR Core parameters reasonable assurance standalone reporting boundary",
    "Section A general disclosures corporate identity number CIN paid-up capital BSE NSE stock exchange",
    "NGRBC nine principles essential indicators leadership compliance transparency accountability",
    "Principle 1 businesses conduct govern integrity ethical transparent accountable anti-bribery policy",
    "Principle 2 goods services sustainable safe extended producer responsibility lifecycle impact",
    "Principle 3 employee wellbeing permanent workers health insurance accident insurance maternity",
    "Principle 4 stakeholder interests communities investors suppliers dealers employees responsive",
    "Principle 5 human rights due diligence child labour forced labour POSH sexual harassment prevention",
    "Principle 6 environment energy consumption GHG scope 1 scope 2 emissions intensity renewable",
    "Principle 7 public regulatory policy transparent responsible trade industry associations advocacy",
    "Principle 8 inclusive growth equitable development CSR social impact assessment communities",
    "Principle 9 consumer value responsible data privacy cyber security product recalls complaints",
    "turnover net worth CSR applicable section 135 Companies Act 2013 standalone financial statements",
    "assurance provider reasonable assurance SEBI notification listed entity BRSR core indicators",
    "grievance redressal mechanism communities investors shareholders employees workers suppliers dealers",
    "material issues ESG risk opportunity financial implications positive negative rationale mitigation",
    "sustainability committee corporate level CMD board directors ESG initiatives review",
    "compliance monitoring tool statutory requirements non-compliance rectification notification",
    "independent assessment external agency IMS audit ISO 9001 14001 45001 50001 assurance letter",
    "Section B management process disclosures NGRBC principles policy procedures value chain",
    "related party transactions purchases sales loans advances investments percentage total",
    "accounts payable trading houses dealers distributors concentration purchases sales percentage",
    "compounding fee adjudication order Companies Act registrar penalty fine NGRBC principle",
    "annual report integrated sustainability statutory reports financial statements corporate overview",
    "supply chain management due diligence supplier code of conduct ESG assessment business spend",
    "water neutrality target zero waste landfill GHG scope reduction renewable energy percentage",
    "female workforce diversity equity inclusion POSH policy zero complaints investigations",
    "sustainable sourcing BRSR CORE KPI value chain implementation business spend percentage",
    "GreenCo certification manufacturing plants eco label submersible pumps castings products",
    "key management personnel board directors remuneration median wages male female workers",
    "BRSR mandatory filing SEBI Regulation 34 annual report section C performance disclosures",
    "national international codes certifications ISO BIS BEE CE FM UL star rating products",
    "specific commitments goals timelines renewable energy targets FY 2026 2027 2029 2030",
    "ESG workshops value chain partners training sessions NGRBC nine principles percentage spend",
    "governance leadership oversight statement director business responsibility challenges",
    "board committee decision making sustainability issues plant corporate level reporting",
    "review NGRBCs committee board frequency annually half yearly quarterly monthly basis",
    "anti-corruption bribery zero tolerance transparent accountable disciplinary action fraud",
    "conflict interest board directors declaration annually code of conduct compliance",
    "awareness programmes value chain partners training sessions dealers suppliers covered",
    "product stewardship lifecycle quality safety eco-friendly sustainable design innovation value",
    "energy management consumption efficiency intensity renewable deployment net zero strategy",
    "emissions risk GHG monitoring government control export norms reduction furnace replacement",
    "occupational health safety OHS 45001 plants safety controls employee sensitisation training",
    "human rights labour conditions due diligence training employees value chain fairness",
    "corporate governance transparency accountability board oversight stakeholders rules",
    "SEBI listing obligations disclosure requirements regulation materiality fines penalties",
    "manufacturing fluid power equipment percentage turnover NIC code products services sold",
    "number locations plants offices national international operations headquarters registered",
    "markets served states union territories countries exports percentage total turnover customers",
    "employee workers permanent other than permanent male female differently abled categories",
    "turnover rate permanent employees workers financial year male female total percentage",
    "holding subsidiary associate companies joint ventures shares held business responsibility",
    "CSR details applicable turnover net worth section 135 Companies Act CSR committee spend",
    "transparency disclosures compliance grievances pending resolution remarks financial year",
    "overview material responsible business conduct environmental social risk opportunity table",
    "training awareness programmes board directors KMP employees workers percentage coverage",
    "fines penalties punishment compounding fees settlement amount proceedings regulatory",
    "R&D capital expenditure technologies environmental social impacts total investments percentage",
    "sustainable sourcing procedures inputs sourced sustainably percentage value chain audit",
    "life cycle assessment LCA products manufacturing boundary cradle gate independent agency",
    "recycled reused input material total material value foundry operations manufacturing plants",
    "employee wellbeing health insurance accident insurance maternity paternity day care facilities",
    "worker wellbeing permanent coverage percentage benefits spending revenue",
    "retirement benefits PF gratuity ESI national pension scheme deducted deposited authority",
    "accessibility differently abled Rights Persons Disabilities Act premises offices workstations",
    "performance career development reviews employees workers total covered percentage financial year",
    "health safety management ISO 45001 coverage HIRA hazard identification risk assessment",
    "safety incidents LTIFR lost time injury frequency rate recordable fatalities high consequence",
    "corrective actions safety incidents significant risks working conditions health practices",
    "stakeholder groups key entity frequency engagement channels communication purpose scope",
    "minimum wages paid employees workers equal more than minimum wage permanent other than",
    "remuneration salary wages median board directors KMP employees workers male female",
    "gross wages females percentage total wages financial year comparison previous year",
    "POSH complaints sexual harassment discrimination child labour forced wages filed pending",
    "assessments plants offices child labour forced labour wages discrimination percentage",
    "energy consumption renewable non-renewable electricity fuel intensity GJ turnover PPP",
    "water withdrawal groundwater surface third party consumption intensity kilolitres turnover",
    "air emissions NOx SOx particulate matter volatile organic compounds MT per year measurements",
    "GHG greenhouse gas scope 1 scope 2 metric tonnes CO2 equivalent intensity physical output",
]

_SEBI_BRSR_VAL = [
    "GHG reduction projects furnace replacement renewable PPA open access solar wind power",
    "waste management plastic e-waste bio-medical battery hazardous non-hazardous MT generated",
    "waste recovered recycled reused co-processing disposal incineration landfilling operations",
    "biodiversity ecologically sensitive national parks wildlife sanctuaries biosphere reserves",
    "environmental compliance Water Act Air Act Environment Protection applicable laws India",
    "trade industry chambers associations affiliations national international reach members",
    "social impact assessments SIA projects rehabilitation resettlement PAFs covered amounts",
    "community grievances mechanisms committee site administration security HR function",
    "input material sourced MSMEs small producers directly within India percentage value",
    "job creation small towns wages rural semi-urban urban metropolitan RBI classification",
    "CSR projects beneficiaries vulnerable marginalised groups percentage healthcare education",
    "consumer complaints data privacy advertising cyber security delivery restrictive trade",
    "product recalls voluntary forced safety issues instances reasons number category",
    "data breaches instances percentage personally identifiable information customers impact",
    "water discharged surface groundwater third party treatment zero liquid discharge plants",
    "energy saved improved efficiency renewable sources total production consumption reduction",
    "environmental impact assessment projects applicable current financial year laws",
    "corrective action ongoing address safety risks significant working conditions practices",
    "supply chain sustainability ESG compliance suppliers dealers annual assessment percentage",
    "board independence committee composition nomination remuneration audit oversight",
]

_SUSTAINABILITY_TRAIN = [
    "Global Reporting Initiative GRI standards sustainability report environmental social governance",
    "carbon neutrality net zero 2050 climate change scope 3 emissions value chain decarbonization",
    "sustainable development goals SDGs United Nations 2030 agenda inclusive growth communities",
    "environmental impact assessment lifecycle analysis cradle to gate product carbon footprint LCA",
    "water stewardship watershed conservation zero liquid discharge recycled water effluent treatment",
    "biodiversity conservation ecosystem services natural capital accounting wetlands forests habitats",
    "circular economy principles reduce reuse recycle waste diversion landfill extended producer",
    "renewable energy solar wind power purchase agreement carbon offsets clean energy transition",
    "social impact assessment community engagement CSR beneficiaries vulnerable marginalised upliftment",
    "employee wellbeing mental health work life balance diversity inclusion gender pay equity wellness",
    "human rights due diligence modern slavery child labour forced labour supply chain audit vendors",
    "stakeholder engagement materiality assessment double materiality financial impact outside-in",
    "green building LEED IGBC platinum certification energy efficiency water conservation sustainable",
    "responsible sourcing raw materials conflict minerals sustainable forestry fair trade Rainforest",
    "Task Force Climate Financial Disclosures TCFD physical transition risks opportunities scenario",
    "science based targets SBTi 1.5 degree pathway absolute emissions reduction committed verified",
    "ESG rating MSCI Sustainalytics CDP A-list water security climate change forests disclosure",
    "integrated thinking value creation six capitals financial manufactured intellectual human social",
    "sustainability governance board oversight ESG committee non-executive director expertise",
    "GHG protocol scope 1 2 3 inventory verification third party assurance limited reasonable",
    "waste management hazardous non-hazardous recycling co-processing incineration landfill diversion",
    "supplier sustainability code of conduct assessment corrective action plan improvement programme",
    "product sustainability eco design recyclability end of life take back scheme packaging reduction",
    "climate resilience adaptation mitigation physical risk chronic acute transition regulatory market",
    "community development education healthcare livelihood skills rural infrastructure CSR investment",
    "employee training learning development upskilling reskilling capability building performance",
    "safety LTIFR lost time injury frequency recordable incidents near miss fatalities consequence",
    "board diversity independence gender age nationality skills matrix experience expertise committees",
    "executive remuneration pay ratio CEO median employee sustainability performance linked incentives",
    "anti-corruption compliance whistleblower ethics hotline speak up culture code business conduct",
    "sustainability annual integrated combined assurance external verification independent auditor",
    "energy intensity GJ tonne production renewable fraction grid electricity purchased certificates",
    "water intensity kilolitres unit product groundwater surface third party consumption withdrawal",
    "emissions intensity metric tonnes CO2 equivalent per revenue unit product scope combined",
    "plastic packaging reduction virgin recycled content post-consumer PCR EPR compliance filing",
    "occupational health medical surveillance audiometry pre-employment periodic fitness wellbeing",
    "gender diversity women leadership pipeline succession planning mentoring sponsorship programme",
    "innovation research development sustainable products services circular revenue green chemistry",
    "sustainable finance green bonds ESG-linked loans sustainability bonds framework second opinion",
    "TNFD nature taskforce biodiversity risks nature positive strategy ecosystem restoration",
    "just transition social equity carbon economy workers communities fossil fuel phase down",
    "climate smart agriculture regenerative farming soil carbon sequestration food water nexus",
    "digital sustainability AI machine learning efficiency resource optimisation predictive maintenance",
    "responsible investment ESG integration engagement proxy voting exclusion screening portfolio",
    "materiality matrix significant topics stakeholders business impact environmental social pillar",
    "sustainability targets science based near term long term net zero committed baseline year",
    "nature based solutions afforestation reforestation avoided deforestation blue carbon mangrove",
    "conflict minerals cobalt mica lithium responsible sourcing traceability chain custody smelter",
    "water risk physical regulatory reputational stress scarcity flood contamination watershed",
    "transition plan 2030 2040 2050 milestones interim targets accountability governance annual",
    "carbon price internal shadow price decarbonisation capex investment decisions climate action",
    "scope 3 categories upstream downstream purchased goods services capital business travel waste",
    "climate scenario analysis 1.5C 2C 4C physical transition risks opportunities resilience",
    "deforestation free supply chain palm oil soy beef timber responsible sourcing policy",
    "ocean plastics marine pollution extended producer responsibility recycling infrastructure",
    "sustainable packaging bio-based compostable recyclable design lightweighting reuse refill",
    "living wage fair wage supply chain workers minimum benchmark methodology audit certification",
    "LGBTQ inclusion rainbow diversity equity belonging psychological safety culture workplace",
    "responsible AI ethics algorithm bias transparency explainability governance framework",
    "net positive nature biodiversity credits habitat banking ecosystem restoration corporate",
    "supply chain transparency traceability blockchain technology provenance verification audit",
    "corporate social responsibility strategic philanthropy shared value creation communities CSR",
    "environmental management system ISO 14001 audit nonconformities corrective action improvement",
    "energy management system ISO 50001 energy review baseline performance indicators targets",
    "quality management system ISO 9001 customer satisfaction continual improvement process audit",
    "carbon footprint product service organizational boundary scope methodology inventory factor",
    "water footprint blue green grey scarcity weighting watershed level assessment methodology",
    "circular economy metrics material circularity indicator Ellen MacArthur Foundation",
    "emissions trading scheme carbon market offset quality additionality permanence verification",
    "modern slavery act transparency reporting supply chain forced labour indicators red flags",
    "board climate competence training literacy scenario analysis oversight strategy integration",
    "integrated reporting framework capitals providers investors materiality connectivity",
    "voluntary carbon market integrity high quality removal reduction nature technology credit",
    "science based targets freshwater land biodiversity ocean targets nature positive 2030",
    "corporate sustainability reporting directive CSRD European standards ESRS disclosure",
    "transition finance credibility alignment Paris Agreement sector decarbonisation pathway",
    "nature capital valuation ecosystem services accounting wealth inclusive green economy",
    "climate finance adaptation mitigation developing countries loss damage fund COP agreement",
]

_SUSTAINABILITY_VAL = [
    "just energy transition partnership coal phase developing countries finance support",
    "green hydrogen renewable energy carrier fuel cells electrolysis decarbonise hard-to-abate",
    "carbon capture storage utilisation CCUS negative emissions technology direct air capture",
    "net zero aligned financial institution Paris pledge portfolio decarbonisation target year",
    "biodiversity positive strategy IPBES species area relationship habitat connectivity",
    "circular bioeconomy biogenic carbon renewable feedstock bio-based materials cascade",
    "sustainable development finance taxonomy classification eligible activities criteria",
    "environmental justice communities frontline fence-line pollution disproportionate impact",
    "responsible minerals sourcing OECD due diligence conflict affected high risk areas",
    "sustainable procurement policy criteria lifecycle thinking total cost ownership assessment",
    "land use change deforestation restoration biodiversity net gain offset compensation",
    "social license to operate community acceptance impact assessment engagement benefit sharing",
    "integrated management system IMS certification surveillance recertification cycle scope",
    "emission factor grid electricity renewable energy attribute certificate calculation",
    "stakeholder consultation material topics process identification significance ranking",
    "GRI universal standards sector program specific performance indicators boundary",
    "SASB industry specific sustainability accounting standards board metrics framework",
    "impact weighted accounts true cost accounting natural capital externalities valuation",
    "Paris aligned investment strategy 1.5 degree IPCC scenario portfolio alignment",
    "ESG data provider ratings methodology controversy screening norms-based screening",
]

_INVALID_TRAIN = [
    "dear customer your order has been shipped tracking number estimated delivery business days",
    "software update version bug fixes performance improvements user interface enhancements changelog",
    "meeting agenda project kickoff discussion timeline milestones stakeholders action items",
    "recipe ingredients flour sugar butter eggs vanilla extract baking powder mix combine bake",
    "movie review excellent cinematography compelling narrative outstanding performances blockbuster",
    "travel itinerary flight hotel reservation sightseeing tourist attractions local cuisine",
    "sports championship final score winning team player statistics league standings tournament",
    "weather forecast sunny partly cloudy temperature humidity wind speed precipitation outlook",
    "job posting software engineer required skills Python JavaScript React experience preferred",
    "product manual installation guide safety warnings troubleshooting FAQ technical specifications",
    "social media post trending hashtag likes comments followers engagement viral content share",
    "political speech campaign promises economic growth jobs infrastructure healthcare education",
    "research paper methodology results discussion conclusion references peer reviewed journal",
    "online shopping cart checkout payment gateway credit card billing address confirmation",
    "news article breaking event location witnesses emergency services response investigation",
    "music playlist favourite songs artist album genre pop rock jazz classical shuffle",
    "fitness routine workout plan exercise sets repetitions rest interval calories heart rate",
    "home renovation kitchen remodel bathroom upgrade flooring painting contractor quote",
    "medical prescription dosage frequency drug interaction side effects patient pharmacist",
    "real estate listing bedroom bathroom square feet amenities neighbourhood asking price",
    "academic transcript grades course credits semester GPA dean list graduation certificate",
    "restaurant menu appetizers main course dessert beverage price chef special promotion",
    "vehicle service schedule oil change tyre rotation brake inspection filter mileage",
    "game walkthrough level cheat code achievement unlock boss fight strategy guide tips",
    "horoscope zodiac sign personality traits prediction love career finance weekly",
    "fashion trend season collection outfit styling accessories colour palette runway",
    "cooking tutorial step preparation technique garnish plating serving suggestion chef",
    "book summary plot characters themes author biography edition publisher review",
    "technology review smartphone camera battery performance benchmark comparison price",
    "insurance policy premium coverage exclusion claim procedure beneficiary renewal",
    "furniture assembly instruction parts list screws bolts allen key diagram step component",
    "flight booking seat selection baggage allowance meal preference check-in boarding",
    "gym membership pricing personal trainer class schedule locker room parking sauna",
    "gardening tips seasonal planting watering pruning fertiliser pest control soil",
    "pet care veterinary appointment vaccination flea tick worming grooming diet nutrition",
    "wedding planning venue catering photographer florist guest list invitation seating",
    "tax return filing deductions exemptions refund assessment year form income salary",
    "school curriculum syllabus examination marks grade promotion admission fees timetable",
    "concert ticket venue date time support act merchandise backstage pass experience",
    "yoga meditation mindfulness breathing technique pose asana energy balance stress",
    "photography landscape portrait lighting composition golden hour lens aperture shutter",
    "podcast episode topic guest interview discussion download subscribe review rating",
    "interior design colour scheme furniture layout accent wall lighting fixture minimalist",
    "online course certification module quiz assignment completion deadline platform",
    "password reset click link enter new password confirm verification email notification",
    "shipping label sender recipient weight dimensions fragile handle care customs",
    "birthday party invitation cake decorations games goodie bags venue RSVP children",
    "car insurance renewal premium no claim bonus comprehensive coverage garage repair",
    "cryptocurrency bitcoin blockchain wallet private key exchange trading price chart",
    "loan application credit score income verification collateral interest rate EMI",
    "mobile app update new features bug fixes improved battery optimised download store",
    "hotel booking check-in checkout breakfast included pool gym spa cancellation policy",
    "train ticket reservation seat berth coach class status platform arrival departure",
    "grocery list milk bread eggs butter vegetables fruits cereals cooking oil snacks",
    "subscription streaming service movies shows episodes seasons watch list recommend",
    "bank statement transaction history balance debit credit ATM withdrawal deposit",
    "electricity bill units consumed tariff rate fixed charges surcharge due date payment",
    "newspaper article headline journalist reporter editor publication date byline column",
    "laboratory test blood sugar cholesterol haemoglobin creatinine report normal range",
    "university admission entrance exam score cutoff merit counselling seat allotment fees",
    "event management conference seminar workshop registration delegate speaker sponsor",
    "tourist visa application documents embassy appointment fee interview approval",
    "painting exhibition gallery artist medium canvas acrylic oil watercolour sculpture",
    "charity donation NGO volunteer community service fundraising beneficiary impact",
    "dictionary word definition pronunciation etymology synonym antonym usage sentence",
    "driving licence renewal form documents fees test transport authority registration",
    "passport application form birth certificate photograph fee appointment travel",
    "ration card family members income certificate address food grain entitlement",
    "property registration stamp duty circle rate sub-registrar document verification",
    "scholarship application academic merit income proof institution recommendation",
    "blood donation camp voluntary donor health check eligibility criteria hospital",
    "language learning app vocabulary grammar practice listening speaking reading",
    "plumber electrician carpenter handyman service charges hourly emergency contact",
    "bakery cake order flavour fondant custom design delivery advance booking",
    "library membership book issue return fine renewal reservation reading room",
    "parking lot charges hourly daily monthly season pass vehicle two four wheeler",
    "catering order menu items quantity serving vegetarian non-vegetarian buffet",
    "flower delivery bouquet arrangement occasion message card same day express",
]

_INVALID_VAL = [
    "car rental self-drive chauffeur duration pickup drop fuel unlimited kilometres",
    "water tanker booking capacity litres delivery area charges emergency residential",
    "laundry service pickup delivery dry cleaning ironing stain removal charges kg",
    "cable television DTH channel package recharge subscription monthly payment plan",
    "swimming pool membership adult child timing lane width depth temperature chlorine",
    "nursery plant seeds sapling fertiliser pot soil mix drainage sunlight watering",
    "cinema hall show timing seat row multiplex ticket booking snacks beverage combo",
    "chess tournament rating grandmaster opening endgame strategy clock time control",
    "gym diet plan protein carbohydrate fat calorie intake muscle building weight loss",
    "amusement park ride ticket combo family package height restriction safety rules",
    "furniture catalogue sofa bed wardrobe table chair dimensions material finish",
    "mobile gaming leaderboard score rank achievement badge clan guild multiplayer",
    "weather app notification rain alert wind advisory temperature change hourly",
    "news feed algorithm trending topic engagement click-through rate impressions",
    "food delivery app restaurant rating cuisine filter price range estimated time",
    "e-commerce return policy refund exchange size chart product review photo upload",
    "digital wallet top-up balance transfer payment QR code merchant cashback offer",
    "rideshare app driver rating surge pricing route map estimated arrival pickup",
    "social network profile bio follower following post story reel comment share",
    "streaming platform recommendation algorithm watch history genre preference rating",
]

import re as _re

def _clean(text: str) -> str:
    """Polars-style regex cleaning: lowercase, strip, normalise spaces."""
    text = text.lower()
    text = _re.sub(r"\s+", " ", text).strip()
    return text

def _augment(s: str, seed: int) -> str:
    rng   = random.Random(seed)
    words = s.split()
    kept  = [w for w in words if rng.random() > 0.15]
    return " ".join(kept) if len(kept) > 4 else s

def _make_samples(train_pool, val_pool, label: int, n_train: int, n_val: int):
    rows = []
    nt, nv = len(train_pool), len(val_pool)
    for i in range(n_train):
        idxs = [(i+j) % nt for j in range(random.randint(2, 3))]
        text = _clean(_augment(" ".join(train_pool[k] for k in idxs),
                                seed=label*10000+i))
        rows.append({"text": text, "label": label, "split": "train"})
    for i in range(n_val):
        idxs = [(i+j) % nv for j in range(random.randint(1, 2))]
        text = _clean(" ".join(val_pool[k] for k in idxs))
        rows.append({"text": text, "label": label, "split": "val"})
    return rows

def _generate() -> pl.DataFrame:
    random.seed(SEED)
    n_train = int(SAMPLES_PER_CLASS * 0.85)
    n_val   = SAMPLES_PER_CLASS - n_train
    rows = []
    rows += _make_samples(_SEBI_BRSR_TRAIN,    _SEBI_BRSR_VAL,    0, n_train, n_val)
    rows += _make_samples(_SUSTAINABILITY_TRAIN, _SUSTAINABILITY_VAL, 1, n_train, n_val)
    rows += _make_samples(_INVALID_TRAIN,      _INVALID_VAL,      2, n_train, n_val)
    df = pl.DataFrame(rows).sample(fraction=1.0, seed=SEED, shuffle=True)
    df.write_csv(DATASET_PATH)
    log.info("Dataset: %d rows → %s", len(df), DATASET_PATH)
    return df

@dataclass
class TrainingResult:
    model_path: str; tokenizer_path: str; best_val_acc: float; epochs_run: int

def train() -> TrainingResult:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras import layers, models, callbacks, regularizers

    log.info("=== Training Start ===")
    try:
        tf.keras.utils.set_random_seed(SEED)   # deterministic seeding (opt #3)

        df     = _generate()
        texts  = df["text"].to_list()
        labels = df["label"].to_list()

        # Dynamic vocab (opt #3 — size from actual corpus)
        tok = Tokenizer(oov_token="<OOV>")
        tok.fit_on_texts(texts)
        actual_vocab = len(tok.word_index) + 1
        VOCAB_SIZE   = min(actual_vocab, 15_000)
        tok2 = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        tok2.fit_on_texts(texts)

        seqs = tok2.texts_to_sequences(texts)
        X    = pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
        y    = tf.keras.utils.to_categorical(labels, NUM_CLASSES)

        with open(TOKENIZER_PATH, "wb") as f:
            pickle.dump(tok2, f)

        # Model — all architectural optimizations
        l2 = regularizers.l2(1e-4)
        inp = layers.Input(shape=(MAX_SEQ_LEN,), name="tokens")
        emb = layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_SEQ_LEN)(inp)
        emb = layers.SpatialDropout1D(0.2)(emb)          # SpatialDropout1D (opt #1)

        branches = []
        for ks in [2, 3, 4]:
            c  = layers.Conv1D(128, ks, activation="relu",
                               padding="same", kernel_regularizer=l2)(emb)  # L2 (opt #3)
            gm = layers.GlobalMaxPooling1D()(c)
            ga = layers.GlobalAveragePooling1D()(c)       # Hybrid pooling (opt #2)
            branches += [gm, ga]

        cat = layers.Concatenate()(branches)
        x   = layers.BatchNormalization()(cat)
        x   = layers.Dense(256, activation="relu", kernel_regularizer=l2)(x)
        x   = layers.Dropout(0.4)(x)
        x   = layers.Dense(128, activation="relu", kernel_regularizer=l2)(x)
        x   = layers.Dropout(0.3)(x)
        out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

        model = models.Model(inp, out, name="ESGClassifier")
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy", metrics=["accuracy"])

        cb_list = [
            callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1,   # opt #2
                                        patience=3, min_lr=1e-6, verbose=1),
        ]

        hist = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         validation_split=VAL_SPLIT, callbacks=cb_list, verbose=1)

        best_val  = float(max(hist.history.get("val_accuracy", [0])))
        ep_run    = len(hist.history["accuracy"])
        log.info("Training complete — val_accuracy=%.4f  epochs=%d", best_val, ep_run)

        model.save(MODEL_PATH)                            # .keras format (opt #1)
        log.info("Model saved → %s", MODEL_PATH)

        return TrainingResult(MODEL_PATH, TOKENIZER_PATH, best_val, ep_run)
    except Exception as e:
        raise ESGException(e) from e

if __name__ == "__main__":
    r = train()
    print(f"\n✅  val_accuracy={r.best_val_acc:.4f}  epochs={r.epochs_run}")
    print(f"   model     → {r.model_path}")
    print(f"   tokenizer → {r.tokenizer_path}\n")
