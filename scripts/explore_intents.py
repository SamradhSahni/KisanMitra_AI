import json
import os
import sys
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

log_path = Path("logs/explore_intents.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")


# ── Load Dataset ─────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> pd.DataFrame:
    records = []
    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            break
        except UnicodeDecodeError:
            records = []
            continue
    return pd.DataFrame(records)


# ── Keyword Banks ────────────────────────────────────────────────────
# These are BROAD discovery keywords — we use them to find what's in
# the data, then you confirm which intents to keep

INTENT_KEYWORDS = {

    "pest_id": {
        "english": [
            "pest", "insect", "worm", "larvae", "aphid", "thrips",
            "army worm", "fall army", "caterpillar", "beetle", "mite",
            "whitefly", "jassid", "termite", "locust", "grasshopper",
            "stem borer", "shoot borer", "leaf folder", "sucking pest",
            "mealybug", "scale insect", "bug", "weevil", "nematode"
        ],
        "hindi": [
            "कीट", "पतंग", "इल्ली", "सुंडी", "माहू", "थ्रिप्स",
            "दीमक", "टिड्डी", "मच्छर", "सफेद मक्खी", "तना छेदक",
            "पत्ती मोड़क", "मिलीबग", "घुन", "सूत्रकृमि", "जैसिड"
        ]
    },

    "disease": {
        "english": [
            "disease", "blight", "fungus", "rust", "wilt", "virus",
            "bacteria", "rot", "mildew", "smut", "scab", "canker",
            "mosaic", "yellowing", "leaf spot", "damping off",
            "sheath blight", "blast", "bacterial leaf", "powdery",
            "downy", "stem rot", "root rot", "crown rot", "neck rot",
            "charcoal rot", "collar rot", "white mold", "early blight",
            "late blight", "alternaria", "cercospora", "fusarium",
            "phytophthora", "pythium", "rhizoctonia", "sclerotinia"
        ],
        "hindi": [
            "रोग", "झुलसा", "ब्लाइट", "फंगस", "फफूंद", "वायरस",
            "बैक्टीरिया", "सड़न", "गलन", "पीला", "धब्बा", "किट्ट",
            "उकठा", "मुरझाना", "चूर्णिल", "मृदु रोमिल", "कंड",
            "आर्द्र गलन", "आभासी कंड"
        ]
    },

    "crop_advisory": {
        "english": [
            "sowing", "irrigation", "fertilizer", "harvest", "seed",
            "cultivation", "planting", "spacing", "thinning", "pruning",
            "variety", "hybrid", "nursery", "transplant", "intercrop",
            "crop rotation", "soil preparation", "land preparation",
            "basal dose", "top dressing", "drip", "sprinkler",
            "flood irrigation", "recommended variety", "seed rate",
            "germination", "crop stage", "maturity", "yield",
            "production", "package of practice", "package", "agronomy",
            "weed", "weeding", "herbicide", "mulching", "staking",
            "trellis", "pollination", "grafting", "budding", "cutting"
        ],
        "hindi": [
            "बुवाई", "सिंचाई", "खाद", "उर्वरक", "कटाई", "फसल",
            "बीज", "रोपाई", "प्रत्यारोपण", "किस्म", "प्रजाति",
            "नर्सरी", "मिश्रित खेती", "खरपतवार", "निराई", "गुड़ाई",
            "मल्चिंग", "परागण", "कलम", "छंटाई", "विरलन",
            "अनुशंसित", "उत्पादन", "उपज", "पैदावार", "पौध",
            "भूमि तैयारी", "आधार खुराक", "टॉप ड्रेसिंग"
        ]
    },

    "msp_price": {
        "english": [
            "msp", "minimum support price", "price", "market rate",
            "rate", "procurement", "mandi", "market price", "selling",
            "purchase", "government price", "support price", "pmfby",
            "pm fasal bima", "crop insurance", "insurance", "compensation",
            "claim", "subsidy", "pm kisan", "kcc", "kisan credit",
            "loan", "credit", "bank", "financial", "scheme benefit"
        ],
        "hindi": [
            "समर्थन मूल्य", "एमएसपी", "कीमत", "दाम", "बाजार",
            "खरीद", "मंडी", "बिक्री", "सरकारी मूल्य", "बीमा",
            "मुआवजा", "सब्सिडी", "ऋण", "कर्ज", "बैंक", "योजना",
            "लाभ", "पीएम किसान", "फसल बीमा", "क्षतिपूर्ति"
        ]
    },

    "weather_sowing": {
        "english": [
            "weather", "rain", "temperature", "wind", "humidity",
            "forecast", "rainfall", "monsoon", "drought", "flood",
            "hail", "frost", "cold wave", "heat wave", "cloud",
            "sunny", "cloudy", "climate", "season", "kharif", "rabi",
            "zaid", "pre kharif", "sowing time", "best time to sow"
        ],
        "hindi": [
            "मौसम", "बारिश", "तापमान", "हवा", "आर्द्रता",
            "पूर्वानुमान", "वर्षा", "मानसून", "सूखा", "बाढ़",
            "ओला", "पाला", "शीत लहर", "लू", "खरीफ", "रबी",
            "जायद", "बुवाई का समय", "मेघ", "धूप"
        ]
    },

    "soil_water": {
        "english": [
            "soil", "soil health", "soil test", "ph", "organic matter",
            "compost", "vermicompost", "green manure", "biofertilizer",
            "micronutrient", "zinc", "boron", "iron deficiency",
            "salinity", "alkalinity", "waterlogging", "drainage",
            "soil moisture", "groundwater", "borewell", "well",
            "canal", "water management", "water conservation",
            "rainwater harvesting", "fertigation", "soil erosion"
        ],
        "hindi": [
            "मिट्टी", "भूमि", "मृदा", "पीएच", "जैविक खाद",
            "कंपोस्ट", "वर्मी कंपोस्ट", "हरी खाद", "जैव उर्वरक",
            "सूक्ष्म पोषक", "जस्ता", "बोरान", "लोहे की कमी",
            "लवणता", "क्षारीयता", "जलभराव", "जल निकास",
            "नमी", "भूजल", "बोरवेल", "कुआं", "नहर", "जल संरक्षण"
        ]
    },

    "government_scheme": {
        "english": [
            "scheme", "government scheme", "yojana", "pradhan mantri",
            "pm", "subsidy scheme", "grant", "beneficiary", "apply",
            "application", "registration", "portal", "online apply",
            "common service centre", "csc", "agriculture department",
            "krishi vibhag", "nabard", "atma", "kvk", "kisan call",
            "helpline", "toll free", "contact number", "department"
        ],
        "hindi": [
            "योजना", "सरकारी योजना", "प्रधानमंत्री", "सब्सिडी",
            "अनुदान", "लाभार्थी", "आवेदन", "पंजीकरण", "पोर्टल",
            "ऑनलाइन", "सामान्य सेवा केंद्र", "कृषि विभाग",
            "नाबार्ड", "आत्मा", "केवीके", "हेल्पलाइन",
            "टोल फ्री", "संपर्क", "विभाग"
        ]
    },

    "post_harvest": {
        "english": [
            "storage", "post harvest", "grading", "sorting", "packaging",
            "cold storage", "warehouse", "godown", "milling", "processing",
            "value addition", "drying", "cleaning", "threshing",
            "winnowing", "moisture content", "shelf life", "export",
            "marketing", "agro processing", "food processing"
        ],
        "hindi": [
            "भंडारण", "कटाई के बाद", "ग्रेडिंग", "छंटाई",
            "पैकेजिंग", "शीत भंडार", "गोदाम", "प्रसंस्करण",
            "मूल्य संवर्धन", "सुखाना", "ओसाई", "नमी",
            "मार्केटिंग", "खाद्य प्रसंस्करण", "निर्यात"
        ]
    },

    "animal_husbandry": {
        "english": [
            "animal", "cattle", "cow", "buffalo", "goat", "sheep",
            "poultry", "fish", "fishery", "dairy", "veterinary",
            "vaccination", "livestock", "fodder", "feed", "milk",
            "breeding", "disease treatment animal", "pashu", "murgi",
            "bakri", "machli", "dugdh"
        ],
        "hindi": [
            "पशु", "गाय", "भैंस", "बकरी", "भेड़", "मुर्गी",
            "मछली", "मत्स्य", "डेयरी", "पशु चिकित्सा", "टीकाकरण",
            "पशुपालन", "चारा", "दूध", "प्रजनन", "पशु रोग",
            "मुर्गीपालन", "बकरीपालन"
        ]
    },

    "horticulture": {
        "english": [
            "horticulture", "fruit", "vegetable", "flower", "orchard",
            "mango", "banana", "citrus", "guava", "papaya", "pomegranate",
            "apple", "grape", "strawberry", "tomato", "potato",
            "onion", "garlic", "capsicum", "brinjal", "cauliflower",
            "cabbage", "cucumber", "gourd", "spice", "turmeric",
            "ginger", "chilli", "floriculture", "rose", "marigold"
        ],
        "hindi": [
            "बागवानी", "फल", "सब्जी", "फूल", "बाग", "आम",
            "केला", "नींबू", "अमरूद", "पपीता", "अनार", "सेब",
            "अंगूर", "टमाटर", "आलू", "प्याज", "लहसुन",
            "मिर्च", "बैंगन", "फूलगोभी", "पत्तागोभी",
            "हल्दी", "अदरक", "मसाले", "पुष्प उत्पादन"
        ]
    },

    "organic_farming": {
        "english": [
            "organic", "organic farming", "natural farming", "zero budget",
            "zbnf", "jeevamrit", "beejamrit", "natural pesticide",
            "bio pesticide", "neem", "neem oil", "cow dung", "cow urine",
            "botanical", "pheromone trap", "sticky trap", "light trap",
            "ipm", "integrated pest management", "integrated farming"
        ],
        "hindi": [
            "जैविक", "प्राकृतिक खेती", "शून्य बजट", "जीवामृत",
            "बीजामृत", "नीम", "नीम तेल", "गोबर", "गोमूत्र",
            "वनस्पति", "फेरोमोन ट्रैप", "जैव कीटनाशक",
            "एकीकृत कीट प्रबंधन", "एकीकृत खेती"
        ]
    },

    "equipment_machinery": {
        "english": [
            "tractor", "machine", "equipment", "implement", "tool",
            "thresher", "harvester", "combine", "sprayer", "pump",
            "drip system", "sprinkler system", "power tiller",
            "rotavator", "plough", "seed drill", "reaper", "baler",
            "custom hiring", "farm machinery", "mechanization",
            "rental", "hire center"
        ],
        "hindi": [
            "ट्रैक्टर", "मशीन", "उपकरण", "यंत्र", "थ्रेशर",
            "हार्वेस्टर", "कंबाइन", "स्प्रेयर", "पंप",
            "ड्रिप सिस्टम", "पावर टिलर", "रोटावेटर", "हल",
            "सीड ड्रिल", "रीपर", "कस्टम हायरिंग", "किराया"
        ]
    },

    "seed_variety": {
        "english": [
            "seed variety", "improved variety", "certified seed",
            "hybrid seed", "foundation seed", "breeder seed",
            "seed treatment", "seed rate", "seed selection",
            "seed germination", "seed health", "quality seed",
            "new variety", "high yielding variety", "disease resistant"
        ],
        "hindi": [
            "बीज किस्म", "उन्नत किस्म", "प्रमाणित बीज",
            "संकर बीज", "बीज उपचार", "बीज दर", "बीज चयन",
            "बीज अंकुरण", "गुणवत्ता बीज", "नई किस्म",
            "अधिक उपज", "रोग प्रतिरोधी"
        ]
    },
}


# ── Intent Discovery ─────────────────────────────────────────────────
def detect_intent(text: str, keywords: dict) -> str:
    """Match text against keyword banks, return matched intent or 'unknown'."""
    if not text or not isinstance(text, str):
        return "unknown"
    text_lower = text.lower()
    for intent, kw_banks in keywords.items():
        for lang, words in kw_banks.items():
            for word in words:
                if word.lower() in text_lower:
                    return intent
    return "unknown"


def run_intent_discovery(df: pd.DataFrame, sample_size: int = 200000):
    """
    Run intent detection on both query and answer fields.
    Use a sample for speed on large datasets.
    """
    logger.info(f"Running intent discovery on {min(sample_size, len(df)):,} records...")

    # Sample for speed
    sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()

    # Detect intent from query (English) and answer (Hindi)
    sample["intent_from_query"]  = sample["query"].apply(
        lambda x: detect_intent(str(x), INTENT_KEYWORDS)
    )
    sample["intent_from_answer"] = sample["answer"].apply(
        lambda x: detect_intent(str(x), INTENT_KEYWORDS)
    )

    # Combine: prefer query detection, fallback to answer detection
    sample["final_intent"] = sample.apply(
        lambda row: row["intent_from_query"]
        if row["intent_from_query"] != "unknown"
        else row["intent_from_answer"],
        axis=1
    )

    return sample


# ── Print Intent Report ──────────────────────────────────────────────
def print_intent_report(sample: pd.DataFrame, total_records: int):

    sep = "=" * 60

    print(f"\n{sep}")
    print("  KisanMitra AI — Intent Discovery Report")
    print(sep)
    print(f"  Sample size : {len(sample):,} records")
    print(f"  Total dataset: {total_records:,} records")

    # ── Intent from Query field ──
    print(f"\n{sep}")
    print("  Intent detected from QUERY field (English text)")
    print(sep)
    q_counts = sample["intent_from_query"].value_counts()
    for intent, count in q_counts.items():
        pct  = (count / len(sample)) * 100
        bar  = "█" * int(pct / 1.5)
        estimated = int((count / len(sample)) * total_records)
        print(f"  {intent:<25} {count:>8,}  ({pct:5.1f}%)  ~{estimated:,} total  {bar}")

    # ── Intent from Answer field ──
    print(f"\n{sep}")
    print("  Intent detected from ANSWER field (Hindi text)")
    print(sep)
    a_counts = sample["intent_from_answer"].value_counts()
    for intent, count in a_counts.items():
        pct  = (count / len(sample)) * 100
        bar  = "█" * int(pct / 1.5)
        estimated = int((count / len(sample)) * total_records)
        print(f"  {intent:<25} {count:>8,}  ({pct:5.1f}%)  ~{estimated:,} total  {bar}")

    # ── Final Combined Intent ──
    print(f"\n{sep}")
    print("  Final Combined Intent Distribution")
    print(sep)
    f_counts = sample["final_intent"].value_counts()
    for intent, count in f_counts.items():
        pct  = (count / len(sample)) * 100
        bar  = "█" * int(pct / 1.5)
        estimated = int((count / len(sample)) * total_records)
        print(f"  {intent:<25} {count:>8,}  ({pct:5.1f}%)  ~{estimated:,} total  {bar}")

    # ── Unknown breakdown — what are we missing? ──
    print(f"\n{sep}")
    print("  'Unknown' Intent — Sample Queries (help identify gaps)")
    print(sep)
    unknowns = sample[sample["final_intent"] == "unknown"]
    print(f"  Total unknown: {len(unknowns):,} ({(len(unknowns)/len(sample))*100:.1f}%)")
    print(f"\n  20 sample unknown queries:")
    for i, row in unknowns.head(20).iterrows():
        print(f"  [{i}] Q: {str(row['query'])[:80]}")
        print(f"       A: {str(row['answer'])[:80]}")
        print()

    # ── Per-crop intent breakdown ──
    print(f"\n{sep}")
    print("  Intent Distribution by Top 10 Crops")
    print(sep)
    top_crops = sample["crop"].value_counts().head(10).index.tolist()
    for crop in top_crops:
        crop_df = sample[sample["crop"] == crop]
        top_intent = crop_df["final_intent"].value_counts().head(3)
        print(f"\n  {crop} ({len(crop_df):,} records):")
        for intent, count in top_intent.items():
            pct = (count / len(crop_df)) * 100
            print(f"    {intent:<25} {count:>6,}  ({pct:5.1f}%)")

    # ── Intent coverage by state ──
    print(f"\n{sep}")
    print("  Intent Distribution by State")
    print(sep)
    for state in sample["state"].value_counts().head(8).index:
        state_df = sample[sample["state"] == state]
        top_intent = state_df["final_intent"].value_counts().head(3)
        print(f"\n  {state} ({len(state_df):,} records):")
        for intent, count in top_intent.items():
            pct = (count / len(state_df)) * 100
            print(f"    {intent:<25} {count:>6,}  ({pct:5.1f}%)")

    print(f"\n{sep}")
    print("  Intent Discovery Complete")
    print(sep)

    return f_counts


# ── Save Intent Counts ───────────────────────────────────────────────
def save_intent_report(f_counts: pd.Series, sample_size: int, total_records: int):
    out = {
        "sample_size": sample_size,
        "total_records": total_records,
        "intent_distribution": {},
    }
    for intent, count in f_counts.items():
        pct = round((count / sample_size) * 100, 2)
        estimated = int((count / sample_size) * total_records)
        out["intent_distribution"][intent] = {
            "sample_count": int(count),
            "sample_pct": pct,
            "estimated_total": estimated
        }

    out_path = Path("data/processed/intent_distribution.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.success(f"Intent report saved to: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_path = os.getenv("RAW_DATA_PATH", "./data/raw/kcc_dataset.jsonl")

    logger.info("Loading dataset for intent exploration...")
    df = load_jsonl(raw_path)
    logger.success(f"Loaded {len(df):,} records")

    sample = run_intent_discovery(df, sample_size=200000)

    f_counts = print_intent_report(sample, total_records=len(df))

    save_intent_report(f_counts, sample_size=len(sample), total_records=len(df))

    logger.success("Intent exploration complete.")