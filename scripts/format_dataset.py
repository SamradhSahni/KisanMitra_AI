import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

log_path = Path("logs/format_dataset.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

INPUT_PATH  = "./data/processed/translated_dataset.jsonl"
OUTPUT_PATH = "./data/processed/formatted_dataset.jsonl"

# ── Crop name Hindi mapping ──────────────────────────────────────────
# Translates English crop names to Hindi for the instruction template
CROP_HINDI_MAP = {
    "others":                                          "अन्य",
    "paddy (dhan)":                                    "धान",
    "wheat":                                           "गेहूं",
    "maize (makka)":                                   "मक्का",
    "green gram (moong bean/ moong)":                  "मूंग",
    "pearl millet (bajra/bulrush millet/spiked millet)": "बाजरा",
    "mustard":                                         "सरसों",
    "sugarcane (noble cane)":                          "गन्ना",
    "cotton (kapas)":                                  "कपास",
    "groundnut (pea nut/mung phalli)":                 "मूंगफली",
    "mango":                                           "आम",
    "potato":                                          "आलू",
    "onion":                                           "प्याज",
    "bengal gram (gram/chick pea/kabuli/chana)":       "चना",
    "tomato":                                          "टमाटर",
    "guar":                                            "ग्वार",
    "soybean (bhat)":                                  "सोयाबीन",
    "cumin":                                           "जीरा",
    "black gram (urd bean)":                           "उड़द",
    "chillies":                                        "मिर्च",
    "barley":                                          "जौ",
    "lentil (masoor)":                                 "मसूर",
    "arhar (tur/red gram/pigeon pea)":                 "अरहर",
    "coriander":                                       "धनिया",
    "fennel":                                          "सौंफ",
    "garlic":                                          "लहसुन",
    "sunflower":                                       "सूरजमुखी",
    "sesame (til)":                                    "तिल",
    "linseed":                                         "अलसी",
    "castor":                                          "अरंडी",
    "isabgol":                                         "ईसबगोल",
    "asaliya":                                         "असलिया",
    "tinda":                                           "टिंडा",
    "bitter gourd":                                    "करेला",
    "bottle gourd":                                    "लौकी",
    "brinjal":                                         "बैंगन",
    "cabbage":                                         "पत्तागोभी",
    "cauliflower":                                     "फूलगोभी",
    "cucumber":                                        "खीरा",
    "pea":                                             "मटर",
    "spinach":                                         "पालक",
    "radish":                                          "मूली",
    "carrot":                                          "गाजर",
    "capsicum":                                        "शिमला मिर्च",
    "papaya":                                          "पपीता",
    "banana":                                          "केला",
    "guava":                                           "अमरूद",
    "pomegranate":                                     "अनार",
    "apple":                                           "सेब",
    "grape":                                           "अंगूर",
    "citrus":                                          "नींबू",
    "coconut":                                         "नारियल",
    "turmeric":                                        "हल्दी",
    "ginger":                                          "अदरक",
    "rose":                                            "गुलाब",
    "marigold":                                        "गेंदा",
}

# ── Intent Hindi display names ───────────────────────────────────────
INTENT_HINDI_MAP = {
    "weather_sowing":       "मौसम एवं बुवाई",
    "crop_advisory":        "फसल सलाह",
    "pest_id":              "कीट प्रबंधन",
    "disease":              "रोग प्रबंधन",
    "nutrient_management":  "पोषक तत्व प्रबंधन",
    "msp_price":            "मूल्य एवं बाजार",
    "government_scheme":    "सरकारी योजना",
    "horticulture":         "बागवानी",
    "soil_water":           "मृदा एवं जल प्रबंधन",
    "animal_husbandry":     "पशुपालन",
    "equipment_machinery":  "कृषि यंत्र",
    "unknown":              "सामान्य",
}

# ── State Hindi display names ────────────────────────────────────────
STATE_HINDI_MAP = {
    "UTTAR PRADESH":   "उत्तर प्रदेश",
    "RAJASTHAN":       "राजस्थान",
    "MADHYA PRADESH":  "मध्य प्रदेश",
    "BIHAR":           "बिहार",
    "HARYANA":         "हरियाणा",
    "JHARKHAND":       "झारखंड",
    "UTTARAKHAND":     "उत्तराखंड",
    "CHHATTISGARH":    "छत्तीसगढ़",
    "HIMACHAL PRADESH": "हिमाचल प्रदेश",
    "DELHI":           "दिल्ली",
}


# ── Build instruction template ───────────────────────────────────────
def build_instruction(record: dict) -> str:
    """
    Build the full mT5 input string from a record.
    All context is in Hindi so the model learns Hindi-in, Hindi-out.
    """
    state  = record.get("state", "")
    crop   = record.get("crop", "others")
    intent = record.get("intent", "unknown")
    query  = record.get("query", "")

    # Map to Hindi display names
    state_hi  = STATE_HINDI_MAP.get(state.upper(), state)
    crop_hi   = CROP_HINDI_MAP.get(crop.lower().strip(), crop)
    intent_hi = INTENT_HINDI_MAP.get(intent, "सामान्य")

    instruction = (
        f"निर्देश: आप एक कृषि विशेषज्ञ हैं। किसान की समस्या का उत्तर हिंदी में दें।\n"
        f"राज्य: {state_hi}\n"
        f"फसल: {crop_hi}\n"
        f"समस्या का प्रकार: {intent_hi}\n"
        f"किसान का प्रश्न: {query}\n"
        f"उत्तर:"
    )
    return instruction


# ── Format single record ─────────────────────────────────────────────
def format_record(record: dict) -> dict:
    """
    Transform a cleaned record into the instruction-tuning format.
    Returns a dict with:
      - input_text  : full instruction prompt (for model input)
      - target_text : Hindi answer (model must generate this)
      - metadata    : state, crop, intent, original query
    """
    instruction = build_instruction(record)
    answer      = str(record.get("answer", "")).strip()

    formatted = {
        "input_text":       instruction,
        "target_text":      answer,
        "state":            record.get("state", ""),
        "crop":             record.get("crop", "others"),
        "intent":           record.get("intent", "unknown"),
        "query_hindi":      record.get("query", ""),
        "query_original":   record.get("query_original", ""),
        "source":           record.get("source", "kcc"),
    }
    return formatted


# ── Validate formatted record ────────────────────────────────────────
def is_valid_record(record: dict,
                    min_input: int = 20,
                    max_input: int = 600,
                    min_target: int = 10,
                    max_target: int = 512) -> bool:
    """
    Final quality gate before saving.
    Drops records where input or target is outside acceptable lengths.
    """
    inp = record.get("input_text", "")
    tgt = record.get("target_text", "")

    if not inp or not tgt:
        return False
    if len(inp) < min_input or len(inp) > max_input:
        return False
    if len(tgt) < min_target or len(tgt) > max_target:
        return False

    # Target must have Devanagari
    dev_chars = sum(1 for c in tgt if '\u0900' <= c <= '\u097F')
    if dev_chars / max(len(tgt), 1) < 0.3:
        return False

    # Input must have Devanagari (from Hindi query)
    dev_in = sum(1 for c in inp if '\u0900' <= c <= '\u097F')
    if dev_in / max(len(inp), 1) < 0.1:
        return False

    return True


# ── Load JSONL ───────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    logger.info(f"Loaded {len(records):,} records")
    return records


# ── Save JSONL ───────────────────────────────────────────────────────
def save_jsonl(records: list, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in tqdm(records, desc=f"Saving → {Path(filepath).name}"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.success(f"Saved {len(records):,} → {filepath}")


# ── Print sample formatted records ──────────────────────────────────
def print_samples(records: list, n: int = 3):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  Sample Formatted Records")
    print(sep)
    import random
    samples = random.sample(records, min(n, len(records)))
    for i, r in enumerate(samples, 1):
        print(f"\n── Record {i} ──────────────────────────────────────────")
        print(f"INPUT:\n{r['input_text']}")
        print(f"\nTARGET:\n{r['target_text'][:200]}{'...' if len(r['target_text']) > 200 else ''}")
        print(f"\nMeta → State: {r['state']} | Crop: {r['crop']} | Intent: {r['intent']}")
    print(f"\n{sep}")


# ── Print formatting report ──────────────────────────────────────────
def print_format_report(total_in: int, valid: list, dropped: int):
    from collections import Counter

    sep = "=" * 65
    print(f"\n{sep}")
    print("  Formatting Pipeline — Report")
    print(sep)
    print(f"  Input records   : {total_in:,}")
    print(f"  Valid records   : {len(valid):,}")
    print(f"  Dropped         : {dropped:,}  ({(dropped/total_in)*100:.1f}%)")

    # Input/target length stats
    inp_lens = [len(r["input_text"]) for r in valid]
    tgt_lens = [len(r["target_text"]) for r in valid]

    print(f"\n  Input text length (chars):")
    print(f"    Min    : {min(inp_lens)}")
    print(f"    Max    : {max(inp_lens)}")
    print(f"    Mean   : {sum(inp_lens)/len(inp_lens):.1f}")

    print(f"\n  Target text length (chars):")
    print(f"    Min    : {min(tgt_lens)}")
    print(f"    Max    : {max(tgt_lens)}")
    print(f"    Mean   : {sum(tgt_lens)/len(tgt_lens):.1f}")

    # Intent distribution
    intents = [r["intent"] for r in valid]
    intent_counts = Counter(intents)
    print(f"\n  Intent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(valid)) * 100
        bar = "█" * int(pct / 2)
        print(f"    {intent:<25} {count:>8,}  ({pct:.1f}%)  {bar}")

    print(f"\n{sep}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("KisanMitra AI — Format Dataset (Task 6a)")
    logger.info("=" * 65)

    records = load_jsonl(INPUT_PATH)
    total_in = len(records)

    valid_records = []
    dropped = 0

    for record in tqdm(records, desc="Formatting"):
        formatted = format_record(record)
        if is_valid_record(formatted):
            valid_records.append(formatted)
        else:
            dropped += 1

    print_format_report(total_in, valid_records, dropped)
    print_samples(valid_records, n=3)
    save_jsonl(valid_records, OUTPUT_PATH)

    logger.success(f"Task 6a complete — {len(valid_records):,} records formatted.")