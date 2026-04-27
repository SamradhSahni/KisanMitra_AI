import json
import os
import sys
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
sys.path.insert(0, ".")
sys.path.insert(0, "./IndicTrans2")

log_path = Path("logs/translate_queries.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="50 MB", encoding="utf-8")


# ── Config ───────────────────────────────────────────────────────────
MODEL_NAME    = "ai4bharat/indictrans2-en-indic-dist-200M"
SRC_LANG      = "eng_Latn"
TGT_LANG      = "hin_Deva"
BATCH_SIZE    = 32     # reduce to 16 if you get CUDA OOM errors
MAX_SEQ_LEN   = 128
NUM_BEAMS     = 4
INPUT_PATH    = "./data/processed/dedup_dataset.jsonl"
OUTPUT_PATH   = "./data/processed/translated_dataset.jsonl"
CACHE_PATH    = "./data/processed/translation_cache.json"
CHECKPOINT_PATH = "./data/processed/translation_checkpoint.json"


# ── Load JSONL ────────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading records"):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    logger.info(f"Loaded {len(records):,} records")
    return records


# ── Load / Save translation cache ────────────────────────────────────
def load_cache(cache_path: str) -> dict:
    """Load previously translated queries to avoid re-translating."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"Loaded translation cache: {len(cache):,} entries")
        return cache
    logger.info("No existing cache found — starting fresh")
    return {}


def save_cache(cache: dict, cache_path: str):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


# ── Load model ───────────────────────────────────────────────────────
def load_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    try:
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        logger.error("IndicTransToolkit not found.")
        logger.error("Run: pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git")
        raise

    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()

    ip = IndicProcessor(inference=True)

    logger.success("Model loaded successfully")
    return model, tokenizer, ip, device


# ── Translate a batch ─────────────────────────────────────────────────
def translate_batch(
    sentences: list,
    model,
    tokenizer,
    ip,
    device: str
) -> list:
    """Translate a batch of English sentences to Hindi."""
    try:
        # Preprocess
        batch = ip.preprocess_batch(
            sentences,
            src_lang=SRC_LANG,
            tgt_lang=TGT_LANG,
        )

        # Tokenize
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        # Generate
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=MAX_SEQ_LEN,
                num_beams=NUM_BEAMS,
                num_return_sequences=1,
            )

        # Decode
        with tokenizer.as_target_tokenizer():
            decoded = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess
        translations = ip.postprocess_batch(decoded, lang=TGT_LANG)
        return translations

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning("CUDA OOM — clearing cache and retrying with smaller batch")
            torch.cuda.empty_cache()
            # Retry with half the batch size
            mid = len(sentences) // 2
            left  = translate_batch(sentences[:mid], model, tokenizer, ip, device)
            right = translate_batch(sentences[mid:], model, tokenizer, ip, device)
            return left + right
        else:
            logger.error(f"Translation error: {e}")
            # Return original text as fallback
            return sentences


# ── Validate Hindi translation ────────────────────────────────────────
def is_valid_hindi_translation(original: str, translated: str) -> bool:
    """
    Check that:
    1. Translation has Devanagari characters
    2. Translation is not empty
    3. Translation is not identical to original (untranslated)
    4. Has minimum Devanagari ratio
    """
    if not translated or not isinstance(translated, str):
        return False

    translated = translated.strip()
    if len(translated) == 0:
        return False

    # Check not same as original (untranslated)
    if translated.strip().lower() == original.strip().lower():
        return False

    # Check Devanagari ratio
    devanagari = sum(1 for c in translated if '\u0900' <= c <= '\u097F')
    total_alpha = sum(1 for c in translated if c.isalpha())

    if total_alpha == 0:
        return False

    ratio = devanagari / total_alpha
    return ratio >= 0.4


# ── Main translation pipeline ─────────────────────────────────────────
def translate_all_queries(records: list) -> list:
    """
    Main pipeline:
    1. Extract unique English queries
    2. Translate unique queries only (huge efficiency gain)
    3. Map translations back to all records
    4. Validate each translation
    5. Return final records
    """

    # ── Step 1: Extract unique queries ──
    logger.info("Extracting unique English queries...")
    unique_queries = list(set(
        str(r.get("query", "")).strip()
        for r in records
        if r.get("query", "").strip()
    ))
    logger.info(f"Total records  : {len(records):,}")
    logger.info(f"Unique queries : {len(unique_queries):,}")
    logger.info(f"Duplicate rate : {((len(records) - len(unique_queries)) / len(records)) * 100:.1f}%")

    # ── Step 2: Load existing cache ──
    cache = load_cache(CACHE_PATH)

    # Filter out already-cached queries
    to_translate = [q for q in unique_queries if q not in cache]
    logger.info(f"Already cached : {len(unique_queries) - len(to_translate):,}")
    logger.info(f"To translate   : {len(to_translate):,}")

    # ── Step 3: Load model ──
    if to_translate:
        model, tokenizer, ip, device = load_model()

        # ── Step 4: Translate in batches ──
        logger.info(f"Translating {len(to_translate):,} unique queries...")
        logger.info(f"Batch size: {BATCH_SIZE} | Estimated batches: {len(to_translate)//BATCH_SIZE + 1:,}")
        logger.info("⏳ This will take ~30–60 minutes on RTX 4050. Safe to leave running.")

        translated_count = 0
        failed_count     = 0

        for i in tqdm(range(0, len(to_translate), BATCH_SIZE), desc="Translating"):
            batch_originals = to_translate[i : i + BATCH_SIZE]

            # Translate
            batch_translations = translate_batch(
                batch_originals, model, tokenizer, ip, device
            )

            # Store in cache
            for original, translated in zip(batch_originals, batch_translations):
                if is_valid_hindi_translation(original, translated):
                    cache[original] = translated
                    translated_count += 1
                else:
                    # Fallback: keep original query — will be flagged in validation
                    cache[original] = None
                    failed_count += 1

            # Save cache every 500 batches (~16K queries)
            if (i // BATCH_SIZE) % 500 == 0 and i > 0:
                save_cache(cache, CACHE_PATH)
                logger.info(f"Cache saved at batch {i//BATCH_SIZE:,} | "
                           f"Translated: {translated_count:,} | Failed: {failed_count:,}")

        # Final cache save
        save_cache(cache, CACHE_PATH)
        logger.info(f"Translation complete: {translated_count:,} success, {failed_count:,} failed")

    # ── Step 5: Map translations back to records ──
    logger.info("Mapping translations back to all records...")

    valid_records   = []
    invalid_records = []

    for record in tqdm(records, desc="Mapping translations"):
        original_query = str(record.get("query", "")).strip()
        translated     = cache.get(original_query)

        if translated and is_valid_hindi_translation(original_query, translated):
            record["query"]          = translated
            record["query_original"] = original_query   # keep original for reference
            record["translated"]     = True
            valid_records.append(record)
        else:
            record["translated"] = False
            invalid_records.append(record)

    logger.info(f"Valid translations   : {len(valid_records):,}")
    logger.info(f"Invalid/failed       : {len(invalid_records):,}")

    return valid_records, invalid_records


# ── Print final report ────────────────────────────────────────────────
def print_translation_report(
    total: int,
    valid: list,
    invalid: list
):
    from collections import Counter

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Translation Pipeline — Final Report")
    print(sep)
    print(f"  Input records       : {total:,}")
    print(f"  Successfully translated : {len(valid):,}  ({(len(valid)/total)*100:.1f}%)")
    print(f"  Failed / skipped    : {len(invalid):,}  ({(len(invalid)/total)*100:.1f}%)")

    # Sample translations
    print(f"\n  Sample translations (Original EN → Translated HI):")
    print(f"  {'─'*56}")
    for record in valid[:10]:
        print(f"\n  EN: {record.get('query_original', 'N/A')[:70]}")
        print(f"  HI: {record.get('query', 'N/A')[:70]}")

    # Intent distribution of valid records
    intents = [r.get("intent", "unknown") for r in valid]
    intent_counts = Counter(intents)
    print(f"\n  Intent distribution (valid translated records):")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(valid)) * 100
        bar = "█" * int(pct / 2)
        print(f"    {intent:<25} {count:>8,}  ({pct:.1f}%)  {bar}")

    # Sample failures
    if invalid:
        print(f"\n  Sample failed translations (kept original):")
        for record in invalid[:5]:
            print(f"    Q: {record.get('query', 'N/A')[:70]}")

    print(f"\n{sep}")


# ── Save JSONL ────────────────────────────────────────────────────────
def save_jsonl(records: list, filepath: str):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in tqdm(records, desc="Saving"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.success(f"Saved {len(records):,} records to: {filepath}")


# ── Validate final dataset script check ──────────────────────────────
def validate_final_dataset(records: list):
    """
    Final check: both query AND answer must be in Hindi/Devanagari.
    Print counts and show any remaining issues.
    """
    sep = "=" * 60
    both_hindi    = 0
    query_not_hi  = 0
    answer_not_hi = 0

    for r in records:
        q_chars = sum(1 for c in str(r.get("query","")) if '\u0900' <= c <= '\u097F')
        a_chars = sum(1 for c in str(r.get("answer","")) if '\u0900' <= c <= '\u097F')
        q_valid = q_chars / max(len(str(r.get("query",""))), 1) >= 0.3
        a_valid = a_chars / max(len(str(r.get("answer",""))), 1) >= 0.3

        if q_valid and a_valid:
            both_hindi += 1
        elif not q_valid:
            query_not_hi += 1
        elif not a_valid:
            answer_not_hi += 1

    total = len(records)
    print(f"\n{sep}")
    print("  Final Dataset — Script Validation")
    print(sep)
    print(f"  Both query+answer Hindi  : {both_hindi:,}  ({(both_hindi/total)*100:.1f}%)")
    print(f"  Query not Hindi          : {query_not_hi:,}  ({(query_not_hi/total)*100:.1f}%)")
    print(f"  Answer not Hindi         : {answer_not_hi:,}  ({(answer_not_hi/total)*100:.1f}%)")
    print(f"{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("KisanMitra AI — Translation Pipeline (Task 5b)")
    logger.info("=" * 60)

    # Load deduplicated dataset
    records = load_jsonl(INPUT_PATH)
    total   = len(records)

    # Translate
    valid_records, invalid_records = translate_all_queries(records)

    # Report
    print_translation_report(total, valid_records, invalid_records)

    # Script validation
    validate_final_dataset(valid_records)

    # Save valid translated records
    save_jsonl(valid_records, OUTPUT_PATH)

    # Also save failures separately for inspection
    if invalid_records:
        save_jsonl(invalid_records, "./data/processed/translation_failures.jsonl")
        logger.info(f"Failures saved to: data/processed/translation_failures.jsonl")

    logger.success(f"Task 5b complete — {len(valid_records):,} Hindi-Hindi records ready.")
    logger.success(f"Output: {OUTPUT_PATH}")