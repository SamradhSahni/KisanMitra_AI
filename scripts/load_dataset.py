import json
import os
import sys
import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# ── Logger setup ────────────────────────────────────────────────────
log_path = Path("logs/load_dataset.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")


# ── Loader ──────────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> pd.DataFrame:
    """
    Load a JSONL file into a pandas DataFrame.
    Handles UTF-8 and UTF-8-SIG encodings.
    """
    records = []
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    file_size_mb = filepath.stat().st_size / 1024 / 1024
    logger.info(f"Loading: {filepath.name} ({file_size_mb:.2f} MB)")

    # Try UTF-8-SIG first (handles BOM), fallback to UTF-8
    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num} skipped — JSON error: {e}")
            logger.success(f"Loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            logger.warning(f"Encoding {encoding} failed, trying next...")
            records = []
            continue

    if not records:
        raise ValueError("Could not load file with any encoding.")

    df = pd.DataFrame(records)
    logger.success(f"Total records loaded: {len(df):,}")
    return df


# ── Field Validator ─────────────────────────────────────────────────
def validate_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check required fields exist and report missing values.
    """
    required_fields = ["query", "answer", "crop", "state", "language", "source", "section"]
    logger.info("\n── Field Validation ──")

    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
        raise ValueError(f"Dataset missing fields: {missing_fields}")

    logger.success(f"All required fields present: {required_fields}")

    # Report nulls per field
    null_counts = df[required_fields].isnull().sum()
    logger.info("\nNull counts per field:")
    for field, count in null_counts.items():
        pct = (count / len(df)) * 100
        if count > 0:
            logger.warning(f"  {field}: {count:,} nulls ({pct:.1f}%)")
        else:
            logger.success(f"  {field}: 0 nulls ✅")

    return df


# ── Stats Printer ───────────────────────────────────────────────────
def print_dataset_stats(df: pd.DataFrame):
    """
    Print comprehensive stats about the dataset.
    """

    separator = "=" * 55

    # ── Basic Stats ──
    print(f"\n{separator}")
    print("  KisanMitra AI — Dataset Exploration Report")
    print(separator)
    print(f"\n{'Total Records':<30} {len(df):>10,}")
    print(f"{'Total Columns':<30} {len(df.columns):>10}")
    print(f"{'Columns':<30} {list(df.columns)}")

    # ── Language Distribution ──
    print(f"\n{separator}")
    print("  Language Distribution")
    print(separator)
    if "language" in df.columns:
        lang_counts = df["language"].value_counts()
        for lang, count in lang_counts.items():
            pct = (count / len(df)) * 100
            bar = "█" * int(pct / 2)
            print(f"  {str(lang):<15} {count:>8,}  ({pct:5.1f}%)  {bar}")

    # ── State Distribution ──
    print(f"\n{separator}")
    print("  State Distribution (Top 15)")
    print(separator)
    if "state" in df.columns:
        state_counts = df["state"].value_counts().head(15)
        for state, count in state_counts.items():
            pct = (count / len(df)) * 100
            bar = "█" * int(pct / 1.5)
            print(f"  {str(state):<25} {count:>8,}  ({pct:5.1f}%)  {bar}")

    # ── Crop Distribution ──
    print(f"\n{separator}")
    print("  Top 20 Crops")
    print(separator)
    if "crop" in df.columns:
        crop_counts = df["crop"].value_counts().head(20)
        for crop, count in crop_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {str(crop):<30} {count:>8,}  ({pct:5.1f}%)")

    # ── Section Distribution ──
    print(f"\n{separator}")
    print("  Section Distribution")
    print(separator)
    if "section" in df.columns:
        section_counts = df["section"].value_counts()
        for section, count in section_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {str(section):<20} {count:>8,}  ({pct:5.1f}%)")

    # ── Source Distribution ──
    print(f"\n{separator}")
    print("  Source Distribution")
    print(separator)
    if "source" in df.columns:
        source_counts = df["source"].value_counts()
        for source, count in source_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {str(source):<20} {count:>8,}  ({pct:5.1f}%)")

    # ── Text Length Stats ──
    print(f"\n{separator}")
    print("  Text Length Analysis")
    print(separator)
    df["query_len"]  = df["query"].astype(str).apply(len)
    df["answer_len"] = df["answer"].astype(str).apply(len)

    print(f"\n  Query length (characters):")
    print(f"    Min    : {df['query_len'].min()}")
    print(f"    Max    : {df['query_len'].max()}")
    print(f"    Mean   : {df['query_len'].mean():.1f}")
    print(f"    Median : {df['query_len'].median():.1f}")

    print(f"\n  Answer length (characters):")
    print(f"    Min    : {df['answer_len'].min()}")
    print(f"    Max    : {df['answer_len'].max()}")
    print(f"    Mean   : {df['answer_len'].mean():.1f}")
    print(f"    Median : {df['answer_len'].median():.1f}")

    # ── Query Language Detection (English vs Hindi) ──
    print(f"\n{separator}")
    print("  Query Script Detection (English vs Hindi)")
    print(separator)
    detect = detect_query_script(df)
    for script, count in detect.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {script:<20} {count:>8,}  ({pct:5.1f}%)  {bar}")

    # ── Answer Script Detection ──
    print(f"\n{separator}")
    print("  Answer Script Detection (Should be Hindi)")
    print(separator)
    detect_ans = detect_answer_script(df)
    for script, count in detect_ans.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {script:<20} {count:>8,}  ({pct:5.1f}%)  {bar}")

    # ── Sample Records ──
    print(f"\n{separator}")
    print("  5 Sample Records")
    print(separator)
    for i, row in df.head(5).iterrows():
        print(f"\n  Record #{i+1}")
        print(f"  State   : {row.get('state', 'N/A')}")
        print(f"  Crop    : {row.get('crop', 'N/A')}")
        print(f"  Query   : {str(row.get('query', ''))[:100]}")
        print(f"  Answer  : {str(row.get('answer', ''))[:120]}")
        print(f"  Lang    : {row.get('language', 'N/A')}")
        print(f"  {'─'*50}")

    print(f"\n{separator}")
    print("  Report Complete")
    print(separator)


# ── Script Detection Helpers ─────────────────────────────────────────
def is_devanagari(text: str, threshold: float = 0.2) -> bool:
    """
    Returns True if enough characters in text are Devanagari (U+0900–U+097F).
    """
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) == 0:
        return False
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    ratio = devanagari_chars / len(text)
    return ratio >= threshold


def detect_query_script(df: pd.DataFrame) -> dict:
    """
    Classify query field as Hindi (Devanagari), English, or Mixed.
    """
    hindi   = 0
    english = 0
    mixed   = 0

    for query in df["query"].astype(str):
        has_dev = any('\u0900' <= c <= '\u097F' for c in query)
        has_lat = any(c.isascii() and c.isalpha() for c in query)
        if has_dev and has_lat:
            mixed += 1
        elif has_dev:
            hindi += 1
        else:
            english += 1

    return {"Hindi (Devanagari)": hindi, "English / Latin": english, "Mixed": mixed}


def detect_answer_script(df: pd.DataFrame) -> dict:
    """
    Classify answer field as Hindi (Devanagari), English, or Mixed.
    """
    hindi   = 0
    english = 0
    mixed   = 0

    for answer in df["answer"].astype(str):
        has_dev = any('\u0900' <= c <= '\u097F' for c in answer)
        has_lat = any(c.isascii() and c.isalpha() for c in answer)
        if has_dev and has_lat:
            mixed += 1
        elif has_dev:
            hindi += 1
        else:
            english += 1

    return {"Hindi (Devanagari)": hindi, "English / Latin": english, "Mixed": mixed}


# ── Save Summary to File ─────────────────────────────────────────────
def save_summary(df: pd.DataFrame):
    """Save a JSON summary of the dataset stats for later reference."""
    summary = {
        "total_records": len(df),
        "columns": list(df.columns),
        "state_distribution": df["state"].value_counts().to_dict() if "state" in df.columns else {},
        "language_distribution": df["language"].value_counts().to_dict() if "language" in df.columns else {},
        "crop_distribution": df["crop"].value_counts().head(20).to_dict() if "crop" in df.columns else {},
        "section_distribution": df["section"].value_counts().to_dict() if "section" in df.columns else {},
        "query_len_stats": {
            "min": int(df["query"].astype(str).apply(len).min()),
            "max": int(df["query"].astype(str).apply(len).max()),
            "mean": round(df["query"].astype(str).apply(len).mean(), 2),
        },
        "answer_len_stats": {
            "min": int(df["answer"].astype(str).apply(len).min()),
            "max": int(df["answer"].astype(str).apply(len).max()),
            "mean": round(df["answer"].astype(str).apply(len).mean(), 2),
        },
    }

    out_path = Path("data/processed/dataset_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.success(f"Summary saved to: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_path = os.getenv("RAW_DATA_PATH", "./data/raw/kcc_dataset.jsonl")

    logger.info("Starting dataset load and exploration...")

    # Load
    df = load_jsonl(raw_path)

    # Validate
    df = validate_fields(df)

    # Print full stats report
    print_dataset_stats(df)

    # Save summary JSON
    save_summary(df)

    logger.success("Task 3a complete.")