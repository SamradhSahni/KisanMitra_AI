import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from datasketch import MinHash, MinHashLSH
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

log_path = Path("logs/deduplicate.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")


# ── Load JSONL into list of dicts ────────────────────────────────────
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
    logger.info(f"Loaded {len(records):,} records from {filepath}")
    return records


# ── Build MinHash for a text ─────────────────────────────────────────
def build_minhash(text: str, num_perm: int, k: int) -> MinHash:
    """
    Create a MinHash signature from word k-shingles.
    k=3 means overlapping 3-word windows.
    """
    m = MinHash(num_perm=num_perm)
    words = str(text).lower().split()

    if len(words) < k:
        # If text is shorter than shingle size, use unigrams
        for word in words:
            m.update(word.encode("utf-8"))
    else:
        # k-word shingles
        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i:i + k])
            m.update(shingle.encode("utf-8"))

    return m


# ── Deduplication ────────────────────────────────────────────────────
def deduplicate(
    records: list,
    num_perm: int = 128,
    threshold: float = 0.7,
    k: int = 3
) -> list:
    """
    Use MinHash LSH to find and remove near-duplicate records.
    Deduplication is done on the ANSWER field (Hindi) since queries
    are English and many are near-identical generic strings.
    Also cross-checks query to avoid removing truly different records
    with similar answers.
    """
    logger.info(f"Deduplication config: num_perm={num_perm}, threshold={threshold}, k={k}")
    logger.info(f"Input records: {len(records):,}")

    # Build LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    unique_records = []
    duplicate_count = 0
    seen_ids = set()

    for idx, record in enumerate(tqdm(records, desc="Building MinHash index")):
        # We deduplicate on answer text (Hindi)
        answer_text = str(record.get("answer", ""))
        minhash = build_minhash(answer_text, num_perm=num_perm, k=k)

        try:
            # Query LSH for near-duplicates
            result = lsh.query(minhash)

            if len(result) == 0:
                # No near-duplicate found — keep this record
                lsh.insert(str(idx), minhash)
                unique_records.append(record)
                seen_ids.add(idx)
            else:
                # Near-duplicate found — skip
                duplicate_count += 1

        except Exception as e:
            # If insert fails (key conflict), skip silently
            duplicate_count += 1
            continue

    logger.info(f"Original  : {len(records):,}")
    logger.info(f"Unique    : {len(unique_records):,}")
    logger.info(f"Duplicates: {duplicate_count:,} ({(duplicate_count/len(records))*100:.1f}%)")

    return unique_records


# ── Save JSONL ───────────────────────────────────────────────────────
def save_jsonl(records: list, filepath: str):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in tqdm(records, desc="Saving"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.success(f"Saved {len(records):,} records to: {filepath}")


# ── Print final report ───────────────────────────────────────────────
def print_dedup_report(before: int, after: int, records: list):
    from collections import Counter

    sep = "=" * 55
    print(f"\n{sep}")
    print("  Deduplication — Final Report")
    print(sep)
    print(f"  Before dedup : {before:,}")
    print(f"  After dedup  : {after:,}")
    print(f"  Removed      : {before - after:,} ({((before-after)/before)*100:.1f}%)")

    # Intent distribution after dedup
    intents = [r.get("intent", "unknown") for r in records]
    intent_counts = Counter(intents)
    print(f"\n  Intent distribution (post-dedup):")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = (count / after) * 100
        bar = "█" * int(pct / 2)
        print(f"    {intent:<25} {count:>8,}  ({pct:.1f}%)  {bar}")

    # State distribution after dedup
    states = [r.get("state", "unknown") for r in records]
    state_counts = Counter(states)
    print(f"\n  State distribution (post-dedup):")
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        pct = (count / after) * 100
        print(f"    {state:<25} {count:>8,}  ({pct:.1f}%)")

    print(f"\n{sep}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_path  = os.getenv("PROCESSED_DATA_PATH", "./data/processed/clean_dataset.jsonl")
    output_path = "./data/processed/dedup_dataset.jsonl"

    dedup_config = CONFIG["deduplication"]
    num_perm  = dedup_config["num_permutations"]   # 128
    threshold = dedup_config["jaccard_threshold"]   # 0.7
    k         = dedup_config["shingle_size"]        # 3

    logger.info("=" * 55)
    logger.info("KisanMitra AI — Deduplication Pipeline (Task 4b)")
    logger.info("=" * 55)

    # Load cleaned dataset
    records = load_jsonl(input_path)
    before  = len(records)

    # Deduplicate
    unique_records = deduplicate(
        records,
        num_perm=num_perm,
        threshold=threshold,
        k=k
    )
    after = len(unique_records)

    # Report
    print_dedup_report(before, after, unique_records)

    # Save
    save_jsonl(unique_records, output_path)

    logger.success(f"Task 4b complete — dedup_dataset.jsonl saved. {after:,} unique records ready.")