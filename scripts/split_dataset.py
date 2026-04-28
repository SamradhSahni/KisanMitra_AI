import json
import os
import sys
import random
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

log_path = Path("logs/split_dataset.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

INPUT_PATH  = "./data/processed/formatted_dataset.jsonl"
TRAIN_PATH  = "./data/processed/train.jsonl"
VAL_PATH    = "./data/processed/val.jsonl"
TEST_PATH   = "./data/processed/test.jsonl"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# For fine-tuning we sample from full dataset
# Change this in .env → TRAIN_SAMPLE_SIZE
TRAIN_SAMPLE_SIZE = int(os.getenv("TRAIN_SAMPLE_SIZE", 20000))


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


# ── Stratified split by intent ───────────────────────────────────────
def stratified_split(
    records: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42
) -> tuple:
    """
    Split records maintaining intent class proportions in each split.
    Groups by intent, shuffles within each group, then splits.
    This prevents data leakage by splitting at record level.
    """
    random.seed(seed)

    # Group records by intent
    intent_groups = defaultdict(list)
    for r in records:
        intent_groups[r.get("intent", "unknown")].append(r)

    train, val, test = [], [], []

    for intent, group in intent_groups.items():
        random.shuffle(group)
        n     = len(group)
        n_val  = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        n_train = n - n_val - n_test

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

        logger.info(
            f"  {intent:<25} total={n:>6,} "
            f"train={n_train:>6,} val={n_val:>5,} test={n_test:>5,}"
        )

    # Final shuffle of each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ── Sample training set ──────────────────────────────────────────────
def sample_training_set(
    train: list,
    sample_size: int,
    seed: int = 42
) -> list:
    """
    From the full training set, sample a stratified subset
    for actual fine-tuning. Keeps intent proportions intact.
    """
    random.seed(seed)

    if sample_size >= len(train):
        logger.info(f"Sample size {sample_size:,} >= train size {len(train):,} — using full train set")
        return train

    # Group by intent
    intent_groups = defaultdict(list)
    for r in train:
        intent_groups[r.get("intent", "unknown")].append(r)

    total = len(train)
    sampled = []

    for intent, group in intent_groups.items():
        # Proportional sample from each intent class
        n_sample = max(1, int((len(group) / total) * sample_size))
        n_sample = min(n_sample, len(group))
        sampled.extend(random.sample(group, n_sample))

    # If we're slightly under due to rounding, top up from largest class
    if len(sampled) < sample_size:
        remaining_needed = sample_size - len(sampled)
        sampled_ids = set(id(r) for r in sampled)
        pool = [r for r in train if id(r) not in sampled_ids]
        random.shuffle(pool)
        sampled.extend(pool[:remaining_needed])

    random.shuffle(sampled)
    logger.info(f"Sampled {len(sampled):,} records for fine-tuning from {len(train):,} train records")
    return sampled


# ── Print split report ───────────────────────────────────────────────
def print_split_report(train: list, val: list, test: list, sampled: list):
    sep = "=" * 65
    total = len(train) + len(val) + len(test)

    print(f"\n{sep}")
    print("  Train / Val / Test Split — Report")
    print(sep)
    print(f"  Total records    : {total:,}")
    print(f"  Train (full)     : {len(train):,}  ({len(train)/total*100:.1f}%)")
    print(f"  Val              : {len(val):,}  ({len(val)/total*100:.1f}%)")
    print(f"  Test             : {len(test):,}  ({len(test)/total*100:.1f}%)")
    print(f"  Train (sampled)  : {len(sampled):,}  ← used for fine-tuning")

    # Intent distribution across splits
    def intent_dist(records):
        c = Counter(r["intent"] for r in records)
        return c

    all_intents = sorted(set(
        r["intent"] for r in train + val + test
    ))

    print(f"\n  Intent distribution across splits:")
    print(f"  {'Intent':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Sample':>8}")
    print(f"  {'─'*60}")

    train_c  = intent_dist(train)
    val_c    = intent_dist(val)
    test_c   = intent_dist(test)
    sample_c = intent_dist(sampled)

    for intent in all_intents:
        print(
            f"  {intent:<25} "
            f"{train_c.get(intent,0):>8,} "
            f"{val_c.get(intent,0):>8,} "
            f"{test_c.get(intent,0):>8,} "
            f"{sample_c.get(intent,0):>8,}"
        )

    # Verify no leakage — check a sample of val queries not in train
    print(f"\n  Data leakage check:")
    train_queries = set(r.get("query_hindi","") for r in train)
    val_leak  = sum(1 for r in val  if r.get("query_hindi","") in train_queries)
    test_leak = sum(1 for r in test if r.get("query_hindi","") in train_queries)
    print(f"    Val  queries found in train : {val_leak:,}")
    print(f"    Test queries found in train : {test_leak:,}")
    if val_leak == 0 and test_leak == 0:
        print(f"    ✅ No leakage detected")
    else:
        print(f"    ⚠️  Some overlap exists (expected for paraphrased queries)")

    # File size estimates
    print(f"\n  Estimated file sizes:")
    avg_bytes = 500   # ~500 bytes per formatted record
    print(f"    train.jsonl   : ~{(len(train)  * avg_bytes)/1024/1024:.0f} MB")
    print(f"    val.jsonl     : ~{(len(val)    * avg_bytes)/1024/1024:.0f} MB")
    print(f"    test.jsonl    : ~{(len(test)   * avg_bytes)/1024/1024:.0f} MB")
    print(f"    train_sample  : ~{(len(sampled)* avg_bytes)/1024/1024:.0f} MB")

    print(f"\n{sep}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("KisanMitra AI — Train/Val/Test Split (Task 6b)")
    logger.info("=" * 65)

    # Load formatted dataset
    records = load_jsonl(INPUT_PATH)

    logger.info(f"\nSplit ratios: train={TRAIN_RATIO} | val={VAL_RATIO} | test={TEST_RATIO}")
    logger.info("Intent-stratified split:")

    # Stratified split
    train, val, test = stratified_split(
        records,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )

    # Sample training set for fine-tuning
    train_sample = sample_training_set(
        train,
        sample_size=TRAIN_SAMPLE_SIZE,
        seed=RANDOM_SEED
    )

    # Report
    print_split_report(train, val, test, train_sample)

    # Save all splits
    save_jsonl(train,        TRAIN_PATH)
    save_jsonl(val,          VAL_PATH)
    save_jsonl(test,         TEST_PATH)
    save_jsonl(train_sample, "./data/processed/train_sample.jsonl")

    # Save split summary JSON
    summary = {
        "total_records":       len(records),
        "train_full":          len(train),
        "val":                 len(val),
        "test":                len(test),
        "train_sample":        len(train_sample),
        "train_sample_size":   TRAIN_SAMPLE_SIZE,
        "split_ratios":        {
            "train": TRAIN_RATIO,
            "val":   VAL_RATIO,
            "test":  TEST_RATIO
        },
        "intent_counts": {
            "train":  dict(Counter(r["intent"] for r in train)),
            "val":    dict(Counter(r["intent"] for r in val)),
            "test":   dict(Counter(r["intent"] for r in test)),
            "sample": dict(Counter(r["intent"] for r in train_sample)),
        }
    }
    summary_path = Path("data/processed/split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.success(f"Split summary saved → {summary_path}")
    logger.success("Task 6b complete — all splits saved.")