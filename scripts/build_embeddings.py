import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

log_path = Path("logs/build_embeddings.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="50 MB", encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = CONFIG["rag"]["embedding_model"]   # intfloat/multilingual-e5-small
EMBEDDING_DIM    = CONFIG["rag"]["embedding_dim"]      # 384
KB_SIZE          = CONFIG["rag"]["kb_size"]            # 50000
BATCH_SIZE       = 128                                 # safe for 6GB VRAM

SOURCE_PATH      = "./data/processed/translated_dataset.jsonl"
EMBEDDINGS_DIR   = Path("./data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

KB_JSONL_PATH    = EMBEDDINGS_DIR / "kb_records.jsonl"
EMBEDDINGS_PATH  = EMBEDDINGS_DIR / "kb_embeddings.npy"
KB_META_PATH     = EMBEDDINGS_DIR / "kb_metadata.json"


# ── Load source dataset ───────────────────────────────────────────────
def load_jsonl(filepath: str) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading source"):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    logger.info(f"Loaded {len(records):,} records from {filepath}")
    return records


# ── Sample knowledge base records ────────────────────────────────────
def sample_kb_records(records: list, kb_size: int) -> list:
    """
    Stratified sample by intent to build a balanced KB.
    Ensures all intent classes are represented.
    """
    import random
    from collections import defaultdict

    random.seed(42)

    intent_groups = defaultdict(list)
    for r in records:
        intent_groups[r.get("intent", "unknown")].append(r)

    sampled = []
    total   = len(records)

    logger.info(f"Sampling {kb_size:,} KB records (stratified by intent):")
    for intent, group in sorted(intent_groups.items()):
        # Proportional allocation
        n = max(10, int((len(group) / total) * kb_size))
        n = min(n, len(group))
        sampled.extend(random.sample(group, n))
        logger.info(f"  {intent:<25} {n:>6,} records")

    # Trim or top up to exact kb_size
    random.shuffle(sampled)
    sampled = sampled[:kb_size]

    logger.info(f"Final KB size: {len(sampled):,} records")
    return sampled


# ── Build text for embedding ──────────────────────────────────────────
def build_embedding_text(record: dict) -> str:
    """
    Build the text string that gets embedded.
    We embed BOTH query and answer concatenated so retrieval
    works whether the user's query matches the question or the answer.

    Format follows multilingual-e5-small's recommended prefix:
    'passage: <text>' for documents being indexed
    'query: <text>'   for queries at retrieval time
    """
    query  = str(record.get("query",  "")).strip()
    answer = str(record.get("answer", "")).strip()
    crop   = str(record.get("crop",   "")).strip()
    state  = str(record.get("state",  "")).strip()
    intent = str(record.get("intent", "")).strip()

    # Combine into a rich passage for indexing
    text = f"passage: {query} {answer} फसल: {crop} राज्य: {state} विषय: {intent}"
    return text[:512]   # cap at 512 chars


# ── Load embedding model ──────────────────────────────────────────────
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    logger.info("First run will download ~120MB...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(EMBEDDING_MODEL, device=device)

    logger.success(f"Embedding model loaded on {device}")
    logger.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ── Generate embeddings in batches ───────────────────────────────────
def embed_records(
    records: list,
    model,
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """
    Embed all KB records in batches.
    Returns L2-normalized embeddings as numpy array (N, 384).
    """
    texts = [build_embedding_text(r) for r in records]
    all_embeddings = []

    logger.info(f"Embedding {len(texts):,} records in batches of {batch_size}...")

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]

        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 normalize for cosine similarity
        )
        all_embeddings.append(embeddings)

        # Clear CUDA cache periodically
        if torch.cuda.is_available() and i % (batch_size * 20) == 0:
            torch.cuda.empty_cache()

    embeddings_matrix = np.vstack(all_embeddings)
    logger.success(
        f"Embeddings generated — shape: {embeddings_matrix.shape} "
        f"dtype: {embeddings_matrix.dtype}"
    )
    return embeddings_matrix


# ── Save KB records as JSONL ──────────────────────────────────────────
def save_kb_records(records: list):
    with open(KB_JSONL_PATH, "w", encoding="utf-8") as f:
        for r in tqdm(records, desc="Saving KB records"):
            # Save only fields needed for RAG retrieval
            out = {
                "id":      r.get("id", ""),
                "query":   str(r.get("query",  "")).strip(),
                "answer":  str(r.get("answer", "")).strip(),
                "crop":    str(r.get("crop",   "")).strip(),
                "state":   str(r.get("state",  "")).strip(),
                "intent":  str(r.get("intent", "")).strip(),
                "source":  str(r.get("source", "kcc")).strip(),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    logger.success(f"KB records saved → {KB_JSONL_PATH}")


# ── Save embeddings as numpy array ───────────────────────────────────
def save_embeddings(embeddings: np.ndarray):
    np.save(str(EMBEDDINGS_PATH), embeddings)
    size_mb = EMBEDDINGS_PATH.stat().st_size / 1024**2
    logger.success(f"Embeddings saved → {EMBEDDINGS_PATH} ({size_mb:.1f} MB)")


# ── Save metadata ─────────────────────────────────────────────────────
def save_metadata(records: list, embeddings: np.ndarray):
    from collections import Counter

    meta = {
        "kb_size":         len(records),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim":   int(embeddings.shape[1]),
        "normalized":      True,
        "intent_counts":   dict(Counter(r.get("intent","") for r in records)),
        "state_counts":    dict(Counter(r.get("state","")  for r in records)),
        "files": {
            "records":    str(KB_JSONL_PATH),
            "embeddings": str(EMBEDDINGS_PATH),
        }
    }
    with open(KB_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.success(f"Metadata saved → {KB_META_PATH}")


# ── Verify embeddings ─────────────────────────────────────────────────
def verify_embeddings(embeddings: np.ndarray, records: list, model):
    """
    Quick sanity check:
    1. Verify shape
    2. Check norms are ~1.0 (L2 normalized)
    3. Run a test similarity search
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("  Embedding Verification")
    print(sep)

    # Shape check
    print(f"  Shape     : {embeddings.shape}")
    print(f"  Dtype     : {embeddings.dtype}")
    assert embeddings.shape[1] == EMBEDDING_DIM, \
        f"Expected dim {EMBEDDING_DIM}, got {embeddings.shape[1]}"
    print(f"  ✅ Shape correct: ({len(records)}, {EMBEDDING_DIM})")

    # Norm check
    norms = np.linalg.norm(embeddings[:100], axis=1)
    avg_norm = float(np.mean(norms))
    print(f"  Avg L2 norm (first 100): {avg_norm:.4f}  (should be ~1.0)")
    if 0.99 <= avg_norm <= 1.01:
        print(f"  ✅ Embeddings are L2 normalized")
    else:
        print(f"  ⚠️  Norm deviates from 1.0 — check normalize_embeddings=True")

    # Test similarity search
    print(f"\n  ── Test Similarity Search ───────────────────────────")
    test_queries = [
        "query: मक्का में कीट नियंत्रण कैसे करें?",
        "query: गेहूं की बुवाई का सही समय क्या है?",
        "query: किसान क्रेडिट कार्ड के लिए आवेदन",
    ]

    for test_q in test_queries:
        q_emb = model.encode(
            [test_q],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        # Cosine similarity = dot product (since both are L2 normalized)
        scores = np.dot(embeddings, q_emb.T).flatten()
        top3   = np.argsort(scores)[::-1][:3]

        print(f"\n  Query : {test_q[7:]}")   # strip 'query: ' prefix
        for rank, idx in enumerate(top3, 1):
            rec = records[idx]
            print(f"  Top {rank}: [{scores[idx]:.4f}] "
                  f"{rec.get('query','')[:60]} | "
                  f"{rec.get('intent','')}")

    print(f"\n{sep}")


# ── Print final report ────────────────────────────────────────────────
def print_embedding_report(records: list, embeddings: np.ndarray):
    from collections import Counter

    sep = "=" * 65
    print(f"\n{sep}")
    print("  KB Embedding Pipeline — Final Report")
    print(sep)
    print(f"  Total KB records   : {len(records):,}")
    print(f"  Embedding shape    : {embeddings.shape}")
    print(f"  Embedding model    : {EMBEDDING_MODEL}")
    print(f"  L2 normalized      : Yes")

    size_mb = embeddings.nbytes / 1024**2
    print(f"  Memory (numpy)     : {size_mb:.1f} MB")

    print(f"\n  Intent distribution in KB:")
    for intent, count in sorted(
        Counter(r.get("intent","") for r in records).items(),
        key=lambda x: -x[1]
    ):
        pct = count / len(records) * 100
        bar = "█" * int(pct / 2)
        print(f"  {intent:<25} {count:>6,}  ({pct:.1f}%)  {bar}")

    print(f"\n  Files saved:")
    print(f"    {KB_JSONL_PATH}")
    print(f"    {EMBEDDINGS_PATH}")
    print(f"    {KB_META_PATH}")
    print(f"\n{sep}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("KisanMitra AI — Build KB Embeddings (Task 11a)")
    logger.info("=" * 65)

    # Load full translated dataset
    records = load_jsonl(SOURCE_PATH)

    # Add sequential IDs
    for i, r in enumerate(records):
        r["id"] = i

    # Sample KB records (stratified)
    kb_records = sample_kb_records(records, kb_size=KB_SIZE)

    # Load embedding model
    embed_model = load_embedding_model()

    # Generate embeddings
    embeddings = embed_records(kb_records, embed_model, batch_size=BATCH_SIZE)

    # Save everything
    save_kb_records(kb_records)
    save_embeddings(embeddings)
    save_metadata(kb_records, embeddings)

    # Verify
    verify_embeddings(embeddings, kb_records, embed_model)

    # Report
    print_embedding_report(kb_records, embeddings)

    logger.success("Task 11a complete — KB embeddings ready for indexing.")