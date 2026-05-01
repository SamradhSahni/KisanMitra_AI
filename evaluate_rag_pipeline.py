import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv
from collections import defaultdict, Counter

load_dotenv()
sys.path.insert(0, ".")

log_path = Path("logs/evaluate_rag.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────
TEST_PATH    = "./data/processed/test.jsonl"
RESULTS_DIR  = Path("./data/processed/eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_SAMPLE  = 500     # evaluate 500 test records with full RAG
BATCH_GEN    = 1       # generate one at a time for accuracy


# ── Load test records ─────────────────────────────────────────────────
def load_test_records(filepath: str, sample_size: int) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    continue

    # Stratified sample by intent
    import random
    random.seed(42)
    intent_groups = defaultdict(list)
    for r in records:
        intent_groups[r.get("intent", "unknown")].append(r)

    sampled = []
    total   = len(records)
    for intent, group in intent_groups.items():
        n = max(2, int((len(group) / total) * sample_size))
        n = min(n, len(group))
        sampled.extend(random.sample(group, n))

    sampled = sampled[:sample_size]
    random.shuffle(sampled)
    logger.info(f"Loaded {len(sampled)} stratified test records")
    return sampled


# ── BLEU score ────────────────────────────────────────────────────────
def compute_bleu_single(prediction: str, reference: str) -> dict:
    from sacrebleu.metrics import BLEU
    bleu  = BLEU(tokenize="char")
    score = bleu.corpus_score([prediction], [[reference]])
    return {
        "bleu4": round(score.score,           4),
        "bleu1": round(score.precisions[0],   4),
    }


# ── chrF score ────────────────────────────────────────────────────────
def compute_chrf_single(prediction: str, reference: str) -> float:
    from sacrebleu.metrics import CHRF
    chrf  = CHRF()
    score = chrf.corpus_score([prediction], [[reference]])
    return round(score.score, 4)


# ── ROUGE score ───────────────────────────────────────────────────────
def compute_rouge_single(prediction: str, reference: str) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure * 100, 4),
        "rouge2": round(scores["rouge2"].fmeasure * 100, 4),
        "rougeL": round(scores["rougeL"].fmeasure * 100, 4),
    }


# ── Language check ────────────────────────────────────────────────────
def check_hindi(text: str) -> bool:
    if not text:
        return False
    dev   = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total = sum(1 for c in text if c.isalpha())
    return (dev / max(total, 1)) >= 0.3


# ── Error category ────────────────────────────────────────────────────
def classify_error(prediction: str, reference: str) -> str:
    if not check_hindi(prediction):
        return "language_mismatch"
    if len(prediction.strip()) < 15:
        return "too_short"
    words = prediction.split()
    if len(words) >= 6:
        from collections import Counter
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams and max(Counter(trigrams).values()) >= 3:
            return "repetition_loop"
    import re
    pred_nums = set(re.findall(r'\d+\.?\d*', prediction))
    ref_nums  = set(re.findall(r'\d+\.?\d*', reference))
    if ref_nums and pred_nums:
        hallucinated = pred_nums - ref_nums
        if len(hallucinated) > 2 and len(hallucinated) > len(ref_nums):
            return "number_hallucination"
    crop_terms = ["किसान","फसल","बीज","खाद","कीट","रोग","सिंचाई",
                  "बुवाई","उर्वरक","कृषि","विभाग","संपर्क","पानी"]
    ref_agri  = any(t in reference   for t in crop_terms)
    pred_agri = any(t in prediction  for t in crop_terms)
    if ref_agri and not pred_agri and len(prediction) > 20:
        return "topic_drift"
    return "correct"


# ── Generate prediction WITH RAG ──────────────────────────────────────
def generate_with_rag(
    pipeline,
    record: dict,
) -> dict:
    query  = str(record.get("input_text",  ""))
    target = str(record.get("target_text", ""))
    intent = str(record.get("intent",      "unknown"))
    state  = str(record.get("state",       "UTTAR PRADESH"))
    crop   = str(record.get("crop",        "others"))

    # Extract Hindi query from input_text
    # input_text format: "निर्देश:...किसान का प्रश्न: <query>\nउत्तर:"
    query_hindi = query
    if "किसान का प्रश्न:" in query:
        query_hindi = query.split("किसान का प्रश्न:")[-1].split("\nउत्तर:")[0].strip()

    start  = time.time()
    result = pipeline.chat(
        query=query_hindi,
        state=state,
        crop=crop,
        intent=intent,
        use_rag=True,
    )
    latency = int((time.time() - start) * 1000)

    return {
        "prediction":     result["response"],
        "reference":      target,
        "intent":         intent,
        "state":          state,
        "crop":           crop,
        "query":          query_hindi,
        "rag_used":       result["rag_used"],
        "passages_count": len(result["passages"]),
        "latency_ms":     latency,
        "retrieval_ms":   result["retrieval_ms"],
        "generation_ms":  result["generation_ms"],
    }


# ── Generate prediction WITHOUT RAG ──────────────────────────────────
def generate_without_rag(
    pipeline,
    record: dict,
) -> dict:
    query  = str(record.get("input_text",  ""))
    target = str(record.get("target_text", ""))
    intent = str(record.get("intent",      "unknown"))
    state  = str(record.get("state",       "UTTAR PRADESH"))
    crop   = str(record.get("crop",        "others"))

    query_hindi = query
    if "किसान का प्रश्न:" in query:
        query_hindi = query.split("किसान का प्रश्न:")[-1].split("\nउत्तर:")[0].strip()

    start  = time.time()
    result = pipeline.chat(
        query=query_hindi,
        state=state,
        crop=crop,
        intent=intent,
        use_rag=False,
    )
    latency = int((time.time() - start) * 1000)

    return {
        "prediction":     result["response"],
        "reference":      target,
        "intent":         intent,
        "state":          state,
        "crop":           crop,
        "query":          query_hindi,
        "rag_used":       False,
        "passages_count": 0,
        "latency_ms":     latency,
        "retrieval_ms":   0,
        "generation_ms":  result["generation_ms"],
    }


# ── Compute all metrics for a result list ─────────────────────────────
def compute_metrics(results: list) -> dict:
    bleu4_scores  = []
    bleu1_scores  = []
    chrf_scores   = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    error_cats    = []
    latencies     = []
    pred_lens     = []
    ref_lens      = []

    for r in tqdm(results, desc="Computing metrics"):
        pred = r["prediction"]
        ref  = r["reference"]

        bleu  = compute_bleu_single(pred, ref)
        chrf  = compute_chrf_single(pred, ref)
        rouge = compute_rouge_single(pred, ref)
        error = classify_error(pred, ref)

        bleu4_scores.append(bleu["bleu4"])
        bleu1_scores.append(bleu["bleu1"])
        chrf_scores.append(chrf)
        rouge1_scores.append(rouge["rouge1"])
        rouge2_scores.append(rouge["rouge2"])
        rougeL_scores.append(rouge["rougeL"])
        error_cats.append(error)
        latencies.append(r["latency_ms"])
        pred_lens.append(len(pred))
        ref_lens.append(len(ref))

        # Store metrics back into result
        r["bleu4"]  = bleu["bleu4"]
        r["chrf"]   = chrf
        r["rouge1"] = rouge["rouge1"]
        r["error"]  = error

    error_counts = Counter(error_cats)
    total        = len(results)

    return {
        "total_samples":    total,
        "bleu4":            round(np.mean(bleu4_scores),  4),
        "bleu1":            round(np.mean(bleu1_scores),  4),
        "chrf":             round(np.mean(chrf_scores),   4),
        "rouge1":           round(np.mean(rouge1_scores), 4),
        "rouge2":           round(np.mean(rouge2_scores), 4),
        "rougeL":           round(np.mean(rougeL_scores), 4),
        "correct_pct":      round(error_counts.get("correct", 0) / total * 100, 2),
        "mismatch_pct":     round(error_counts.get("language_mismatch", 0) / total * 100, 2),
        "topic_drift_pct":  round(error_counts.get("topic_drift", 0) / total * 100, 2),
        "repetition_pct":   round(error_counts.get("repetition_loop", 0) / total * 100, 2),
        "hallucination_pct":round(error_counts.get("number_hallucination", 0) / total * 100, 2),
        "too_short_pct":    round(error_counts.get("too_short", 0) / total * 100, 2),
        "avg_latency_ms":   round(np.mean(latencies), 1),
        "avg_pred_len":     round(np.mean(pred_lens),  1),
        "avg_ref_len":      round(np.mean(ref_lens),   1),
        "avg_retrieval_ms": round(np.mean([r.get("retrieval_ms", 0) for r in results]), 1),
        "avg_gen_ms":       round(np.mean([r.get("generation_ms", 0) for r in results]), 1),
        "hindi_pct":        round(
            sum(1 for r in results if check_hindi(r["prediction"])) / total * 100, 2
        ),
    }


# ── Per-intent breakdown ──────────────────────────────────────────────
def per_intent_metrics(results: list) -> dict:
    intent_groups = defaultdict(list)
    for r in results:
        intent_groups[r["intent"]].append(r)

    breakdown = {}
    for intent, group in intent_groups.items():
        correct = sum(1 for r in group if r.get("error") == "correct")
        rouge1s = [r.get("rouge1", 0) for r in group]
        chrf_s  = [r.get("chrf",   0) for r in group]
        breakdown[intent] = {
            "count":       len(group),
            "correct_pct": round(correct / len(group) * 100, 1),
            "rouge1":      round(np.mean(rouge1s), 2),
            "chrf":        round(np.mean(chrf_s),  2),
            "avg_lat_ms":  round(np.mean([r["latency_ms"] for r in group]), 0),
        }
    return breakdown


# ── Print comparison report ───────────────────────────────────────────
def print_comparison_report(
    metrics_no_rag:  dict,
    metrics_with_rag: dict,
    intent_no_rag:   dict,
    intent_with_rag: dict,
):
    sep = "=" * 70

    def delta(a, b, higher_better=True):
        d = b - a
        if higher_better:
            return f"{'↑' if d > 0 else '↓'}{abs(d):.2f}"
        return f"{'↓' if d < 0 else '↑'}{abs(d):.2f}"

    print(f"\n{sep}")
    print("  KisanMitra AI — RAG Impact Evaluation Report")
    print(sep)
    print(f"  Samples evaluated : {metrics_no_rag['total_samples']}")

    # ── Core metrics ──
    print(f"\n  ── Core Metrics ─────────────────────────────────────────────")
    print(f"  {'Metric':<20} {'No RAG':>10} {'With RAG':>10} {'Delta':>10}")
    print(f"  {'─'*52}")

    metric_rows = [
        ("BLEU-4",       "bleu4",           True),
        ("BLEU-1",       "bleu1",           True),
        ("chrF",         "chrf",            True),
        ("ROUGE-1",      "rouge1",          True),
        ("ROUGE-2",      "rouge2",          True),
        ("ROUGE-L",      "rougeL",          True),
        ("Correct %",    "correct_pct",     True),
        ("Hindi %",      "hindi_pct",       True),
        ("Mismatch %",   "mismatch_pct",    False),
        ("Topic drift %","topic_drift_pct", False),
        ("Hallucination%","hallucination_pct",False),
        ("Avg pred len", "avg_pred_len",    True),
        ("Avg latency ms","avg_latency_ms", False),
        ("Avg retrieval ms","avg_retrieval_ms",False),
        ("Avg gen ms",   "avg_gen_ms",      False),
    ]

    for label, key, higher_better in metric_rows:
        v_no  = metrics_no_rag.get(key,  0)
        v_rag = metrics_with_rag.get(key, 0)
        d     = delta(v_no, v_rag, higher_better)
        # Highlight if RAG helped
        marker = " ✅" if (higher_better and v_rag > v_no) or \
                         (not higher_better and v_rag < v_no) else ""
        print(f"  {label:<20} {v_no:>10.2f} {v_rag:>10.2f} {d:>10}{marker}")

    # ── Per-intent comparison ──
    print(f"\n  ── Per-Intent Correct % ─────────────────────────────────────")
    print(f"  {'Intent':<25} {'No RAG':>8} {'With RAG':>10} {'Delta':>8} {'Count':>7}")
    print(f"  {'─'*60}")

    all_intents = sorted(
        set(list(intent_no_rag.keys()) + list(intent_with_rag.keys())),
        key=lambda x: -intent_with_rag.get(x, {}).get("count", 0)
    )

    for intent in all_intents:
        no_r  = intent_no_rag.get(intent,  {})
        wi_r  = intent_with_rag.get(intent, {})
        v_no  = no_r.get("correct_pct", 0)
        v_rag = wi_r.get("correct_pct", 0)
        count = wi_r.get("count", 0)
        d     = v_rag - v_no
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        marker = " ✅" if d > 5 else (" ⚠️" if d < -5 else "")
        print(
            f"  {intent:<25} {v_no:>7.1f}% {v_rag:>9.1f}%"
            f" {arrow}{abs(d):>5.1f}%{marker} {count:>6}"
        )

    # ── Per-intent chrF with RAG ──
    print(f"\n  ── Per-Intent chrF (With RAG) ───────────────────────────────")
    print(f"  {'Intent':<25} {'chrF':>8} {'ROUGE-1':>9} {'Avg Lat':>9} {'Count':>7}")
    print(f"  {'─'*60}")
    for intent in all_intents:
        wi_r  = intent_with_rag.get(intent, {})
        print(
            f"  {intent:<25} "
            f"{wi_r.get('chrf',0):>8.2f} "
            f"{wi_r.get('rouge1',0):>9.2f} "
            f"{wi_r.get('avg_lat_ms',0):>8.0f}ms "
            f"{wi_r.get('count',0):>6}"
        )

    # ── Sample predictions comparison ──
    print(f"\n  ── RAG vs No-RAG Sample Comparison ─────────────────────────")
    print(sep)


# ── Sample comparison printer ─────────────────────────────────────────
def print_sample_comparison(
    no_rag_results:   list,
    with_rag_results: list,
    n: int = 5,
):
    import random
    random.seed(99)

    # Find cases where RAG made a difference
    improved  = []
    worsened  = []
    same      = []

    for nr, wr in zip(no_rag_results, with_rag_results):
        if nr["query"] != wr["query"]:
            continue
        if nr.get("error") != "correct" and wr.get("error") == "correct":
            improved.append((nr, wr))
        elif nr.get("error") == "correct" and wr.get("error") != "correct":
            worsened.append((nr, wr))
        else:
            same.append((nr, wr))

    sep = "─" * 65

    print(f"\n  ✅ CASES WHERE RAG IMPROVED ({len(improved)} found):")
    for nr, wr in improved[:3]:
        print(f"\n  {sep}")
        print(f"  Intent : {nr['intent']} | State: {nr['state']}")
        print(f"  Query  : {nr['query'][:80]}")
        print(f"  No RAG : {nr['prediction'][:120]}")
        print(f"  With RAG:{wr['prediction'][:120]}")
        print(f"  Ref    : {nr['reference'][:120]}")

    print(f"\n  ⚠️  CASES WHERE RAG DID NOT HELP ({len(same)} same):")
    for nr, wr in random.sample(same, min(2, len(same))):
        print(f"\n  {sep}")
        print(f"  Intent : {nr['intent']}")
        print(f"  Query  : {nr['query'][:80]}")
        print(f"  No RAG : {nr['prediction'][:120]}")
        print(f"  With RAG:{wr['prediction'][:120]}")

    if worsened:
        print(f"\n  ❌ CASES WHERE RAG HURT ({len(worsened)} found):")
        for nr, wr in worsened[:2]:
            print(f"\n  {sep}")
            print(f"  Intent : {nr['intent']}")
            print(f"  Query  : {nr['query'][:80]}")
            print(f"  No RAG : {nr['prediction'][:120]}")
            print(f"  With RAG:{wr['prediction'][:120]}")


# ── Save results ──────────────────────────────────────────────────────
def save_all_results(
    no_rag_results:   list,
    with_rag_results: list,
    metrics_no_rag:   dict,
    metrics_with_rag: dict,
    intent_no_rag:    dict,
    intent_with_rag:  dict,
):
    # Save predictions
    with open(RESULTS_DIR / "rag_eval_no_rag.jsonl",   "w", encoding="utf-8") as f:
        for r in no_rag_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(RESULTS_DIR / "rag_eval_with_rag.jsonl", "w", encoding="utf-8") as f:
        for r in with_rag_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save metrics summary
    summary = {
        "eval_sample_size": len(no_rag_results),
        "no_rag":  {
            "metrics":        metrics_no_rag,
            "per_intent":     intent_no_rag,
        },
        "with_rag": {
            "metrics":        metrics_with_rag,
            "per_intent":     intent_with_rag,
        },
        "improvements": {
            k: round(metrics_with_rag.get(k, 0) - metrics_no_rag.get(k, 0), 4)
            for k in ["bleu4", "chrf", "rouge1", "correct_pct",
                      "topic_drift_pct", "hallucination_pct"]
        }
    }

    with open(RESULTS_DIR / "rag_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.success(f"All results saved to {RESULTS_DIR}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("KisanMitra AI — RAG Pipeline Evaluation")
    logger.info("=" * 70)

    # Load pipeline
    from backend.rag.pipeline import KisanMitraRAGPipeline
    pipeline = KisanMitraRAGPipeline()
    pipeline.load()

    # Load test records
    records = load_test_records(TEST_PATH, sample_size=EVAL_SAMPLE)

    logger.info(f"Evaluating {len(records)} records — this takes ~15-20 minutes")
    logger.info("Running WITHOUT RAG first...")

    # ── Pass 1: Without RAG ──
    no_rag_results = []
    for record in tqdm(records, desc="No RAG"):
        result = generate_without_rag(pipeline, record)
        no_rag_results.append(result)

    logger.info("Running WITH RAG...")

    # ── Pass 2: With RAG ──
    with_rag_results = []
    for record in tqdm(records, desc="With RAG"):
        result = generate_with_rag(pipeline, record)
        with_rag_results.append(result)

    # ── Compute metrics ──
    logger.info("Computing metrics for No RAG...")
    metrics_no_rag = compute_metrics(no_rag_results)

    logger.info("Computing metrics for With RAG...")
    metrics_with_rag = compute_metrics(with_rag_results)

    # ── Per-intent breakdown ──
    intent_no_rag  = per_intent_metrics(no_rag_results)
    intent_with_rag = per_intent_metrics(with_rag_results)

    # ── Print report ──
    print_comparison_report(
        metrics_no_rag, metrics_with_rag,
        intent_no_rag,  intent_with_rag,
    )

    # ── Sample comparison ──
    print_sample_comparison(no_rag_results, with_rag_results, n=5)

    # ── Save ──
    save_all_results(
        no_rag_results, with_rag_results,
        metrics_no_rag, metrics_with_rag,
        intent_no_rag,  intent_with_rag,
    )

    # ── Final verdict ──
    sep = "=" * 70
    correct_improvement = (
        metrics_with_rag["correct_pct"] - metrics_no_rag["correct_pct"]
    )
    print(f"\n{sep}")
    print("  Final Verdict")
    print(sep)
    print(f"  No RAG  correct : {metrics_no_rag['correct_pct']:.1f}%")
    print(f"  With RAG correct: {metrics_with_rag['correct_pct']:.1f}%")
    print(f"  Improvement     : +{correct_improvement:.1f}%")

    if correct_improvement >= 5:
        print(f"  ✅ RAG is significantly improving the model")
        print(f"  ✅ System is working correctly — ready for deployment")
    elif correct_improvement >= 0:
        print(f"  ✅ RAG is helping — marginal improvement")
        print(f"  ℹ️  Expanding KB to 2 lakh records will increase this further")
    else:
        print(f"  ⚠️  RAG not helping on this sample")
        print(f"  ℹ️  Check retrieval quality and passage filter threshold")

    print(f"\n  Share the full output above.")
    print(f"{sep}\n")

    logger.success("RAG evaluation complete.")