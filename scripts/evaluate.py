import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

log_path = Path("logs/evaluate.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────────────
FINAL_MODEL_DIR = os.getenv("FINETUNED_MODEL_PATH", "./model/final")
TEST_PATH       = "./data/processed/test.jsonl"
RESULTS_DIR     = Path("./data/processed/eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# How many test samples to evaluate
# Full test set = 77K — use 1000 for speed, increase for accuracy
EVAL_SAMPLE_SIZE = 1000
BATCH_SIZE       = 8
MAX_INPUT_LEN    = int(os.getenv("MAX_INPUT_LENGTH",  256))
MAX_NEW_TOKENS   = int(os.getenv("MAX_OUTPUT_LENGTH", 128))


# ── Dataset ───────────────────────────────────────────────────────────
class EvalDataset(Dataset):
    def __init__(self, filepath: str, tokenizer, sample_size: int = None):
        self.tokenizer = tokenizer
        self.records   = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if sample_size and sample_size < len(self.records):
            import random
            random.seed(42)
            # Stratified sample by intent
            from collections import defaultdict
            intent_groups = defaultdict(list)
            for r in self.records:
                intent_groups[r.get("intent", "unknown")].append(r)

            sampled = []
            for intent, group in intent_groups.items():
                n = max(1, int((len(group) / len(self.records)) * sample_size))
                sampled.extend(random.sample(group, min(n, len(group))))

            # Top up if under
            if len(sampled) < sample_size:
                pool = [r for r in self.records if r not in sampled]
                random.shuffle(pool)
                sampled.extend(pool[:sample_size - len(sampled)])

            self.records = sampled[:sample_size]

        logger.info(f"Eval dataset: {len(self.records):,} records from {filepath}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        inputs = self.tokenizer(
            str(record.get("input_text", "")),
            max_length=MAX_INPUT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "target_text":    str(record.get("target_text", "")),
            "intent":         str(record.get("intent", "unknown")),
            "query":          str(record.get("query_hindi", "")),
            "state":          str(record.get("state", "")),
            "crop":           str(record.get("crop", "")),
        }


# ── Load merged model ─────────────────────────────────────────────────
def load_model(model_dir: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info(f"Loading merged model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    logger.success("Model loaded for evaluation")
    return model, tokenizer


# ── Generate predictions ──────────────────────────────────────────────
def generate_predictions(model, tokenizer, dataloader, device) -> list:
    """
    Run beam search generation on all test batches.
    Returns list of dicts with prediction, reference, intent, etc.
    """
    all_results = []

    for batch in tqdm(dataloader, desc="Generating predictions"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_texts   = batch["target_text"]
        intents        = batch["intent"]
        queries        = batch["query"]
        states         = batch["state"]
        crops          = batch["crop"]

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        # Decode predictions
        predictions = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for pred, ref, intent, query, state, crop in zip(
            predictions, target_texts, intents, queries, states, crops
        ):
            all_results.append({
                "prediction": pred.strip(),
                "reference":  ref.strip(),
                "intent":     intent,
                "query":      query,
                "state":      state,
                "crop":       crop,
            })

    logger.info(f"Generated {len(all_results):,} predictions")
    return all_results


# ── Compute BLEU ──────────────────────────────────────────────────────
def compute_bleu(results: list) -> dict:
    from sacrebleu.metrics import BLEU

    bleu = BLEU(tokenize="char")   # char-level for Hindi

    predictions = [r["prediction"] for r in results]
    references  = [[r["reference"] for r in results]]

    score = bleu.corpus_score(predictions, references)

    return {
        "bleu_4":  round(score.score, 4),
        "bleu_1":  round(score.precisions[0], 4),
        "bleu_2":  round(score.precisions[1], 4),
        "bleu_3":  round(score.precisions[2], 4),
        "bp":      round(score.bp, 4),
    }


# ── Compute chrF ──────────────────────────────────────────────────────
def compute_chrf(results: list) -> dict:
    from sacrebleu.metrics import CHRF

    chrf = CHRF()

    predictions = [r["prediction"] for r in results]
    references  = [[r["reference"] for r in results]]

    score = chrf.corpus_score(predictions, references)

    return {
        "chrf": round(score.score, 4),
    }


# ── Compute ROUGE ─────────────────────────────────────────────────────
def compute_rouge(results: list) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,    # no stemmer for Hindi
    )

    r1_scores, r2_scores, rL_scores = [], [], []

    for r in results:
        scores = scorer.score(r["reference"], r["prediction"])
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": round(np.mean(r1_scores) * 100, 4),
        "rouge2": round(np.mean(r2_scores) * 100, 4),
        "rougeL": round(np.mean(rL_scores) * 100, 4),
    }


# ── Compute language mismatch ─────────────────────────────────────────
def compute_language_mismatch(results: list) -> dict:
    """
    Check what % of predictions are in Hindi vs English.
    A good model should have 0% language mismatch.
    """
    hindi    = 0
    english  = 0
    mixed    = 0

    for r in results:
        pred = r["prediction"]
        has_dev = any('\u0900' <= c <= '\u097F' for c in pred)
        has_lat = any(c.isascii() and c.isalpha() for c in pred)

        if has_dev and not has_lat:
            hindi += 1
        elif has_dev and has_lat:
            mixed += 1
        else:
            english += 1

    total = len(results)
    return {
        "hindi_pct":   round(hindi   / total * 100, 2),
        "mixed_pct":   round(mixed   / total * 100, 2),
        "english_pct": round(english / total * 100, 2),
        "mismatch_pct": round((english) / total * 100, 2),
    }


# ── Compute per-intent metrics ────────────────────────────────────────
def compute_per_intent_metrics(results: list) -> dict:
    from collections import defaultdict
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    intent_groups = defaultdict(list)
    for r in results:
        intent_groups[r["intent"]].append(r)

    per_intent = {}
    for intent, group in intent_groups.items():
        r1_scores = []
        rL_scores = []
        pred_lens = []
        ref_lens  = []

        for r in group:
            scores = scorer.score(r["reference"], r["prediction"])
            r1_scores.append(scores["rouge1"].fmeasure)
            rL_scores.append(scores["rougeL"].fmeasure)
            pred_lens.append(len(r["prediction"]))
            ref_lens.append(len(r["reference"]))

        per_intent[intent] = {
            "count":       len(group),
            "rouge1":      round(np.mean(r1_scores) * 100, 2),
            "rougeL":      round(np.mean(rL_scores) * 100, 2),
            "avg_pred_len": round(np.mean(pred_lens), 1),
            "avg_ref_len":  round(np.mean(ref_lens),  1),
        }

    return per_intent


# ── Compute avg prediction length ────────────────────────────────────
def compute_length_stats(results: list) -> dict:
    pred_lens = [len(r["prediction"]) for r in results]
    ref_lens  = [len(r["reference"])  for r in results]
    return {
        "avg_prediction_len": round(np.mean(pred_lens), 1),
        "avg_reference_len":  round(np.mean(ref_lens),  1),
        "min_prediction_len": min(pred_lens),
        "max_prediction_len": max(pred_lens),
    }


# ── Print full evaluation report ──────────────────────────────────────
def print_eval_report(
    bleu:         dict,
    chrf:         dict,
    rouge:        dict,
    lang:         dict,
    per_intent:   dict,
    length_stats: dict,
    sample_size:  int,
):
    sep = "=" * 65

    print(f"\n{sep}")
    print("  KisanMitra AI — Evaluation Report")
    print(sep)
    print(f"  Test sample size : {sample_size:,}")

    # ── Core metrics ──
    print(f"\n  ── Core Metrics ───────────────────────────────────────")
    print(f"  BLEU-4      : {bleu['bleu_4']:>8.4f}")
    print(f"  BLEU-1      : {bleu['bleu_1']:>8.4f}")
    print(f"  BLEU-2      : {bleu['bleu_2']:>8.4f}")
    print(f"  BLEU-3      : {bleu['bleu_3']:>8.4f}")
    print(f"  chrF        : {chrf['chrf']:>8.4f}")
    print(f"  ROUGE-1     : {rouge['rouge1']:>8.4f}")
    print(f"  ROUGE-2     : {rouge['rouge2']:>8.4f}")
    print(f"  ROUGE-L     : {rouge['rougeL']:>8.4f}")

    # ── Language mismatch ──
    print(f"\n  ── Language Check ─────────────────────────────────────")
    print(f"  Hindi predictions  : {lang['hindi_pct']:>6.1f}%")
    print(f"  Mixed predictions  : {lang['mixed_pct']:>6.1f}%")
    print(f"  English predictions: {lang['english_pct']:>6.1f}%")
    print(f"  Language mismatch  : {lang['mismatch_pct']:>6.1f}%  (target: 0%)")

    if lang["mismatch_pct"] == 0:
        print(f"  ✅ Zero language mismatch!")
    else:
        print(f"  ⚠️  Language mismatch detected")

    # ── Length stats ──
    print(f"\n  ── Prediction Length ──────────────────────────────────")
    print(f"  Avg prediction : {length_stats['avg_prediction_len']:>6.1f} chars")
    print(f"  Avg reference  : {length_stats['avg_reference_len']:>6.1f} chars")
    print(f"  Min prediction : {length_stats['min_prediction_len']:>6} chars")
    print(f"  Max prediction : {length_stats['max_prediction_len']:>6} chars")

    # ── Per-intent ──
    print(f"\n  ── Per-Intent ROUGE-1 ─────────────────────────────────")
    print(f"  {'Intent':<25} {'Count':>6} {'ROUGE-1':>8} {'ROUGE-L':>8} {'AvgPredLen':>10}")
    print(f"  {'─'*60}")
    for intent, m in sorted(per_intent.items(), key=lambda x: -x[1]["count"]):
        print(
            f"  {intent:<25} "
            f"{m['count']:>6,} "
            f"{m['rouge1']:>8.2f} "
            f"{m['rougeL']:>8.2f} "
            f"{m['avg_pred_len']:>10.1f}"
        )

    # ── Baseline comparison ──
    print(f"\n  ── Baseline Comparison ────────────────────────────────")
    print(f"  {'Metric':<12} {'Baseline A':>12} {'Baseline B':>12} {'Fine-tuned':>12}")
    print(f"  {'─'*50}")
    print(f"  {'BLEU-4':<12} {'~0.00':>12} {'~0.00':>12} {bleu['bleu_4']:>12.4f}")
    print(f"  {'chrF':<12} {'~1.00':>12} {'~1.00':>12} {chrf['chrf']:>12.4f}")
    print(f"  {'ROUGE-1':<12} {'~0.00':>12} {'~0.00':>12} {rouge['rouge1']:>12.4f}")
    print(f"  {'Lang miss':<12} {'100%':>12} {'100%':>12} {lang['mismatch_pct']:>11.1f}%")
    print(f"\n  Baseline A = zero-shot Mistral-7B (English output)")
    print(f"  Baseline B = five-shot Mistral-7B (English output)")

    print(f"\n{sep}")


# ── Save all results ──────────────────────────────────────────────────
def save_results(results: list, metrics: dict):
    # Save predictions JSONL
    pred_path = RESULTS_DIR / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.success(f"Predictions saved → {pred_path}")

    # Save metrics JSON
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.success(f"Metrics saved → {metrics_path}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("KisanMitra AI — Evaluation (Task 9a)")
    logger.info("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model, tokenizer = load_model(FINAL_MODEL_DIR)

    # Dataset + DataLoader
    dataset    = EvalDataset(TEST_PATH, tokenizer, sample_size=EVAL_SAMPLE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Generate predictions
    results = generate_predictions(model, tokenizer, dataloader, device)

    # Compute all metrics
    logger.info("Computing metrics...")
    bleu         = compute_bleu(results)
    chrf         = compute_chrf(results)
    rouge        = compute_rouge(results)
    lang         = compute_language_mismatch(results)
    per_intent   = compute_per_intent_metrics(results)
    length_stats = compute_length_stats(results)

    # Full metrics dict
    metrics = {
        "bleu":         bleu,
        "chrf":         chrf,
        "rouge":        rouge,
        "language":     lang,
        "length":       length_stats,
        "per_intent":   per_intent,
        "sample_size":  EVAL_SAMPLE_SIZE,
    }

    # Print report
    print_eval_report(bleu, chrf, rouge, lang, per_intent, length_stats, EVAL_SAMPLE_SIZE)

    # Save everything
    save_results(results, metrics)

    logger.success("Task 9a complete — metrics saved.")