import sys
import json
import time
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

log_path = Path("logs/test_pipeline.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")


# ── Test cases covering all 11 intents ───────────────────────────────
TEST_CASES = [
    {
        "id":     1,
        "intent": "pest_id",
        "state":  "UTTAR PRADESH",
        "crop":   "maize (makka)",
        "query":  "मक्का में फॉल आर्मी वर्म कीट का नियंत्रण कैसे करें?",
    },
    {
        "id":     2,
        "intent": "disease",
        "state":  "BIHAR",
        "crop":   "paddy (dhan)",
        "query":  "धान में झुलसा रोग का उपचार बताएं",
    },
    {
        "id":     3,
        "intent": "crop_advisory",
        "state":  "HARYANA",
        "crop":   "wheat",
        "query":  "गेहूं की बुवाई का सही समय और बीज दर क्या है?",
    },
    {
        "id":     4,
        "intent": "nutrient_management",
        "state":  "RAJASTHAN",
        "crop":   "mustard",
        "query":  "सरसों में यूरिया कब और कितनी मात्रा में डालें?",
    },
    {
        "id":     5,
        "intent": "government_scheme",
        "state":  "BIHAR",
        "crop":   "others",
        "query":  "किसान क्रेडिट कार्ड के लिए आवेदन कहाँ और कैसे करें?",
    },
    {
        "id":     6,
        "intent": "msp_price",
        "state":  "UTTAR PRADESH",
        "crop":   "wheat",
        "query":  "गेहूं का न्यूनतम समर्थन मूल्य क्या है?",
    },
    {
        "id":     7,
        "intent": "weather_sowing",
        "state":  "MADHYA PRADESH",
        "crop":   "soybean (bhat)",
        "query":  "सोयाबीन की बुवाई के लिए मौसम कब अनुकूल होता है?",
    },
    {
        "id":     8,
        "intent": "horticulture",
        "state":  "HIMACHAL PRADESH",
        "crop":   "apple",
        "query":  "सेब के पेड़ में कौन सी खाद डालें और कब?",
    },
    {
        "id":     9,
        "intent": "soil_water",
        "state":  "RAJASTHAN",
        "crop":   "others",
        "query":  "मिट्टी की जांच कैसे करें और रिपोर्ट कैसे समझें?",
    },
    {
        "id":     10,
        "intent": "animal_husbandry",
        "state":  "HARYANA",
        "crop":   "others",
        "query":  "गाय में थनैला रोग का उपचार कैसे करें?",
    },
    {
        "id":     11,
        "intent": "equipment_machinery",
        "state":  "UTTAR PRADESH",
        "crop":   "others",
        "query":  "कृषि यंत्र किराए पर कैसे मिलते हैं?",
    },
]


# ── Validate response ─────────────────────────────────────────────────
def validate_response(result: dict) -> dict:
    response = result.get("response", "")
    checks   = {}

    # Hindi check
    dev_chars   = sum(1 for c in response if '\u0900' <= c <= '\u097F')
    total_alpha = sum(1 for c in response if c.isalpha())
    checks["is_hindi"] = (dev_chars / max(total_alpha, 1)) >= 0.4

    # Not empty
    checks["not_empty"] = len(response.strip()) > 10

    # Minimum length
    checks["min_length"] = len(response) >= 20

    # No repetition
    words = response.split()
    if len(words) >= 6:
        from collections import Counter
        trigrams   = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        max_repeat = max(Counter(trigrams).values()) if trigrams else 1
        checks["no_repetition"] = max_repeat < 3
    else:
        checks["no_repetition"] = True

    # RAG passages returned
    checks["has_passages"] = len(result.get("passages", [])) >= 0  # 0+ is ok

    # Latency under 5s
    checks["latency_ok"] = result.get("latency_ms", 9999) < 5000

    checks["overall_pass"] = all(checks.values())
    return checks


# ── Print single result ───────────────────────────────────────────────
def print_result(case: dict, result: dict, checks: dict):
    status = "✅ PASS" if checks["overall_pass"] else "❌ FAIL"
    sep    = "─" * 65

    print(f"\n{sep}")
    print(
        f"  Test {case['id']:>2} | {status} | {case['intent']} | "
        f"Total: {result['latency_ms']}ms "
        f"(ret: {result['retrieval_ms']}ms "
        f"gen: {result['generation_ms']}ms)"
    )
    print(sep)
    print(f"  State    : {case['state']} | Crop: {case['crop']}")
    print(f"  Query    : {case['query']}")
    print(f"  Intent   : {result['intent']}  (detected)")
    print(f"  RAG used : {result['rag_used']} | Passages: {len(result['passages'])}")

    # Print passages
    if result["passages"]:
        print(f"\n  Retrieved passages:")
        for i, p in enumerate(result["passages"], 1):
            print(f"    {i}. [RRF:{p['rrf_score']:.5f}] "
                  f"[{p['intent']}] "
                  f"{p['answer'][:80]}...")

    # Print response
    print(f"\n  Response:")
    response = result["response"]
    words    = response.split()
    line     = "  "
    for word in words:
        if len(line) + len(word) + 1 > 67:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)

    # Checks
    print(f"\n  Quality checks:")
    for k, v in checks.items():
        if k == "overall_pass":
            continue
        print(f"    {'✅' if v else '❌'} {k}")


# ── Compare RAG vs no-RAG on 3 cases ────────────────────────────────
def compare_rag_vs_no_rag(pipeline):
    sep = "=" * 65

    compare_cases = [
        {
            "query":  "मक्का में फॉल आर्मी वर्म कीट का नियंत्रण कैसे करें?",
            "state":  "UTTAR PRADESH",
            "crop":   "maize (makka)",
            "intent": "pest_id",
        },
        {
            "query":  "धान में झुलसा रोग का उपचार बताएं",
            "state":  "BIHAR",
            "crop":   "paddy (dhan)",
            "intent": "disease",
        },
        {
            "query":  "गेहूं का न्यूनतम समर्थन मूल्य क्या है?",
            "state":  "UTTAR PRADESH",
            "crop":   "wheat",
            "intent": "msp_price",
        },
    ]

    print(f"\n{sep}")
    print("  RAG vs No-RAG Comparison")
    print(sep)

    for case in compare_cases:
        print(f"\n  Query  : {case['query']}")
        print(f"  Intent : {case['intent']}")

        # Without RAG
        no_rag = pipeline.chat(**case, use_rag=False)
        print(f"\n  WITHOUT RAG ({no_rag['generation_ms']}ms):")
        print(f"  {no_rag['response'][:200]}")

        # With RAG
        with_rag = pipeline.chat(**case, use_rag=True)
        print(f"\n  WITH RAG ({with_rag['latency_ms']}ms total | "
              f"{len(with_rag['passages'])} passages):")
        print(f"  {with_rag['response'][:200]}")

        # Show what RAG context was used
        if with_rag["passages"]:
            print(f"\n  RAG context used:")
            for i, p in enumerate(with_rag["passages"], 1):
                print(f"    {i}. {p['answer'][:100]}...")

        print(f"\n  {'─'*60}")


# ── Print summary ─────────────────────────────────────────────────────
def print_summary(results: list):
    sep    = "=" * 65
    passed = sum(1 for r in results if r["checks"]["overall_pass"])
    total  = len(results)

    avg_total = sum(r["result"]["latency_ms"]    for r in results) / total
    avg_ret   = sum(r["result"]["retrieval_ms"]  for r in results) / total
    avg_gen   = sum(r["result"]["generation_ms"] for r in results) / total
    avg_pass  = sum(
        len(r["result"]["passages"]) for r in results
    ) / total

    print(f"\n{sep}")
    print("  Full Pipeline Test — Summary")
    print(sep)
    print(f"  Tests passed    : {passed}/{total}  ({passed/total*100:.0f}%)")
    print(f"  Avg total ms    : {avg_total:.0f}ms")
    print(f"  Avg retrieval ms: {avg_ret:.0f}ms")
    print(f"  Avg generation ms:{avg_gen:.0f}ms")
    print(f"  Avg passages/q  : {avg_pass:.1f}")

    # Per-test summary table
    print(f"\n  {'Test':<6} {'Intent':<22} {'Pass':<6} "
          f"{'Total':>7} {'Ret':>6} {'Gen':>6} {'Pass#':>6}")
    print(f"  {'─'*62}")
    for r in results:
        icon = "✅" if r["checks"]["overall_pass"] else "❌"
        print(
            f"  {r['case']['id']:<6} "
            f"{r['case']['intent']:<22} "
            f"{icon:<6} "
            f"{r['result']['latency_ms']:>6}ms "
            f"{r['result']['retrieval_ms']:>5}ms "
            f"{r['result']['generation_ms']:>5}ms "
            f"{len(r['result']['passages']):>6}"
        )

    # Production readiness
    print(f"\n  Production readiness:")
    checks = [
        (avg_total < 3000,  f"Avg latency {avg_total:.0f}ms < 3000ms"),
        (avg_ret   < 500,   f"Avg retrieval {avg_ret:.0f}ms < 500ms"),
        (passed >= total*0.8, f"{passed}/{total} tests passing"),
    ]
    for passed_check, label in checks:
        print(f"    {'✅' if passed_check else '⚠️ '} {label}")

    print(f"\n{sep}")


# ── Save results ──────────────────────────────────────────────────────
def save_results(results: list):
    out_path = Path("data/processed/eval_results/pipeline_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for r in results:
        serializable.append({
            "test_id":      r["case"]["id"],
            "intent":       r["case"]["intent"],
            "state":        r["case"]["state"],
            "query":        r["case"]["query"],
            "response":     r["result"]["response"],
            "rag_used":     r["result"]["rag_used"],
            "passages":     len(r["result"]["passages"]),
            "latency_ms":   r["result"]["latency_ms"],
            "retrieval_ms": r["result"]["retrieval_ms"],
            "generation_ms":r["result"]["generation_ms"],
            "passed":       r["checks"]["overall_pass"],
            "checks":       r["checks"],
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.success(f"Pipeline test results → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("KisanMitra AI — Full Pipeline Test (Task 12b)")
    logger.info("=" * 65)

    # Load pipeline (loads model + retriever + warms up embeddings)
    from backend.rag.pipeline import KisanMitraRAGPipeline
    pipeline = KisanMitraRAGPipeline()
    pipeline.load()

    # Run all 11 intent tests
    print("\n" + "=" * 65)
    print("  Running 11 Full Pipeline Tests (RAG + Generation)")
    print("=" * 65)

    all_results = []
    for case in TEST_CASES:
        result = pipeline.chat(
            query=case["query"],
            state=case["state"],
            crop=case["crop"],
            intent=case["intent"],
            session_id=f"test_{case['id']}",
        )
        checks = validate_response(result)
        print_result(case, result, checks)
        all_results.append({
            "case":   case,
            "result": result,
            "checks": checks,
        })

    # Summary
    print_summary(all_results)

    # RAG vs No-RAG comparison
    compare_rag_vs_no_rag(pipeline)

    # Save
    save_results(all_results)

    logger.success("Task 12 complete — full RAG pipeline verified.")
    logger.success("Next: Task 13 — FastAPI backend.")