import os
import sys
import json
import torch
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

log_path = Path("logs/save_model.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

MODEL_NAME      = os.getenv("BASE_MODEL_NAME",        "google/mt5-base")
CHECKPOINT_DIR  = Path(os.getenv("CHECKPOINT_DIR",    "./model/checkpoints"))
FINAL_DIR       = Path(os.getenv("FINETUNED_MODEL_PATH", "./model/final"))
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Load best checkpoint info ─────────────────────────────────────────
def get_best_checkpoint() -> dict:
    best_path = CHECKPOINT_DIR / "best_checkpoint.json"
    if not best_path.exists():
        raise FileNotFoundError(
            "best_checkpoint.json not found. "
            "Make sure training completed at least 1 epoch."
        )
    with open(best_path) as f:
        info = json.load(f)
    logger.info(f"Best checkpoint: epoch={info['epoch']} val_loss={info['val_loss']:.4f}")
    logger.info(f"Checkpoint path: {info['path']}")
    return info


# ── Load training log ─────────────────────────────────────────────────
def print_training_log():
    log_path = CHECKPOINT_DIR / "training_log.json"
    if not log_path.exists():
        logger.warning("training_log.json not found — skipping")
        return

    with open(log_path) as f:
        log = json.load(f)

    sep = "=" * 65
    print(f"\n{sep}")
    print("  Full Training Log")
    print(sep)
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'LR'}")
    print(f"  {'─'*55}")
    for entry in log:
        marker = " ← best" if entry["val_loss"] == min(e["val_loss"] for e in log) else ""
        print(
            f"  {entry['epoch']:<8} "
            f"{entry['train_loss']:<14.4f} "
            f"{entry['val_loss']:<14.4f} "
            f"{entry['lr']:.2e}"
            f"{marker}"
        )
    print(f"{sep}\n")


# ── Merge LoRA adapters into base model ───────────────────────────────
def merge_and_save(checkpoint_path: str):
    """
    Load the best LoRA checkpoint, merge adapters into base model
    weights, and save the full merged model to model/final/.
    This gives a standalone model that doesn't need PEFT at inference.
    """
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info("Loading base model for merging (fp32, no quantization)...")
    # Load in fp32 for merging — quantized models cannot be merged directly
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",           # merge on CPU to avoid VRAM issues
    )

    logger.info(f"Loading LoRA adapters from: {checkpoint_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch.float32,
    )

    logger.info("Merging LoRA adapters into base model weights...")
    merged_model = peft_model.merge_and_unload()
    logger.success("Merge complete — adapters folded into base weights")

    # Save merged model
    logger.info(f"Saving merged model to: {FINAL_DIR}")
    merged_model.save_pretrained(str(FINAL_DIR))

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.save_pretrained(str(FINAL_DIR))

    logger.success(f"Final model saved to: {FINAL_DIR}")
    return merged_model, tokenizer


# ── Quick inference test on merged model ──────────────────────────────
def test_inference(model, tokenizer):
    """
    Run 5 sample inference passes on the merged model.
    No gradients needed — pure generation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    test_cases = [
        {
            "label": "Weather query — Bihar — Wheat",
            "input": (
                "निर्देश: आप एक कृषि विशेषज्ञ हैं। किसान की समस्या का उत्तर हिंदी में दें।\n"
                "राज्य: बिहार\nफसल: गेहूं\nसमस्या का प्रकार: मौसम एवं बुवाई\n"
                "किसान का प्रश्न: गेहूं की बुवाई कब करें?\nउत्तर:"
            ),
        },
        {
            "label": "Pest query — UP — Maize",
            "input": (
                "निर्देश: आप एक कृषि विशेषज्ञ हैं। किसान की समस्या का उत्तर हिंदी में दें।\n"
                "राज्य: उत्तर प्रदेश\nफसल: मक्का\nसमस्या का प्रकार: कीट प्रबंधन\n"
                "किसान का प्रश्न: मक्का में फॉल आर्मी वर्म कीट का नियंत्रण कैसे करें?\nउत्तर:"
            ),
        },
        {
            "label": "Nutrient query — Rajasthan — Mustard",
            "input": (
                "निर्देश: आप एक कृषि विशेषज्ञ हैं। किसान की समस्या का उत्तर हिंदी में दें।\n"
                "राज्य: राजस्थान\nफसल: सरसों\nसमस्या का प्रकार: पोषक तत्व प्रबंधन\n"
                "किसान का प्रश्न: सरसों में यूरिया का छिड़काव कब करें?\nउत्तर:"
            ),
        },
        {
            "label": "Government scheme — Haryana",
            "input": (
                "निर्देश: आप एक कृषि विशेषज्ञ हैं। किसान की समस्या का उत्तर हिंदी में दें।\n"
                "राज्य: हरियाणा\nफसल: अन्य\nसमस्या का प्रकार: सरकारी योजना\n"
                "किसान का प्रश्न: किसान क्रेडिट कार्ड के लिए आवेदन कैसे करें?\nउत्तर:"
            ),
        },
        {
            "label": "Disease query — MP — Soybean",
            "input": (
                "निर्देश: आप एक कृषि विशेषज्ञ हैं। किसान की समस्या का उत्तर हिंदी में दें।\n"
                "राज्य: मध्य प्रदेश\nफसल: सोयाबीन\nसमस्या का प्रकार: रोग प्रबंधन\n"
                "किसान का प्रश्न: सोयाबीन में पीला मोजेक वायरस का उपचार क्या है?\nउत्तर:"
            ),
        },
    ]

    sep = "=" * 65
    print(f"\n{sep}")
    print("  Merged Model — Inference Test")
    print(sep)

    for i, case in enumerate(test_cases, 1):
        inputs = tokenizer(
            case["input"],
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        response = tokenizer.decode(generated[0], skip_special_tokens=True)

        print(f"\n── Test {i}: {case['label']}")
        print(f"   Query    : {case['input'].split('किसान का प्रश्न:')[-1].split('उत्तर:')[0].strip()}")
        print(f"   Response : {response[:250]}")

    print(f"\n{sep}")


# ── Save model card ───────────────────────────────────────────────────
def save_model_card(best_info: dict):
    card = f"""# KisanMitra AI — Fine-tuned mT5-base

## Model Details
- **Base model**: google/mt5-base (580M parameters)
- **Fine-tuning**: QLoRA (4-bit NF4, r=16, alpha=32)
- **Trainable params**: ~6.7M (1.15% of total)
- **Training data**: 20,000 Hindi-Hindi agricultural QA pairs
- **Best val loss**: {best_info['val_loss']:.4f} (epoch {best_info['epoch']})

## Task
Seq2Seq instruction-following for Hindi agricultural advisory.
Input: Hindi instruction with state, crop, intent, query.
Output: Hindi advisory answer.

## Intents Supported
weather_sowing, crop_advisory, pest_id, disease,
nutrient_management, msp_price, government_scheme,
horticulture, soil_water, animal_husbandry, equipment_machinery

## Dataset
- Source: KCC (Kisan Call Centre) helpline logs
- States: UP, Rajasthan, Haryana, Bihar, MP, Chhattisgarh,
          Himachal Pradesh, Jharkhand, Uttarakhand
- Language: Hindi (Devanagari)
- Queries translated from English using IndicTrans2
"""
    card_path = FINAL_DIR / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    logger.success(f"Model card saved → {card_path}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("KisanMitra AI — Save Final Model (Task 8b)")
    logger.info("=" * 65)

    # Print training log
    print_training_log()

    # Get best checkpoint
    best_info = get_best_checkpoint()

    # Merge and save
    merged_model, tokenizer = merge_and_save(best_info["path"])

    # Test inference
    test_inference(merged_model, tokenizer)

    # Save model card
    save_model_card(best_info)

    sep = "=" * 65
    logger.success(f"\n{sep}")
    logger.success(f"Task 8b complete — final model saved to: {FINAL_DIR}")
    logger.success(f"Best val loss   : {best_info['val_loss']:.4f}")
    logger.success(f"Best epoch      : {best_info['epoch']}")
    logger.success("Next: Task 9 — Evaluation (BLEU, chrF, ROUGE).")
    logger.success(sep)