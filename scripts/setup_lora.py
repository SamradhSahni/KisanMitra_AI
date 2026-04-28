import os
import sys
import torch
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

log_path = Path("logs/setup_lora.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

MODEL_NAME = os.getenv("BASE_MODEL_NAME", "google/mt5-base")


# ── Load quantized model + tokenizer ─────────────────────────────────
def load_base(model_name: str):
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BitsAndBytesConfig
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    logger.success(f"Base model loaded: {model_name}")
    return model, tokenizer


# ── Prepare model for kbit training ──────────────────────────────────
def prepare_for_kbit(model):
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,   # saves VRAM during backprop
    )
    logger.success("Model prepared for kbit training (gradient checkpointing enabled)")
    return model


# ── Apply LoRA adapters ───────────────────────────────────────────────
def apply_lora(model):
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = CONFIG["model"]["lora"]

    # Read target modules from config
    target_modules = lora_cfg["target_modules"]

    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,       # mT5 is encoder-decoder
        r=lora_cfg["r"],                        # rank = 16
        lora_alpha=lora_cfg["alpha"],           # alpha = 32
        lora_dropout=lora_cfg["dropout"],       # dropout = 0.05
        target_modules=target_modules,          # q, k, v, o, wi_0, wi_1, wo
        bias="none",                            # don't adapt bias terms
        inference_mode=False,                   # we are training
    )

    model = get_peft_model(model, config)
    logger.success("LoRA adapters applied successfully")

    return model, config


# ── Print trainable parameter stats ──────────────────────────────────
def print_trainable_params(model):
    trainable_params = 0
    total_params     = 0

    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = (trainable_params / total_params) * 100

    sep = "=" * 60
    print(f"\n{sep}")
    print("  LoRA Parameter Summary")
    print(sep)
    print(f"  Total parameters       : {total_params:>12,}")
    print(f"  Trainable (LoRA) params: {trainable_params:>12,}")
    print(f"  Frozen parameters      : {total_params - trainable_params:>12,}")
    print(f"  Trainable %            : {pct:>11.2f}%")
    print(f"\n  Expected: ~6.7M trainable out of 580M total (~1.15%)")

    if pct < 0.5:
        logger.warning("Very few trainable params — check target_modules in config.yaml")
    elif pct > 5:
        logger.warning("More than 5% trainable — LoRA may be too large for 6GB VRAM")
    else:
        logger.success(f"✅ {pct:.2f}% trainable — good for 6GB VRAM training")

    print(sep)


# ── Print LoRA adapter layer names ────────────────────────────────────
def print_lora_layers(model):
    logger.info("\nLoRA adapter layers (requires_grad=True):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  ✅ {name:<60} shape={list(param.shape)}")


# ── Save LoRA config for reference ───────────────────────────────────
def save_lora_config_json(lora_config):
    import json
    out = {
        "task_type":       "SEQ_2_SEQ_LM",
        "r":               lora_config.r,
        "lora_alpha":      lora_config.lora_alpha,
        "lora_dropout":    lora_config.lora_dropout,
        "target_modules":  list(lora_config.target_modules),
        "bias":            lora_config.bias,
        "base_model":      MODEL_NAME,
    }
    out_path = Path("model/checkpoints/lora_config.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.success(f"LoRA config saved → {out_path}")


# ── Quick forward pass with LoRA ──────────────────────────────────────
def verify_lora_forward(model, tokenizer):
    logger.info("\nVerifying LoRA forward pass with gradient computation...")

    test_input  = (
        "निर्देश: आप एक कृषि विशेषज्ञ हैं। किसान की समस्या का उत्तर हिंदी में दें।\n"
        "राज्य: बिहार\nफसल: गेहूं\nसमस्या का प्रकार: मौसम एवं बुवाई\n"
        "किसान का प्रश्न: गेहूं की बुवाई कब करें?\nउत्तर:"
    )
    test_target = "गेहूं की बुवाई नवंबर के पहले सप्ताह में करें।"

    device = next(model.parameters()).device

    inputs  = tokenizer(
        test_input,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    ).to(device)

    targets = tokenizer(
        test_target,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)

    # This time we DO compute gradients (training mode)
    model.train()
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=targets["input_ids"],
    )

    loss = outputs.loss
    loss.backward()   # confirm gradients flow through LoRA layers

    # Check gradients exist on LoRA params
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break

    if has_grad:
        logger.success(f"✅ Gradients flow through LoRA layers — loss: {loss.item():.4f}")
    else:
        logger.error("❌ No gradients on LoRA params — check config")

    # Reset gradients
    model.zero_grad()
    model.eval()

    return has_grad


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("KisanMitra AI — LoRA Setup (Task 7b)")
    logger.info("=" * 60)

    # Load base quantized model
    model, tokenizer = load_base(MODEL_NAME)

    # Prepare for kbit training
    model = prepare_for_kbit(model)

    # Apply LoRA
    model, lora_config = apply_lora(model)

    # Print stats
    print_trainable_params(model)
    print_lora_layers(model)

    # Save config
    save_lora_config_json(lora_config)

    # Verify forward + backward pass
    verify_lora_forward(model, tokenizer)

    sep = "=" * 60
    logger.success(f"\n{sep}")
    logger.success("Task 7b complete — LoRA adapters verified and ready.")
    logger.success("Next: Task 8 — Training loop.")
    logger.success(sep)