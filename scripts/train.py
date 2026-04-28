import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from tqdm import tqdm

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

# ── Paths & Hyperparameters ───────────────────────────────────────────
MODEL_NAME      = os.getenv("BASE_MODEL_NAME",        "google/mt5-base")
TRAIN_PATH      = "./data/processed/train_sample.jsonl"
VAL_PATH        = "./data/processed/val.jsonl"
CHECKPOINT_DIR  = Path(os.getenv("CHECKPOINT_DIR",    "./model/checkpoints"))
FINAL_DIR       = Path(os.getenv("FINETUNED_MODEL_PATH", "./model/final"))
LOG_PATH        = Path("logs/training.log")

LOG_PATH.parent.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
logger.add(str(LOG_PATH), rotation="50 MB", encoding="utf-8")

# Training hyperparameters — read from .env / config
MAX_INPUT_LEN   = int(os.getenv("MAX_INPUT_LENGTH",  256))
MAX_TARGET_LEN  = int(os.getenv("MAX_OUTPUT_LENGTH", 128))
TRAIN_BATCH     = int(os.getenv("TRAIN_BATCH_SIZE",  4))
EVAL_BATCH      = int(os.getenv("EVAL_BATCH_SIZE",   4))
GRAD_ACCUM      = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 8))
NUM_EPOCHS      = int(os.getenv("NUM_EPOCHS",        3))
LR              = float(os.getenv("LEARNING_RATE",   1e-4))
WARMUP_RATIO    = float(os.getenv("WARMUP_RATIO",    0.05))
PATIENCE        = int(os.getenv("EARLY_STOPPING_PATIENCE", 3))
VAL_SAMPLE_SIZE = 2000    # use subset of val for speed during training


# ── Dataset ───────────────────────────────────────────────────────────
class KisanMitraDataset(Dataset):
    """
    PyTorch Dataset for mT5 instruction fine-tuning.
    Tokenizes input_text → input_ids and target_text → labels.
    """

    def __init__(
        self,
        filepath: str,
        tokenizer,
        max_input_len: int,
        max_target_len: int,
        sample_size: int = None,
    ):
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.records        = []

        # Load JSONL
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Optional sampling for val speed
        if sample_size and sample_size < len(self.records):
            import random
            random.seed(42)
            self.records = random.sample(self.records, sample_size)

        logger.info(f"Dataset loaded: {filepath} — {len(self.records):,} records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        input_text  = str(record.get("input_text",  ""))
        target_text = str(record.get("target_text", ""))

        # Tokenize input
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        # Replace padding token id in labels with -100
        # so they are ignored in loss computation
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         label_ids,
        }


# ── Load model + tokenizer ────────────────────────────────────────────
def load_model_and_tokenizer():
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    logger.info(f"Loading model with 4-bit NF4 quantization: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Apply LoRA
    lora_cfg = CONFIG["model"]["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    return model, tokenizer


# ── Evaluation ────────────────────────────────────────────────────────
def evaluate(model, dataloader, device):
    """
    Compute average loss over validation set.
    Returns mean loss across all batches.
    """
    model.eval()
    total_loss  = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss  += outputs.loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    model.train()
    return avg_loss


# ── Training loop ─────────────────────────────────────────────────────
def train(model, tokenizer, train_loader, val_loader, device):
    """
    Full training loop with:
    - AdamW optimizer
    - Cosine LR schedule with warmup
    - Gradient accumulation
    - Early stopping on val loss
    - Checkpoint saving on best val loss
    """

    # Effective batch size = TRAIN_BATCH * GRAD_ACCUM = 4 * 8 = 32
    total_steps    = (len(train_loader) * NUM_EPOCHS) // GRAD_ACCUM
    warmup_steps   = int(total_steps * WARMUP_RATIO)

    logger.info(f"\nTraining config:")
    logger.info(f"  Epochs            : {NUM_EPOCHS}")
    logger.info(f"  Train batches/ep  : {len(train_loader):,}")
    logger.info(f"  Val batches       : {len(val_loader):,}")
    logger.info(f"  Batch size        : {TRAIN_BATCH}")
    logger.info(f"  Grad accumulation : {GRAD_ACCUM}")
    logger.info(f"  Effective batch   : {TRAIN_BATCH * GRAD_ACCUM}")
    logger.info(f"  Total steps       : {total_steps:,}")
    logger.info(f"  Warmup steps      : {warmup_steps:,}")
    logger.info(f"  Learning rate     : {LR}")
    logger.info(f"  Early stop after  : {PATIENCE} epochs no improvement")

    # Optimizer — only update trainable (LoRA) params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.01,
    )

    # Cosine LR schedule with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training state
    best_val_loss     = float("inf")
    patience_counter  = 0
    global_step       = 0
    train_losses      = []
    val_losses        = []

    # Training log file
    training_log = []

    sep = "=" * 65
    print(f"\n{sep}")
    print("  KisanMitra AI — Training Started")
    print(sep)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_steps   = 0
        optimizer.zero_grad()

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
        )

        for step, batch in progress:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Scale loss by grad accumulation steps
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            epoch_loss  += outputs.loss.item()
            epoch_steps += 1

            # Gradient accumulation step
            if (step + 1) % GRAD_ACCUM == 0:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Update progress bar
            avg_train_loss = epoch_loss / epoch_steps
            current_lr     = scheduler.get_last_lr()[0]
            progress.set_postfix({
                "loss": f"{avg_train_loss:.4f}",
                "lr":   f"{current_lr:.2e}",
                "step": global_step,
            })

        # ── End of epoch: evaluate ──
        avg_train_loss = epoch_loss / epoch_steps
        avg_val_loss   = evaluate(model, val_loader, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Log epoch results
        log_entry = {
            "epoch":          epoch,
            "train_loss":     round(avg_train_loss, 4),
            "val_loss":       round(avg_val_loss, 4),
            "global_step":    global_step,
            "lr":             round(current_lr, 8),
        }
        training_log.append(log_entry)

        print(f"\n{'─'*65}")
        print(f"  Epoch {epoch}/{NUM_EPOCHS} Complete")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val Loss   : {avg_val_loss:.4f}")
        print(f"  LR         : {current_lr:.2e}")
        print(f"  Step       : {global_step:,}")

        # Save training log after every epoch
        log_path = CHECKPOINT_DIR / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        # ── Checkpoint: save if best val loss ──
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0

            checkpoint_path = CHECKPOINT_DIR / f"epoch_{epoch}_valloss_{avg_val_loss:.4f}"
            model.save_pretrained(str(checkpoint_path))
            tokenizer.save_pretrained(str(checkpoint_path))

            # Save best marker
            best_info = {
                "epoch":     epoch,
                "val_loss":  avg_val_loss,
                "train_loss": avg_train_loss,
                "path":      str(checkpoint_path),
            }
            with open(CHECKPOINT_DIR / "best_checkpoint.json", "w") as f:
                json.dump(best_info, f, indent=2)

            print(f"  ✅ New best val loss: {avg_val_loss:.4f} — checkpoint saved")
            print(f"     → {checkpoint_path}")

        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"\n  🛑 Early stopping triggered after {epoch} epochs.")
                logger.info(f"Early stopping at epoch {epoch} — best val loss: {best_val_loss:.4f}")
                break

        print(f"{'─'*65}\n")

    print(f"\n{sep}")
    print(f"  Training Complete")
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Total Epochs  : {epoch}")
    print(f"  Total Steps   : {global_step:,}")
    print(f"{sep}\n")

    return best_val_loss, training_log


# ── Print training summary ────────────────────────────────────────────
def print_training_summary(training_log: list):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  Training Log Summary")
    print(sep)
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'LR':<12} {'Step'}")
    print(f"  {'─'*60}")
    for entry in training_log:
        print(
            f"  {entry['epoch']:<8} "
            f"{entry['train_loss']:<14.4f} "
            f"{entry['val_loss']:<14.4f} "
            f"{entry['lr']:<12.2e} "
            f"{entry['global_step']:,}"
        )
    print(f"{sep}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("KisanMitra AI — Training (Task 8a)")
    logger.info("=" * 65)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Datasets
    logger.info("Loading datasets...")
    train_dataset = KisanMitraDataset(
        TRAIN_PATH, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN
    )
    val_dataset = KisanMitraDataset(
        VAL_PATH, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN,
        sample_size=VAL_SAMPLE_SIZE,    # use 2K subset for fast eval
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        num_workers=0,              # keep 0 on Windows
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    logger.info(f"Train samples : {len(train_dataset):,}")
    logger.info(f"Val samples   : {len(val_dataset):,}")

    # Train
    best_val_loss, training_log = train(
        model, tokenizer, train_loader, val_loader, device
    )

    # Print summary table
    print_training_summary(training_log)

    logger.success(f"Task 8a complete — best val loss: {best_val_loss:.4f}")
    logger.success("Run Task 8b next to monitor and save the final model.")