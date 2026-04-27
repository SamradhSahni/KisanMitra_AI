import sys
import importlib
from loguru import logger

def check_package(name, import_name=None):
    try:
        mod = importlib.import_module(import_name or name)
        version = getattr(mod, "__version__", "unknown")
        logger.success(f"✅ {name} — v{version}")
        return True
    except ImportError:
        logger.error(f"❌ {name} — NOT FOUND")
        return False

def main():
    logger.info("=" * 50)
    logger.info("KisanMitra AI — Environment Verification")
    logger.info("=" * 50)

    # Python version
    logger.info(f"\nPython: {sys.version}")

    # Core packages
    logger.info("\n── Core ML ──")
    check_package("torch")
    check_package("transformers")
    check_package("datasets")
    check_package("peft")
    check_package("bitsandbytes")
    check_package("accelerate")
    check_package("trl")

    logger.info("\n── Data Processing ──")
    check_package("pandas")
    check_package("numpy")
    check_package("datasketch")
    check_package("sacrebleu")
    check_package("rouge_score", "rouge_score")
    check_package("sentence_transformers", "sentence_transformers")

    logger.info("\n── Backend ──")
    check_package("fastapi")
    check_package("uvicorn")
    check_package("pydantic")
    check_package("redis")
    check_package("psycopg2", "psycopg2")
    check_package("boto3")
    check_package("elasticsearch")

    logger.info("\n── Utilities ──")
    check_package("dotenv", "dotenv")
    check_package("yaml", "yaml")
    check_package("tqdm")
    check_package("loguru")

    # GPU check
    logger.info("\n── GPU / CUDA ──")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.success(f"✅ CUDA available")
            logger.success(f"✅ GPU: {gpu_name}")
            logger.success(f"✅ VRAM: {vram:.1f} GB")
        else:
            logger.warning("⚠️  CUDA not available — will train on CPU (very slow)")
    except Exception as e:
        logger.error(f"❌ GPU check failed: {e}")

    # Config loader check
    logger.info("\n── Project Config ──")
    try:
        sys.path.insert(0, ".")
        from utils.config_loader import CONFIG
        logger.success(f"✅ config.yaml loaded")
        logger.success(f"✅ Project: {CONFIG['project']['name']}")
        logger.success(f"✅ Target states: {len(CONFIG['target_states'])} states")
        logger.success(f"✅ Intents: {list(CONFIG['intents'].keys())}")
    except Exception as e:
        logger.error(f"❌ Config load failed: {e}")

    # .env check
    logger.info("\n── Environment Variables ──")
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        keys_to_check = [
            "APP_NAME", "BASE_MODEL_NAME",
            "RAW_DATA_PATH", "ES_HOST", "DB_HOST"
        ]
        for key in keys_to_check:
            val = os.getenv(key)
            if val:
                logger.success(f"✅ {key} = {val}")
            else:
                logger.warning(f"⚠️  {key} not set")
    except Exception as e:
        logger.error(f"❌ .env check failed: {e}")

    logger.info("\n" + "=" * 50)
    logger.info("Verification complete!")

if __name__ == "__main__":
    main()