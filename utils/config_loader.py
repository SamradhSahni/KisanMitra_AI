import os
import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

def load_config(config_path: str = "config.yaml") -> dict:
    """Load the master YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config

def get_env(key: str, default=None, required: bool = False):
    """Safely fetch an environment variable."""
    value = os.getenv(key, default)
    if required and value is None:
        raise EnvironmentError(f"Required env variable '{key}' is not set in .env")
    return value

# Pre-load for import convenience
CONFIG = load_config()