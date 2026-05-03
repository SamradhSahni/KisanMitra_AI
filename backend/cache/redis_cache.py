import os
import json
import redis
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST    = os.getenv("REDIS_HOST",        "localhost")
REDIS_PORT    = int(os.getenv("REDIS_PORT",    6379))
TTL_MSP       = int(os.getenv("REDIS_TTL_MSP",     86400))   # 24 hours
TTL_SESSION   = int(os.getenv("REDIS_TTL_SESSION",  3600))   # 1 hour

# ── Singleton Redis client ────────────────────────────────────────────
_redis_client = None

def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
        )
        # Test connection
        _redis_client.ping()
        logger.success(f"Redis connected at {REDIS_HOST}:{REDIS_PORT}")
    return _redis_client


def is_redis_available() -> bool:
    try:
        get_redis().ping()
        return True
    except Exception:
        return False


# ── MSP Cache (TTL: 24 hours) ─────────────────────────────────────────
def get_cached_msp(crop: str) -> dict:
    """Get MSP price from Redis cache."""
    if not is_redis_available():
        return None
    try:
        key  = f"msp:{crop.lower().strip()}"
        data = get_redis().get(key)
        if data:
            logger.debug(f"Cache HIT: {key}")
            return json.loads(data)
        logger.debug(f"Cache MISS: {key}")
        return None
    except Exception as e:
        logger.warning(f"Redis get error: {e}")
        return None


def set_cached_msp(crop: str, data: dict):
    """Cache MSP price in Redis for 24 hours."""
    if not is_redis_available():
        return
    try:
        key = f"msp:{crop.lower().strip()}"
        get_redis().setex(key, TTL_MSP, json.dumps(data))
        logger.debug(f"Cached MSP: {key} (TTL: {TTL_MSP}s)")
    except Exception as e:
        logger.warning(f"Redis set error: {e}")


# ── Session Context Cache (TTL: 1 hour) ──────────────────────────────
def get_session_context(session_id: str) -> dict:
    """Get session context (state, crop, recent intent) from Redis."""
    if not is_redis_available():
        return None
    try:
        key  = f"session:{session_id}"
        data = get_redis().get(key)
        if data:
            logger.debug(f"Session cache HIT: {key}")
            return json.loads(data)
        return None
    except Exception as e:
        logger.warning(f"Redis session get error: {e}")
        return None


def set_session_context(session_id: str, context: dict):
    """Cache session context for 1 hour."""
    if not is_redis_available():
        return
    try:
        key = f"session:{session_id}"
        get_redis().setex(key, TTL_SESSION, json.dumps(context))
        logger.debug(f"Cached session: {key} (TTL: {TTL_SESSION}s)")
    except Exception as e:
        logger.warning(f"Redis session set error: {e}")


def refresh_session_ttl(session_id: str):
    """Extend session TTL on each interaction."""
    if not is_redis_available():
        return
    try:
        key = f"session:{session_id}"
        get_redis().expire(key, TTL_SESSION)
    except Exception as e:
        logger.warning(f"Redis TTL refresh error: {e}")


def delete_session(session_id: str):
    """Clear session from cache."""
    if not is_redis_available():
        return
    try:
        get_redis().delete(f"session:{session_id}")
    except Exception as e:
        logger.warning(f"Redis delete error: {e}")


# ── Cache stats ───────────────────────────────────────────────────────
def get_cache_stats() -> dict:
    if not is_redis_available():
        return {"available": False}
    try:
        r     = get_redis()
        info  = r.info()
        msp_keys     = len(r.keys("msp:*"))
        session_keys = len(r.keys("session:*"))
        return {
            "available":     True,
            "msp_cached":    msp_keys,
            "sessions_live": session_keys,
            "memory_used":   info.get("used_memory_human", "N/A"),
            "uptime_days":   info.get("uptime_in_days", 0),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}