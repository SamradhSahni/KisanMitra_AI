import os
import psycopg2
import psycopg2.pool
from contextlib import contextmanager
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST",     "localhost")
DB_PORT = os.getenv("DB_PORT",     "5432")
DB_NAME = os.getenv("DB_NAME",     "kisanmitra")
DB_USER = os.getenv("DB_USER",     "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "your_password_here")

# ── Connection pool ───────────────────────────────────────────────────
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
        )
        logger.success("PostgreSQL connection pool created")
    return _pool


@contextmanager
def get_db():
    """Context manager for DB connections from pool."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"DB error: {e}")
        raise
    finally:
        pool.putconn(conn)


# ── Session operations ────────────────────────────────────────────────
def upsert_session(session_id: str, state: str, crop: str, language: str = "hi"):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO sessions (session_id, state, crop, language, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (session_id)
            DO UPDATE SET
                state      = EXCLUDED.state,
                crop       = EXCLUDED.crop,
                updated_at = NOW()
        """, (session_id, state, crop, language))
        cur.close()


def get_session(session_id: str) -> dict:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT session_id, state, crop, language, created_at
            FROM sessions WHERE session_id = %s
        """, (session_id,))
        row = cur.fetchone()
        cur.close()
    if not row:
        return None
    return {
        "session_id": row[0],
        "state":      row[1],
        "crop":       row[2],
        "language":   row[3],
        "created_at": row[4],
    }


# ── Message operations ────────────────────────────────────────────────
def save_message(
    session_id:     str,
    role:           str,
    content:        str,
    intent:         str   = None,
    rag_used:       bool  = False,
    passages_count: int   = 0,
    latency_ms:     int   = None,
):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO messages
                (session_id, role, content, intent, rag_used, passages_count, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (session_id, role, content, intent, rag_used, passages_count, latency_ms))
        msg_id = cur.fetchone()[0]
        cur.close()
    return msg_id


def get_session_messages(session_id: str, limit: int = 20) -> list:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT role, content, intent, rag_used, latency_ms, created_at
            FROM messages
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (session_id, limit))
        rows = cur.fetchall()
        cur.close()
    return [
        {
            "role":       r[0],
            "content":    r[1],
            "intent":     r[2],
            "rag_used":   r[3],
            "latency_ms": r[4],
            "created_at": r[5],
        }
        for r in reversed(rows)
    ]


# ── Feedback operations ───────────────────────────────────────────────
def save_feedback(
    session_id: str,
    query:      str,
    response:   str,
    rating:     int,
    comment:    str  = None,
    intent:     str  = None,
    state:      str  = None,
    crop:       str  = None,
) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO feedback
                (session_id, query, response, rating, comment, intent, state, crop)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (session_id, query, response, rating, comment, intent, state, crop))
        feedback_id = cur.fetchone()[0]
        cur.close()
    return feedback_id


# ── MSP operations ────────────────────────────────────────────────────
def get_msp_from_db(crop: str) -> dict:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT crop, price, unit, season, year, source
            FROM msp_prices
            WHERE LOWER(crop) = LOWER(%s)
            LIMIT 1
        """, (crop,))
        row = cur.fetchone()
        cur.close()
    if not row:
        return None
    return {
        "crop":   row[0],
        "price":  float(row[1]),
        "unit":   row[2],
        "season": row[3],
        "year":   row[4],
        "source": row[5],
    }


# ── Stats query ───────────────────────────────────────────────────────
def get_usage_stats() -> dict:
    with get_db() as conn:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM messages WHERE role = 'user'")
        total_queries = cur.fetchone()[0]

        cur.execute("SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL")
        avg_rating = cur.fetchone()[0]

        cur.execute("""
            SELECT intent, COUNT(*) as cnt
            FROM messages WHERE role = 'user' AND intent IS NOT NULL
            GROUP BY intent ORDER BY cnt DESC LIMIT 5
        """)
        top_intents = cur.fetchall()

        cur.close()

    return {
        "total_sessions": total_sessions,
        "total_queries":  total_queries,
        "avg_rating":     round(float(avg_rating), 2) if avg_rating else None,
        "top_intents":    [{"intent": r[0], "count": r[1]} for r in top_intents],
    }