import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

log_path = Path("logs/db.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

DB_HOST = os.getenv("DB_HOST",     "localhost")
DB_PORT = os.getenv("DB_PORT",     "5432")
DB_NAME = os.getenv("DB_NAME",     "kisanmitra")
DB_USER = os.getenv("DB_USER",     "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "your_password_here")


# ── Create database if not exists ─────────────────────────────────────
def create_database():
    """Connect to default postgres DB and create kisanmitra DB."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database="postgres",    # connect to default first
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Check if DB exists
    cur.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (DB_NAME,)
    )
    exists = cur.fetchone()

    if not exists:
        cur.execute(f"CREATE DATABASE {DB_NAME}")
        logger.success(f"Database '{DB_NAME}' created")
    else:
        logger.info(f"Database '{DB_NAME}' already exists")

    cur.close()
    conn.close()


# ── Create all tables ─────────────────────────────────────────────────
def create_tables():
    """Create all required tables in the kisanmitra database."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
    )
    cur = conn.cursor()

    # ── Sessions table ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          SERIAL PRIMARY KEY,
            session_id  VARCHAR(100) UNIQUE NOT NULL,
            state       VARCHAR(50),
            crop        VARCHAR(100),
            language    VARCHAR(10)  DEFAULT 'hi',
            created_at  TIMESTAMP    DEFAULT NOW(),
            updated_at  TIMESTAMP    DEFAULT NOW()
        );
    """)
    logger.info("Table 'sessions' ready")

    # ── Messages table ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              SERIAL PRIMARY KEY,
            session_id      VARCHAR(100) REFERENCES sessions(session_id),
            role            VARCHAR(10)  NOT NULL CHECK (role IN ('user', 'assistant')),
            content         TEXT         NOT NULL,
            intent          VARCHAR(50),
            rag_used        BOOLEAN      DEFAULT FALSE,
            passages_count  INTEGER      DEFAULT 0,
            latency_ms      INTEGER,
            created_at      TIMESTAMP    DEFAULT NOW()
        );
    """)
    logger.info("Table 'messages' ready")

    # ── MSP prices table ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS msp_prices (
            id          SERIAL PRIMARY KEY,
            crop        VARCHAR(100) NOT NULL,
            price       NUMERIC(10,2),
            unit        VARCHAR(50)  DEFAULT '₹/quintal',
            season      VARCHAR(20),
            year        VARCHAR(10),
            source      VARCHAR(200),
            updated_at  TIMESTAMP    DEFAULT NOW()
        );
    """)
    logger.info("Table 'msp_prices' ready")

    # ── Feedback table ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          SERIAL PRIMARY KEY,
            session_id  VARCHAR(100),
            query       TEXT,
            response    TEXT,
            rating      INTEGER      CHECK (rating BETWEEN 1 AND 5),
            comment     TEXT,
            intent      VARCHAR(50),
            state       VARCHAR(50),
            crop        VARCHAR(100),
            created_at  TIMESTAMP    DEFAULT NOW()
        );
    """)
    logger.info("Table 'feedback' ready")

    # ── Indexes ──
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_created
        ON messages(created_at);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_session
        ON feedback(session_id);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_rating
        ON feedback(rating);
    """)

    conn.commit()
    cur.close()
    conn.close()
    logger.success("All tables and indexes created")


# ── Seed MSP prices ───────────────────────────────────────────────────
def seed_msp_prices():
    """Insert MSP prices into the database."""
    MSP_DATA = [
        ("wheat",        2275,  "₹/quintal", "Rabi",   "2024-25"),
        ("paddy",        2300,  "₹/quintal", "Kharif", "2024-25"),
        ("maize",        2225,  "₹/quintal", "Kharif", "2024-25"),
        ("mustard",      5650,  "₹/quintal", "Rabi",   "2024-25"),
        ("soybean",      4892,  "₹/quintal", "Kharif", "2024-25"),
        ("groundnut",    6783,  "₹/quintal", "Kharif", "2024-25"),
        ("cotton",       7121,  "₹/quintal", "Kharif", "2024-25"),
        ("sugarcane",    340,   "₹/quintal", "Annual", "2024-25"),
        ("arhar",        7550,  "₹/quintal", "Kharif", "2024-25"),
        ("moong",        8682,  "₹/quintal", "Kharif", "2024-25"),
        ("urad",         7400,  "₹/quintal", "Kharif", "2024-25"),
        ("barley",       1735,  "₹/quintal", "Rabi",   "2024-25"),
        ("gram",         5440,  "₹/quintal", "Rabi",   "2024-25"),
        ("lentil",       6425,  "₹/quintal", "Rabi",   "2024-25"),
        ("bajra",        2625,  "₹/quintal", "Kharif", "2024-25"),
        ("jowar",        3371,  "₹/quintal", "Kharif", "2024-25"),
        ("ragi",         4290,  "₹/quintal", "Kharif", "2024-25"),
        ("sunflower",    7280,  "₹/quintal", "Rabi",   "2024-25"),
        ("sesamum",      9267,  "₹/quintal", "Kharif", "2024-25"),
        ("safflower",    5800,  "₹/quintal", "Rabi",   "2024-25"),
    ]

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS,
        database=DB_NAME,
    )
    cur = conn.cursor()

    # Clear existing and re-seed
    cur.execute("DELETE FROM msp_prices")

    for crop, price, unit, season, year in MSP_DATA:
        cur.execute("""
            INSERT INTO msp_prices (crop, price, unit, season, year, source)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (crop, price, unit, season, year, "CCEA, Government of India"))

    conn.commit()
    cur.close()
    conn.close()
    logger.success(f"Seeded {len(MSP_DATA)} MSP price records")


# ── Verify tables ──────────────────────────────────────────────────────
def verify_tables():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS,
        database=DB_NAME,
    )
    cur = conn.cursor()

    tables = ["sessions", "messages", "msp_prices", "feedback"]
    sep = "=" * 55

    print(f"\n{sep}")
    print("  PostgreSQL — Table Verification")
    print(sep)

    for table in tables:
        cur.execute(
            f"SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_name = %s",
            (table,)
        )
        exists = cur.fetchone()[0] > 0
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count  = cur.fetchone()[0]
        icon   = "✅" if exists else "❌"
        print(f"  {icon} {table:<20} ({count} rows)")

    # Test insert + delete
    cur.execute("""
        INSERT INTO sessions (session_id, state, crop)
        VALUES ('test_session', 'BIHAR', 'wheat')
        ON CONFLICT (session_id) DO NOTHING
    """)
    cur.execute("DELETE FROM sessions WHERE session_id = 'test_session'")
    conn.commit()
    print(f"\n  ✅ Insert/delete test passed")
    print(f"  ✅ PostgreSQL ready at {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"{sep}\n")

    cur.close()
    conn.close()


if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("KisanMitra AI — PostgreSQL Setup (Task 14a)")
    logger.info("=" * 55)

    create_database()
    create_tables()
    seed_msp_prices()
    verify_tables()

    logger.success("Task 14a complete — PostgreSQL ready")