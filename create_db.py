"""
create_db.py
------------
Creates database.db with Users and History tables, and safely migrates
existing databases to support risk_level, confidence, risk_score, model_version, and inputs_json.

Usage:
    python create_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


def create_or_migrate_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ---------- Users table ----------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ---------- History table ----------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            prediction_type TEXT NOT NULL,
            disease TEXT NOT NULL,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )
    """)

    # ---------- Health Profiles table ----------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT UNIQUE NOT NULL,
            age INTEGER,
            gender TEXT,
            height_cm REAL,
            weight_kg REAL,
            bmi REAL,
            smoking_status TEXT,
            physical_activity TEXT,
            existing_conditions TEXT,
            family_history TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )
    """)

    # Safe Schema Migrations for existing history records
    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(history)").fetchall()]

    migrations = [
        ("risk_level", "TEXT"),
        ("confidence", "REAL"),
        ("risk_score", "INTEGER"),
        ("model_version", "TEXT DEFAULT 'v1.0'"),
        ("inputs_json", "TEXT"),
        ("bmi", "REAL")
    ]

    for col_name, col_type in migrations:
        if col_name not in existing_cols:
            print(f"Migrating history table: Adding column '{col_name}'...")
            cursor.execute(f"ALTER TABLE history ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()
    print(f"Database setup & migration complete at: {DB_PATH}")


if __name__ == "__main__":
    create_or_migrate_database()
