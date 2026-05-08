from __future__ import annotations

import psycopg2
from psycopg2 import sql


DB_HOST = "localhost"
DB_PORT = 5418
DB_USER = "postgres"
DB_PASSWORD = "123456"
DEFAULT_DB = "postgres"
TARGET_DB = "csdl_dpt"


def create_database_if_missing() -> None:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DEFAULT_DB,
    )
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (TARGET_DB,))
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(
                    sql.SQL("CREATE DATABASE {};").format(sql.Identifier(TARGET_DB))
                )
                print(f"[OK] Created database: {TARGET_DB}")
            else:
                print(f"[OK] Database already exists: {TARGET_DB}")
    finally:
        conn.close()


def create_schema() -> None:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=TARGET_DB,
    )
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id BIGSERIAL PRIMARY KEY,
                    image_name VARCHAR(255) NOT NULL,
                    image_metadata JSONB,
                    vector VECTOR,
                    image_location TEXT NOT NULL
                );
                """
            )
            print("[OK] Extension and table created/verified in csdl_dpt.")
    finally:
        conn.close()


def main() -> None:
    create_database_if_missing()
    create_schema()
    print("[DONE] PostgreSQL setup completed.")


if __name__ == "__main__":
    main()
