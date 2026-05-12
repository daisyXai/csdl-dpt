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


def _create_images_indexes(cur) -> None:
    """
    Build secondary indexes for `images`.

    - B-tree on image_name: exact / prefix lookups by file name.
    - HNSW on vector with cosine ops: approximate nearest neighbor for
      similarity search in SQL (ORDER BY vector <=> query), aligned with
      cosine distance used in application code.
    """
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_images_image_name
        ON images (image_name);
        """
    )
    try:
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_images_vector_hnsw_cosine
            ON images USING hnsw (vector vector_cosine_ops);
            """
        )
    except psycopg2.Error as exc:
        print(
            "[WARN] HNSW index on vector failed (need pgvector with HNSW support). "
            f"Error: {exc}\n"
            "       Fallback (IVFFlat, train after bulk load):\n"
            "       CREATE INDEX idx_images_vector_ivfflat_cosine ON images "
            "       USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);"
        )


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
                    vector vector(52),
                    image_location TEXT NOT NULL,
                    original_image_location TEXT
                );
                """
            )
            _create_images_indexes(cur)
            print("[OK] Extension, table, and indexes created/verified in csdl_dpt.")
    finally:
        conn.close()


def main() -> None:
    create_database_if_missing()
    create_schema()
    print("[DONE] PostgreSQL setup completed.")


if __name__ == "__main__":
    main()
