"""
Ingest the raw CSV exports into a SQLite database (database/inventory.db).

Tables created (one per raw file): vendor_invoice, purchase_prices, begin_inventory, end_inventory.
Run:  python scripts/ingestion_db.py
"""
from __future__ import annotations

import sqlite3
import time

import pandas as pd

from config import DATA_DIR, DB_PATH, RAW_FILES, get_logger

log = get_logger("ingestion_db", "ingestion_db.log")

CHUNK = 50_000
DTYPES = {"Approval": "string", "Volume": "string", "Description": "string", "City": "string"}
INDEXES = {
    "vendor_invoice": ["VendorNumber", "PONumber"],
    "purchase_prices": ["Brand", "VendorNumber"],
    "begin_inventory": ["Brand", "Store"],
    "end_inventory": ["Brand", "Store"],
}


def ingest_table(conn: sqlite3.Connection, table: str, csv_path) -> int:
    """Load one CSV into SQLite in chunks (replaces the table). Returns rows loaded."""
    rows = 0
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK, dtype=DTYPES)):
        chunk.to_sql(table, conn, if_exists="replace" if i == 0 else "append", index=False)
        rows += len(chunk)
    for col in INDEXES.get(table, []):
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON "{table}" ("{col}")')
    conn.commit()
    return rows


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    missing = [f for f in RAW_FILES.values() if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw file(s) in {DATA_DIR}: {missing}")

    with sqlite3.connect(DB_PATH) as conn:
        for table, fname in RAW_FILES.items():
            t0 = time.time()
            n = ingest_table(conn, table, DATA_DIR / fname)
            log.info("Ingested %-16s %9d rows from %s (%.1fs)", table, n, fname, time.time() - t0)
        # sanity check: row counts in the DB equal row counts in the CSVs
        for table, fname in RAW_FILES.items():
            db_n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            with open(DATA_DIR / fname, "rb") as fh:
                csv_n = sum(1 for _ in fh) - 1
            assert db_n == csv_n, f"{table}: db rows {db_n} != csv rows {csv_n}"
    log.info("Ingestion complete in %.1fs -> %s", time.time() - start, DB_PATH)


if __name__ == "__main__":
    main()
