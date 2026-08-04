#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Load official LDBC SNB BI composite-merged-fk Parquet into the DuckDB LDBC schema"
    )
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()

    ldbc_root = Path(__file__).resolve().parents[1]
    schema = ldbc_root / "schema.sql"
    loader = Path(__file__).resolve().parent / "snb-bi-load.sql"
    if not args.duckdb.is_file():
        raise ValueError(f"DuckDB executable does not exist: {args.duckdb}")
    if not args.data.is_dir():
        raise ValueError(f"Parquet data directory does not exist: {args.data}")
    if args.database.exists():
        raise ValueError(f"refusing to overwrite existing database: {args.database}")
    if args.archive and not args.archive.is_file():
        raise ValueError(f"source archive does not exist: {args.archive}")
    args.database.parent.mkdir(parents=True, exist_ok=True)

    sql = schema.read_text()
    sql += "\n" + loader.read_text().replace("{{DATA}}", str(args.data.resolve()))
    sql += "\nANALYZE;\nCHECKPOINT;\n"
    started = time.perf_counter()
    process = subprocess.run(
        [str(args.duckdb.resolve()), str(args.database.resolve()), "-bail", "-batch"],
        input=sql,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"database creation failed with status {process.returncode}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )

    version = subprocess.run(
        [str(args.duckdb.resolve()), "-version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "source_url": args.source_url,
        "archive": str(args.archive.resolve()) if args.archive else None,
        "archive_bytes": args.archive.stat().st_size if args.archive else None,
        "archive_sha256": sha256_file(args.archive) if args.archive else None,
        "data": str(args.data.resolve()),
        "parquet_files": len(list(args.data.rglob("*.parquet"))),
        "duckdb": str(args.duckdb.resolve()),
        "duckdb_version": version,
        "duckdb_sha256": sha256_file(args.duckdb),
        "database": str(args.database.resolve()),
        "database_bytes": args.database.stat().st_size,
        "database_sha256": sha256_file(args.database),
        "load_seconds": elapsed,
    }
    manifest_path = args.database.with_suffix(args.database.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exception:
        print(f"error: {exception}", file=sys.stderr)
        sys.exit(1)
