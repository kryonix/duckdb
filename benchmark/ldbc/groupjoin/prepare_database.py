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
        description="Create a reproducible LDBC database with a selected DuckDB revision"
    )
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    schema = root / "schema.sql"
    loader = root / "snb-load.sql"
    if not args.duckdb.is_file():
        raise ValueError(f"DuckDB executable does not exist: {args.duckdb}")
    if not args.data.is_dir():
        raise ValueError(f"LDBC data directory does not exist: {args.data}")
    if args.database.exists():
        raise ValueError(f"refusing to overwrite existing database: {args.database}")
    args.database.parent.mkdir(parents=True, exist_ok=True)

    sql = (
        schema.read_text()
        + "\n"
        + loader.read_text().replace("PATHVAR", str(args.data.resolve()))
    )
    if args.analyze:
        sql += "\nANALYZE;\n"
    sql += "\nCHECKPOINT;\n"
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

    files = []
    for path in sorted(args.data.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(args.data)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    version = subprocess.run(
        [str(args.duckdb.resolve()), "-version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "source_url": args.source_url,
        "data": str(args.data.resolve()),
        "files": files,
        "duckdb": str(args.duckdb.resolve()),
        "duckdb_version": version,
        "duckdb_sha256": sha256_file(args.duckdb),
        "database": str(args.database.resolve()),
        "database_bytes": args.database.stat().st_size,
        "database_sha256": sha256_file(args.database),
        "analyzed": args.analyze,
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
