#!/usr/bin/env python3

import argparse
import csv
import fnmatch
import hashlib
import io
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


TIMER_PATTERN = re.compile(r"Run Time \(s\): real ([0-9.]+)")
TIMER_LINE_PATTERN = re.compile(r"^Run Time \(s\):.*(?:\n|$)", re.MULTILINE)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sql_literal(value):
    return "'" + value.replace("'", "''") + "'"


def canonical_result_sha256(csv_text):
    # SQL result order is undefined unless the complete result is ordered.
    rows = sorted(tuple(row) for row in csv.reader(io.StringIO(csv_text)))
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def parse_parameters(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid parameter {value!r}; expected NAME=VALUE")
        name, parameter_value = value.split("=", 1)
        if not name:
            raise ValueError(f"invalid empty parameter name in {value!r}")
        result[name] = parameter_value
    return result


def render_query(text, parameters):
    required = set(re.findall(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}", text))
    missing = sorted(required - parameters.keys())
    if missing:
        raise ValueError(f"missing query parameters: {', '.join(missing)}")
    for name in required:
        text = text.replace("{{" + name + "}}", sql_literal(parameters[name]))
    return text.strip()


def load_queries(directory, parameters, include_patterns):
    metadata_path = directory / "queries.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    queries = []
    for path in sorted(directory.rglob("*.sql")):
        relative = str(path.relative_to(directory))
        if include_patterns and not any(
            fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.stem, pattern)
            for pattern in include_patterns
        ):
            continue
        query_metadata = metadata.get(relative, metadata.get(path.name, {}))
        queries.append(
            {
                "name": path.stem,
                "path": str(path),
                "relative_path": relative,
                "sql": render_query(path.read_text(), parameters),
                "equivalence_group": query_metadata.get("equivalence_group", relative),
                "unchanged": query_metadata.get("unchanged", True),
            }
        )
    if not queries:
        raise ValueError(f"no SQL queries found under {directory}")
    return queries


def load_manifest(path, selected_variants):
    manifest = json.loads(path.read_text())
    builds = manifest.get("builds", {})
    variants = manifest.get("variants", {})
    selected = {}
    for name in selected_variants:
        if name not in variants:
            raise ValueError(f"variant {name!r} is absent from {path}")
        variant = variants[name]
        build_name = variant.get("build")
        if build_name not in builds:
            raise ValueError(
                f"variant {name!r} references missing build {build_name!r}"
            )
        executable = Path(builds[build_name]["path"]).resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"DuckDB executable is not runnable: {executable}")
        selected[name] = {
            "build_name": build_name,
            "build": builds[build_name],
            "path": executable,
            "setup": variant.get("setup", []),
        }
    return manifest, selected


def prepare_output(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run_cli(variant, database, query, threads, timer=False):
    command = [
        str(variant["path"]),
        str(database),
        "-readonly",
        "-bail",
        "-batch",
        "-csv",
        "-noheader",
        "-nullvalue",
        "NULL",
    ]
    setup = [f"SET threads={threads}"] + variant["setup"]
    for statement in setup:
        command.extend(["-cmd", statement])
    if timer:
        command.extend(["-cmd", ".timer on"])
    command.extend(["-c", query])
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"DuckDB failed with status {process.returncode}\n"
            f"command: {command}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    query_seconds = None
    result_stdout = process.stdout
    if timer:
        matches = TIMER_PATTERN.findall(process.stdout + process.stderr)
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one timer result, found {matches}\n"
                f"command: {command}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
            )
        query_seconds = float(matches[0])
        result_stdout = TIMER_LINE_PATTERN.sub("", process.stdout)
    return {
        "command": command,
        "stdout": result_stdout,
        "stderr": process.stderr,
        "query_seconds": query_seconds,
        "process_seconds": elapsed,
        "result_sha256": canonical_result_sha256(result_stdout),
        "raw_result_sha256": hashlib.sha256(result_stdout.encode()).hexdigest(),
    }


def run_prepared_cli(variant, database, query, threads, warmups, samples):
    command = [
        str(variant["path"]),
        str(database),
        "-readonly",
        "-bail",
        "-batch",
        "-csv",
        "-noheader",
        "-nullvalue",
        "NULL",
    ]
    setup = [f"SET threads={threads}"] + variant["setup"]
    for statement in setup:
        command.extend(["-cmd", statement])

    statement = query.rstrip().rstrip(";")
    executions = warmups + samples
    script = [
        f".output {os.devnull}",
        f"PREPARE groupjoin_evidence AS {statement};",
        ".timer on",
    ]
    script.extend("EXECUTE groupjoin_evidence;" for _ in range(executions))
    started = time.perf_counter()
    process = subprocess.run(
        command, input="\n".join(script) + "\n", capture_output=True, text=True
    )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"prepared DuckDB batch failed with status {process.returncode}\n"
            f"command: {command}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    timings = [
        float(value) for value in TIMER_PATTERN.findall(process.stdout + process.stderr)
    ]
    if len(timings) != executions:
        raise RuntimeError(
            f"expected {executions} timer results, found {timings}\n"
            f"command: {command}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return {
        "command": command,
        "timings": timings,
        "process_seconds": elapsed,
    }


def operator_flags(plan):
    upper = plan.upper().replace(" ", "_")
    return {
        "factorized_group_join": "FACTORIZED_GROUP_JOIN" in upper,
        "group_join": "GROUP_JOIN" in upper,
        "hash_join": "HASH_JOIN" in upper,
        "recursive_cte": "RECURSIVE_CTE" in upper or "REC_CTE" in upper,
    }


def environment_record(args, manifest, variants, database):
    build_records = {}
    for name, variant in variants.items():
        executable = variant["path"]
        version = subprocess.run(
            [str(executable), "-version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        build_records[name] = {
            "build_name": variant["build_name"],
            "manifest": variant["build"],
            "executable": str(executable),
            "executable_sha256": sha256_file(executable),
            "version": version,
            "setup": variant["setup"],
        }
    environment = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "created_unix_seconds": time.time(),
        "platform": platform.platform(),
        "python": sys.version,
        "database": str(database),
        "database_bytes": database.stat().st_size,
        "database_sha256": sha256_file(database),
        "threads": args.threads,
        "manifest": manifest,
        "resolved_variants": build_records,
    }
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if git_root.returncode == 0:
        environment["git"] = {
            "root": git_root.stdout.strip(),
            "head": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "branch": subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "status": subprocess.run(
                ["git", "status", "--short"], capture_output=True, text=True, check=True
            ).stdout,
        }
    return environment


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_census(args, queries, variants, output):
    records = []
    errors = []
    plans = output / "plans"
    results = output / "results"
    plans.mkdir()
    results.mkdir()
    for query in queries:
        reference_hash = None
        for variant_name, variant in variants.items():
            cell = f"{query['name']}--{variant_name}"
            try:
                plan = run_cli(
                    variant, args.database, "EXPLAIN " + query["sql"], args.threads
                )
                (plans / f"{cell}.csv").write_text(plan["stdout"])
                record = {
                    "query": query["name"],
                    "query_path": query["relative_path"],
                    "variant": variant_name,
                    **operator_flags(plan["stdout"]),
                }
                if not args.plans_only:
                    result = run_cli(variant, args.database, query["sql"], args.threads)
                    (results / f"{cell}.csv").write_text(result["stdout"])
                    if reference_hash is None:
                        reference_hash = result["result_sha256"]
                    record.update(
                        {
                            "result_sha256": result["result_sha256"],
                            "matches_first_variant": result["result_sha256"]
                            == reference_hash,
                            "process_seconds": result["process_seconds"],
                        }
                    )
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            except Exception as exception:
                error = {
                    "query": query["name"],
                    "variant": variant_name,
                    "error": str(exception),
                }
                errors.append(error)
                print(json.dumps(error, sort_keys=True), file=sys.stderr, flush=True)
                if args.strict:
                    raise
    write_json(output / "census.json", {"records": records, "errors": errors})
    return 1 if errors else 0


def median_absolute_deviation(values):
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def bootstrap_relative_change(left, right, seed, iterations=10000):
    generator = random.Random(seed)
    changes = []
    for _ in range(iterations):
        left_sample = [generator.choice(left) for _ in left]
        right_sample = [generator.choice(right) for _ in right]
        changes.append(
            100.0
            * (statistics.median(left_sample) - statistics.median(right_sample))
            / statistics.median(right_sample)
        )
    changes.sort()
    return [changes[int(iterations * 0.025)], changes[int(iterations * 0.975)]]


def summarize_measurements(records, seed):
    cells = {}
    for record in records:
        cell = record["cell"]
        cells.setdefault(cell, []).append(record["query_seconds"])
    summaries = {}
    for cell, values in cells.items():
        summaries[cell] = {
            "samples": len(values),
            "median_seconds": statistics.median(values),
            "mad_seconds": median_absolute_deviation(values),
            "minimum_seconds": min(values),
            "maximum_seconds": max(values),
        }
    comparisons = []
    cell_names = sorted(cells)
    for left_index, left_name in enumerate(cell_names):
        left_group = left_name.split("/", 1)[0]
        for right_name in cell_names[left_index + 1 :]:
            if right_name.split("/", 1)[0] != left_group:
                continue
            left = cells[left_name]
            right = cells[right_name]
            change = (
                100.0
                * (statistics.median(left) - statistics.median(right))
                / statistics.median(right)
            )
            comparisons.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "left_relative_change_percent": change,
                    "bootstrap_95_percent": bootstrap_relative_change(
                        left, right, f"{seed}:{left_name}:{right_name}"
                    ),
                }
            )
    return {"cells": summaries, "comparisons": comparisons}


def write_measurement_summary(output, records, seed):
    summary = summarize_measurements(records, seed)
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        fields = (
            "samples",
            "median_seconds",
            "mad_seconds",
            "minimum_seconds",
            "maximum_seconds",
        )
        writer.writerow(["cell", *fields])
        for cell, values in sorted(summary["cells"].items()):
            writer.writerow([cell] + [values[key] for key in fields])


def verify_equivalent_results(args, queries, variants):
    groups = {}
    records = []
    for query in queries:
        for variant_name, variant in variants.items():
            result = run_cli(variant, args.database, query["sql"], args.threads)
            group = query["equivalence_group"]
            expected = groups.setdefault(group, result["result_sha256"])
            record = {
                "query": query["name"],
                "query_path": query["relative_path"],
                "variant": variant_name,
                "equivalence_group": group,
                "result_sha256": result["result_sha256"],
                "matches_group": result["result_sha256"] == expected,
            }
            records.append(record)
            if not record["matches_group"]:
                raise RuntimeError(
                    f"result mismatch for {query['name']}/{variant_name} in equivalence group {group}"
                )
    return records


def run_measure(args, queries, variants, output):
    verification = verify_equivalent_results(args, queries, variants)
    write_json(output / "verification.json", verification)
    if args.protocol == "prepared":
        return run_prepared_measure(args, queries, variants, output, verification)

    jobs = []
    for query in queries:
        for variant_name in variants:
            for repetition in range(args.warmups):
                jobs.append(("warmup", repetition, query, variant_name))
    generator = random.Random(args.seed)
    generator.shuffle(jobs)
    measured_jobs = []
    for query in queries:
        for variant_name in variants:
            for repetition in range(args.samples):
                measured_jobs.append(("sample", repetition, query, variant_name))
    generator.shuffle(measured_jobs)
    jobs.extend(measured_jobs)

    records = []
    raw_path = output / "raw.jsonl"
    with raw_path.open("w") as raw_file:
        for phase, repetition, query, variant_name in jobs:
            variant = variants[variant_name]
            result = run_cli(
                variant, args.database, query["sql"], args.threads, timer=True
            )
            record = {
                "phase": phase,
                "repetition": repetition,
                "protocol": "fresh-process",
                "equivalence_group": query["equivalence_group"],
                "query": query["name"],
                "query_path": query["relative_path"],
                "unchanged": query["unchanged"],
                "variant": variant_name,
                "cell": f"{query['equivalence_group']}/{query['name']}/{variant_name}",
                "query_seconds": result["query_seconds"],
                "process_seconds": result["process_seconds"],
                "result_sha256": result["result_sha256"],
            }
            raw_file.write(json.dumps(record, sort_keys=True) + "\n")
            raw_file.flush()
            if phase == "sample":
                records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    write_measurement_summary(output, records, args.seed)
    return 0


def run_prepared_measure(args, queries, variants, output, verification):
    result_hashes = {
        (record["query_path"], record["variant"]): record["result_sha256"]
        for record in verification
    }
    cells = [(query, variant_name) for query in queries for variant_name in variants]
    generator = random.Random(args.seed)
    generator.shuffle(cells)

    records = []
    raw_path = output / "raw.jsonl"
    with raw_path.open("w") as raw_file:
        for query, variant_name in cells:
            result = run_prepared_cli(
                variants[variant_name],
                args.database,
                query["sql"],
                args.threads,
                args.warmups,
                args.samples,
            )
            for repetition, query_seconds in enumerate(result["timings"]):
                phase = "warmup" if repetition < args.warmups else "sample"
                phase_repetition = (
                    repetition if phase == "warmup" else repetition - args.warmups
                )
                record = {
                    "phase": phase,
                    "repetition": phase_repetition,
                    "protocol": "prepared",
                    "equivalence_group": query["equivalence_group"],
                    "query": query["name"],
                    "query_path": query["relative_path"],
                    "unchanged": query["unchanged"],
                    "variant": variant_name,
                    "cell": f"{query['equivalence_group']}/{query['name']}/{variant_name}",
                    "query_seconds": query_seconds,
                    "batch_process_seconds": result["process_seconds"],
                    "result_sha256": result_hashes[
                        (query["relative_path"], variant_name)
                    ],
                }
                raw_file.write(json.dumps(record, sort_keys=True) + "\n")
                raw_file.flush()
                if phase == "sample":
                    records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    write_measurement_summary(output, records, args.seed)
    return 0


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run reproducible public LDBC GroupJoin evidence"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("census", "measure"):
        subparser = subparsers.add_parser(mode)
        subparser.add_argument("--manifest", type=Path, required=True)
        subparser.add_argument("--database", type=Path, required=True)
        subparser.add_argument("--queries", type=Path, required=True)
        subparser.add_argument("--variants", required=True)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--parameter", action="append", default=[])
        subparser.add_argument(
            "--include",
            action="append",
            default=[],
            help="Only run matching query names or relative paths",
        )
        subparser.add_argument("--threads", type=int, default=4)
    census = subparsers.choices["census"]
    census.add_argument("--strict", action="store_true")
    census.add_argument("--plans-only", action="store_true")
    measure = subparsers.choices["measure"]
    measure.add_argument("--warmups", type=int, default=2)
    measure.add_argument("--samples", type=int, default=15)
    measure.add_argument("--seed", type=int, default=20260803)
    measure.add_argument(
        "--protocol", choices=("fresh-process", "prepared"), default="fresh-process"
    )
    return parser


def main():
    args = create_parser().parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if args.mode == "measure" and (args.warmups < 0 or args.samples < 1):
        raise ValueError("warmups must be non-negative and samples must be positive")
    if not args.database.is_file():
        raise ValueError(f"database does not exist: {args.database}")
    selected_names = [name for name in args.variants.split(",") if name]
    if not selected_names:
        raise ValueError("at least one variant is required")
    parameters = parse_parameters(args.parameter)
    queries = load_queries(args.queries, parameters, args.include)
    manifest, variants = load_manifest(args.manifest, selected_names)
    prepare_output(args.output)
    environment = environment_record(args, manifest, variants, args.database)
    if args.mode == "measure":
        environment["measurement_protocol"] = args.protocol
    environment["queries"] = [
        {key: value for key, value in query.items() if key != "sql"}
        for query in queries
    ]
    write_json(args.output / "context.json", environment)
    if args.mode == "census":
        return run_census(args, queries, variants, args.output)
    return run_measure(args, queries, variants, args.output)


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
