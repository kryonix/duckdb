# Public LDBC GroupJoin evidence harness

This directory contains workload diagnostics for deciding whether GroupJoin is
useful on unchanged public queries. It is not a benchmark result by itself.

The harness verifies the complete CSV result before timing, stores raw samples
and plans, and reports medians and median absolute deviations. Fresh-process
timing remains the default. Prepared timing keeps one connection per
query/variant cell, prepares once, and measures only repeated executions. Keep
its output outside Git.

Create a manifest that names revision-isolated executables:

```json
{
  "builds": {
    "baseline": {"path": "/tmp/groupjoin-builds/baseline/duckdb"},
    "binary": {"path": "/tmp/groupjoin-builds/binary/duckdb"},
    "factorized": {"path": "/tmp/groupjoin-builds/factorized/duckdb"}
  },
  "variants": {
    "baseline": {"build": "baseline", "setup": []},
    "disabled": {
      "build": "factorized",
      "setup": ["SET debug_group_join_strategy='disabled'"]
    },
    "auto": {
      "build": "factorized",
      "setup": ["SET debug_group_join_strategy='auto'"]
    },
    "binary": {
      "build": "factorized",
      "setup": ["SET debug_group_join_strategy='hash'"]
    },
    "factorized": {
      "build": "factorized",
      "setup": ["SET debug_group_join_strategy='factorized'"]
    }
  }
}
```

Create the shared database with the oldest compared revision so every newer
revision can open exactly the same immutable storage file:

```bash
python3 benchmark/ldbc/groupjoin/prepare_database.py \
  --duckdb /tmp/groupjoin-builds/baseline/duckdb \
  --data benchmark/ldbc/sf0.1 \
  --database /tmp/ldbc-sf0.1.duckdb --analyze \
  --source-url https://github.com/duckdb/duckdb-data/releases/download/v1.0/ldbc-snb-sf0.1.tar.gz
```

The loader writes a sidecar manifest containing the source-file, executable,
and database SHA-256 hashes. It refuses to overwrite an existing database.

Official SNB BI archives contain composite-merged-fk Parquet. Load them into
the same compatibility schema with the oldest compared revision:

```bash
python3 benchmark/ldbc/groupjoin/prepare_bi_database.py \
  --duckdb /tmp/groupjoin-builds/baseline/duckdb \
  --data /tmp/bi-sf1-raw/graphs/parquet/raw/composite-merged-fk \
  --archive /tmp/bi-sf1-raw.tar.zst \
  --database /tmp/ldbc-bi-sf1.duckdb \
  --source-url https://datasets.ldbcouncil.org/bi-pre-audit/bi-sf1-raw.tar.zst
```

Run the plan census on the bundled BI queries:

```bash
python3 benchmark/ldbc/groupjoin/run_evidence.py census \
  --manifest /tmp/groupjoin-builds/manifest.json \
  --database /tmp/ldbc-sf0.1.duckdb \
  --queries benchmark/ldbc/queries \
  --variants baseline,disabled,auto,binary,factorized \
  --output /tmp/groupjoin-evidence/census-sf0.1
```

Run the equivalent Q5 forms with two warmups and 15 measured fresh processes:

```bash
python3 benchmark/ldbc/groupjoin/run_evidence.py measure \
  --manifest /tmp/groupjoin-builds/manifest.json \
  --database /tmp/ldbc-bi-sf1.duckdb \
  --queries benchmark/ldbc/groupjoin/queries/q5 \
  --variants disabled,auto,binary,factorized \
  --parameter tag=Sikh_Empire --threads 4 \
  --warmups 2 --samples 15 --seed 20260803 \
  --output /tmp/groupjoin-evidence/q5-sf1
```

Repeat the same comparison without process startup or repeated planning:

```bash
python3 benchmark/ldbc/groupjoin/run_evidence.py measure \
  --manifest /tmp/groupjoin-builds/manifest.json \
  --database /tmp/ldbc-bi-sf1.duckdb \
  --queries benchmark/ldbc/groupjoin/queries/q5 \
  --variants disabled,auto,binary,factorized \
  --parameter tag=Sikh_Empire --threads 4 \
  --warmups 2 --samples 15 --seed 20260803 \
  --protocol prepared \
  --output /tmp/groupjoin-evidence/q5-sf1-prepared
```

`q5-original.sql`, `q5-preaggregate.sql`, and `q5-two-stage.sql` are result
equivalent. The two-stage form is diagnostic: it exposes a per-message
factorized inner aggregate before rolling up to the creator. A win there does
not count as an unchanged-query win.

Each output directory contains `context.json` with the exact command, Git
state, machine, database checksum, build metadata, executable checksums, and
selected queries. Keep the database loader's sidecar manifest with the raw
measurement directory so the public source URL and archive checksum remain
part of the evidence.
