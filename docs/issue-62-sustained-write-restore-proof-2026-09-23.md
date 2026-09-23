# Issue #62 sustained-write and restore proof

This is a bounded, repeatable operational proof against a newly created local
Chroma store and temporal SQLite sidecar. It uses synthetic text, deterministic
UUID5 record IDs, a deterministic local hashing embedder, the real HTTP `/store`
handler, and a simulated embedding outage. It makes no calls to external model
services and does not read or copy the user's existing Fidelis store.

Run from the repository root:

```sh
python scripts/sustained_write_restore_proof.py
```

The command creates an isolated temporary home, store, queue, backup, and
restore location; asserts each invariant; and emits JSON. The recorded run,
including all direct IDs, duplicate-to-existing-ID mappings, queued IDs, and
correction IDs/text/target links, is in
[`issue-62-sustained-write-restore-proof-2026-09-23.json`](issue-62-sustained-write-restore-proof-2026-09-23.json).

## Recorded run

The script was run on upstream `f8055e7ff5e68b90898edf208c1153d5e6cf4ff4`
with the environment's configured Python 3.14. It wrote and verified 120
direct records, exercised 20 duplicate requests (all mapped to their original
IDs), wrote 12 corrections with 12 matching supersession edges, then queued 16
writes during a controlled embedding outage. After process restart, store and
queue backup, and restore into a separate path, all 16 queued records replayed.
The restored corpus contains exactly 148 unique records and all 148 exact text
payloads verified; queue remaining and dead-letter counts are both zero.

| Measure | Observed |
| --- | ---: |
| Write requests | 168 |
| Write errors / error rate | 0 / 0% |
| Maximum queue depth | 16 |
| Restored direct + corrected + replayed records | 148 |
| Duplicate requests / new records | 20 / 0 |
| Correction links expected / found | 12 / 12 |
| Queued / replayed / remaining / dead-lettered | 16 / 16 / 0 / 0 |
| HTTP `/store` latency p50 / p95 | 3.363 / 5.845 ms |
| Restored vector recall latency p50 / p95 (60 queries) | 0.637 / 0.725 ms |
| Restored Chroma bytes / remaining queue bytes | 1,693,860 / 0 |
| Process peak RSS increase during run | 44,417,024 bytes |

The exact identifiers and correction links are in the JSON artifact rather
than abbreviated here. Latency, storage, and peak RSS are specific to this
machine and bounded sample. This does not establish long-duration uptime,
production capacity, or resolution of issue #60's store-wide budget. Recall is
verified through the installed Chroma vector-store API; this proof does not
exercise the separate HTTP `/query` adapter.
