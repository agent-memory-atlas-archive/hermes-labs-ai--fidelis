"""_chroma_index_lag: collection selection, backlog arithmetic, probe failure.

Both sides of the backlog calculation must be collection-filtered: the
watermark lookup selects cfg["collection"]'s VECTOR segment, and the WAL side
counts only that collection's own rows (topic carries the collection id). A
global MAX(seq_id) minuend charges the configured collection for sibling
collections' writes and reports the full WAL for a collection with no
watermark row. Any probe failure reports None, never a degraded health status.
"""

import sqlite3

from fidelis.server import _chroma_index_lag

# Mirrors the chroma tables the probe touches (FKs and unused columns dropped).
_SCHEMA = """
CREATE TABLE embeddings_queue (
    seq_id INTEGER PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    operation INTEGER NOT NULL,
    topic TEXT NOT NULL,
    id TEXT NOT NULL,
    vector BLOB,
    encoding TEXT,
    metadata TEXT
);
CREATE TABLE max_seq_id (segment_id TEXT PRIMARY KEY, seq_id INTEGER);
CREATE TABLE segments (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    scope TEXT NOT NULL,
    collection TEXT NOT NULL
);
CREATE TABLE collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dimension INTEGER,
    database_id TEXT NOT NULL DEFAULT 'default'
);
"""


def _mk_db(tmp_path):
    con = sqlite3.connect(tmp_path / "chroma.sqlite3")
    con.executescript(_SCHEMA)
    return con


def _add_collection(con, name, vector_watermark=None, metadata_watermark=None,
                    vector_segment=True):
    cid = f"coll-{name}"
    con.execute("INSERT INTO collections (id, name) VALUES (?, ?)", (cid, name))
    for scope, wm in (("VECTOR", vector_watermark), ("METADATA", metadata_watermark)):
        if scope == "VECTOR" and not vector_segment:
            continue
        seg = f"seg-{name}-{scope}"
        con.execute(
            "INSERT INTO segments (id, type, scope, collection) VALUES (?, 'hnsw', ?, ?)",
            (seg, scope, cid),
        )
        if wm is not None:
            con.execute("INSERT INTO max_seq_id (segment_id, seq_id) VALUES (?, ?)", (seg, wm))


def _fill_wal(con, seqs, collection="cogito_main"):
    # Live topic shape: persistent://default/default/<collection-id>.
    topic = f"persistent://default/default/coll-{collection}"
    con.executemany(
        "INSERT INTO embeddings_queue (seq_id, operation, topic, id) VALUES (?, 1, ?, ?)",
        [(seq, topic, f"rec-{seq}") for seq in seqs],
    )


def _cfg(tmp_path, collection="cogito_main"):
    return {"store_path": str(tmp_path), "collection": collection}


# --- collection selection -------------------------------------------------

def test_watermark_uses_configured_collection(tmp_path):
    con = _mk_db(tmp_path)
    # Decoy inserted first: an unfiltered scope='VECTOR' scan would hit it.
    _add_collection(con, "mem0migrations", vector_watermark=110)
    _add_collection(con, "cogito_main", vector_watermark=90)
    _fill_wal(con, range(1, 101), "cogito_main")
    _fill_wal(con, range(101, 121), "mem0migrations")
    con.commit()
    con.close()
    assert _chroma_index_lag(_cfg(tmp_path, "cogito_main")) == 10
    assert _chroma_index_lag(_cfg(tmp_path, "mem0migrations")) == 10


def test_sibling_collection_writes_do_not_inflate_lag(tmp_path):
    # cogito_main fully applied at seq 50; mem0migrations then writes 51..80.
    # A global MAX(seq_id) minuend would report 30 for cogito_main.
    con = _mk_db(tmp_path)
    _add_collection(con, "cogito_main", vector_watermark=50)
    _add_collection(con, "mem0migrations", vector_watermark=None)
    _fill_wal(con, range(1, 51), "cogito_main")
    _fill_wal(con, range(51, 81), "mem0migrations")
    con.commit()
    con.close()
    assert _chroma_index_lag(_cfg(tmp_path, "cogito_main")) == 0


def test_no_watermark_sibling_reports_own_rows_not_global_wal(tmp_path):
    # The live 2026-07-22 failure shape: mem0migrations has no watermark row
    # and no WAL rows of its own; the global minuend reported the full ~103k.
    con = _mk_db(tmp_path)
    _add_collection(con, "cogito_main", vector_watermark=90)
    _add_collection(con, "mem0migrations", vector_watermark=None)
    _fill_wal(con, range(1, 101), "cogito_main")
    con.commit()
    con.close()
    assert _chroma_index_lag(_cfg(tmp_path, "mem0migrations")) == 0


def test_metadata_watermark_never_masks_vector_lag(tmp_path):
    # Live failure shape: metadata segment fully applied, vector segment behind.
    con = _mk_db(tmp_path)
    _add_collection(con, "cogito_main", vector_watermark=40, metadata_watermark=100)
    _fill_wal(con, range(1, 101))
    con.commit()
    con.close()
    assert _chroma_index_lag(_cfg(tmp_path)) == 60


# --- backlog arithmetic ---------------------------------------------------

def test_lag_counts_own_rows_past_vector_watermark(tmp_path):
    con = _mk_db(tmp_path)
    _add_collection(con, "cogito_main", vector_watermark=97)
    _fill_wal(con, range(1, 101))
    con.commit()
    con.close()
    assert _chroma_index_lag(_cfg(tmp_path)) == 3


def test_fully_applied_reports_zero(tmp_path):
    con = _mk_db(tmp_path)
    _add_collection(con, "cogito_main", vector_watermark=50)
    _fill_wal(con, range(1, 51))
    con.commit()
    con.close()
    assert _chroma_index_lag(_cfg(tmp_path)) == 0


def test_watermark_ahead_of_pruned_wal_reports_zero(tmp_path):
    # chroma purges applied WAL rows; the watermark may exceed anything queued.
    con = _mk_db(tmp_path)
    _add_collection(con, "cogito_main", vector_watermark=50)
    con.commit()
    con.close()
    assert _chroma_index_lag(_cfg(tmp_path)) == 0


def test_missing_watermark_counts_all_own_rows(tmp_path):
    # Segment exists but never flushed: nothing applied, the collection's own
    # queued rows are all backlog — sibling rows still excluded.
    con = _mk_db(tmp_path)
    _add_collection(con, "cogito_main", vector_watermark=None)
    _add_collection(con, "mem0migrations", vector_watermark=None)
    _fill_wal(con, range(1, 8), "cogito_main")
    _fill_wal(con, range(8, 10), "mem0migrations")
    con.commit()
    con.close()
    assert _chroma_index_lag(_cfg(tmp_path)) == 7


def test_collection_without_vector_segment_counts_own_rows(tmp_path):
    # Degenerate but deterministic: no VECTOR segment = no watermark = nothing
    # applied, so the collection's own queued rows all count.
    con = _mk_db(tmp_path)
    _add_collection(con, "cogito_main", vector_segment=False)
    _fill_wal(con, range(1, 6))
    con.commit()
    con.close()
    assert _chroma_index_lag(_cfg(tmp_path)) == 5


# --- probe failure returns None ------------------------------------------

def test_missing_db_returns_none(tmp_path):
    assert _chroma_index_lag(_cfg(tmp_path)) is None


def test_db_without_chroma_schema_returns_none(tmp_path):
    sqlite3.connect(tmp_path / "chroma.sqlite3").close()
    assert _chroma_index_lag(_cfg(tmp_path)) is None


def test_non_sqlite_file_returns_none(tmp_path):
    (tmp_path / "chroma.sqlite3").write_bytes(b"not a database")
    assert _chroma_index_lag(_cfg(tmp_path)) is None


def test_unknown_collection_returns_none(tmp_path):
    # A store that has never seen the configured collection is a config
    # mismatch, not a zero-backlog signal.
    con = _mk_db(tmp_path)
    _add_collection(con, "cogito_main", vector_watermark=5)
    _fill_wal(con, range(1, 6))
    con.commit()
    con.close()
    assert _chroma_index_lag(_cfg(tmp_path, "ghost_collection")) is None


def test_bare_cfg_returns_none_without_raising():
    # A cfg missing store_path must not blow up /health (the do_GET caller
    # treats any raise as a failed health request, not a null metric).
    assert _chroma_index_lag({}) is None


def test_missing_collection_key_returns_none(tmp_path):
    con = _mk_db(tmp_path)
    _add_collection(con, "cogito_main", vector_watermark=5)
    _fill_wal(con, range(1, 6))
    con.commit()
    con.close()
    assert _chroma_index_lag({"store_path": str(tmp_path)}) is None
