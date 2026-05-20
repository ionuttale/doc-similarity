"""
MPI document-similarity server.

Run:
    mpiexec -n 5 python server.py --source newsgroups --docs 500
    mpiexec -n 5 -machinefile hostfile python server.py --source db --db data/docs.db

Rank 0  — TCP server + MPI master
Ranks 1..N-1 — similarity workers (stay alive between client requests)
"""

import argparse
import pickle
import socket
import struct
import sys

import numpy as np
from mpi4py import MPI

# ── MPI tags ──────────────────────────────────────────────────────────────────
TAG_CTRL   = 0   # "query" | "shutdown"
TAG_TASK   = 1   # (start, end) row range
TAG_RESULT = 2   # list[(idx, score)]

DEFAULT_PORT  = 5555
DEFAULT_TOP_N = 10


# ── Network helpers ───────────────────────────────────────────────────────────

def _send(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv(sock: socket.socket):
    raw = _recv_exact(sock, 4)
    n = struct.unpack(">I", raw)[0]
    return pickle.loads(_recv_exact(sock, n))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client disconnected")
        buf.extend(chunk)
    return bytes(buf)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_documents(args) -> tuple[list, list[str]]:
    """Return (ids, texts). ids are strings or ints depending on source."""
    if args.source == "newsgroups":
        from sklearn.datasets import fetch_20newsgroups
        dataset = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        docs = [(i, d.strip()) for i, d in enumerate(dataset.data) if d.strip()]
        docs = docs[: args.docs]
        return [d[0] for d in docs], [d[1] for d in docs]

    if args.source == "db":
        import sqlite3
        conn = sqlite3.connect(args.db)
        rows = conn.execute(
            "SELECT id, content FROM docs LIMIT ?", (args.docs,)
        ).fetchall()
        conn.close()
        return [r[0] for r in rows], [r[1] for r in rows]

    if args.source == "files":
        from processing.extract import extract_from_directory
        texts = extract_from_directory(args.files_dir)[: args.docs]
        return list(range(len(texts))), texts

    raise ValueError(f"Unknown source: {args.source}")


# ── Workload splitting ────────────────────────────────────────────────────────

def _split_ranges(n: int, n_workers: int) -> list[tuple[int, int]]:
    base, extra = divmod(n, n_workers)
    ranges, start = [], 0
    for i in range(n_workers):
        end = start + base + (1 if i < extra else 0)
        ranges.append((start, min(end, n)))
        start = end
    return ranges


# ── Master (rank 0) ───────────────────────────────────────────────────────────

def run_master(comm, size: int, args, db_ids, db_matrix, vectorizer) -> None:
    n_workers = size - 1
    top_n     = args.top_n

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", args.port))
    server_sock.listen(8)

    n_db = len(db_ids)
    print(
        f"[Master] Listening on :{args.port} | {n_db} docs | {n_workers} worker(s)",
        flush=True,
    )

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"[Master] Connection from {addr}", flush=True)
            try:
                _handle_client(conn, comm, size, db_ids, db_matrix, vectorizer,
                                n_workers, top_n)
            except Exception as exc:
                print(f"[Master] Error: {exc}", flush=True)
                try:
                    _send(conn, {"error": str(exc)})
                except Exception:
                    pass
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\n[Master] Shutting down workers…", flush=True)
    finally:
        server_sock.close()
        for w in range(1, size):
            comm.send("shutdown", dest=w, tag=TAG_CTRL)


def _handle_client(conn, comm, size, db_ids, db_matrix, vectorizer,
                   n_workers, top_n) -> None:
    import time

    query_text: str = _recv(conn)
    t0 = time.perf_counter()

    # Transform query with the pre-fitted vectorizer
    query_vec = vectorizer.transform([query_text]).toarray().astype(np.float32)[0]

    if n_workers == 0:
        # Fallback: single-process (no MPI workers)
        results = _local_similarity(db_ids, db_matrix, query_vec, top_n)
    else:
        # Notify workers a query is incoming
        for w in range(1, size):
            comm.send("query", dest=w, tag=TAG_CTRL)

        # Broadcast matrix + query vector to all workers
        comm.bcast((db_matrix, query_vec), root=0)

        # Send each worker its row range
        ranges = _split_ranges(len(db_ids), n_workers)
        for w in range(1, size):
            rng = ranges[w - 1] if w - 1 < len(ranges) else (0, 0)
            comm.send(rng, dest=w, tag=TAG_TASK)

        # Gather partial results
        all_pairs: list[tuple[int, float]] = []
        for w in range(1, size):
            all_pairs.extend(comm.recv(source=w, tag=TAG_RESULT))

        all_pairs.sort(key=lambda x: x[1], reverse=True)
        results = [(db_ids[idx], score) for idx, score in all_pairs[:top_n]]

    elapsed = time.perf_counter() - t0
    print(
        f"[Master] Query processed in {elapsed:.3f}s -> top {len(results)} matches",
        flush=True,
    )
    _send(conn, results)


def _local_similarity(db_ids, db_matrix, query_vec, top_n):
    norms = np.linalg.norm(db_matrix, axis=1)
    norm_q = np.linalg.norm(query_vec)
    denom = norms * norm_q
    sims = np.zeros(len(denom))
    mask = denom > 0
    sims[mask] = (db_matrix @ query_vec)[mask] / denom[mask]
    top_idx = np.argsort(sims)[::-1][:top_n]
    return [(db_ids[i], float(sims[i])) for i in top_idx]


# ── Worker (ranks 1..N-1) ─────────────────────────────────────────────────────

def run_worker(comm, rank: int) -> None:
    while True:
        signal: str = comm.recv(source=0, tag=TAG_CTRL)
        if signal == "shutdown":
            break

        db_matrix, query_vec = comm.bcast(None, root=0)
        start, end = comm.recv(source=0, tag=TAG_TASK)

        pairs: list[tuple[int, float]] = []
        if end > start:
            chunk = db_matrix[start:end]               # (rows, vocab)
            norms = np.linalg.norm(chunk, axis=1)
            norm_q = np.linalg.norm(query_vec)
            denom = norms * norm_q
            dots = chunk @ query_vec
            sims = np.zeros(len(denom))
            mask = denom > 0
            sims[mask] = dots[mask] / denom[mask]
            pairs = [(start + i, float(sims[i])) for i in range(len(sims))]

        comm.send(pairs, dest=0, tag=TAG_RESULT)


# ── Entry point ───────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MPI document-similarity server")
    p.add_argument("--source",   choices=["newsgroups", "db", "files"], default="newsgroups")
    p.add_argument("--docs",     type=int, default=500,         help="Max DB documents to load")
    p.add_argument("--db",       default="data/docs.db",        help="SQLite DB path (--source db)")
    p.add_argument("--files-dir",default="data/raw",            help="Raw files dir (--source files)")
    p.add_argument("--port",     type=int, default=DEFAULT_PORT)
    p.add_argument("--top-n",    type=int, default=DEFAULT_TOP_N)
    p.add_argument("--max-features", type=int, default=10_000,  help="TF-IDF vocabulary size")
    return p


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = build_parser().parse_args()

    print(f"[MPI] rank {rank}/{size} on {MPI.Get_processor_name()}", flush=True)
    comm.Barrier()

    if rank == 0:
        print(f"[Master] Loading documents from source='{args.source}'…", flush=True)
        db_ids, db_texts = load_documents(args)
        print(f"[Master] Loaded {len(db_ids)} documents. Fitting TF-IDF…", flush=True)

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=args.max_features)
        db_matrix = vectorizer.fit_transform(db_texts).toarray().astype(np.float32)
        print(f"[Master] TF-IDF matrix: {db_matrix.shape}", flush=True)

        run_master(comm, size, args, db_ids, db_matrix, vectorizer)
    else:
        run_worker(comm, rank)


if __name__ == "__main__":
    main()
