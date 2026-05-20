import numpy as np


def load_documents(args) -> list[str]:
    if args.source == "newsgroups":
        from sklearn.datasets import fetch_20newsgroups
        dataset = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        docs = [d.strip() for d in dataset.data if d.strip()]
        return docs[: args.docs]

    if args.source == "db":
        import sqlite3
        conn = sqlite3.connect(args.db)
        rows = conn.execute("SELECT content FROM docs LIMIT ?", (args.docs,)).fetchall()
        conn.close()
        return [r[0] for r in rows]

    if args.source == "files":
        from processing.extract import extract_from_directory
        return extract_from_directory(args.files_dir)[: args.docs]

    raise ValueError(f"Unknown source: {args.source}")


def _split_rows(n_rows: int, n_workers: int) -> list[tuple[int, int]]:
    """Return (start, end) row ranges split as evenly as possible."""
    chunk = max(1, n_rows // n_workers)
    ranges = []
    start = 0
    for i in range(n_workers):
        end = start + chunk if i < n_workers - 1 else n_rows
        if start < n_rows:
            ranges.append((start, end))
        start = end
    return ranges


def run_master(comm, size: int, args) -> list[tuple[int, int, float]]:
    print(f"[Master] Loading {args.docs} documents from '{args.source}'...")
    documents = load_documents(args)
    n = len(documents)
    print(f"[Master] Loaded {n} documents. Fitting TF-IDF...")

    from processing.tfidf import fit_tfidf
    tfidf_matrix = fit_tfidf(documents)
    dense = tfidf_matrix.toarray().astype(np.float32)  # (n, vocab)
    print(f"[Master] TF-IDF matrix shape: {dense.shape}. Distributing similarity work to {size - 1} worker(s)...")

    # Single-process fallback — no MPI needed
    if size == 1:
        from mpi_src.similarity import partial_top_pairs, merge_top_n
        pairs = partial_top_pairs(dense, 0, n, args.top_n)
        return merge_top_n(pairs, args.top_n)

    # Broadcast the full TF-IDF matrix to all workers
    comm.bcast(dense, root=0)

    # Scatter row ranges — each worker computes similarity for its assigned rows vs all rows
    n_workers = size - 1
    row_ranges = _split_rows(n, n_workers)

    for worker_rank in range(1, size):
        if worker_rank - 1 < len(row_ranges):
            start, end = row_ranges[worker_rank - 1]
        else:
            start, end = 0, 0  # worker gets no rows (more workers than rows)
        comm.send({"start": start, "end": end, "top_n": args.top_n}, dest=worker_rank, tag=1)

    # Gather partial results
    all_pairs = []
    for worker_rank in range(1, size):
        partial = comm.recv(source=worker_rank, tag=2)
        all_pairs.extend(partial)

    # Send shutdown signal
    for worker_rank in range(1, size):
        comm.send(None, dest=worker_rank, tag=0)

    from mpi_src.similarity import merge_top_n
    return merge_top_n(all_pairs, args.top_n)
