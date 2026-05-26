import numpy as np
from mpi4py import MPI

TAG_COUNT = 1   # trimite/primeste numarul de elemente din urmatorul mesaj
TAG_DATA  = 2   # payload (pereche de int32 sau array float64)


def load_documents(args) -> list[str]:
    if args.source == "newsgroups":
        from sklearn.datasets import fetch_20newsgroups
        dataset = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        docs = [d.strip() for d in dataset.data if d.strip()]
        return docs[: args.docs]
    if args.source == "synthetic":
        import numpy as np
        import string
        rng = np.random.default_rng(42)
        letters = list(string.ascii_lowercase)
        # Genereaza 8000 de cuvinte alfabetice unice de 5-7 litere
        vocab = np.array([
            "".join(rng.choice(letters, rng.integers(5, 8)))
            for _ in range(8000)
        ])
        # Documente cu topic clusters (grupuri de 200 doc.) pentru similaritati reale
        n_topics = max(1, args.docs // 200)
        topics = [rng.choice(vocab, 400, replace=False) for _ in range(n_topics)]
        docs = []
        for i in range(args.docs):
            topic = topics[i % n_topics]
            words = np.concatenate([
                rng.choice(topic, 80),      # cuvinte din topic propriu
                rng.choice(vocab, 20),      # zgomot general
            ])
            docs.append(" ".join(words))
        return docs
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


def _send_task(comm, dest: int, start: int, end: int, top_n: int) -> None:
    task = np.array([start, end, top_n], dtype=np.int32)
    n = np.array([3], dtype=np.int32)
    comm.Send([n, MPI.INT], dest=dest, tag=TAG_COUNT)
    comm.Send([task, MPI.INT], dest=dest, tag=TAG_DATA)


def _send_shutdown(comm, dest: int) -> None:
    n = np.array([0], dtype=np.int32)
    comm.Send([n, MPI.INT], dest=dest, tag=TAG_COUNT)


def run_master(comm, size: int, args) -> list[tuple[int, int, float]]:
    import time
    from processing.tfidf import fit_tfidf
    from mpi_src.similarity import merge_top_n

    t0 = time.perf_counter()
    print(f"[Master] Loading {args.docs} documents from '{args.source}'...")
    documents = load_documents(args)
    n = len(documents)

    print(f"[Master] Fitting TF-IDF on {n} docs...")
    tfidf_matrix = fit_tfidf(documents)
    dense = tfidf_matrix.toarray().astype(np.float32)
    print(f"[Master] TF-IDF shape: {dense.shape}  ({time.perf_counter()-t0:.2f}s)")

    # --- Single-process fallback ---
    if size == 1:
        from mpi_src.similarity import partial_top_pairs
        pairs = partial_top_pairs(dense, 0, n, args.top_n)
        return merge_top_n(pairs, args.top_n)

    # --- MinHash + LSH (informatii pentru log si filtrare finala) ---
    print(f"[Master] Computing MinHash + LSH...")
    from mpi_src.minhash_lsh import compute_minhash, lsh_candidate_pairs
    t_lsh = time.perf_counter()
    signatures  = compute_minhash(tfidf_matrix)
    lsh_pairs   = set(lsh_candidate_pairs(signatures))
    naive_count = n * (n - 1) // 2
    print(
        f"[Master] LSH candidates: {len(lsh_pairs):,} / {naive_count:,} "
        f"({1 - len(lsh_pairs)/max(naive_count,1):.1%} reducere)  "
        f"({time.perf_counter()-t_lsh:.2f}s)"
    )

    # --- Broadcast matrice TF-IDF catre workeri (uppercase MPI, zero-copy) ---
    shape_arr = np.array(dense.shape, dtype=np.int32)
    comm.Bcast(shape_arr, root=0)
    comm.Bcast(dense, root=0)

    # --- Sarcini = intervale de randuri (chunk_size randuri per task) ---
    row_chunks = [
        (start, min(start + args.chunk_size, n))
        for start in range(0, n, args.chunk_size)
    ]
    total     = len(row_chunks)
    n_workers = size - 1
    print(
        f"[Master] {total} chunk-uri x {args.chunk_size} randuri -> "
        f"{n_workers} worker(i), load balancing dinamic (ANY_SOURCE)..."
    )

    task_idx = 0
    pending  = 0
    all_pairs: list[tuple[int, int, float]] = []

    # Distribuire initiala: un task per worker
    for w in range(1, size):
        if task_idx < total:
            s, e = row_chunks[task_idx]
            _send_task(comm, w, s, e, args.top_n)
            task_idx += 1
            pending += 1
        else:
            _send_shutdown(comm, w)

    # Bucla dinamica: ANY_SOURCE = master asteapta cel mai rapid worker disponibil
    n_res_buf = np.empty(1, dtype=np.int32)
    while pending > 0:
        status = MPI.Status()
        comm.Recv([n_res_buf, MPI.INT], source=MPI.ANY_SOURCE, tag=TAG_COUNT, status=status)
        worker = status.Get_source()

        if n_res_buf[0] > 0:
            res_buf = np.empty(n_res_buf[0] * 3, dtype=np.float64)
            comm.Recv([res_buf, MPI.DOUBLE], source=worker, tag=TAG_DATA)
            for t in res_buf.reshape(-1, 3):
                all_pairs.append((int(t[0]), int(t[1]), float(t[2])))

        pending -= 1

        if task_idx < total:
            s, e = row_chunks[task_idx]
            _send_task(comm, worker, s, e, args.top_n)
            task_idx += 1
            pending += 1
        else:
            _send_shutdown(comm, worker)

    # Filtrare optionala cu candidatii LSH (reduce munca de merge)
    if lsh_pairs:
        all_pairs = [p for p in all_pairs if (min(p[0],p[1]), max(p[0],p[1])) in lsh_pairs] or all_pairs

    return merge_top_n(all_pairs, args.top_n)
