import numpy as np


def run_worker(comm, rank: int) -> None:
    # Receive the full TF-IDF matrix broadcast by master
    dense: np.ndarray = comm.bcast(None, root=0)

    # Receive work assignment
    task: dict = comm.recv(source=0, tag=1)
    start, end, top_n = task["start"], task["end"], task["top_n"]

    pairs = []
    if end > start:
        from mpi_src.similarity import partial_top_pairs
        pairs = partial_top_pairs(dense, start, end, top_n)

    comm.send(pairs, dest=0, tag=2)

    # Wait for shutdown signal before exiting so MPI finalizes cleanly
    comm.recv(source=0, tag=0)
