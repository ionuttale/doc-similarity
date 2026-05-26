import numpy as np
from mpi4py import MPI

TAG_COUNT = 1
TAG_DATA  = 2


def run_worker(comm, rank: int) -> None:
    # Primeste matricea TF-IDF via Bcast (uppercase, fara pickle)
    shape_arr = np.empty(2, dtype=np.int32)
    comm.Bcast(shape_arr, root=0)
    dense = np.empty(shape_arr, dtype=np.float32)
    comm.Bcast(dense, root=0)

    from mpi_src.similarity import partial_top_pairs

    n_buf = np.empty(1, dtype=np.int32)

    while True:
        comm.Recv([n_buf, MPI.INT], source=0, tag=TAG_COUNT)
        if n_buf[0] == 0:   # semnal de inchidere
            break

        task_buf = np.empty(n_buf[0], dtype=np.int32)
        comm.Recv([task_buf, MPI.INT], source=0, tag=TAG_DATA)
        start, end, top_n = int(task_buf[0]), int(task_buf[1]), int(task_buf[2])

        # Calcul cosinus vectorizat pentru intervalul de randuri primit
        pairs = partial_top_pairs(dense, start, end, top_n)

        # Trimite rezultatele inapoi ca array float64 plat: [i, j, score, ...]
        if pairs:
            res_arr = np.array(pairs, dtype=np.float64).flatten()
            n_res   = np.array([len(pairs)], dtype=np.int32)
            comm.Send([n_res, MPI.INT], dest=0, tag=TAG_COUNT)
            comm.Send([res_arr, MPI.DOUBLE], dest=0, tag=TAG_DATA)
        else:
            n_res = np.array([0], dtype=np.int32)
            comm.Send([n_res, MPI.INT], dest=0, tag=TAG_COUNT)
