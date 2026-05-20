import argparse
import time
from mpi4py import MPI


def build_parser():
    p = argparse.ArgumentParser(description="Distributed document similarity via MPI")
    p.add_argument("--docs",       type=int, default=500,          help="Number of documents to process")
    p.add_argument("--top-n",      type=int, default=10,           help="Top N similar pairs to return")
    p.add_argument("--chunk-size", type=int, default=50,           help="Rows per worker batch (similarity phase)")
    p.add_argument("--source",     choices=["newsgroups", "db", "files"], default="newsgroups")
    p.add_argument("--db",         default="data/docs.db",         help="SQLite database path")
    p.add_argument("--files-dir",  default="data/raw",             help="Directory with raw documents")
    return p


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = build_parser().parse_args()

    host = MPI.Get_processor_name()
    role = "master" if rank == 0 else "worker"
    print(f"[MPI] rank {rank}/{size}  {role}  on {host}", flush=True)
    comm.Barrier()

    if rank == 0:
        from mpi_src.master import run_master
        t0 = time.perf_counter()
        top_pairs = run_master(comm, size, args)
        elapsed = time.perf_counter() - t0

        print(f"\n{'=' * 52}")
        print(f"Top {len(top_pairs)} similar document pairs")
        print(f"{'=' * 52}")
        for i, (a, b, score) in enumerate(top_pairs, 1):
            print(f"  {i:3d}. doc_{a:04d} <-> doc_{b:04d}  cosine={score:.4f}")

        throughput = args.docs / elapsed if elapsed > 0 else 0
        print(
            f"\n[Stats] docs={args.docs} | ranks={size} | "
            f"chunk={args.chunk_size} | time={elapsed:.3f} | throughput={throughput:.2f}"
        )
    else:
        from mpi_src.worker import run_worker
        run_worker(comm, rank)


if __name__ == "__main__":
    main()
