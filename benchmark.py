"""Benchmark speedup and throughput across different MPI configurations.

Run:
    python benchmark.py

Requires mpiexec (OpenMPI or MPICH) in PATH.
Adjust CONFIGS below to match your machine's CPU count and desired doc sizes.
"""
import itertools
import json
import subprocess
import sys
import time

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

CONFIGS = {
    "docs":       [200, 500, 1000],
    "n_ranks":    [1, 2, 4],      # set max to your logical CPU count
    "chunk_size": [50, 100],
}

SOURCE = "newsgroups"
TOP_N  = 10


def run_config(docs: int, n_ranks: int, chunk_size: int) -> dict | None:
    cmd = [
        "mpiexec", "-n", str(n_ranks),
        sys.executable, "main.py",
        "--docs",       str(docs),
        "--chunk-size", str(chunk_size),
        "--top-n",      str(TOP_N),
        "--source",     SOURCE,
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  ERROR:\n{result.stderr[-400:]}")
        return None

    throughput = None
    for line in result.stdout.splitlines():
        if "throughput=" in line:
            try:
                throughput = float(line.split("throughput=")[1].split()[0])
            except Exception:
                pass

    return {
        "docs":       docs,
        "n_ranks":    n_ranks,
        "chunk_size": chunk_size,
        "time":       elapsed,
        "throughput": throughput if throughput else docs / elapsed,
    }


def main() -> None:
    combos = list(itertools.product(CONFIGS["docs"], CONFIGS["n_ranks"], CONFIGS["chunk_size"]))
    print(f"Running {len(combos)} configurations...\n")

    results = []
    for docs, n_ranks, chunk_size in combos:
        print(f"  docs={docs:5d}  ranks={n_ranks}  chunk={chunk_size:4d}  ... ", end="", flush=True)
        r = run_config(docs, n_ranks, chunk_size)
        if r:
            results.append(r)
            print(f"{r['time']:.2f}s  ({r['throughput']:.1f} docs/s)")
        else:
            print("FAILED")

    if not results:
        print("No results to analyze.")
        return

    # Compute speedup relative to single-rank baseline
    baseline: dict[tuple, float] = {
        (r["docs"], r["chunk_size"]): r["time"]
        for r in results if r["n_ranks"] == 1
    }
    for r in results:
        key = (r["docs"], r["chunk_size"])
        r["speedup"] = baseline[key] / r["time"] if key in baseline else None

    # Print table
    print(f"\n{'docs':>6} {'ranks':>6} {'chunk':>6} {'time(s)':>10} {'throughput':>12} {'speedup':>8}")
    print("-" * 54)
    for r in results:
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else "  N/A"
        print(f"{r['docs']:>6} {r['n_ranks']:>6} {r['chunk_size']:>6} {r['time']:>10.3f} "
              f"{r['throughput']:>12.1f} {sp:>8}")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")

    if HAS_PLOT:
        _plot(results)


def _plot(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for docs in CONFIGS["docs"]:
        subset = sorted(
            [r for r in results if r["docs"] == docs and r["speedup"]],
            key=lambda x: x["n_ranks"],
        )
        if subset:
            ax.plot([r["n_ranks"] for r in subset], [r["speedup"] for r in subset],
                    marker="o", label=f"{docs} docs")
    ranks = CONFIGS["n_ranks"]
    ax.plot(ranks, ranks, "k--", label="Ideal")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Number of Ranks")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    for docs in CONFIGS["docs"]:
        subset = sorted(
            [r for r in results if r["docs"] == docs],
            key=lambda x: x["n_ranks"],
        )
        if subset:
            ax.plot([r["n_ranks"] for r in subset], [r["throughput"] for r in subset],
                    marker="s", label=f"{docs} docs")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Throughput (docs/s)")
    ax.set_title("Throughput vs Number of Ranks")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_speedup.png", dpi=150)
    plt.show()
    print("Plot saved to benchmark_speedup.png")


if __name__ == "__main__":
    main()
