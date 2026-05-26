"""
Benchmark rapid: măsoară speedup real la volum mare de documente.

Rulare:
    .venv\\Scripts\\python.exe quick_bench.py
    .venv\\Scripts\\python.exe quick_bench.py --docs 2000 3000 5000
"""
import argparse
import json
import os
import subprocess
import sys
import time

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


MPIEXEC    = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
PYTHON     = os.path.join(os.path.dirname(sys.executable), "python.exe") or sys.executable
RANK_STEPS = [1, 2, 4, 8]
CHUNK_SIZE = 100
TOP_N      = 10
SOURCE     = "synthetic"   # "synthetic" evita overhead-ul de incarcare date


def run_once(n_ranks: int, docs: int) -> float | None:
    cmd = [
        MPIEXEC, "-n", str(n_ranks),
        PYTHON, "main.py",
        "--docs",       str(docs),
        "--chunk-size", str(CHUNK_SIZE),
        "--top-n",      str(TOP_N),
        "--source",     SOURCE,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"    EROARE: {proc.stderr[-200:]}")
        return None
    return elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, nargs="+", default=[2000, 4000, 6000])
    args = p.parse_args()

    results: list[dict] = []

    print(f"{'='*55}")
    print(f"  Quick Benchmark  —  ranks={RANK_STEPS}  chunk={CHUNK_SIZE}")
    print(f"{'='*55}")

    for docs in args.docs:
        print(f"\n  Documente: {docs}")
        t1 = None
        for ranks in RANK_STEPS:
            print(f"    p={ranks:2d}  ... ", end="", flush=True)
            elapsed = run_once(ranks, docs)
            if elapsed is None:
                continue
            if t1 is None:
                t1 = elapsed
            speedup = t1 / elapsed
            print(f"{elapsed:6.2f}s   speedup={speedup:.2f}x")
            results.append({
                "docs": docs, "n_ranks": ranks,
                "time": round(elapsed, 3),
                "speedup": round(speedup, 3),
                "throughput": round(docs / elapsed, 1),
            })

    print(f"\n{'─'*55}")
    print(f"  {'docs':>6}  {'p':>4}  {'time(s)':>8}  {'speedup':>9}  {'docs/s':>8}")
    print(f"{'─'*55}")
    for r in results:
        print(f"  {r['docs']:>6}  {r['n_ranks']:>4}  {r['time']:>8.2f}  "
              f"  {r['speedup']:>6.2f}x  {r['throughput']:>8.1f}")

    out_json = "quick_bench_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRezultate salvate: {out_json}")

    if HAS_PLOT and results:
        _plot(results, args.docs)


def _plot(results: list[dict], docs_list: list[int]) -> None:
    COLORS  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    MARKERS = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Similaritate Documente — Speedup Real (MPI)", fontsize=14, fontweight="bold")

    # ── Speedup ──────────────────────────────────────────────────────────────
    ax = axes[0]
    for i, docs in enumerate(docs_list):
        pts = sorted([r for r in results if r["docs"] == docs], key=lambda x: x["n_ranks"])
        if pts:
            ax.plot([p["n_ranks"] for p in pts], [p["speedup"] for p in pts],
                    marker=MARKERS[i], color=COLORS[i], linewidth=2.5, markersize=8,
                    label=f"{docs} documente")

    max_rank = max(RANK_STEPS)
    ax.plot([1, max_rank], [1, max_rank], "k--", linewidth=1, alpha=0.4, label="Ideal")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Număr procese MPI  (p)", fontsize=11)
    ax.set_ylabel("Speedup   S = T₁ / Tₚ", fontsize=11)
    ax.set_title("Speedup", fontsize=12, fontweight="bold")
    ax.set_xticks(RANK_STEPS)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)

    # ── Timp de execuție ─────────────────────────────────────────────────────
    ax = axes[1]
    for i, docs in enumerate(docs_list):
        pts = sorted([r for r in results if r["docs"] == docs], key=lambda x: x["n_ranks"])
        if pts:
            ax.plot([p["n_ranks"] for p in pts], [p["time"] for p in pts],
                    marker=MARKERS[i], color=COLORS[i], linewidth=2.5, markersize=8,
                    label=f"{docs} documente")

    ax.set_xlabel("Număr procese MPI  (p)", fontsize=11)
    ax.set_ylabel("Timp execuție  (s)", fontsize=11)
    ax.set_title("Timp de execuție", fontsize=12, fontweight="bold")
    ax.set_xticks(RANK_STEPS)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)

    # ── Throughput ───────────────────────────────────────────────────────────
    ax = axes[2]
    all_ranks = sorted(set(r["n_ranks"] for r in results))
    for i, ranks in enumerate(all_ranks):
        pts = sorted([r for r in results if r["n_ranks"] == ranks], key=lambda x: x["docs"])
        if pts:
            ax.plot([p["docs"] for p in pts], [p["throughput"] for p in pts],
                    marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                    linewidth=2.5, markersize=8, label=f"p = {ranks}")

    ax.set_xlabel("Număr documente", fontsize=11)
    ax.set_ylabel("Throughput  (doc/s)", fontsize=11)
    ax.set_title("Throughput", fontsize=12, fontweight="bold")
    ax.set_xticks(sorted(set(r["docs"] for r in results)))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    out_png = "quick_bench_speedup.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafic salvat: {out_png}")


if __name__ == "__main__":
    main()
