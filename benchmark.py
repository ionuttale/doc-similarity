"""Benchmark: local parallelism vs distributed (multi-machine) MPI execution.

Phases:
  1. Connectivity check  — ping each host in hostfile
  2. Sequential baseline — n=1 (no MPI overhead)
  3. Local parallel      — n=2..LOCAL_CORES on this machine only
  4. Distributed         — n=LOCAL_CORES+REMOTE_CORES across both machines

Run:
    python benchmark.py
    python benchmark.py --skip-distributed   (if Mac is not available)
    python benchmark.py --docs 200 500 --chunk-size 50
"""
import argparse
import itertools
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

# Resolve mpiexec — fallback to MSMPI default install path on Windows
def _find_mpiexec() -> str:
    import shutil
    if shutil.which("mpiexec"):
        return "mpiexec"
    fallback = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        "mpiexec not found. Add it to PATH or install MSMPI from "
        "https://github.com/microsoft/Microsoft-MPI/releases"
    )

# ── Configuration ────────────────────────────────────────────────────────────

HOSTFILE      = "hostfile"
LOCAL_IP      = "192.168.0.127"
LOCAL_CORES   = 12
REMOTE_IP     = "192.168.0.193"
REMOTE_CORES  = 8
TOTAL_CORES   = LOCAL_CORES + REMOTE_CORES

SOURCE  = "newsgroups"
TOP_N   = 10

# Local ranks to test (powers of 2 up to LOCAL_CORES, plus LOCAL_CORES itself)
LOCAL_RANK_STEPS  = [1, 2, 4, 8, LOCAL_CORES]
# Distributed ranks to test
DIST_RANK_STEPS   = [LOCAL_CORES, LOCAL_CORES + 4, TOTAL_CORES]

DEFAULT_DOCS       = [200, 500, 1000]
DEFAULT_CHUNK_SIZE = [50, 100]

# ── Helpers ──────────────────────────────────────────────────────────────────

def ping(host: str) -> bool:
    result = subprocess.run(
        ["ping", "-n", "2", "-w", "1000", host],
        capture_output=True,
    )
    return result.returncode == 0


def check_hosts() -> dict[str, bool]:
    hosts = {LOCAL_IP: True}  # always reachable (it's us)
    print(f"  {LOCAL_IP:<18} LOCAL  ✓")
    for ip in [REMOTE_IP]:
        ok = ping(ip)
        status = "✓ reachable" if ok else "✗ unreachable"
        print(f"  {ip:<18} REMOTE {status}")
        hosts[ip] = ok
    return hosts


def run_mpi(n_ranks: int, docs: int, chunk_size: int, extra_args: list[str]) -> dict | None:
    import threading

    cmd = [
        _find_mpiexec(), "-n", str(n_ranks),
        *extra_args,
        sys.executable, "main.py",
        "--docs",       str(docs),
        "--chunk-size", str(chunk_size),
        "--top-n",      str(TOP_N),
        "--source",     SOURCE,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stop_spinner = threading.Event()

    def spinner():
        while not stop_spinner.wait(timeout=1.0):
            print(".", end="", flush=True)

    t = threading.Thread(target=spinner, daemon=True)
    t.start()
    t0 = time.perf_counter()
    stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - t0
    stop_spinner.set()
    t.join()

    if proc.returncode != 0:
        print(f"\n    ERROR: {stderr[-300:]}")
        return None

    throughput = None
    hosts_seen: dict[str, int] = {}
    for line in stdout.splitlines():
        if "throughput=" in line:
            try:
                throughput = float(line.split("throughput=")[1].split()[0])
            except Exception:
                pass
        if line.startswith("[MPI] rank"):
            try:
                host = line.split(" on ")[-1].strip()
                hosts_seen[host] = hosts_seen.get(host, 0) + 1
            except Exception:
                pass

    hosts_str = "  [" + ", ".join(f"{h}×{n}" for h, n in hosts_seen.items()) + "]" if hosts_seen else ""

    return {
        "docs":       docs,
        "n_ranks":    n_ranks,
        "chunk_size": chunk_size,
        "time":       round(elapsed, 4),
        "throughput": throughput if throughput else round(docs / elapsed, 2),
        "hosts":      hosts_str,
    }


def add_speedup(results: list[dict]) -> None:
    baseline: dict[tuple, float] = {
        (r["docs"], r["chunk_size"]): r["time"]
        for r in results if r["n_ranks"] == 1
    }
    for r in results:
        key = (r["docs"], r["chunk_size"])
        b = baseline.get(key)
        r["speedup"] = round(b / r["time"], 3) if b else None


def print_table(results: list[dict], title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    print(f"{'docs':>6} {'ranks':>6} {'chunk':>6} {'time(s)':>9} {'docs/s':>10} {'speedup':>9}")
    print(f"{'─'*60}")
    for r in results:
        sp = f"{r['speedup']:.2f}x" if r.get("speedup") else "   N/A"
        print(f"{r['docs']:>6} {r['n_ranks']:>6} {r['chunk_size']:>6} "
              f"{r['time']:>9.3f} {r['throughput']:>10.1f} {sp:>9}")


# ── Benchmark phases ─────────────────────────────────────────────────────────

def run_local(docs_list: list[int], chunk_sizes: list[int]) -> list[dict]:
    """Sequential + local parallel runs (no hostfile)."""
    results = []
    combos = list(itertools.product(LOCAL_RANK_STEPS, docs_list, chunk_sizes))
    print(f"\n[Local] {len(combos)} configurations")

    for n_ranks, docs, chunk_size in combos:
        print(f"  ranks={n_ranks:2d}  docs={docs:5d}  chunk={chunk_size:4d}  ... ",
              end="", flush=True)
        r = run_mpi(n_ranks, docs, chunk_size, extra_args=[])
        if r:
            r["mode"] = "local"
            results.append(r)
            print(f"{r['time']:.3f}s  ({r['throughput']:.1f} docs/s){r['hosts']}")
        else:
            print("FAILED")

    return results


def run_distributed(docs_list: list[int], chunk_sizes: list[int]) -> list[dict]:
    """Distributed runs across all hosts in hostfile."""
    if not os.path.exists(HOSTFILE):
        print(f"\n[Distributed] hostfile '{HOSTFILE}' not found — skipping.")
        return []

    results = []
    combos = list(itertools.product(DIST_RANK_STEPS, docs_list, chunk_sizes))
    print(f"\n[Distributed] {len(combos)} configurations  (hostfile: {HOSTFILE})")

    # MSMPI uses -machinefile; OpenMPI uses -hostfile — try both
    mf_flag = "-machinefile"

    for n_ranks, docs, chunk_size in combos:
        print(f"  ranks={n_ranks:2d}  docs={docs:5d}  chunk={chunk_size:4d}  ... ",
              end="", flush=True)
        r = run_mpi(n_ranks, docs, chunk_size, extra_args=[mf_flag, HOSTFILE])
        if r:
            r["mode"] = "distributed"
            results.append(r)
            print(f"{r['time']:.3f}s  ({r['throughput']:.1f} docs/s){r['hosts']}")
        else:
            print("FAILED")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def _avg_over_chunks(data: list[dict]) -> list[dict]:
    """Average time/throughput/speedup over chunk_size variants for the same (docs, n_ranks)."""
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in data:
        groups[(r["docs"], r["n_ranks"])].append(r)
    result = []
    for (docs, n_ranks), rows in groups.items():
        sp_vals = [r["speedup"] for r in rows if r.get("speedup") is not None]
        result.append({
            "docs":       docs,
            "n_ranks":    n_ranks,
            "time":       sum(r["time"] for r in rows) / len(rows),
            "throughput": sum(r["throughput"] for r in rows) / len(rows),
            "speedup":    sum(sp_vals) / len(sp_vals) if sp_vals else None,
            "mode":       rows[0].get("mode", "local"),
        })
    return result


def plot(local: list[dict], distributed: list[dict], docs_list: list[int], out_png: str = "benchmark_speedup.png") -> None:
    local = _avg_over_chunks(local)
    distributed = _avg_over_chunks(distributed)

    all_docs  = sorted(set(r["docs"]    for r in local))
    all_ranks = sorted(set(r["n_ranks"] for r in local))

    COLORS  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    MARKERS = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Similaritate Documente — Analiză Performanță MPI", fontsize=14, fontweight="bold")

    # ── ① Speedup S = T₁ / Tₚ ────────────────────────────────────────────────
    ax = axes[0]
    for i, docs in enumerate(all_docs):
        pts = sorted([r for r in local if r["docs"] == docs and r.get("speedup") is not None],
                     key=lambda x: x["n_ranks"])
        if pts:
            ax.plot([p["n_ranks"] for p in pts], [p["speedup"] for p in pts],
                    marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=7,
                    label=f"{docs} documente")
        if distributed:
            pts_d = sorted([r for r in distributed if r["docs"] == docs and r.get("speedup") is not None],
                           key=lambda x: x["n_ranks"])
            if pts_d:
                ax.plot([p["n_ranks"] for p in pts_d], [p["speedup"] for p in pts_d],
                        marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=7,
                        linestyle="--", label=f"{docs} doc. (dist.)")

    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, label="S = 1  (nicio îmbunătățire)")
    ax.set_xlabel("Număr procese MPI  (p)", fontsize=11)
    ax.set_ylabel("Speedup   S = T₁ / Tₚ", fontsize=11)
    ax.set_title("Speedup", fontsize=12, fontweight="bold")
    ax.set_xticks(all_ranks)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    # ── ② Timp de execuție vs număr procese ───────────────────────────────────
    ax = axes[1]
    for i, docs in enumerate(all_docs):
        pts = sorted([r for r in local if r["docs"] == docs], key=lambda x: x["n_ranks"])
        if pts:
            ax.plot([p["n_ranks"] for p in pts], [p["time"] for p in pts],
                    marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=7,
                    label=f"{docs} doc. (local)")
        if distributed:
            pts_d = sorted([r for r in distributed if r["docs"] == docs], key=lambda x: x["n_ranks"])
            if pts_d:
                ax.plot([p["n_ranks"] for p in pts_d], [p["time"] for p in pts_d],
                        marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=7,
                        linestyle="--", label=f"{docs} doc. (dist.)")

    ax.set_xlabel("Număr procese MPI  (p)", fontsize=11)
    ax.set_ylabel("Timp execuție  (s)", fontsize=11)
    ax.set_title("Timp de execuție", fontsize=12, fontweight="bold")
    ax.set_xticks(all_ranks)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    # ── ③ Throughput vs număr documente ──────────────────────────────────────
    ax = axes[2]
    for i, ranks in enumerate(all_ranks):
        pts = sorted([r for r in local if r["n_ranks"] == ranks], key=lambda x: x["docs"])
        if pts:
            ax.plot([p["docs"] for p in pts], [p["throughput"] for p in pts],
                    marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                    linewidth=2, markersize=7, label=f"p = {ranks}")

    ax.set_xlabel("Număr documente", fontsize=11)
    ax.set_ylabel("Throughput  (doc/s)", fontsize=11)
    ax.set_title("Throughput", fontsize=12, fontweight="bold")
    ax.set_xticks(all_docs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGrafic salvat: {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--docs",             type=int, nargs="+", default=DEFAULT_DOCS)
    p.add_argument("--chunk-size",       type=int, nargs="+", default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--skip-distributed", action="store_true")
    p.add_argument("--skip-local",       action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 60)
    print("  MPI Document Similarity — Benchmark")
    print("=" * 60)

    print("\n[Connectivity check]")
    hosts = check_hosts()
    remote_ok = hosts.get(REMOTE_IP, False)
    if not remote_ok:
        print(f"  Warning: {REMOTE_IP} unreachable — distributed phase will be skipped.")

    local_results = []
    dist_results  = []

    if not args.skip_local:
        local_results = run_local(args.docs, args.chunk_size)
        add_speedup(local_results)
        print_table(local_results, "Local results")

    if not args.skip_distributed and remote_ok:
        dist_results = run_distributed(args.docs, args.chunk_size)
        add_speedup(dist_results)
        print_table(dist_results, "Distributed results")
    elif not args.skip_distributed and not remote_ok:
        print("\n[Distributed] Skipped (remote host unreachable).")

    all_results = local_results + dist_results
    if not all_results:
        print("\nNo results collected.")
        return

    v = 1
    while os.path.exists(f"benchmark_results_v{v}.json"):
        v += 1
    out_json = f"benchmark_results_v{v}.json"
    out_png  = f"benchmark_speedup_v{v}.png"

    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_json} ({len(all_results)} entries)")

    if HAS_PLOT:
        plot(local_results, dist_results, args.docs, out_png)
    else:
        print("Install matplotlib to generate plots: pip install matplotlib")


if __name__ == "__main__":
    main()
