"""
Client-side benchmark: measures latency and throughput against the MPI server.

Workflow:
  1. Start the server on Mac with a given N_RANKS, e.g.:
       docker run -p 5555:5555 -e N_RANKS=4 similarity-server

  2. Run this script:
       python benchmark_client.py --host 192.168.0.193
       python benchmark_client.py --host 192.168.0.193 --queries 20 --sizes 500 2000 8000

  3. Change N_RANKS in Docker, repeat step 2, compare the printed tables.
"""

import argparse
import json
import os
import pickle
import socket
import struct
import time
import random
import string

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ── Network helpers (same as client.py) ──────────────────────────────────────

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
            raise ConnectionError("server disconnected")
        buf.extend(chunk)
    return bytes(buf)


# ── Query helpers ─────────────────────────────────────────────────────────────

_WORDS = (
    "the quick brown fox jumps over the lazy dog machine learning "
    "neural network deep learning natural language processing text "
    "classification clustering similarity document retrieval search "
    "vector space model cosine distance tfidf embedding representation"
).split()


def _make_query(n_words: int) -> str:
    return " ".join(random.choices(_WORDS, k=n_words))


def query_server(host: str, port: int, text: str) -> tuple[float, int]:
    """Send one query; return (latency_seconds, n_results)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        t0 = time.perf_counter()
        _send(sock, text)
        results = _recv(sock)
        latency = time.perf_counter() - t0
    if isinstance(results, dict) and "error" in results:
        raise RuntimeError(f"Server error: {results['error']}")
    return latency, len(results)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(host: str, port: int, n_queries: int,
                  word_counts: list[int], warmup: int = 2) -> list[dict]:
    results = []

    for wc in word_counts:
        latencies = []

        # Warmup
        for _ in range(warmup):
            try:
                query_server(host, port, _make_query(wc))
            except Exception:
                pass

        # Measured queries
        print(f"  words={wc:5d}  ", end="", flush=True)
        for i in range(n_queries):
            text = _make_query(wc)
            try:
                lat, n_res = query_server(host, port, text)
                latencies.append(lat)
                print(".", end="", flush=True)
            except Exception as exc:
                print(f"E({exc})", end="", flush=True)

        print()

        if not latencies:
            continue

        avg   = sum(latencies) / len(latencies)
        mn    = min(latencies)
        mx    = max(latencies)
        p95   = sorted(latencies)[int(len(latencies) * 0.95)]
        qps   = len(latencies) / sum(latencies)

        results.append({
            "word_count": wc,
            "n_queries":  len(latencies),
            "avg_s":      round(avg,  4),
            "min_s":      round(mn,   4),
            "max_s":      round(mx,   4),
            "p95_s":      round(p95,  4),
            "qps":        round(qps,  3),
        })

    return results


# ── Output ─────────────────────────────────────────────────────────────────────

def print_table(results: list[dict], n_ranks: int | None) -> None:
    title = f"N_RANKS={n_ranks}" if n_ranks else "results"
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")
    print(f"  {'words':>6}  {'avg(s)':>8}  {'min(s)':>8}  "
          f"{'p95(s)':>8}  {'max(s)':>8}  {'q/s':>7}")
    print(f"{'─'*62}")
    for r in results:
        print(f"  {r['word_count']:>6}  {r['avg_s']:>8.4f}  {r['min_s']:>8.4f}  "
              f"  {r['p95_s']:>8.4f}  {r['max_s']:>8.4f}  {r['qps']:>7.2f}")


def plot_results(all_runs: list[dict], out_png: str = "benchmark_client.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Server Benchmark — Latency vs Query Size", fontsize=12)

    for run in all_runs:
        label = f"N_RANKS={run['n_ranks']}"
        xs = [r["word_count"] for r in run["results"]]
        avgs = [r["avg_s"]     for r in run["results"]]
        qps  = [r["qps"]       for r in run["results"]]

        axes[0].plot(xs, avgs, marker="o", label=label)
        axes[1].plot(xs, qps,  marker="o", label=label)

    axes[0].set_xlabel("Query size (words)")
    axes[0].set_ylabel("Avg latency (s)")
    axes[0].set_title("Latency")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel("Query size (words)")
    axes[1].set_ylabel("Queries / second")
    axes[1].set_title("Throughput")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"\nPlot saved to {out_png}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark the similarity server")
    p.add_argument("--host",     default="127.0.0.1")
    p.add_argument("--port",     type=int,   default=5555)
    p.add_argument("--queries",  type=int,   default=10,
                   help="Number of measured queries per word-count")
    p.add_argument("--sizes",    type=int,   nargs="+",
                   default=[100, 500, 1000, 2000],
                   help="Query sizes in words to test")
    p.add_argument("--warmup",   type=int,   default=2)
    p.add_argument("--n-ranks",  type=int,   default=None,
                   help="N_RANKS the server is running with (for labelling only)")
    p.add_argument("--out-json", default=None,
                   help="Append results to this JSON file (for multi-run comparison)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    print(f"Benchmark -> {args.host}:{args.port}")
    print(f"  queries={args.queries}  sizes={args.sizes}  warmup={args.warmup}")
    if args.n_ranks:
        print(f"  server N_RANKS={args.n_ranks} (label only — set in Docker)")
    print()

    results = run_benchmark(args.host, args.port, args.queries,
                            args.sizes, args.warmup)
    print_table(results, args.n_ranks)

    if args.out_json:
        runs = []
        if os.path.exists(args.out_json):
            with open(args.out_json) as f:
                runs = json.load(f)
        runs.append({"n_ranks": args.n_ranks, "results": results})
        with open(args.out_json, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"\nAppended to {args.out_json}")

        if HAS_PLOT and len(runs) > 1:
            v = 1
            while os.path.exists(f"benchmark_client_v{v}.png"):
                v += 1
            plot_results(runs, f"benchmark_client_v{v}.png")
    elif HAS_PLOT:
        if results:
            plot_results([{"n_ranks": args.n_ranks, "results": results}])


if __name__ == "__main__":
    main()
