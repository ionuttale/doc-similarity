"""
Document similarity client.

Usage:
    python client.py --file my_document.txt
    python client.py --file my_document.txt --host 192.168.0.127 --port 5555
"""

import argparse
import pickle
import socket
import struct
import sys
import time


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
            raise ConnectionError("server disconnected")
        buf.extend(chunk)
    return bytes(buf)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Send a file to the similarity server")
    p.add_argument("--file",    required=True, help="Path to the file to check")
    p.add_argument("--host",    default="127.0.0.1")
    p.add_argument("--port",    type=int, default=5555)
    p.add_argument("--encoding",default="utf-8")
    return p


def main() -> None:
    args = build_parser().parse_args()

    import os
    ext = os.path.splitext(args.file)[1].lower()
    try:
        if ext == ".pdf":
            import pdfplumber
            with pdfplumber.open(args.file) as pdf:
                content = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif ext in (".html", ".htm"):
            from bs4 import BeautifulSoup
            with open(args.file, encoding=args.encoding, errors="ignore") as f:
                content = BeautifulSoup(f.read(), "html.parser").get_text(separator=" ", strip=True)
        else:
            with open(args.file, "r", encoding=args.encoding) as f:
                content = f.read()
    except FileNotFoundError:
        print(f"[Client] ERROR: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"[Client] File: {args.file} ({len(content):,} chars)")
    print(f"[Client] Connecting to {args.host}:{args.port}…")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect((args.host, args.port))
        except ConnectionRefusedError:
            print(f"[Client] ERROR: connection refused — is the server running?", file=sys.stderr)
            sys.exit(1)

        t0 = time.perf_counter()
        _send(sock, content)
        results = _recv(sock)
        elapsed = time.perf_counter() - t0

    if isinstance(results, dict) and "error" in results:
        print(f"[Client] Server error: {results['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"[Client] Response in {elapsed:.3f}s — {len(results)} match(es)\n")

    if not results:
        print("  (no matches returned)")
        return

    print(f"  {'Rank':>4}  {'Doc ID':>8}  {'Score':>8}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}")
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"  {i:>4}  {str(doc_id):>8}  {score:>8.4f}")


if __name__ == "__main__":
    main()
