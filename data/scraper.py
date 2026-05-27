"""Collect documents and store in SQLite.

Usage:
    python data/scraper.py --source newsgroups --docs 2000
    python data/scraper.py --source web --urls data/urls.txt --docs 100
"""
import argparse
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "docs.db")


def init_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            source  TEXT,
            title   TEXT,
            author  TEXT,
            content TEXT NOT NULL
        )
    """)
    # migrare pentru baze de date vechi fără coloana author
    cols = [r[1] for r in conn.execute("PRAGMA table_info(docs)").fetchall()]
    if "author" not in cols:
        conn.execute("ALTER TABLE docs ADD COLUMN author TEXT")
    conn.commit()
    return conn


def scrape_newsgroups(conn: sqlite3.Connection, limit: int) -> None:
    from sklearn.datasets import fetch_20newsgroups
    print(f"Fetching 20 Newsgroups (up to {limit} docs)...")
    dataset = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    docs = [(d.strip(),) for d in dataset.data if d.strip()][:limit]
    conn.executemany("INSERT INTO docs (source, content) VALUES ('newsgroups', ?)", docs)
    conn.commit()
    print(f"Inserted {len(docs)} newsgroup documents.")


def scrape_urls(conn: sqlite3.Connection, urls: list[str], limit: int) -> None:
    import requests
    from bs4 import BeautifulSoup

    inserted = 0
    for url in urls[:limit]:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            text = soup.get_text(separator=" ", strip=True)
            if len(text) > 200:
                conn.execute(
                    "INSERT INTO docs (source, title, content) VALUES ('web', ?, ?)",
                    (title, text),
                )
                inserted += 1
                print(f"  [{inserted:4d}] {url[:70]}")
        except Exception as e:
            print(f"  SKIP {url[:60]}: {e}")

    conn.commit()
    print(f"Inserted {inserted} web documents.")


def scrape_arxiv(conn: sqlite3.Connection, query: str, limit: int) -> None:
    import time
    import requests
    import xml.etree.ElementTree as ET

    BATCH = 100  # arXiv recomandă max 100 per request
    headers = {"User-Agent": "doc-similarity-project/1.0 (student research)"}
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    inserted = 0
    start = 0

    print(f"Fetching arXiv papers (query='{query}', limit={limit})...")

    while inserted < limit:
        batch = min(BATCH, limit - inserted)
        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query={query}&start={start}&max_results={batch}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Request failed: {e}. Stopping.")
            break

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ns)
        if not entries:
            break

        for entry in entries:
            title_el   = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            if title_el is None or summary_el is None:
                continue

            title    = " ".join(title_el.text.split())
            abstract = " ".join(summary_el.text.split())
            authors  = ", ".join(
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            )

            if len(abstract) > 100:
                conn.execute(
                    "INSERT INTO docs (source, title, author, content) VALUES ('arxiv', ?, ?, ?)",
                    (title, authors, abstract),
                )
                inserted += 1
                print(f"  [{inserted:4d}] {title[:70]}")

        conn.commit()
        start += batch

        if inserted < limit and len(entries) == batch:
            time.sleep(3)  # arXiv cere pauză între request-uri

    print(f"Inserted {inserted} arXiv papers.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["newsgroups", "web", "arxiv"], default="newsgroups")
    parser.add_argument("--docs",   type=int, default=1000)
    parser.add_argument("--urls",   default="data/urls.txt", help="File with one URL per line (--source web)")
    parser.add_argument("--query",  default="cat:cs.DC", help="arXiv query string (--source arxiv)")
    parser.add_argument("--db",     default=DB_PATH)
    args = parser.parse_args()

    conn = init_db(args.db)

    if args.source == "newsgroups":
        scrape_newsgroups(conn, args.docs)
    elif args.source == "web":
        with open(args.urls) as f:
            urls = [line.strip() for line in f if line.strip()]
        scrape_urls(conn, urls, args.docs)
    elif args.source == "arxiv":
        scrape_arxiv(conn, args.query, args.docs)

    conn.close()
    print(f"Database saved to: {args.db}")


if __name__ == "__main__":
    main()
