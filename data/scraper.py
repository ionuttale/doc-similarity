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
            content TEXT NOT NULL
        )
    """)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["newsgroups", "web"], default="newsgroups")
    parser.add_argument("--docs",   type=int, default=1000)
    parser.add_argument("--urls",   default="data/urls.txt", help="File with one URL per line")
    parser.add_argument("--db",     default=DB_PATH)
    args = parser.parse_args()

    conn = init_db(args.db)

    if args.source == "newsgroups":
        scrape_newsgroups(conn, args.docs)
    elif args.source == "web":
        with open(args.urls) as f:
            urls = [line.strip() for line in f if line.strip()]
        scrape_urls(conn, urls, args.docs)

    conn.close()
    print(f"Database saved to: {args.db}")


if __name__ == "__main__":
    main()
