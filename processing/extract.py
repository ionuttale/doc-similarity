"""Extract plain text from .txt, .pdf, .html, and .htm files."""
import os


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".html", ".htm"}


def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return _from_pdf(filepath)
    if ext in (".html", ".htm"):
        return _from_html(filepath)
    return _from_text(filepath)


def extract_from_directory(directory: str) -> list[str]:
    documents = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTENSIONS:
                continue
            path = os.path.join(root, fname)
            try:
                text = extract_text(path).strip()
                if text:
                    documents.append(text)
            except Exception as e:
                print(f"  [extract] SKIP {fname}: {e}")
    return documents


def _from_text(filepath: str) -> str:
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        return f.read()


def _from_pdf(filepath: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pip install pdfplumber")
    with pdfplumber.open(filepath) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _from_html(filepath: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("pip install beautifulsoup4")
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        return BeautifulSoup(f.read(), "html.parser").get_text(separator=" ", strip=True)
