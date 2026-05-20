import numpy as np


def partial_top_pairs(matrix: np.ndarray, row_start: int, row_end: int, top_n: int) -> list:
    """
    Compute cosine similarity for rows [row_start:row_end] vs all rows.
    Only considers pairs (i, j) where i < j to avoid duplicates.
    Returns at most top_n * 5 pairs (master merges across workers).
    """
    n = matrix.shape[0]

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = matrix / norms  # (n, vocab)

    chunk = normed[row_start:row_end]  # (chunk_rows, vocab)
    scores = chunk @ normed.T          # (chunk_rows, n)

    keep = min(top_n * 5, max(1, n - 1))
    pairs = []

    for local_i, global_i in enumerate(range(row_start, row_end)):
        row = scores[local_i].copy()
        row[: global_i + 1] = -2.0  # mask self and lower triangle

        remaining = n - global_i - 1
        if remaining <= 0:
            continue

        k = min(keep, remaining)
        top_j = np.argpartition(row, -k)[-k:]
        for j in top_j:
            if row[j] > -2.0:
                pairs.append((global_i, int(j), float(row[j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:keep]


def merge_top_n(all_pairs: list, top_n: int) -> list:
    """Deduplicate and return the global top-N pairs from merged worker results."""
    seen = set()
    result = []
    for a, b, score in sorted(all_pairs, key=lambda x: x[2], reverse=True):
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            result.append((a, b, score))
        if len(result) >= top_n:
            break
    return result
