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
    normed = matrix / norms

    chunk = normed[row_start:row_end]
    # un singur produs matriceal calculează toate scorurile cosinus dintr-odată
    scores = chunk @ normed.T

    # returnăm mai mult decât top_n ca masterul să aibă cu ce lucra la merge
    keep = min(top_n * 5, max(1, n - 1))
    pairs = []

    for local_i, global_i in enumerate(range(row_start, row_end)):
        row = scores[local_i].copy()
        # mascăm tot ce e la stânga (inclusiv documentul cu el însuși) ca să nu raportăm fiecare pereche de două ori
        row[: global_i + 1] = -2.0

        remaining = n - global_i - 1
        if remaining <= 0:
            continue

        k = min(keep, remaining)
        # argpartition găsește top-k fără să sorteze tot array-ul - mai rapid decât argsort
        top_j = np.argpartition(row, -k)[-k:]
        for j in top_j:
            if row[j] > -2.0:
                pairs.append((global_i, int(j), float(row[j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:keep]


def compute_pair_scores(matrix: np.ndarray, pairs: np.ndarray) -> list[tuple[int, int, float]]:
    """Compute cosine similarity for an explicit list of (i, j) candidate pairs."""
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    results = []
    for i, j in pairs:
        score = float(np.dot(matrix[i], matrix[j]) / (norms[i] * norms[j]))
        results.append((int(i), int(j), score))
    return results


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
