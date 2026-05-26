"""MinHash + LSH for fast approximate Jaccard-based candidate pair generation."""
import numpy as np
from scipy.sparse import spmatrix

N_HASHES = 128
N_BANDS = 16
_PRIME = (1 << 31) - 1  # Mersenne prime for universal hashing


def compute_minhash(tfidf_matrix: spmatrix, n_hashes: int = N_HASHES, seed: int = 42) -> np.ndarray:
    """Return MinHash signature matrix of shape (n_docs, n_hashes)."""
    n_docs, n_terms = tfidf_matrix.shape
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=n_hashes, dtype=np.int64)
    b = rng.integers(0, _PRIME, size=n_hashes, dtype=np.int64)

    terms = np.arange(n_terms, dtype=np.int64)
    # hash_table[h, t] = hash value for term t under hash function h
    hash_table = ((a[:, None] * terms[None, :] + b[:, None]) % _PRIME)  # (n_hashes, n_terms)

    signatures = np.full((n_docs, n_hashes), _PRIME, dtype=np.int64)
    cx = tfidf_matrix.tocsr()
    rows, cols = cx.nonzero()

    for h in range(n_hashes):
        np.minimum.at(signatures[:, h], rows, hash_table[h, cols])

    return signatures


def lsh_candidate_pairs(signatures: np.ndarray, n_bands: int = N_BANDS) -> list[tuple[int, int]]:
    """
    LSH banding: two docs sharing a bucket in ANY band are candidate pairs.
    Threshold Jaccard ≈ (1/n_bands)^(1/rows_per_band).
    """
    n_docs, n_hashes = signatures.shape
    rpb = n_hashes // n_bands
    candidates: set[tuple[int, int]] = set()

    for band in range(n_bands):
        band_sigs = signatures[:, band * rpb : (band + 1) * rpb]
        buckets: dict[bytes, list[int]] = {}
        for i in range(n_docs):
            key = band_sigs[i].tobytes()
            buckets.setdefault(key, []).append(i)
        for group in buckets.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    lo, hi = (group[i], group[j]) if group[i] < group[j] else (group[j], group[i])
                    candidates.add((lo, hi))

    return sorted(candidates)
