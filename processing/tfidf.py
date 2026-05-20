from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(
    documents: list[str],
    max_features: int = 10_000,
    min_df: int = 2,
    max_df: float = 0.95,
) -> spmatrix:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,       # replace tf with 1 + log(tf) — reduces impact of high-freq terms
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )
    return vectorizer.fit_transform(documents)
