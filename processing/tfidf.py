from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(
    documents: list[str],
    max_features: int = 10_000,
    min_df: int = 2,      # ignorăm cuvintele rare (apar într-un singur document - de obicei greșeli sau cod)
    max_df: float = 0.95, # ignorăm cuvintele prea comune (apar în aproape tot - nu ajută la diferențiere)
) -> spmatrix:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,       # folosim log pe frecvență ca un cuvânt repetat de 100x să nu bată unul apărut de 10x
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",  # exclude cifre și cuvinte de o literă
    )
    return vectorizer.fit_transform(documents)
