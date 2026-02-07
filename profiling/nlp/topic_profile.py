from typing import List, Dict
from utils.trace import trace


@trace
def compute_topic_profile(
    documents: List[str],
    max_topics: int = 5,
    top_terms: int = 6,
) -> Dict:
    """
    Build a lightweight topic profile using TF-IDF + KMeans.
    Returns top terms per cluster and cluster sizes.
    """

    docs = [d for d in documents if d and len(d) > 20]
    if len(docs) < 3:
        return {
            "topics": [],
            "num_topics": 0,
            "num_docs": len(docs),
        }

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=800,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vectorizer.fit_transform(docs)
    n_docs = X.shape[0]
    if n_docs < 3:
        return {
            "topics": [],
            "num_topics": 0,
            "num_docs": n_docs,
        }

    # heuristic: 2–5 clusters based on doc count
    k = min(max_topics, max(2, n_docs // 5))
    k = min(k, n_docs)

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)

    terms = vectorizer.get_feature_names_out()
    topics = []
    for i in range(k):
        idxs = (labels == i).nonzero()[0]
        size = int(len(idxs))
        if size == 0:
            continue
        centroid = model.cluster_centers_[i]
        top_idx = centroid.argsort()[::-1][:top_terms]
        top = [terms[j] for j in top_idx]
        topics.append({
            "cluster": i,
            "size": size,
            "top_terms": top,
        })

    topics.sort(key=lambda t: t["size"], reverse=True)

    return {
        "topics": topics,
        "num_topics": len(topics),
        "num_docs": n_docs,
    }
