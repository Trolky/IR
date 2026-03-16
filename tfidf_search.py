import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


def tokenize_whitespace_lower(text: str) -> List[str]:
    """Tokenize text by whitespace and lowercase it.

    This is intentionally a minimal tokenizer for the homework examples where
    documents/queries are already naturally separated by spaces.

    Args:
        text: Input text.

    Returns:
        A list of lowercase tokens. Empty tokens are filtered out.
    """
    return [t for t in text.lower().split() if t]


def tf_raw(tokens: Sequence[str]) -> Dict[str, float]:
    """Compute raw term frequency (TF) as token occurrence counts.

    Args:
        tokens: Sequence of tokens (e.g., a list of strings).

    Returns:
        A mapping term -> occurrence count (float for convenience in later math).
    """
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    return tf


def l2_norm(vec: Dict[str, float]) -> float:
    """Return the L2 norm (Euclidean length) of a sparse vector.

    Args:
        vec: Sparse vector as dict term -> weight.

    Returns:
        The L2 norm: sqrt(sum_i v_i^2). Returns 0.0 for an empty vector.
    """
    return math.sqrt(sum(v * v for v in vec.values()))


def l2_normalize(vec: Dict[str, float]) -> Dict[str, float]:
    """Return an L2-normalized copy of a sparse vector.

    Args:
        vec: Sparse vector as dict term -> weight.

    Returns:
        A new dict term -> weight / ||vec||_2.
        If the norm is 0.0 (empty / all-zero vector), returns {}.
    """
    n = l2_norm(vec)
    if n == 0.0:
        return {}
    return {k: v / n for k, v in vec.items()}


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors.

    The dot product is computed over the intersection of keys (iterating over
    the smaller dict for efficiency), then divided by the product of L2 norms.

    Args:
        a: Sparse vector A as dict term -> weight.
        b: Sparse vector B as dict term -> weight.

    Returns:
        Cosine similarity. With non-negative weights it will be in [0, 1].
        If either vector is empty or has zero norm, returns 0.0.
    """
    if not a or not b:
        return 0.0

    # dot
    if len(a) < len(b):
        smaller, larger = a, b
    else:
        smaller, larger = b, a

    dot = 0.0
    for k, v in smaller.items():
        dot += v * larger.get(k, 0.0)

    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class SearchResult:
    """Search result item.

    Attributes:
        doc_id: Document identifier (e.g., "d1" or a URL).
        score: Cosine similarity score (higher = more relevant).
    """

    doc_id: str
    score: float


class TfidfIndex:
    """A simple TF-IDF index with cosine similarity.

    Conventions:

    - TF = raw counts (number of occurrences of a term in a document)
    - DF(term) = number of documents containing the term at least once
    - IDF(term) = log10(N / DF(term))
    - TF-IDF(term, doc) = TF(term, doc) * IDF(term)
    - Similarity = cosine(query_tfidf, doc_tfidf)
    """

    def __init__(self):
        """Create an empty index."""
        self._doc_ids: List[str] = []
        self._doc_tfidf: List[Dict[str, float]] = []
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._N: int = 0

    @property
    def N(self) -> int:
        """Number of indexed documents."""
        return self._N

    @property
    def df(self) -> Dict[str, int]:
        """Document frequency for each term (returns a copy)."""
        return dict(self._df)

    @property
    def idf(self) -> Dict[str, float]:
        """Inverse document frequency for each term (returns a copy)."""
        return dict(self._idf)

    def build(self, documents: Sequence[Tuple[str, Sequence[str]]]) -> None:
        """Build DF/IDF statistics and TF-IDF vectors for all documents.

        Args:
            documents: Sequence of (doc_id, tokens) pairs.
                - doc_id: arbitrary string identifier
                - tokens: tokenized document content

        Returns:
            None. Results are stored in the instance (df/idf and document vectors).

        Notes:
            An empty collection is allowed (N=0, empty df/idf).
        """
        self._doc_ids = []
        self._doc_tfidf = []
        self._df = {}
        self._idf = {}
        self._N = len(documents)

        # 1) compute df
        doc_tfs: List[Dict[str, float]] = []
        for doc_id, tokens in documents:
            self._doc_ids.append(doc_id)
            tf = tf_raw(tokens)
            doc_tfs.append(tf)
            for term in tf.keys():
                self._df[term] = self._df.get(term, 0) + 1

        # 2) compute idf
        # Note: when N=0, _df is empty, so this loop is safely skipped.
        for term, df in self._df.items():
            self._idf[term] = math.log10(self._N / df)

        # 3) compute doc tf-idf
        for tf in doc_tfs:
            vec: Dict[str, float] = {}
            for term, c in tf.items():
                vec[term] = c * self._idf.get(term, 0.0)
            self._doc_tfidf.append(vec)

    def vectorize_query(self, query_tokens: Sequence[str], normalize: bool = False) -> Dict[str, float]:
        """Convert query tokens into a TF-IDF vector using the index IDF.

        Terms not present in the index vocabulary (i.e., without IDF) are ignored.

        Args:
            query_tokens: Query tokens.
            normalize: If True, returns an L2-normalized TF-IDF vector.

        Returns:
            Query TF-IDF sparse vector as dict term -> weight.
            If no query term is in the vocabulary, returns {}.
        """
        q_tf = tf_raw(query_tokens)
        q_vec: Dict[str, float] = {}
        for term, c in q_tf.items():
            if term in self._idf:
                q_vec[term] = c * self._idf[term]
        return l2_normalize(q_vec) if normalize else q_vec

    def search(self, query_tokens: Sequence[str], k: int = 5, normalize: bool = True) -> List[SearchResult]:
        """Return top-k documents by cosine similarity to the query.

        Args:
            query_tokens: Query tokens.
            k: Number of results to return.
            normalize: If True, explicitly L2-normalizes both query and document
                TF-IDF vectors, and then computes cosine on unit vectors.
                This yields the same final score as cosine on raw TF-IDF, but
                makes intermediate "normalized tf-idf" weights explicit.

        Returns:
            List of `SearchResult` sorted by descending score.
            If the query is empty / out-of-vocabulary, scores will be 0.0.
        """
        if normalize:
            q_vec = self.vectorize_query(query_tokens, normalize=True)
            scored: List[SearchResult] = []
            for doc_id, d_vec in zip(self._doc_ids, self._doc_tfidf):
                d_norm = l2_normalize(d_vec)
                scored.append(SearchResult(doc_id=doc_id, score=cosine_sparse(q_vec, d_norm)))
        else:
            q_vec = self.vectorize_query(query_tokens, normalize=False)
            scored = [
                SearchResult(doc_id=doc_id, score=cosine_sparse(q_vec, d_vec))
                for doc_id, d_vec in zip(self._doc_ids, self._doc_tfidf)
            ]

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def doc_vector(self, doc_id: str, normalize: bool = False) -> Dict[str, float]:
        """Return a document TF-IDF vector by its document ID.

        Args:
            doc_id: Document ID used in `build()`.
            normalize: If True, returns an L2-normalized version.

        Returns:
            Sparse TF-IDF vector for the document.

        Raises:
            ValueError: If `doc_id` is not present in the index.
        """
        i = self._doc_ids.index(doc_id)
        v = dict(self._doc_tfidf[i])
        return l2_normalize(v) if normalize else v


def pretty_vec(vec: Dict[str, float]) -> str:
    """Pretty-print a sparse vector (sorted terms, 3 decimal places).

    Args:
        vec: Sparse vector as dict term -> weight.

    Returns:
        A formatted string like "{term: 0.123, ...}".
    """
    items = sorted(vec.items())
    return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in items) + "}"
