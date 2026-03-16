import json
from typing import List, Tuple

from tfidf_search import TfidfIndex, pretty_vec, tokenize_whitespace_lower


def example_1() -> None:
    """Run and print the fully-worked TF-IDF + cosine similarity Example 1.

    This reproduces the Czech assignment example with documents about cities and
    the query "krásné město".

    Printed intermediate results:
        - N, DF, IDF
        - Query tokens and TF-IDF weights
        - Per-document TF-IDF weights
        - Cosine similarity scores for all documents

    Returns:
        None. Results are printed to stdout.
    """
    print("=== Příklad 1 ===")
    docs = {
        "d1": "Plzeň je krásné město a je to krásné místo",
        "d2": "Ostrava je ošklivé místo",
        "d3": "Praha je také krásné město Plzeň je hezčí",
    }
    q1 = "krásné město"

    documents: List[Tuple[str, List[str]]] = [(k, tokenize_whitespace_lower(v)) for k, v in docs.items()]
    idx = TfidfIndex()
    idx.build(documents)

    print(f"N = {idx.N}")
    print("DF:", idx.df)
    print("IDF:", {k: round(v, 3) for k, v in idx.idf.items()})

    q_tokens = tokenize_whitespace_lower(q1)
    q_vec = idx.vectorize_query(q_tokens, normalize=False)

    print("q1 tokens:", q_tokens)
    print("q1 tf-idf:", pretty_vec(q_vec))

    for doc_id in docs.keys():
        d_vec = idx.doc_vector(doc_id, normalize=False)
        print(f"{doc_id} tf-idf:", pretty_vec(d_vec))

    results = idx.search(q_tokens, k=3, normalize=True)
    for r in results:
        print(f"cos(q1, {r.doc_id}) = {r.score:.3f}")


def example_2() -> None:
    """Run and print the fully-worked TF-IDF + cosine similarity Example 2.

    This reproduces the English assignment example about "tropical fish" and
    computes results for two queries:
        - q1 = "tropical fish sea"
        - q2 = "tropical fish"

    Printed intermediate results:
        - N, full DF and IDF tables (sorted by term)
        - Query TF-IDF (raw) and query TF-IDF after L2 normalization
        - Cosine similarities for each query against all documents

    Returns:
        None. Results are printed to stdout.
    """
    print("\n=== Příklad 2 ===")
    docs = {
        "d1": "tropical fish include fish found in tropical enviroments",
        "d2": "fish live in a sea",
        "d3": "tropical fish are popular aquarium fish",
        "d4": "fish also live in Czechia",
        "d5": "Czechia is a country",
    }
    q1 = "tropical fish sea"
    q2 = "tropical fish"

    documents: List[Tuple[str, List[str]]] = [(k, tokenize_whitespace_lower(v)) for k, v in docs.items()]
    idx = TfidfIndex()
    idx.build(documents)

    print(f"N = {idx.N}")

    # Full DF/IDF tables for debugging/verification (sorted by term)
    print("DF", dict(sorted(idx.df.items())))
    print("IDF", {t: round(v, 3) for t, v in sorted(idx.idf.items())})

    for q in [q1, q2]:
        print(f"\nQuery: {q}")
        q_tokens = tokenize_whitespace_lower(q)
        q_vec = idx.vectorize_query(q_tokens, normalize=False)
        q_vec_normed = idx.vectorize_query(q_tokens, normalize=True)

        print("q tokens:", q_tokens)
        print("q tf-idf:", pretty_vec(q_vec))
        print("q tf-idf normalized:", pretty_vec(q_vec_normed))

        results = idx.search(q_tokens, k=5, normalize=True)
        for r in results:
            print(f"cos({q}, {r.doc_id}) = {r.score:.3f}")


def wowhead_search() -> None:
    """Build a TF-IDF index over the local Wowhead dataset and run sample queries.

    Expected input:
        A local JSONL file named "wowhead_articles.jsonl" where each line is a JSON
        object containing at least title/content and usually url.

    Processing:
        - Uses `WowheadPreprocessor` to clean and tokenize title + content.
        - Builds a `TfidfIndex` over the tokenized documents.
        - Executes a few hard-coded queries and prints top-5 results.

    Returns:
        None. Results are printed to stdout.

    Raises:
        FileNotFoundError: If "wowhead_articles.jsonl" is not present.
    """
    print("\n=== Wowhead ===")
    # Lazy import so this file works even if some NLP deps aren't installed.
    from preprocessing import WowheadPreprocessor

    input_file = "wowhead_articles.jsonl"
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    pre = WowheadPreprocessor(use_lemmatization=True, remove_diacritics=True)

    documents: List[Tuple[str, List[str]]] = []
    for i, rec in enumerate(data):
        doc_id = rec.get("url") or f"doc_{i}"
        text = " ".join(filter(None, [rec.get("title", ""), rec.get("content", "")]))
        cleaned = pre.clean_text(text)
        tokens = cleaned.split()
        if tokens:
            documents.append((doc_id, tokens))

    idx = TfidfIndex()
    idx.build(documents)
    print(f"Naindexováno dokumentů: {idx.N}")
    print(f"Velikost slovníku (unikátní termy): {len(idx.df)}")

    queries = [
        "patch content",
        "player housing",
        "raid boss",
        "prop hunt",
    ]

    for q in queries:
        q_tokens = pre.clean_text(q).split()
        results = idx.search(q_tokens, k=5)
        print(f"\nDotaz: {q} -> top {len(results)}")
        for r in results:
            print(f"  {r.score:.3f}  {r.doc_id}")


def main() -> None:
    """Entry point: run both homework examples and the Wowhead search demo."""
    example_1()
    example_2()
    wowhead_search()


if __name__ == "__main__":
    main()
