import json
from typing import List, Tuple

from boolean_search import BooleanIndex


def wowhead_boolean_search() -> None:
    """Build a boolean inverted index over Wowhead and run a few example queries."""
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
        tokens = pre.clean_text(text).split()
        if tokens:
            documents.append((doc_id, tokens))

    idx = BooleanIndex()
    idx.add_documents(documents)
    print(f"Indexed documents: {len(idx.all_docs)}")

    # IMPORTANT: query terms must be normalized the same way as documents.
    # We keep boolean operators and parentheses, but normalize the rest via pre.clean_text.
    def normalize_boolean_query(q: str) -> str:
        parts = q.replace("(", " ( ").replace(")", " ) ").split()
        out_parts: List[str] = []
        for p in parts:
            up = p.upper()
            if up in {"AND", "OR", "NOT", "(", ")"}:
                out_parts.append(up if up in {"AND", "OR", "NOT"} else p)
            else:
                cleaned = pre.clean_text(p)
                # clean_text can turn a token into multiple tokens (rare here), keep the first if so
                if cleaned:
                    out_parts.append(cleaned.split()[0])
        return " ".join(out_parts)

    queries = [
        "patch AND content",
        "raid AND boss",
        "housing OR (player AND housing)",
        "NOT pvp AND (raid OR dungeon)",
    ]

    for q in queries:
        nq = normalize_boolean_query(q)
        hits = idx.evaluate(nq)
        # show top few doc_ids (unordered set -> sort for stable output)
        shown = sorted(list(hits))[:10]
        print(f"\nQuery: {q}")
        print(f"Normalized: {nq}")
        print(f"Hits: {len(hits)}")
        for doc_id in shown:
            print(f"  {doc_id}")


def main() -> None:
    wowhead_boolean_search()


if __name__ == "__main__":
    main()

