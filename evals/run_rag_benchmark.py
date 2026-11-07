"""Compare semantic, BM25 and unweighted RRF on a frozen, isolated RAG index.

Run from the project root:
    python evals/run_rag_benchmark.py

Reads only: evals/chroma_benchmark/ (collection: rag_backend_eval)
Writes only: evals/rag_benchmark_results.json, evals/rag_benchmark_report.md
No Groq/LLM calls; never modifies the ChromaDB collection.
"""

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))
from embed import MODEL_NAME  # noqa: E402

DATASET_PATH = ROOT / "evals" / "golden_set_rag.json"
DB_PATH = ROOT / "evals" / "chroma_benchmark"
COLLECTION_NAME = "rag_backend_eval"
RESULTS_PATH = ROOT / "evals" / "rag_benchmark_results.json"
REPORT_PATH = ROOT / "evals" / "rag_benchmark_report.md"
K = 5
CANDIDATE_K = 15
RRF_CONSTANT = 60
EXPECTED_CHUNK_COUNT = 32


def tokenize(text):
    """Simple lexical baseline; underscores in code identifiers are retained."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def retrieve_bm25(bm25, tokens, ids, candidate_k):
    scores = bm25.get_scores(tokenize(tokens))
    # Same positive-score-only policy as the earlier BM25 baseline.
    order = sorted(
        (i for i, score in enumerate(scores) if score > 0),
        key=lambda i: (-float(scores[i]), ids[i]),
    )[:candidate_k]
    return [ids[i] for i in order]


def rrf(semantic, lexical):
    scores = defaultdict(float)
    for ranking in (semantic, lexical):
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (RRF_CONSTANT + position)
    return sorted(scores, key=lambda key: (-scores[key], key))


def metrics(ranked_ids, relevant, k=K):
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Retrieval metrics require a nonempty relevance set")
    found = set(ranked_ids[:k]) & relevant
    reciprocal_rank = next(
        (1.0 / i for i, chunk_id in enumerate(ranked_ids[:k], start=1)
         if chunk_id in relevant),
        0.0,
    )
    return {
        f"precision@{k}": len(found) / k,
        f"recall@{k}": len(found) / len(relevant),
        f"mrr_contribution@{k}": reciprocal_rank,
        "top1_hit": int(bool(ranked_ids) and ranked_ids[0] in relevant),
    }


def avg(values):
    return sum(values) / len(values) if values else 0.0


def summarize(rows):
    methods = ("Semantic", "BM25", "Hybrid_RRF")
    positives = [row for row in rows if row["answerable"]]
    result = {}
    for method in methods:
        scored = [q["results"][method]["metrics"] for q in positives]
        result[method] = {
            "precision@5": avg([s["precision@5"] for s in scored]),
            "recall@5": avg([s["recall@5"] for s in scored]),
            "mrr@5": avg([s["mrr_contribution@5"] for s in scored]),
            "top1_accuracy": avg([s["top1_hit"] for s in scored]),
        }
    return result


def make_report(payload):
    summary = payload["summary"]
    questions = payload["questions"]
    lines = [
        "# Chat-with-Your-Repo — Retrieval Benchmark",
        "",
        "## Scope and methodology",
        "",
        f"- Corpus: {payload['corpus_chunks']} backend chunks in `{COLLECTION_NAME}`.",
        f"- Questions: {payload['answerable_questions']} answerable and "
        f"{payload['unanswerable_questions']} deliberately out of scope.",
        f"- Model: `{MODEL_NAME}`. Evaluation depth: top-{K}; candidate depth: {CANDIDATE_K}.",
        "- Gold labels are provisional chunk-level relevance judgments from the uploaded manifest, "
        "not independently adjudicated annotations.",
        "- The RRF configuration is unweighted with constant 60. "
        "BM25 uses a simple lowercase, identifier-preserving tokenizer.",
        "- Queries marked out of scope are **excluded from retrieval metrics**. "
        "The current retrievers cannot determine when to abstain; LLM answer abstention "
        "requires a separate experiment.",
        f"- Index fingerprint (SHA-256): `{payload['index_sha256']}`.",
        "",
        "## Scores on answerable questions",
        "",
        "| Method | Precision@5 | Recall@5 | MRR@5 | Top-1 accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("Semantic", "BM25", "Hybrid_RRF"):
        m = summary[method]
        lines.append(
            f"| {method} | {m['precision@5']:.3f} | {m['recall@5']:.3f} | "
            f"{m['mrr@5']:.3f} | {m['top1_accuracy']:.3f} |"
        )
    lines += [
        "",
        "Precision@5 uses five as its denominator for every question. "
        "Most questions have only one or two labelled relevant chunks, so low values "
        "can be a consequence of the annotation design rather than poor retrieval.",
        "",
        "## Questions needing attention",
        "",
        "| ID | Question | Relevant chunks | Semantic RR | BM25 RR | Hybrid RR |",
        "|---|---|---|---:|---:|---:|",
    ]
    hard = []
    for q in questions:
        if not q["answerable"]:
            continue
        rms = {name: q["results"][name]["metrics"]["mrr_contribution@5"]
               for name in ("Semantic", "BM25", "Hybrid_RRF")}
        if min(rms.values()) < 1.0:
            hard.append((q, rms))
    for q, rms in hard:
        safe_question = q["question"].replace("|", "\\|")
        lines.append(
            f"| {q['id']} | {safe_question} | "
            f"{', '.join(q['relevant_chunks'])} | "
            f"{rms['Semantic']:.2f} | {rms['BM25']:.2f} | {rms['Hybrid_RRF']:.2f} |"
        )
    if not hard:
        lines.append("| — | No ranked misses on this benchmark | — | — | — | — |")
    lines += [
        "",
        "## Out-of-scope questions (diagnostic only)",
        "",
    ]
    for q in questions:
        if q["answerable"]:
            continue
        lines.append(f"- **{q['id']}**: {q['question']}")
        lines.append(f"  - Why out of scope: {q['label_rationale']}")
        lines.append(
            f"  - Semantic top result: "
            f"`{(q['results']['Semantic']['ranking'] or ['none'])[0]}`. "
            "This is a nearest neighbour, **not proof of an answer**."
        )
    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Do not treat a perfect score as general AI answer accuracy: this only tests retrieval.",
        "- Historical `Backend/NOTES.md` may mention superseded behavior. "
        "Current executable code is authoritative for current-implementation questions.",
        "- `main.py`, frontend files and any unindexed extensions are outside this corpus.",
        "- Verify partial relevance and annotation completeness before publishing "
        "this benchmark as portfolio-level evidence.",
        "- Compare changes against a held-out benchmark instead of tuning RRF "
        "to these same questions.",
    ]
    return "\n".join(lines) + "\n"


def main():
    if not DB_PATH.is_dir():
        raise FileNotFoundError(f"Missing evaluation DB: {DB_PATH}")
    with DATASET_PATH.open(encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list) or not questions:
        raise ValueError("Expected a nonempty array in the golden set")
    ids_seen = set()
    for q in questions:
        if q["id"] in ids_seen:
            raise ValueError(f"Duplicate question ID {q['id']}")
        ids_seen.add(q["id"])
        if bool(q["relevant_chunks"]) != q["answerable"]:
            raise ValueError(f"Invalid answerability/relevance combination in {q['id']}")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    corpus = collection.get(include=["documents", "metadatas"])
    stored = sorted(zip(corpus["ids"], corpus["documents"]), key=lambda x: x[0])
    if len(stored) != EXPECTED_CHUNK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CHUNK_COUNT} chunks; found {len(stored)}. "
            "Verify the snapshot or update EXPECTED_CHUNK_COUNT explicitly."
        )
    ids = [item[0] for item in stored]
    documents = [item[1] or "" for item in stored]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in evaluation corpus")
    available = set(ids)
    for q in questions:
        unknown = set(q["relevant_chunks"]) - available
        if unknown:
            raise ValueError(f"{q['id']} references missing chunk IDs: {sorted(unknown)}")

    fingerprint = hashlib.sha256()
    for chunk_id, document in stored:
        fingerprint.update(json.dumps([chunk_id, document], ensure_ascii=False).encode("utf-8"))
        fingerprint.update(b"\n")

    bm25 = BM25Okapi([tokenize(doc) for doc in documents])
    model = SentenceTransformer(MODEL_NAME)
    print(f"Loaded {len(ids)} chunks and {len(questions)} questions")
    print("Encoding all questions once...")
    question_embeddings = model.encode([q["question"] for q in questions]).tolist()
    semantic_batch = collection.query(
        query_embeddings=question_embeddings,
        n_results=min(CANDIDATE_K, len(ids)),
        include=["metadatas"],
    )

    evaluated = []
    for index, q in enumerate(questions):
        semantic = semantic_batch["ids"][index]
        lexical = retrieve_bm25(bm25, q["question"], ids, CANDIDATE_K)
        hybrid = rrf(semantic, lexical)
        methods = {"Semantic": semantic, "BM25": lexical, "Hybrid_RRF": hybrid}
        record = dict(q)
        record["results"] = {}
        for method, ranking in methods.items():
            values = {
                "ranking": ranking[:K],
                "metrics": metrics(ranking, q["relevant_chunks"])
                if q["answerable"] else None,
            }
            record["results"][method] = values
        evaluated.append(record)
        if q["answerable"]:
            ranks = {method: (r["metrics"]["mrr_contribution@5"])
                     for method, r in record["results"].items()}
            print(f"{q['id']} | Semantic RR={ranks['Semantic']:.2f} "
                  f"BM25 RR={ranks['BM25']:.2f} "
                  f"Hybrid RR={ranks['Hybrid_RRF']:.2f}")
        else:
            print(f"{q['id']} | OUT OF SCOPE (ranking displayed, unscored)")

    payload = {
        "corpus_chunks": len(ids),
        "index_sha256": fingerprint.hexdigest(),
        "collection": COLLECTION_NAME,
        "embedding_model": MODEL_NAME,
        "answerable_questions": sum(q["answerable"] for q in questions),
        "unanswerable_questions": sum(not q["answerable"] for q in questions),
        "summary": summarize(evaluated),
        "questions": evaluated,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(make_report(payload), encoding="utf-8")

    print("\n=== RAG BACKEND RETRIEVAL COMPARISON ===")
    print(f"Scored answerable queries: {payload['answerable_questions']}")
    print(f"Excluded out-of-scope queries: {payload['unanswerable_questions']}")
    print(f"{'Method':<13} {'P@5':>7} {'R@5':>7} {'MRR@5':>7} {'Top1':>7}")
    for method, m in payload["summary"].items():
        print(f"{method:<13} {m['precision@5']:>7.3f} {m['recall@5']:>7.3f} "
              f"{m['mrr@5']:>7.3f} {m['top1_accuracy']:>7.3f}")
    print(f"Saved detailed results: {RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
