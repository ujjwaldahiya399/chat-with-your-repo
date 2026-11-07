"""Measure warm retrieval-only latency for the frozen 32-chunk benchmark.
Run at project root: python evals/measure_latency.py
This measures embedding+Chroma query; it excludes Groq generation/network latency.
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Backend'))
from embed import MODEL_NAME
import chromadb
from sentence_transformers import SentenceTransformer


def percentile(values, fraction):
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValueError('No observations')
    index = (len(sorted_values)-1) * fraction
    lo = int(index)
    hi = min(lo+1, len(sorted_values)-1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index-lo)


def main():
    questions = json.loads((ROOT/'evals'/'golden_set_rag.json').read_text())
    queries = [q['question'] for q in questions if q['answerable']]
    if len(queries) != 23:
        raise ValueError('Expected 23 frozen answerable questions')
    client = chromadb.PersistentClient(path=str(ROOT/'evals'/'chroma_benchmark'))
    collection = client.get_collection('rag_backend_eval')
    if collection.count() != 32:
        raise ValueError(f'Expected 32 chunks, found {collection.count()}')
    model = SentenceTransformer(MODEL_NAME)
    # Warm both the model and vector index; exclude model/download initialization.
    for _ in range(3):
        v = model.encode([queries[0]]).tolist()
        collection.query(query_embeddings=v, n_results=5)
    rows=[]
    for _ in range(5):
        for question in queries:
            t0 = time.perf_counter()
            v = model.encode([question]).tolist()
            t1 = time.perf_counter()
            collection.query(query_embeddings=v, n_results=5)
            t2 = time.perf_counter()
            rows.append({'embedding_ms':(t1-t0)*1000,'search_ms':(t2-t1)*1000,'total_ms':(t2-t0)*1000})
    print('=== WARM RETRIEVAL-ONLY LATENCY ===')
    print(f'Queries: {len(queries)} | repetitions: 5 | samples: {len(rows)} | top_k: 5')
    summary={}
    for key in ['embedding_ms','search_ms','total_ms']:
        values=[r[key] for r in rows]
        summary[key]={'p50':round(percentile(values,0.50),2),'p95':round(percentile(values,0.95),2),'mean':round(statistics.mean(values),2)}
        print(f'{key}: p50={summary[key]["p50"]:.2f} ms | p95={summary[key]["p95"]:.2f} ms')
    print('Note: warm local retrieval only, no LLM generation, request transport, or network.')
    output=ROOT/'evals'/'week1_latency_results.json'
    output.write_text(json.dumps({'method':'warm local retrieval','samples':len(rows),'top_k':5,'summary_ms':summary},indent=2)+'\n')
    print(f'Wrote {output}')

if __name__=='__main__':
    main()
