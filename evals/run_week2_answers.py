"""Frozen six-case pilot. Run from the project root; --live makes 12 Groq calls.
Default mode verifies local evidence/configuration only. Never rebuilds an index.
Answer and citation quality require human review, not regex-based grading.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import re
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def generation_config(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    prompt = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in n.targets))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'create')
    keywords = {k.arg: k.value for k in call.keywords}
    if set(keywords) != {'model', 'messages', 'stream'} or ast.literal_eval(keywords['stream']) is not True:
        raise ValueError('Generation request changed; review the comparison protocol first')
    return {'model': ast.literal_eval(keywords['model']), 'system_prompt': prompt,
            'stream': True, 'temperature': 'provider default; omitted',
            'max_tokens': 'provider default; omitted', 'seed': 'omitted'}


def validate(cases, snapshot, query_path):
    if sha(query_path) != cases['query_py_sha256']:
        raise ValueError('Current query.py changed since protocol preparation')
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    if manifest['snapshot_id'] != cases['snapshot_id'] or manifest['git_base_commit'] != cases['source_commit']:
        raise ValueError('Wrong source snapshot')
    for item in manifest['files']:
        if sha(snapshot / item['path']) != item['sha256']:
            raise ValueError('Frozen source changed: ' + item['path'])
    questions = cases['questions']
    if len(questions) != 6 or sum(q['answerable'] for q in questions) != 4 or len({q['id'] for q in questions}) != 6:
        raise ValueError('Expected four answerable and two unanswerable cases')
    for q in questions:
        for e in q['evidence']:
            lines = (snapshot / e['file_path']).read_text().splitlines(keepends=True)
            if not 1 <= e['start_line'] <= e['end_line'] <= len(lines):
                raise ValueError('Invalid evidence span')
            content = ''.join(lines[e['start_line'] - 1:e['end_line']])
            if hashlib.sha256(content.encode()).hexdigest() != e['text_sha256']:
                raise ValueError('Expected evidence changed')
    return generation_config(query_path)


def prepare_retrieval(cases, snapshot, candidate_path):
    # Reuse the same verified-record/model helpers as the completed experiment.
    import build_chunking_v2_index as builder
    import experimental_chunk_metadata as metadata
    import experimental_chunker_v2 as chunker
    from evaluate_semantic_reranking import RERANKER, PAIR_LIMIT, rank_scores, weight_digest
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb

    if sha(candidate_path) != cases['candidate_report_sha256']:
        raise ValueError('Candidate report changed')
    reference = json.loads(candidate_path.read_text())
    if sha(sys.modules['evaluate_semantic_reranking'].__file__) != reference['runner_sha256']:
        raise ValueError('Original reranking evaluator changed')
    if (reference['index_fingerprint'] != cases['index_fingerprint']
            or reference['candidate_depth'] != 20 or reference['final_k'] != 5):
        raise ValueError('Candidate configuration differs')
    matches = []
    for path in Path('evals/chroma_chunking_v2').glob('*/experiment.json'):
        run = json.loads(path.read_text())
        if run.get('index_fingerprint') == cases['index_fingerprint']:
            matches.append((path.parent, run))
    if len(matches) != 1:
        raise ValueError('Expected one existing matching experimental index; found ' + str(len(matches)))
    index_dir, run = matches[0]
    config = run['configuration']
    if (run.get('status') != 'complete' or builder.digest(config) != run['experiment_id']
            or builder.digest({'experiment_id': run['experiment_id'], 'vectors_sha256': run['vectors_sha256']}) != cases['index_fingerprint']
            or config['snapshot_id'] != cases['snapshot_id']):
        raise ValueError('Index manifest does not match the locked experiment')
    for name, module in [('splitter', chunker), ('metadata_builder', metadata), ('index_builder', builder)]:
        if builder.file_digest(module.__file__) != config['code_sha256'][name]:
            raise ValueError('Experimental helper changed: ' + name)
    print('Loading and verifying both models once...', flush=True)
    model = SentenceTransformer(config['model']['model_name'])
    snap, sources = chunker.load_source_snapshot(snapshot)
    records = metadata.build_chunk_records(config['repo_id'], snap, sources, model.tokenizer, config['token_budget'])
    if builder.digest(records) != config['records_sha256'] or builder.model_identity(model) != config['model']:
        raise ValueError('Embedding model or records differ')
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(config['collection'], embedding_function=None)
    if (collection.metadata or {}).get('experiment_id') != run['experiment_id']:
        raise ValueError('Collection identity differs')
    builder.verify_collection(collection, records, config['model']['embedding_dimension'], run['vectors_sha256'])
    reranker = CrossEncoder(RERANKER, max_length=PAIR_LIMIT)
    identity = {'model_name': RERANKER, 'weights_sha256': weight_digest(reranker), 'max_length': PAIR_LIMIT,
                'tokenizer_class': type(reranker.tokenizer).__name__,
                'vocabulary_sha256': builder.digest(reranker.tokenizer.get_vocab()),
                'device': str(getattr(reranker, 'device', 'unknown'))}
    if any(identity[k] != reference['reranker'][k] for k in identity if k != 'device'):
        raise ValueError('Reranker differs from candidate')
    expected = {r['chunk_id']: r for r in records}
    batches = []
    for q in cases['questions']:
        if chunker.count_tokens(q['question'], model.tokenizer) > model.max_seq_length:
            raise ValueError('Question would truncate')
        start = perf_counter()
        embedding = model.encode([q['question']], normalize_embeddings=config['normalize_embeddings_argument'],
                                 precision=config['encode_precision']).tolist()
        result = collection.query(query_embeddings=embedding, n_results=min(20, len(records)), include=['metadatas'])
        search_time = perf_counter() - start
        ids, metas = result['ids'][0], result['metadatas'][0]
        if (not ids or len(ids) != len(metas) or len(ids) != len(set(ids))
                or any(k not in expected or m != expected[k]['metadata'] for k, m in zip(ids, metas))):
            raise ValueError('Query results disagree with verified records')
        pairs = [(q['question'], expected[k]['embedding_input']) for k in ids]
        for question, document in pairs:
            if len(reranker.tokenizer(question, document, add_special_tokens=True, truncation=False)['input_ids']) > PAIR_LIMIT:
                raise ValueError('Reranker pair would truncate: ' + q['id'])
        start = perf_counter()
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        chosen = rank_scores(ids, scores)
        rerank_time = perf_counter() - start
        rankings = {'semantic5': ids[:5], 'reranked5': [ids[i] for i in chosen]}
        by_mode = {}
        for mode, keys in rankings.items():
            selected = []
            for number, key in enumerate(keys, 1):
                meta = expected[key]['metadata']
                lines = (snapshot / meta['file_path']).read_text().splitlines(keepends=True)
                a, b = meta['start_line'], meta['end_line']
                if not 1 <= a <= b <= len(lines):
                    raise ValueError('Invalid retrieved source span')
                selected.append({'source_number': number, 'chunk_id': key, 'metadata': meta,
                                 'text': ''.join(lines[a-1:b])})
            context = '\n\n'.join(f"[Source {s['source_number']}: {s['metadata']['file_path']}:{s['metadata']['start_line']}]\n{s['text']}" for s in selected)
            by_mode[mode] = {'sources': selected, 'context': context, 'search_seconds': search_time,
                             'rerank_seconds': rerank_time if mode == 'reranked5' else 0.0}
        batches.append({'question_id': q['id'], 'question': q['question'], 'modes': by_mode})
    return batches, {'experiment_id': run['experiment_id'], 'index_fingerprint': cases['index_fingerprint'],
                     'reranker': identity, 'configuration': config}


def generate(context, question, api_key, config):
    from groq import Groq
    answer, finish, usage, response_model = [], None, None, None
    started = perf_counter()
    first_token = None
    try:
        # Same request fields as current query.py. No answer key enters this call.
        with Groq(api_key=api_key) as client:
            stream = client.chat.completions.create(model=config['model'], messages=[
                {'role': 'system', 'content': config['system_prompt']},
                {'role': 'user', 'content': f'Chunk:\n{context}\n\nQuestion:\n{question}'}], stream=True)
            with stream:
                for event in stream:
                    raw = event.model_dump()
                    usage = raw.get('usage') or (raw.get('x_groq') or {}).get('usage') or usage
                    response_model = raw.get('model') or response_model
                    if event.choices:
                        choice = event.choices[0]
                        finish = choice.finish_reason or finish
                        token = choice.delta.content
                        if token:
                            if first_token is None:
                                first_token = perf_counter() - started
                            answer.append(token)
        return {'status': 'ok', 'answer': ''.join(answer), 'usage': usage,
                'response_model': response_model, 'finish_reason': finish,
                'generation_seconds': perf_counter() - started, 'time_to_first_token_seconds': first_token}
    except Exception as exc:
        # Keep useful diagnostics while removing the credential from the message.
        message = str(exc).replace(api_key, '[REDACTED]') if api_key else str(exc)
        return {'status': 'error', 'error_type': type(exc).__name__,
                'http_status': getattr(exc, 'status_code', None), 'error_message': message[:2000],
                'partial_answer': ''.join(answer),
                'generation_seconds': perf_counter() - started, 'usage': usage}


def review_markdown(report, cases):
    sections = ['# Week 2 answer review', '',
        'Pilot only. Source correctness, completeness, citation support and unsupported claims require manual review.',
        'Citation-number checks below do NOT establish support. Expected claims never enter generation prompts.', '']
    lookup = {q['id']: q for q in cases['questions']}
    for row in report['questions']:
        q = lookup[row['question_id']]
        sections += ['## ' + q['id'], '', q['question'], '', 'Expected claims:', '']
        sections += ['- ' + c for c in q['expected_claims']]
        for mode, result in row['modes'].items():
            sections += ['', '### ' + mode, '', 'Answer:', '', result.get('answer', result.get('partial_answer', '(not generated)')), '',
                         'Manual score: correctness 0/1/2; completeness 0/1/2; citation support 0/1/2 or N/A;',
                         'unsupported claims: list/count; abstention: appropriate/inappropriate; notes: ____', '',
                         'Retrieved sources:', '']
            for s in result['sources']:
                m = s['metadata']
                sections += [f"Source {s['source_number']}: {m['file_path']}:{m['start_line']}-{m['end_line']}", '',
                             '````text', s['text'], '````', '']
    return '\n'.join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Make the 12 generation calls')
    parser.add_argument('--model', help='Explicit generation model for BOTH methods; recorded as an override of the app model')
    parser.add_argument('--cases', type=Path, default=HERE / 'week2_answer_cases.json')
    parser.add_argument('--snapshot', type=Path, default=Path('evals/corpus_snapshots/50c2529de39ff786'))
    parser.add_argument('--candidate-report', type=Path, default=Path('evals/dev_reranking_d1z8s8sx.json'))
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    config = validate(cases, args.snapshot, Path('Backend/query.py'))
    config['app_model'] = config['model']
    config['model_override'] = args.model is not None
    if args.model is not None:
        config['model'] = args.model
    print('Validated six frozen questions, evidence spans, snapshot and generation configuration.', flush=True)
    print('Generation model for both methods:', config['model'], flush=True)
    if not args.live:
        print('No model loading or API calls. Add --live for the paired batch.')
        return
    from dotenv import load_dotenv
    import os
    load_dotenv(Path('Backend/.env'))
    load_dotenv(Path('.env'))
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError('GROQ_API_KEY is missing; use your existing local .env')
    # Check API model availability before loading local models. Listing makes no generation request.
    from groq import Groq
    with Groq(api_key=api_key, max_retries=0) as check_client:
        available = sorted(m.id for m in check_client.models.list().data)
    if config['model'] not in available:
        raise ValueError('Model not listed for this key: ' + config['model'] + '\nAvailable models: ' + ', '.join(available))
    print('Model listing check passed; generation access will be tested by the first answer call.', flush=True)
    # All retrieval checks finish before any generation request.
    rows, provenance = prepare_retrieval(cases, args.snapshot, args.candidate_report)
    run_dir = Path('evals/week2_answer_runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_dir.mkdir(parents=True, exist_ok=False)
    packages = {}
    for package in ['groq', 'sentence-transformers', 'transformers', 'torch', 'chromadb']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    report = {'purpose': cases['purpose'], 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'cases_sha256': sha(args.cases), 'runner_sha256': sha(__file__), 'query_py_sha256': sha('Backend/query.py'),
              'generation_config': config, 'provenance': provenance, 'packages': packages,
              'timing_scope': 'Search: encode+query; rerank: predict only; generation: client creation through stream end. Models and verification excluded. Sum is component time, not measured app end-to-end latency.',
              'limitations': 'Six scenario questions from a known development corpus; not independent holdout. Generation defaults omitted to match app; single stochastic sample per method. No automatic answer/citation quality grade. Costs null unless separately priced; missing usage is unknown, not zero.',
              'cost_usd': None, 'questions': rows}
    path = run_dir / 'answers.json'
    save(path, report)  # Freeze all actual retrievals before generation.
    for i, row in enumerate(rows):
        order = ['semantic5', 'reranked5'] if i % 2 == 0 else ['reranked5', 'semantic5']
        row['generation_order'] = order
        for mode in order:
            result = row['modes'][mode]
            result.update(generate(result['context'], row['question'], api_key, config))
            if result['status'] == 'ok':
                cited = [int(n) for n in re.findall(r'\(Source (\d+)\)', result['answer'])]
                result['citation_number_check_only'] = {'cited': cited, 'invalid': [n for n in cited if not 1 <= n <= len(result['sources'])]}
                result['exact_refusal'] = result['answer'].strip() == "I don't know."
                result['incomplete_generation'] = result['finish_reason'] != 'stop'
                result['component_seconds'] = result['search_seconds'] + result['rerank_seconds'] + result['generation_seconds']
            save(path, report)
            print(f"{row['question_id']} {mode}: {result['status']}", flush=True)
            if result['status'] == 'error':
                print(result['error_type'], result.get('http_status'), result.get('error_message'), flush=True)
            if result['status'] != 'ok':
                (run_dir / 'review.md').write_text(review_markdown(report, cases))
                raise RuntimeError('Stopped after API error; partial report saved at ' + str(path))
    (run_dir / 'review.md').write_text(review_markdown(report, cases))
    print('\nSaved:', path)
    print('Also saved review.md. Upload answers.json here for claim-by-claim scoring.')


if __name__ == '__main__':
    main()
