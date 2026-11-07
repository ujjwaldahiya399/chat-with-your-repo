# Week 2 answer-quality review and development decision

Reviewed November 5, 2026 (America/Vancouver).

## Decision

Do not promote the experimental reranked pipeline into the app on this evidence. Keep the current application retrieval baseline in place and keep semantic20 → cross-encoder → five as an experimental candidate. Close the bounded Week 2 retrieval investigation; do not keep tuning this six-question pilot. A negative or mixed validation result is a valid engineering conclusion.

The earlier development result remains valid: prescribed evidence coverage improved from 31/70 to 41/70. This generation pilot does not establish better answers. On four fresh answerable scenarios, semantic-five produced two fully correct and complete answers, reranked-five produced one. Both methods correctly refused both unanswerable questions. These small counts are observations, not general accuracy estimates.

The pilot ran with Groq `openai/gpt-oss-120b`, explicitly overriding the app's `llama-3.3-70b-versatile` after Groq rejected that model with HTTP 404. Both retrieval modes used the same generation model and system prompt. This does not validate the app's original model or establish that the deployed application has been repaired.

## Evidence and provenance

- Uploaded input: `answers.json`, from `evals/week2_answer_runs/20261105T070646_074364Z/answers.json`.
- Snapshot: `50c2529de39ff78609626aad1fcbe0b8cebd49c4c552175b16041c8f3c6a7674`.
- Source commit: `5d28da6d4c499ddb6b887965e25d52a080278bc4`.
- Experimental index fingerprint: `8a4a5fbf5c087c0c4caababd33ef3242be3b5c409d3ad3b1f01226f7d3eb2767`.
- Case-file and runner hashes match the prepared files.
- All six question IDs/texts match the frozen case file; all twelve calls completed with status `ok` and finish reason `stop`.
- All 60 supplied source passages match the corresponding exact line ranges in the frozen source snapshot.
- Labels and expected claims were not sent to generation.

This is an assistant-reviewed confirmation pilot from a known development corpus. It is not an independent holdout or an external human annotation study. The judgments below are manual evaluations against implementation and cited text, not automatic test results.

## Scoring

Correctness: 0 = no answer or wrong central conclusion; 1 = partly correct with a material error; 2 = all material claims correct.

Completeness: 0 = none of the three expected claim groups; 1 = some; 2 = all. A claim that is stated incorrectly does not count as covered.

Citation support: 0 = no substantive assertions reliably supported; 1 = some material assertions supported, with gaps or mismatches; 2 = all material assertions supported by the cited retrieved passages. N/A = no substantive claims because the response is a refusal. Valid-looking citation numbers alone do not establish support.

| Case | Mode | Correctness /2 | Completeness /2 | Citation support /2 | Outcome |
|---|---|---:|---:|---:|---|
| A01 branch replacement | semantic5 | 2 | 2 | 1 | Correct conclusion, incomplete direct action evidence |
| A01 branch replacement | reranked5 | 0 | 1 | 1 | Wrong deletion interpretation and central conclusion |
| A02 mixed ingestion failures | semantic5 | 0 | 0 | N/A | Exact refusal; final return absent from retrieved context |
| A02 mixed ingestion failures | reranked5 | 0 | 0 | N/A | Exact refusal; final return absent from retrieved context |
| A03 empty Python/Markdown | semantic5 | 1 | 1 | 1 | Wrong Python return shape; rejection behavior correct |
| A03 empty Python/Markdown | reranked5 | 1 | 1 | 0 | Wrong Markdown return shape; citations do not support embedding behavior |
| A04 whitespace URL | semantic5 | 2 | 2 | 1 | Correct behavior, canonicalization citation mismatch |
| A04 whitespace URL | reranked5 | 2 | 2 | 2 | Correct and supported |
| U01 deployed rate limit | semantic5 | N/A | N/A | N/A | Appropriate exact refusal; no unsupported claims |
| U01 deployed rate limit | reranked5 | N/A | N/A | N/A | Appropriate exact refusal; no unsupported claims |
| U02 production p95 | semantic5 | N/A | N/A | N/A | Appropriate exact refusal; no unsupported claims |
| U02 production p95 | reranked5 | N/A | N/A | N/A | Appropriate exact refusal; no unsupported claims |

Answerable correctness totals: semantic 5/8, reranked 3/8. Completeness totals: semantic 5/8, reranked 4/8. These are rubric points, not accuracy percentages. Citation scores are shown per answer and are not averaged across differing refusal denominators.

Fully correct and complete answerable responses: semantic 2/4; reranked 1/4. Appropriate exact refusals on corpus-unanswerable questions: 2/2 for each method.

## Claim-by-claim findings

### A01: can two branches coexist for the same repository/file?

Expected: IDs contain the branch, but deletion matches only `file_path` and `repo_url`. After a successful second indexing, the first branch's records for that file are removed. The decisive full implementation is `Backend/embed.py:64–80`.

Semantic: correctly explains distinct branch IDs and the branch-free delete filter, and reaches the correct replacement conclusion. Source 3 supports IDs; Source 1 supports the filter. Source 1 ends at line 78, immediately before the actual `collection.delete` and `upsert` at 79–80. Therefore direct execution/action evidence is incomplete, even though the filter comments and variable support the interpretation. Treat the action-order claim as a retrieved-evidence gap, not a false implementation claim.

Reranked: correctly describes distinct IDs, then incorrectly treats `repo_url` as implicitly branch-specific and concludes that main/feature records coexist. Source 2 explicitly shows a filter with no branch condition; it does not support this claim. Source 3 supports the IDs but distinct IDs cannot protect against a broad delete operation.

Unsupported claim families in reranked answer: (1) repo_url/filter implicitly scopes deletion by branch; (2) the earlier branch's records remain. Repeated paraphrases are not counted as additional families. Semantic has one direct-evidence gap concerning the actual deletion/upsert execution, noted above.

Both modes retrieved the same core ID and filter passages, among three shared chunks, but used different surrounding context and ordering. This is an observed answer regression. With a single stochastic output per mode, it does not isolate reranking as the causal source of the reasoning error.

### A02: two file failures with twelve existing chunks

Expected: both exceptions are caught; walking continues; the return is `(0, 12)` under the stated assumptions. Implementation: `Backend/embed_repo.py:24–63`.

Both outputs are exactly `I don't know.` No unsupported assertions. Both retrieved initialization and exception handling, but neither retrieved the final return at line 63. Thus both miss a corpus-answerable task, while refusing is appropriate for the missing part of the actual retrieved context. Do not count this as a successful answer or assume it proves the generator could not reason over complete context.

### A03: zero-byte Python versus Markdown

Expected: `get_chunks(empty.py)` yields `[(1, "")]`; `get_chunks(empty.md)` yields `[]`. Both are rejected by `embed_file`'s empty-content guard before model creation/database access. Implementation: `Backend/ingest.py:15–34,68–94`; `Backend/embed.py:51–60`.

Semantic: incorrectly says both files produce an empty iterable/no chunks. That is false for Python, and the cited embed_file call site does not define the chunker's return shape. It correctly concludes both are rejected and no embedding/write happens; the actual guard is retrieved in Source 1. The claim about model/write ordering has incomplete direct retrieved implementation evidence, since the selected source ends at the guard. Unsupported claim families: (1) Python returns no chunks, (2) the precise model/write execution ordering beyond the visible guard is asserted without the full execution excerpt. The second family is factually correct against the complete snapshot.

Reranked: gets Python's nonempty empty-content list broadly right, but incorrectly extends it to Markdown. The cited Source 1 is a truncated historical note about an ask.py empty-file guard before query(), not an implementation of embed_file's embedding guard. The answer wrongly uses that note to establish no model loading/no Chroma writes. Source 5 provides dispatch only, not the chunker bodies. Unsupported claim families: (1) Markdown returns a nonempty list with an empty chunk, (2) the cited note establishes embed_file's model/write rejection path. The second family reaches the correct final behavior using inadequate supporting evidence.

This case exposes both missing implementation context and a misleading historical note. It does not justify rewriting frozen notes or re-indexing during this pilot.

### A04: whitespace in a repository URL

Expected: regex matching uses `url.strip()`; git receives the original `url`; after a successful clone the returned canonical repo_url is `https://github.com/acme/widget`. No claim is made that git accepts the original spaced string. Implementation: `Backend/embed_repo.py:73–100`.

Both modes correctly describe all three transformations. Neither claims actual clone success for the spaced input.

Semantic: Source 4 supports stripping/regex, Source 1 supports passing the original string and returning repo_url. However its final canonicalization claim cites Sources 2 and 4, while the actual constructor is in retrieved Source 3 (`74–77`), which it does not cite. One unsupported/misattributed citation family: exact canonical URL construction. The claim itself is correct and its evidence is available in the retrieved context.

Reranked: Source 1 supports validation; grouped Sources 2 and 4 support original-argument flow; Source 5 supports canonical construction. The earlier Source 4 also includes the clone return. Citation support is complete at the material-claim level. No unsupported claims.

Several citations use narrow Unicode spaces, grouped source numbers or occur inside code blocks. The runner's narrow `(Source N)` regex misses some. The manual review recognized the references; the format check must not be treated as a factual-support score.

### U01/U02: unavailable deployment facts

All four outputs are exactly `I don't know.` after whitespace trimming. The snapshot cannot establish a deployed per-user rate-limit number or production p95 latency for the preceding day. No fabricated configuration or telemetry is supplied. These four refusals pass both semantic abstention and the exact-format requirement for these two cases only.

## Usage and local timing

| Measurement, six questions per method | semantic5 | reranked5 |
|---|---:|---:|
| Provider prompt tokens, total | 5,612 | 5,793 |
| Provider completion tokens, total | 2,655 | 2,280 |
| Median query encode + search | 0.009 s | 0.009 s, shared candidate search |
| Median added reranker prediction | — | 0.056 s |
| Median generation time, client creation through stream end | 1.120 s | 1.097 s |
| Median sum of measured components per question | 1.130 s | 1.198 s |

Completion token counts include the provider's recorded reasoning tokens. Lower completion totals do not establish better quality or a stable cost advantage. No current dollar rates have been applied; cost remains unpriced. The 12 calls total 11,405 prompt and 4,935 completion tokens.

These timings exclude model loading, verification and other application work. Component sums are not measured app end-to-end latency, a production SLA or p95. The single small batch does not establish a stable latency ranking. The prior development run's 49 ms median reranker prediction and this pilot's 56 ms are different observations.

## Week 2 retrospective

What worked: token-aware source representation; reproducible metadata/index identities; evidence-level development comparison; candidate-capacity diagnosis; batch evaluation; explicit generation-model migration after an API access failure; separate factual/citation/abstention review.

What failed or stayed uncertain: reranking's development coverage gain did not establish an answer-quality win; two answerable scenarios lacked key selected implementation passages; the model made a branch-deletion reasoning error despite seeing the filter; stale documentation was overgeneralized; some correct facts had misassigned citations. The narrow citation regex could not assess real support. Single-sample provider defaults do not isolate causal effects.

A packaging mistake placed runner instructions in the repository root README when the ZIP was extracted. The tracked README can be restored from Git, but any prior unstaged README edits are not recoverable from the shown diff alone. Preserve the runner instructions under evals before restoring the worktree README from Git's index. Do not reset staged changes or other files.

## Next actions and honest completion state

Completed: bounded six-question answer-quality pilot, source integrity checks, manual scoring, development decision, and this evaluation/retrospective note.

Still open:

- Recover the project README and install this review under `docs/`; neither action has been performed on the learner's Mac by the assistant.
- Complete a short independent Python/RRF exercise and explain the result without treating assistant scripts as independent coding evidence.
- Confirm outstanding Week 1 test/validator and Git closeout status. Do not rebuild any index merely for closeout.
- Record the walkthrough and actual applications/follow-ups; submissions/recordings remain unconfirmed.
- Inspect existing QA Copilot code before drafting Week 3 changes. No v2 QA Copilot implementation is claimed.
- App generation still names the rejected Llama model; the successful evaluation override did not alter app code or deployment. Repair this separately when inspecting app integration.

Defer unless separately justified: further retrieval tuning on these questions, deep hybrid search, context expansion, broad multi-repository evaluation, tracing integration, and new reranker deployment. These remain open limitations, not completed features and not reasons to keep repeating Week 2 experiments.

## Safe local documentation recovery

First exit Git's pager with `q` if it is still open. From the project root:

```bash
cp README.md evals/WEEK2_VALIDATION_RUNNER_NOTES.md
git restore --worktree -- README.md
git --no-pager diff -- README.md
```

This preserves the overwritten runner instructions and restores README from Git's current index, leaving staged content intact. A clean worktree diff is expected afterward. This recovers the tracked/index version, not any unknown pre-existing unstaged edits.

After downloading this report, copy it into the project without touching root README:

```bash
mkdir -p docs
cp "$HOME/Downloads/WEEK2_ANSWER_QUALITY_REVIEW.md" docs/WEEK2_ANSWER_QUALITY_REVIEW.md
```
