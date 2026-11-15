# Micrograd Claim-Faithfulness Pilot

## Result

The manual claim-level review found 42 of 43 factual claims supported by their retrieved and cited sources. One F03 claim was contradicted because `_prev` stores one distinct predecessor for `x + x`, although the generated answer said the set contains two operands. The two unanswerable questions both produced the required exact refusal, `I don't know.`

| Metric | Result |
|---|---:|
| Supported-claim rate | 42/43 (97.7%) |
| Contradicted claims | 1 |
| Claims not in retrieved sources | 0 |
| Citation coverage | 42/43 (97.7%) |
| Fully faithful answerable responses | 7/8 (87.5%) |
| Correct exact abstentions | 2/2 (100%) |
| False refusals on answerable questions | 0/8 |
| Invalid citation numbers | 0 |
| Correct question-level outcomes | 9/10 (90%) |

## What was measured

Each answer was split into atomic factual claims. Every claim was checked only against the five retrieved sources stored with that question. A claim was labelled:

- `SUPPORTED` when a cited source directly supported it or it followed through simple deterministic arithmetic from the cited code.
- `CONTRADICTED` when the retrieved evidence conflicted with it.
- `NOT_IN_SOURCE` when the retrieved evidence did not establish it.

Citation coverage required each factual claim to have at least one answer citation pointing to a retrieved source that supported that claim.

## Question-level outcome

| Question | Claims | Supported | Outcome |
|---|---:|---:|---|
| F01 | 5 | 5 | Fully faithful |
| F02 | 5 | 5 | Fully faithful |
| F03 | 5 | 4 | Not fully faithful |
| F04 | 6 | 6 | Fully faithful |
| F05 | 4 | 4 | Fully faithful |
| F06 | 6 | 6 | Fully faithful |
| F07 | 6 | 6 | Fully faithful |
| F08 | 6 | 6 | Fully faithful |
| U01 | 0 | N/A | Correct exact refusal |
| U02 | 0 | N/A | Correct exact refusal |

## Interpretation

The current retrieval-plus-generation path produced seven fully faithful answerable responses and two correct abstentions on this frozen micrograd pilot. F03's final gradient was correct, but its explanation misstated the number of distinct predecessors stored in `_prev` for `x + x`.

The result does **not** establish production accuracy. The evaluation contains only ten questions, was designed from a known development corpus, and used one generation sample per question. It is evidence that the evaluation machinery and current behavior work on this pilot, not proof that hallucinations are solved.

## Provenance

- Repository: `https://github.com/karpathy/micrograd`
- Frozen commit: `7bc720e951fe422b8f8814aa5aa1b64121d26b4c`
- Snapshot ID: `2d319a78dffd90aa6fa26bf69d8c92a258b3ed4179b6d2129e8fbde280c55541`
- Generation model: `openai/gpt-oss-120b`
- Run timestamp: `2026-11-15T07:02:31.155576+00:00`
- Review method: manual claim-by-claim annotation against only the retrieved sources stored in the run

## Recommended next step

Keep this pilot as the first frozen baseline. Next, add a second independently authored holdout set containing harder multi-source questions, plausible distractors, and unsupported questions whose wording closely resembles facts that do exist in the corpus. That will test how well this pilot behavior generalizes instead of merely reflecting a well-aligned development set.
