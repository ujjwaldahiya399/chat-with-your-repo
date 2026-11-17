# Micrograd AST-expanded claim-faithfulness experiment

This post-hoc retrieval experiment expanded semantic results through direct Python AST relationships while keeping the final context at five chunks. Required-evidence coverage increased from 6/8 to 8/8 answerable questions. Manual review found higher claim support, citation coverage, and correct question outcomes than the original frozen holdout.

The original holdout remains the unbiased result. These graph-expanded results describe one stochastic generation after the retrieval strategy was selected by inspecting holdout failures.

| Metric | Original holdout | AST-expanded run |
|---|---:|---:|
| Required-evidence retrieval | 6/8 (75.0%) | 8/8 (100.0%) |
| Supported factual claims | 52/58 (89.7%) | 57/59 (96.6%) |
| Claims absent from retrieved evidence | 5 | 1 |
| Contradicted claims | 1 | 1 |
| Citation-covered claims | 46/58 (79.3%) | 55/59 (93.2%) |
| Fully faithful answerable responses | 4/8 (50.0%) | 6/8 (75.0%) |
| Correct exact refusals | 4/4 (100.0%) | 4/4 (100.0%) |
| False refusals on answerable questions | 0 | 0 |
| Invalid citation numbers | 0 | 0 |
| Correct question outcomes | 8/12 (66.7%) | 10/12 (83.3%) |

Atomic-claim totals differ because the generated wording changed. Question-level faithfulness is the primary before-and-after comparison.

## Question-level findings

| Question | Evidence | Supported claims | Citation coverage | Finding |
|---|---|---:|---:|---|
| H01 | Full | 6/6 | 6/6 | Fully faithful. The answer correctly limits its dependency claim to package declarations. |
| H02 | Full | 11/11 | 11/11 | All claims are supported, but the answer omits the explicit use of `zip` and the addition closure's updates to both operands. |
| H03 | Full | 7/7 | 7/7 | Fully faithful. Both length-mismatch cases and the absence of validation are explained. |
| H04 | Full | 7/7 | 5/7 | Fully faithful. Two exact `Value` type claims are supported by retrieved sources but lack supporting citations. |
| H05 | Full | 6/6 | 6/6 | Fully faithful. The complete `MLP -> Layer -> Neuron -> weights and bias` path is established. |
| H06 | Full | 8/10 | 8/10 | Required values are correct. Repeated calls add 3 rather than repeatedly doubling, and the training-loop convention is absent from the sources. |
| H07 | Full | 6/6 | 6/6 | Fully faithful. Exponent constraints, predecessor storage, and base-only differentiation are correct. |
| H08 | Full | 6/6 | 6/6 | Fully faithful. Initialization, parameter order, and exact representation are correct. |
| HU01 | N/A | N/A | N/A | Exact refusal. No seed or sampled weights are present. |
| HU02 | N/A | N/A | N/A | Exact refusal. PyTorch is mentioned without a version. |
| HU03 | N/A | N/A | N/A | Exact refusal after context expansion. Exact training settings remain absent. |
| HU04 | N/A | N/A | N/A | Exact refusal. The retrieved sources contain no LICENSE text. |

## Retrieval repair findings

### H02

The graph-expanded context supplied the missing `Value` implementation. Unsupported claims fell from three to zero, and citation coverage increased from 4/11 to 11/11. The response still omitted two expected details even though the evidence was available. This is an answer-completeness issue rather than a retrieval failure.

### H05

The graph-expanded context supplied `Layer.parameters` and `Neuron.parameters`. The response then explained the full parameter chain and correctly established that `zero_grad` reaches every weight and bias. H05 changed from incomplete and partially unsupported to fully faithful.

## Other answer behavior

- H01 improved through different generation wording even though its required evidence was already present.
- H03, H04, H07, and H08 remained fully faithful.
- H06 answered every requested part correctly but introduced a false generalization that gradients double on every call. The implementation produces 3, 6, 9, 12, and so on. It also added a training-loop convention that the five sources do not state.
- HU01 and HU03 received changed contexts and still returned the exact required refusal. HU02 and HU04 also refused exactly.

## Integration decision

The retrieval results justify implementing the smallest reusable AST relationship-expansion layer in the application path. H02 and H05 improved, previously fully faithful answerable cases did not regress, and all four unanswerable cases continued to refuse correctly.

Integration should include regression tests for:

- H02 evidence coverage across `Neuron` and `Value`
- H05 parameter traversal across `MLP`, `Layer`, and `Neuron`
- the four unanswerable cases, especially HU01 and HU03
- preservation of the five-chunk final context budget
- deterministic relationship selection with at most two hops

Prompt or answer-validation work should be tracked separately for H02 completeness, H04 citation precision, and H06 overgeneralization.

## Method and scope

Distinct factual propositions were scored against only the five sources stored with each question. Expected claims served as a completeness rubric. A response was fully faithful when it addressed the required parts and every factual claim was supported. Citation coverage was measured independently. Exact refusals contributed zero factual claims.

This was a post-hoc experiment on the frozen holdout. It recognizes direct `ast.Name` relationships to indexed top-level Python definitions, follows at most two hops, and can displace lower-ranked semantic chunks. The run used one stochastic generation per question and does not estimate production-scale latency.

## Provenance

- Frozen repository: `https://github.com/karpathy/micrograd`
- Frozen commit: `7bc720e951fe422b8f8814aa5aa1b64121d26b4c`
- Snapshot ID: `2d319a78dffd90aa6fa26bf69d8c92a258b3ed4179b6d2129e8fbde280c55541`
- Model: `openai/gpt-oss-120b`
- Run timestamp: `2026-11-16T08:17:45.605512+00:00`
- Run directory: `20261116T081745_605187Z`
- Source answers SHA-256: `b88c7cf6d62fc369031b494fc6669fdb28b3f4a99bbbfc0d6d010551933af710`
- Annotation file: `claim_faithfulness_holdout_graph_expanded_manual_annotations.json`
