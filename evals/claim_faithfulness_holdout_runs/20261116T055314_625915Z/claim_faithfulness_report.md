# Micrograd claim-faithfulness holdout

The retriever supplied all required evidence for six of eight answerable questions. All four unanswerable questions received the exact required refusal. Both questions with incomplete retrieval received substantive answers that included claims their evidence did not establish.

| Metric | Result |
|---|---:|
| Required evidence retrieval on answerable questions | 6/8 (75.0%) |
| Supported factual claims | 52/58 (89.7%) |
| Claims absent from retrieved evidence | 5 |
| Contradicted claims | 1 |
| Claims covered by supporting citations | 46/58 (79.3%) |
| Fully faithful answerable responses | 4/8 (50.0%) |
| Correct exact refusals | 4/4 (100.0%) |
| False refusals on answerable questions | 0/8 |
| Invalid citation numbers | 0 |
| Correct question outcomes | 8/12 (66.7%) |

## Question-level findings

| Question | Required evidence retrieved | Supported claims | Citation coverage | Finding |
|---|---|---:|---:|---|
| H01 | Full | 5/6 | 5/6 | All requested facts addressed. One broader runtime assertion exceeds the cited evidence. |
| H02 | Partial | 8/11 | 4/11 | Mixed arithmetic is illustrated in the README. Engine-specific claims and omissions remain. |
| H03 | Full | 7/7 | 7/7 | Both truncation cases and the lack of validation are explained. |
| H04 | Full | 7/7 | 5/7 | Shapes and data flow are correct. The citations cover the shapes and omit the evidence needed for the exact Value types. |
| H05 | Partial | 4/5 | 4/5 | The answer reaches the layer call, then asserts full weight and bias coverage without retrieved support. |
| H06 | Full | 9/10 | 9/10 | The gradients and helper are correct. The blanket no-reset statement conflicts with the root assignment. |
| H07 | Full | 6/6 | 6/6 | The exponent constraint, base predecessor, and base-only update are supported. |
| H08 | Full | 6/6 | 6/6 | Initialization, parameter order, and the exact representation are supported. |
| HU01 | N/A | N/A | N/A | Exact refusal. The neuron constructor gives a random initializer, with no specific draws for the requested instance. |
| HU02 | N/A | N/A | N/A | Exact refusal. The README identifies PyTorch as a test dependency and supplies no exact version. |
| HU03 | N/A | N/A | N/A | Exact refusal. The README points to the demo notebook and names SGD. The two exact training settings are absent. |
| HU04 | N/A | N/A | N/A | Exact refusal. The five sources contain no LICENSE file text or the requested copyright details. |

## Failure details

- **H02** received the Neuron expression and README examples. The examples substantiate mixed arithmetic and broad autodiff behavior. The retrieved sources do not establish the named engine methods or that ReLU allocates a fresh output object. The answer also leaves out scalar-to-Value wrapping and the addition closure's gradient updates.
- **H05** received Module.zero_grad and MLP.parameters. Those excerpts establish that zero_grad assigns 0 to each returned parameter and that MLP flattens the layer results. The remaining links through Layer.parameters and Neuron.parameters were missing, so the claim of complete weight and bias coverage was unsupported in the supplied context.
- **H01** correctly reports what setup.py declares and why the README calls for PyTorch during tests. I scored its additional assertion about actual runtime behavior separately because it is stronger than the setup metadata.
- **H06** correctly derives x.grad = 3 and then 6. I read the parenthetical about any other node literally. The root y gets assigned a gradient of 1 before the closure loop, which the answer also states correctly. H06 uses bracketed citation markers in place of the prompt's parenthesized example. The references count toward semantic citation coverage where they support a claim.

## Method and scope

I scored distinct factual claims against the five retrieved sources for each question. The expected claims were a completeness rubric. Support and citation coverage have the same denominator and differ when retrieved evidence exists outside the references used for a claim. A fully faithful answerable response covers the required parts and has only supported claims. Correct question outcomes combine fully faithful responses with exact refusals on the unanswerable questions. The four exact refusals contributed zero claims.

The harder holdout was authored after the pilot and preregistered before retrieval and generation. It uses the same frozen repository and one generation per question. These results characterize this set and run.

## Next experiment

Keep the model and prompt fixed and replay H02 and H05 with the missing evidence supplied from the frozen snapshot. Treat this as a diagnostic run. Check whether the answers now supply the omitted implementation links and cite them. Use those results to decide whether to prioritize retrieval changes, then test partial-evidence behavior separately with the original frozen contexts.

## Provenance

- Frozen micrograd commit `7bc720e951fe422b8f8814aa5aa1b64121d26b4c`
- Model `openai/gpt-oss-120b`
- Run timestamp `2026-11-16T05:53:14.626263+00:00`
- Run directory `20261116T055314_625915Z`
- Raw answer commit `55dc511`
- Claim-level record `claim_faithfulness_manual_annotations.json`
