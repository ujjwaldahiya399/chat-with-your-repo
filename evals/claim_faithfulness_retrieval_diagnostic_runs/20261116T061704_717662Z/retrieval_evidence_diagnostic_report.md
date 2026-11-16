# Micrograd retrieval-evidence diagnostic

## Result

Supplying the missing implementation evidence fixed both targeted answers. H02 and H05 went from **0/2 fully faithful** in the original holdout to **2/2 fully faithful** in the diagnostic run.

| Metric | Original H02 + H05 | Diagnostic H02 + H05 |
|---|---:|---:|
| Questions with all required evidence | 0/2 | 2/2 |
| Supported factual claims | 12/16 (75.0%) | 19/19 (100%) |
| Claims absent from supplied evidence | 4 | 0 |
| Contradicted claims | 0 | 0 |
| Claims covered by supporting citations | 8/16 (50.0%) | 19/19 (100%) |
| Fully faithful answers | 0/2 | 2/2 |
| Invalid citation numbers | 0 | 0 |

Atomic-claim totals differ because the stochastic generations used different wording and detail. The cleanest comparison is the question-level result. Both originally incomplete answers became complete and fully supported after the missing evidence was supplied.

## H02

The original context contained the neuron expression but omitted the implementation of `Value.__mul__` and `Value.__add__`. The model still named those methods and described their behavior, producing three claims that were not established by the retrieved sources. It also omitted two requested implementation details.

The diagnostic added the frozen `micrograd/engine.py:2-94` chunk. The new answer correctly explains that:

- `Value.__mul__` wraps an ordinary number in a `Value`.
- The multiplication output stores a closure that accumulates gradients into both operands.
- `sum` begins with the `Value` bias and combines terms through `Value.__add__`.
- The addition closure propagates the upstream gradient to both operands.
- The resulting activation remains part of the autograd graph, with conditional ReLU handled as another `Value` operation.

All 11 factual claims were supported and citation-covered.

## H05

The original context showed `Module.zero_grad()` and `MLP.parameters()`, but it stopped at `layer.parameters()`. The model asserted that the call reached every weight and bias without receiving the two methods needed to prove that path.

The diagnostic added the frozen `Neuron.parameters()` and `Layer.parameters()` chunks. The new answer traces the complete chain:

```text
MLP → layers → neurons → weights and bias → grad = 0
```

All eight factual claims were supported and citation-covered.

## Interpretation

This diagnostic isolates evidence availability as the main cause of the H02 and H05 failures. With the model, prompt, questions, repository snapshot, and single-sample generation procedure held fixed, adding the missing implementation evidence changed both answers from incomplete or unsupported to fully faithful.

The result does not measure retrieval quality because the missing chunks were supplied manually after the holdout was reviewed. It also does not prove a deterministic causal effect because the provider generation was stochastic and only one new sample was collected per question. It is strong diagnostic evidence that retrieval coverage should be improved before changing the generation model.

## Recommended engineering direction

Improve retrieval for cross-chunk implementation questions. The two failures share the same structure:

- H02 needed code from `Neuron.__call__` and `Value` arithmetic methods.
- H05 needed code across the `MLP → Layer → Neuron` parameter hierarchy.

The next retrieval experiment should expand from an initially retrieved symbol to directly referenced or structurally related definitions, then rerun the frozen 12-question holdout preflight. Generation should remain frozen until the retrieval-only comparison is complete.

## Provenance

- Repository `https://github.com/karpathy/micrograd`
- Frozen commit `7bc720e951fe422b8f8814aa5aa1b64121d26b4c`
- Model `openai/gpt-oss-120b`
- Diagnostic run timestamp `2026-11-16T06:17:04.717970+00:00`
- Diagnostic run directory `20261116T061704_717662Z`
- Raw answer SHA-256 `f26b95bdd18f509cc8b38a3090e00ab41f56aea2fce9a59cd36e5efcf6e0bb3f`
- Review method manual claim-by-claim scoring against only the five sources stored with each diagnostic question
- Claim-level record `retrieval_evidence_diagnostic_annotations.json`
