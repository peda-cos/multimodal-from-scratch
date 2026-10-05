# 63 — Multimodal Evaluation

Training a multimodal model is only half the loop. The other half is measuring
what it learned on data it did not train on. This module implements three
evaluation surfaces from primitives and runs them end to end:

- **Image-caption retrieval:** R@1, R@5, R@10 in both directions.
- **Visual question answering:** exact-match accuracy over a fixed answer
  vocabulary.
- **Image captioning:** multi-reference sentence BLEU-4 with add-one smoothing.

Everything runs on CPU with NumPy. There are no downloads, API keys, external
metric libraries, or test-time network calls. The model is a self-contained
copy of the lesson-62 vision-language stack because track modules must verify
independently.

This is module `63-multimodal-eval` of the *Multimodal From Scratch* track and
implements the [Multimodal Evaluation capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/multimodal-eval).

## Quickstart

From this directory:

```bash
./verify.sh
.venv/bin/python code/main.py
```

`verify.sh` creates an ignored `.venv`, installs the two pinned dependencies,
runs the tests and mypy, then runs the demo. It is safe to run twice.

If NumPy is already available, the demo and tests also run directly:

```bash
python3 code/main.py
python3 -m unittest code.test_main -v
```

## What is measured

The demo trains a freshly initialized model for 50 steps on 200 synthetic
pairs, then evaluates a 50-sample suite generated from a different RNG seed.
Measured locally on CPU in a single run (seed 62, batch 16; no hyperparameter
search):

| Metric | Random baseline | Before | After 50 steps |
| --- | ---: | ---: | ---: |
| R@1 text → image | 0.0200 | 0.0200 | **0.3400** |
| R@5 text → image | 0.1000 | 0.1600 | **0.7800** |
| R@10 text → image | 0.2000 | 0.2400 | **0.8800** |
| R@1 image → text | 0.0200 | 0.0200 | **0.3200** |
| R@5 image → text | 0.1000 | 0.0800 | **0.7400** |
| R@10 image → text | 0.2000 | 0.2200 | **0.9000** |
| VQA exact match | 0.0156 | 0.0200 | **0.3800** |
| BLEU-4 | n/a | 0.0673 | **0.2673** |

BLEU has no closed-form random baseline; the before column is the untrained
model's score against the suite references.

Full-corpus training losses:

| Loss | Before | After |
| --- | ---: | ---: |
| Contrastive | 5.3260 | 1.5217 |
| Language modeling | 6.5184 | 3.5368 |
| Caption readout | 4.1865 | 1.3968 |
| VQA readout | 4.1579 | 1.4297 |
| Total | 20.1888 | 7.8852 |

The demo prints the same table and asserts that every surface improves and
that R@1/R@5/R@10 and VQA beat their random baselines. These are measurements
of this synthetic fixture, not benchmark results.

## Architecture

```mermaid
flowchart TB
  subgraph data[Deterministic synthetic data]
    C[200 training pairs] --> B
    S[50 held-out pairs<br/>retrieval + VQA + 3 caption refs] --> E
  end

  subgraph model[NumPy model]
    I[8x8 image] --> V[SmallViT<br/>17 tokens]
    V --> VT[vision patch tokens]
    VT --> P[token projector]
    P --> CLS[projected CLS]
    P --> MEM[projected memory tokens]
    Cap[8-token caption] --> T[text tower]
    Cap --> D[causal cross-attention decoder]
    MEM --> D
  end

  B[contrastive + LM + heads] --> model
  CLS --> R[cosine matrix]
  T --> R
  R --> RK[R@1 / R@5 / R@10<br/>both directions]

  VT --> CH[caption head<br/>per-slot MLP]
  VT --> VH[VQA head<br/>patch + question-slot MLP]
  CH --> BLEU[BLEU-4 vs 3 references]
  VH --> EM[exact-match accuracy]
```

Default sizes:

| Piece | Configuration |
| --- | --- |
| Image | 8×8 grayscale, 2×2 patches |
| ViT | 2 layers, hidden 32, 2 heads, CLS + 16 patches |
| Projector | 32 → 64 → 32, GELU |
| Text tower | 512-token embedding, hidden 32, masked mean pool |
| Decoder | 2 cross-attention layers, hidden 64, 4 heads |
| Content vocabulary | 64 ids: 3–66 |
| Caption | BOS + 6 content tokens + EOS + 2 PAD = 10 |
| Caption head | 32 → 64 → 64, GELU |
| VQA head | patch 32 + question 32 → 64 → 64, GELU |
| Parameters | 223,713 |
| Base optimizer | Adam, lr 3e-3, global gradient-norm clip 1.0 |
| Head optimizer | Adam, lr 3e-2 |

### Why the heads read vision tokens, not projected tokens

Retrieval uses the projected CLS vector. The caption and VQA heads need the
identity of individual content patches, which the token projector is not
trained to preserve: it is optimized for the CLS-to-text contrastive objective
and the decoder cross-attention. In a local probe, a fresh MLP fit projected
patch tokens poorly after base training, while the pre-projection ViT patch
tokens remained separable. The heads therefore read `vision_tokens` and
backpropagate through the ViT; the projector and CLS stay dedicated to
retrieval and LM.

The heads use a separate, higher learning rate because they are randomly
initialized and must calibrate quickly while the base features are still
moving. A local 50-step comparison with everything else fixed:

| Head lr | Caption CE | VQA CE | VQA EM | BLEU-4 |
| ---: | ---: | ---: | ---: | ---: |
| 3e-3 (base rate) | 1.9678 | 1.9298 | 0.3000 | 0.2200 |
| 3e-2 | 1.3967 | 1.4295 | 0.3800 | 0.2673 |

This is a recipe comparison, not a hyperparameter search.

## Metrics

### Recall@K

For a square similarity matrix `S` with row `i` the caption and column `i` the
matching image:

$$
\text{R@K} = \frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\left[i \in \operatorname{topK}(S_{i,:})\right].
$$

`direction="image_to_text"` applies the same statistic to `S.T`. Ties are
broken by stable sort order; the tests use strict margins or a diagonal-worst
matrix so the expected value is unambiguous.

### VQA exact match

$$
\text{EM} = \frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\left[\hat{a}_i = a_i\right].
$$

The synthetic question is a single token naming a content slot. The VQA head
fuses that slot's ViT token with a learned slot embedding and predicts one of
64 answer tokens. Predictions and references are integer ids, so this is the
single-answer form; VQA v2's soft accuracy is documented as a limitation and a
natural extension, not simulated as if it were measured.

### BLEU-4

Modified n-gram precision clips each generated n-gram count by the maximum
count in any single reference, then divides by the number of generated
n-grams. The score is the geometric mean of $p_1,\ldots,p_4$ times the brevity
penalty:

$$
\text{BP} =
\begin{cases}
1 & g \ge r,\\
\exp(1-r/g) & g < r,
\end{cases}
\qquad
\text{BLEU-4} = \text{BP}\cdot
\exp\left(\frac{1}{4}\sum_{n=1}^{4}\log p_n\right).
$$

`r` is the length of the reference closest to the generated length, following
the standard best-match-length convention. When an n-gram count is zero, the
implementation adds one to both numerator and denominator (Chen and Cherry's
add-one rule). A completely disjoint hypothesis returns `0.0` even with
smoothing, matching the lesson's test contract.

Generated and reference special tokens are stripped before caption scoring.
The references are three permutations of the canonical content tokens, which
makes BLEU sensitive to token order without requiring a second language model.

## Held-out synthetic suite

`build_eval_suite(seed, n_samples, vocab_size, max_len)` returns one
`EvalSuite` that unpacks into three lists:

- `pairs`: 50 `(image, caption_ids)` entries for retrieval.
- `vqa`: 50 `(image, question_ids, answer_id)` triples.
- `caps`: 50 `(image, [reference_ids, ...])` entries with three references.

The training corpus uses seed 62 and the suite uses seed 6300. The demo checks
that the suite's 50 ordered content sequences are absent from the 200 training
sequences (measured overlap: 0). The visual codebook and language distribution
are shared, so this is a held-out sample split, not a domain shift.

The suite is deterministic: the same seed reproduces images, captions,
questions, and references byte for byte.

## Offline and real-service boundary

**Runs offline:** data generation, training, all three metrics, tests, mypy,
and the demo. Tests make no network calls and use no secrets. `verify.sh`
needs the network only to install the pinned Python packages into the ignored
local environment.

**Not simulated as a paid service:** there is no hosted evaluator, dataset
download, cloud sandbox, or GPU path. A real run replaces the synthetic
`build_eval_suite` with a dataset loader and swaps the NumPy model for any
object exposing the same three surfaces:

1. projected image/text embeddings for the similarity matrix,
2. per-image vision patch tokens for the caption and VQA heads,
3. answer token ids and generated token ids.

The metric functions (`recall_at_k`, `vqa_exact_match`, `bleu4`) are
benchmark-agnostic and can be lifted into a PyTorch evaluation loop unchanged.
For real VQA v2, replace exact match with the official soft accuracy:
`min(human_count / 3, 1)` per answer, averaged over the ten-choose-nine
annotator sets. For MS-COCO captioning, keep the multi-reference BLEU-4 and
add CIDEr and METEOR; the reference implementation is corpus-level and uses
its own tokenizer, so numbers are not comparable across tokenizations.

## Limitations

- **Synthetic fixture, not a benchmark.** The images are 8×8 rendered codes and
  the captions are token permutations. High absolute numbers mainly show that
  the measurement plumbing and training signal are connected.
- **Small evaluation pool.** N=50 gives coarse R@K resolution and no confidence
  intervals. The numbers are one deterministic run, not a seed-average.
- **Single-answer VQA.** The lesson's exact-match contract is implemented. VQA
  v2 soft accuracy, multiple annotators, and answer normalization are not.
- **Fixed question vocabulary.** Questions name a content slot; this is not
  free-form natural-language question encoding.
- **Parallel caption readout.** The caption head classifies six fixed content
  slots in parallel. It is not an autoregressive caption decoder, so it cannot
  produce variable-length or free-form captions.
- **Sentence-level BLEU.** No corpus-level aggregation, CIDEr, METEOR, SPICE,
  or tokenizer variants are included.
- **NumPy only.** There is no GPU implementation or distributed evaluation.
- **Tie behavior.** R@K uses stable sort, so exact score ties are resolved by
  column order. Real embedding similarities rarely tie exactly.

## Design decisions

- Copy the lesson-62 model stack into this module instead of importing a
  sibling module, as required for independent verification.
- Keep the metrics as pure functions over arrays and id sequences. This makes
  the math testable without the model and portable to real benchmarks.
- Use the closest reference length for the brevity penalty and max-over-
  references clipping, which are the standard multi-reference conventions.
- Use add-one smoothing only for zero-match n-gram orders and explicitly return
  zero for a disjoint hypothesis, following the capstone's stated behavior.
- Read caption/VQA heads from pre-projection vision tokens; see the probe result
  above.
- Use a separate Adam optimizer for the heads (lr 3e-2) and the base stack
  (lr 3e-3). This is a training recipe, not a claim that the rates are optimal.
- Assert metric improvements above random baselines in the demo rather than
  relying on the numbers looking plausible.

## Tests

`code/test_main.py` contains 21 tests, including:

- R@K identity, diagonal-worst, direction, `k = N`, and invalid-input cases.
- Exact BLEU-4, disjoint vocabulary, add-one smoothing, multi-reference
  clipping, brevity penalty, and empty generation.
- VQA exact-match fraction and input validation.
- Suite counts, determinism, seed sensitivity, short-caption handling,
  question/answer ranges, and invalid inputs.
- Content-token rendering into the expected 2×2 patch.
- Forward shapes, loss decomposition, finite head gradients, head-only loss
  reduction, bounded evaluation metrics, reproducible evaluation, and a short
  training run that reduces total, caption, and VQA losses.

## Not included

The capstone exercises (CIDEr, VQA soft accuracy, NaN-safe BLEU variants, MRR,
and checkpoint learning curves) are out of scope for this build. The metric
function boundaries leave room for each without changing the core shapes.

## References

1. [Multimodal Evaluation capstone specification](https://aiengineeringfromscratch.com/learn/capstone-projects/multimodal-eval).
2. Papineni, Roukos, Ward, and Zhu. *BLEU: a Method for Automatic Evaluation of
   Machine Translation.* ACL 2002. <https://aclanthology.org/P02-1040/>
3. Chen and Cherry. *A Systematic Comparison of Smoothing Techniques for
   Sentence-Level BLEU.* WMT 2014. <https://aclanthology.org/W14-3346/>
4. Goyal, Khot, Summers-Stay, Batra, and Parikh. *Making the V in VQA Matter:
   Elevating the Role of Image Understanding in Visual Question Answering.*
   CVPR 2017. <https://arxiv.org/abs/1612.00837> (VQA v2 dataset; the official
   soft-accuracy and normalization rules are at
   <https://visualqa.org/evaluation.html>).
5. Chen, Fang, Lin, Vedantam, Gupta, Dollár, and Zitnick. *Microsoft COCO
   Captions: Data Collection and Evaluation Server.* 2015.
   <https://arxiv.org/abs/1504.00325>
