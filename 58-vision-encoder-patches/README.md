# Vision Encoder Patches

A from-scratch implementation of the image tokenizer at the front of a vision
transformer. The model cuts a `224 x 224` RGB image into `16 x 16` patches,
projects each patch through one linear map, adds a fixed 2D sinusoidal position
signal, and prepends a learned CLS token. The result is a `(1, 197, 768)` token
sequence that a transformer can consume.

Implemented in Python with NumPy and a small `Module` base that mirrors the
callable/parameter surface of `torch.nn.Module`. No pretrained weights, GPU, or
external service is involved: every number in this README comes from the local
fixture and the local test run.

## Architecture

```mermaid
flowchart LR
    Image["NCHW image<br/>(B, 3, 224, 224)"] --> Conv["Conv2d<br/>kernel 16, stride 16"]
    Conv --> Patches["patch tokens<br/>(B, 196, 768)"]
    Patches --> Add["add 2D sinusoidal<br/>(196, 768)"]
    Add --> Cat["prepend CLS<br/>(B, 1, 768)"]
    Cat --> Out["token sequence<br/>(B, 197, 768)"]
```

| Stage | Shape | Trainable parameters |
|---|---|---:|
| Input image | `(B, 3, 224, 224)` | 0 |
| Patch projection `Conv2d` | weight `(768, 3, 16, 16)`, bias `(768,)` | 590,592 |
| Patch tokens | `(B, 196, 768)` | 0 |
| 2D sinusoidal table | `(196, 768)` | 0 |
| CLS token | `(1, 768)` | 768 |
| Output sequence | `(B, 197, 768)` | **591,360** |

The patch grid is `14 x 14 = 196`; the projection maps each flattened
`3 * 16 * 16 = 768`-pixel patch to the 768-dimensional hidden space. Token order
is row-major over the patch grid, matching the position table.

## Design decisions

- **NumPy instead of PyTorch.** The lesson is forward-only and needs one
  dependency, not autograd. `Module`, `Conv2d`, and `PatchEmbed` keep the same
  call/parameter semantics as their PyTorch counterparts. Torch is not
  installed or imported anywhere. The equivalent PyTorch construction is
  `nn.Conv2d(3, 768, kernel_size=16, stride=16)` followed by flatten and
  transpose; the math is identical.
- **One real convolution spelling, one independent test spelling.** Production
  code implements `Conv2d.forward` with `sliding_window_view` plus a matrix
  multiply. Tests contain a separate `Unfold`-style extractor and `Linear`
  reference, so the equivalence check is not a comparison of a function with
  itself. A third, manually looped dot-product check pins the channel/pixel
  ordering.
- **Position layout.** `sinusoidal_2d` splits `dim` in half: the first 384
  channels encode the row with sin/cos at frequencies
  `1 / 10000^(2i/384)`, and the last 384 encode the column. Values are
  interleaved as adjacent sin/cos pairs, computed in float64 and stored as
  float32. A token at `(row, col)` is
  `concat(row_encoding[row], column_encoding[col])`.
- **Uniform position norm.** Each sin/cos pair has unit L2 norm, so every token
  in the table has norm `sqrt(dim / 2)`. This is the sinusoidal signature the
  demo reports: for `dim = 768`, `sqrt(384) = 19.5959`.
- **CLS is unpositioned.** It is a learned vector prepended after position
  addition, so it is identical across batch items and is not polluted by a
  spatial coordinate. This also matches the parameter table: the position
  table is computed, not learned.
- **Deterministic fixtures and weights.** `synthesize_image(seed)` uses its own
  `numpy.random.default_rng` stream, so it never advances the global RNG.
  Model weights use a separate seeded stream. Re-running the demo gives the
  same output; tests assert determinism.
- **Square patches, arbitrary image grids.** Images can be non-square and
  produce `grid_h x grid_w` token grids. Non-square patch shapes are outside
  this lesson's scope.

## Quickstart

Requires Python 3.14 and `uv`. From the module directory:

```bash
cd 58-vision-encoder-patches
./verify.sh
```

`verify.sh` creates an ignored `.venv`, installs `numpy==2.5.5` and
`mypy==2.4.1`, runs the 19 tests, runs mypy, and executes the demo. It is safe
to run repeatedly.

Run the pieces directly after the first verification:

```bash
.venv/bin/python -m unittest discover -s code -p 'test_*.py' -v
.venv/bin/python code/main.py
```

The demo prints the token shape, CLS norm, one row of the position table, the
position-norm range, and median batch-1 forward latency.

Expected demo shape: the `224 x 224` fixture becomes a grid of `14 x 14 = 196`
patches, and the output is `(1, 197, 768)`.

## What runs offline

Everything in the lesson runs offline after the first dependency install:

- fixture synthesis, patch projection, position construction, and the front end;
- all 19 unit tests (no network, no sleeps, no external fixtures);
- the demo and latency measurement;
- mypy over `code/`.

Only `uv pip install` in `verify.sh` contacts PyPI. Pins are exact and the
venv lives in `.venv/`, which is git-ignored. There is no paid API, cloud
sandbox, dataset download, or telemetry.

## Optional real-provider / GPU mode

None is required or simulated for this module. The complete implementation is
the NumPy CPU path above; there is no hidden adapter or service boundary. A GPU
port would replace the patch projection with `torch.nn.Conv2d(3, 768, 16, 16)`
and the test reference with `torch.nn.Unfold` plus `torch.nn.Linear`; the
position table and shapes would stay the same. That port is intentionally not
wired in because it would turn a ~4 MB environment into a torch install without
adding behavior to this lesson.

## Limitations

- This is the front end only. There is no transformer stack, attention, or
  classification head; lesson 59 is the next layer.
- No training, optimizer, or learned position embeddings. The optional
  learned-position and frozen-feature-extractor exercises are not implemented.
- Positions are generated for each requested grid and are not interpolated;
  dynamic resolution works by rebuilding the table for the new grid.
- Patches must be square and image dimensions must be divisible by the patch
  size. Non-square patch kernels are not supported.
- Weights are randomly initialized, not pretrained, and inputs are not
  normalized by the front end. Both are the caller's responsibility.
- The implementation is single-process CPU NumPy. It favors clarity and
  deterministic verification over GPU throughput.

## Measured results

Measured locally with the pinned environment on CPU. Latency is machine- and
BLAS-dependent, varied between the two verification runs, and is not a
benchmark claim.

- `./verify.sh`: **19 tests passed** (`Ran 19 tests`, `OK`); mypy reported
  `Success: no issues found in 5 source files`; demo exited 0.
- Demo image shape: `(1, 3, 224, 224)`
- Patch grid: `14 x 14 = 196`
- Token shape: `(1, 197, 768)`
- CLS norm: `0.541106`
- Position table norm range: `min=19.595919`, `max=19.595919` (uniform, as
  predicted by the sin/cos pairs)
- Position row `0` first eight values: `[0.000000 1.000000 0.000000 1.000000
  0.000000 1.000000 0.000000 1.000000]` (the origin token, where `sin(0)=0` and
  `cos(0)=1`)
- Batch-1 forward median over 7 timed runs after warm-up: `3.907 ms` on the
  first run, `1.697 ms` on the second
- Parameter check: projection `590,592`, CLS `768`, total `591,360`, fixed
  position table `0`

## References

Sources read and used for this implementation and its documentation:

1. [Vision Encoder Patches capstone spec](https://aiengineeringfromscratch.com/learn/capstone-projects/vision-encoder-patches) — task framing, required classes, 224/16/768 geometry, CLS/position table, test expectations.
2. [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) — patch projection as the front end, CLS token, and 2D position embeddings.
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the sinusoidal sin/cos formula adapted here to row and column coordinates.
4. [DINOv2 `patch_embed.py`](https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/patch_embed.py) — production shape of a patch embedder: `Conv2d(kernel=patch, stride=patch)`, flatten, transpose.
5. [NumPy `sliding_window_view` documentation](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html) — the im2col-style view used by the convolution and unfold implementations.

## License

MIT © 2026 Pedro Monteiro. See [LICENSE](LICENSE).
