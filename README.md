# Track

A collection of 6 tested projects built from scratch, one module per capstone lesson.

Each module has its own `README.md`, `verify.sh`, tests, and a deterministic offline demo.

## Modules

| # | Module | What it demonstrates | Verify |
|---:|---|---|---|
| 58 | [Vision Encoder Patches](58-vision-encoder-patches/) | A vision model that reads pixels needs a tokenizer for pixels. Patch embedding is that tokenizer. Cut the image into a grid of squares, flat | `cd 58-vision-encoder-patches && ./verify.sh` |
| 59 | [Vision Transformer Encoder](59-vit-transformer/) | Patches alone do not see. A 12-layer pre-LN transformer with 12 attention heads turns the sequence of patch tokens into a sequence of contex | `cd 59-vit-transformer && ./verify.sh` |
| 60 | [Projection Layer for Modality Alignment](60-projection-layer-modality-align/) | A vision encoder produces image tokens. A text decoder consumes text tokens. The two live in different vector spaces. A small two-layer MLP  | `cd 60-projection-layer-modality-align && ./verify.sh` |
| 61 | [Cross-Attention Fusion](61-cross-attention-fusion/) | The projection layer aligns one image vector with one caption vector. A real vision-language decoder needs every text token to attend to eve | `cd 61-cross-attention-fusion && ./verify.sh` |
| 62 | [Vision-Language Pretraining](62-vision-language-pretraining/) | The encoder, projection, and decoder are wired. Now train them together. Two objectives drive learning: a contrastive image-text loss (InfoN | `cd 62-vision-language-pretraining && ./verify.sh` |
| 63 | [Multimodal Evaluation](63-multimodal-eval/) | Training is half the loop. The other half is measurement. This lesson builds three evaluation surfaces from primitives: image-caption retrie | `cd 63-multimodal-eval && ./verify.sh` |

## Requirements

- Python 3.12+ and `uv` for Python modules
- Node 22+ / Bun 1.4 for TypeScript modules
- No API keys, GPU, or network needed for the test suites

## Verify all

```bash
make verify
```

## References

- Capstone source: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects
