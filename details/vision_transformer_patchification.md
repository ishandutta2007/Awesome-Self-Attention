# Vision-Transformer Visual Token Patchification (ViTs)

Vision Transformers (ViTs) adapt self-attention to computer vision tasks by representing images as sequences of flattened patches.

## Method
An image of size $H \times W \times C$ is partitioned into non-overlapping patches of size $P \times P$. Each patch is flattened and linearly projected into a vector, acting exactly like a text token in a Transformer model.

## Patch Flow

```mermaid
flowchart LR
    Image[H x W Image] --> Patches[Split into P x P patches]
    Patches --> Flatten[Linear Projection]
    Flatten --> PosEmbed[Add Position Embeddings]
    PosEmbed --> Attention[Transformer Encoder Layers]
```

---
[← Back to README](../README.md)
