# Cross-Attention

Cross-attention dynamically mixes information from two different sequences or modalities. It is the key mechanism in encoder-decoder structures.

## Alignment
*   **Queries (Q):** Derived from the target decoder sequence.
*   **Keys (K) & Values (V):** Derived from the source encoder sequence.

This enables the target sequence generation to be conditioned on representations from the source sequence.

## Interaction Flow

```mermaid
flowchart LR
    Source[Encoder Output] --> K[Keys] & V[Values]
    Target[Decoder Hidden State] --> Q[Queries]
    Q & K --> Matmul[MatMul & Softmax]
    Matmul & V --> Context[Context Output]
```

---
[← Back to README](../README.md)
