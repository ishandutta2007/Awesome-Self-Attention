# Scaled Dot-Product Attention

Scaled Dot-Product Attention is the foundational block of the Transformer architecture, introduced by Vaswani et al. in 2017. It allows for complete parallelization of sequence processing.

## Mathematical Formulation
Given query matrix $Q$, key matrix $K$, and value matrix $V$, the attention maps are calculated as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

The scaling factor $\frac{1}{\sqrt{d_k}}$ is critical; it prevents the dot products from growing excessively large in high dimensions, which would push the Softmax function into regions with extremely small gradients.

## Computational Path

```mermaid
flowchart LR
    Q[Queries Q] --> Mul[MatMul]
    K[Keys K] --> Mul
    Mul --> Scale[Scale by 1/sqrt(d_k)]
    Scale --> Softmax[Softmax]
    Softmax --> MulV[MatMul with V]
    V[Values V] --> MulV
    MulV --> Output[Output Context Matrix]
```

---
[← Back to README](../README.md)
