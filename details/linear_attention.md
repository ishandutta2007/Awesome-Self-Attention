# Linear Attention Approximation (Kernel Tricks)

Linear attention avoids calculating the expensive $N \times N$ similarity matrix by altering the execution order of dot-product multiplications.

## Associative Property
Traditional Attention:

$$\text{Softmax}(Q K^T) V$$

By replacing the Softmax function with a feature map $\phi(\cdot)$ such that similarity is $\phi(Q) \phi(K)^T$, we can rewrite the equation as:

$$(\phi(Q) \phi(K)^T) V = \phi(Q) (\phi(K)^T V)$$

This reduces computational complexity from $O(N^2)$ to $O(N)$.

## Computational Order Shift

```mermaid
flowchart LR
    subgraph Traditional [Quadratic O(N^2)]
        Q1[Q] -->|MatMul| K1[K^T] --> Softmax1[Softmax] -->|MatMul| V1[V]
    end
    subgraph Linear [Linear O(N)]
        K2[K^T] -->|MatMul| V2[V] -->|MatMul| Q2[Q]
    end
```

---
[← Back to README](../README.md)
