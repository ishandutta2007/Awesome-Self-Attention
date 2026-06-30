# Multi-Query (MQA) & Grouped-Query Attention (GQA)

As Large Language Models (LLMs) scale, inference becomes bottlenecked by Key-Value (KV) cache memory bandwidth. MQA and GQA were introduced to optimize serving throughput.

## Comparison
*   **Multi-Head Attention (MHA):** Each query head has its own key and value head.
*   **Multi-Query Attention (MQA):** All query heads share a single key and value head. (High compression, slight accuracy degradation).
*   **Grouped-Query Attention (GQA):** Query heads are partitioned into groups, and each group shares a key-value head. (Balances MHA accuracy with MQA speed).

## Head Configurations

```mermaid
flowchart TD
    subgraph MHA [Multi-Head Attention]
        Q1[Q1] --> K1[K1]
        Q2[Q2] --> K2[K2]
        Q3[Q3] --> K3[K3]
        Q4[Q4] --> K4[K4]
    end
    subgraph GQA [Grouped-Query Attention]
        Q5[Q1] --> GK1[K1]
        Q6[Q2] --> GK1
        Q7[Q3] --> GK2[K2]
        Q8[Q4] --> GK2
    end
    subgraph MQA [Multi-Query Attention]
        Q9[Q1] --> MK1[K]
        Q10[Q2] --> MK1
        Q11[Q3] --> MK1
        Q12[Q4] --> MK1
    end
```

---
[← Back to README](../README.md)
