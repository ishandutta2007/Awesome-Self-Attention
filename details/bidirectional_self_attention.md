# Bidirectional Self-Attention (Encoder-Style)

Bidirectional self-attention allows each token in the input sequence to attend to all other tokens, regardless of whether they appear before or after in the sequence.

## Use Cases
It is typical of encoder-style architectures such as BERT. It is ideal for sequence understanding, text classification, and named entity recognition, where context from both left and right is crucial.

## Interaction Matrix

```mermaid
flowchart TD
    Token1[Token 1] <--> Token2[Token 2]
    Token2 <--> Token3[Token 3]
    Token1 <--> Token3
    Token1 <--> Token4[Token 4]
    Token2 <--> Token4
    Token3 <--> Token4
```

---
[← Back to README](../README.md)
