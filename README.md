# Neural Machine Translation with a Transformer 

## Overview

## Architecture

### Data Pre-processing
Byte-pair encoding of source (German) and target (English) sentences. Separate tokenizers were used for the two languages

### Encoder Layer 
1) Self-attention mechanism: multi-head attention with the source input x as the query (Q), key (K), and value (V)
2) Residual connection, followed by layer normalization
3) Dense layer
4) Dropout layer (activated only during training)
5) Residual connection, followed by layer normalization

### Encoder




### Decoder





## Implementation

### Data Wrangling

### Training

### Translating
