# Neural Machine Translation with a Transformer 

## Overview

## Architecture

### Data Pre-processing
Byte-pair encoding of source (German) and target (English) sentences. Separate tokenizers were used for the two languages

### Encoder Layer 
1) Self-attention mechanism: multi-head attention with the input x as the query (Q), key (K), and value (V)
    **Zero-padding is added to sentences shorter than the maximum sentence length
2) Residual connection, followed by layer normalization
3) Dense layer
4) Dropout layer (activated only while training)
5) Residual connection, followed by layer normalization

### Encoder
1) Pass the source input (German) through an Embedding layer, then scale by square root of embedding dimensions
2) Add positional encoding to embedding vectors
3) Dropout layer (activated only while training)
4) Pass through a stack of four Encoder Layers

### Decoder Layer
#### Block 1
1) Self-attention mechanism: multi-head attention with the input x as the query (Q), key (K), and value (V)
    **Look-ahead mask is applied to the input, so the model only sees the sentence up to the word it is predicting
2) Residual connection, followed by layer normalization
#### Block 2
1) Multi-head attention with the output from Block 1 as the query (Q) and the output from the Encoder as the key (K) and value (V)
    **Zero-padding is added to sentences shorter than the maximum sentence length
2) Residual connection, followed by layer normalization
#### Block 3
1) Pass output of Block 2 through Dense layer
2) Dropout layer (activated only while training)
3) Residual connection, followed by layer normalization 

### Decoder
1) Pass the target input (English) through an Embedding layer, then scale by the square root of embedding dimensions
2) Add positional encodings to the embedding vectors
3) Dropout layer (activated only while training)
4) Pass through a stack of four Decoder Layers

### Transformer
1) Pass the source input sentence into the Encoder to get the Encoder output
2) Pass the target input sentence, together with the Encoder output, into the Decoder
3) Pass the Decoder output through a Dense layer, followed by softmax activation, to get the logits




## Implementation

### Data Wrangling

### Training

### Translating
