# Neural Machine Translation with a Transformer 

## Overview

### Overview of the Self-Attention mechanism

- We have a sequence of words, with each word represented by an embedding vector
- The goal is to create a new vector representation for each word, based on the other words in the sentence 
- For each word $x_i$ in the sentence, create a query vector ( $q_i$ ) , key vector ( $k_i$ ) , and value vector ( $v_i$ )
- Then we create a new vector representation $a_i$ for each word, where $a_i$ is a linear combination of all the value vectors: 

$$ a_i = \alpha_1 * v_1 + \alpha_2 * v_2 + ... + \alpha_N * v_N $$

- Each coefficient $\alpha_j$ is computed by projecting $q_i$ onto $k_j$ : $\alpha_j$ = $q_i$ \cdot $k_j$
- The coefficients are normalised using the soft-max function 
- We vectorise these operations in the Transformer architecture for higher performance: 

$$ A = \frac{1}{\sqrt{d_k}}softmax(Q * K^T)V $$

where A, Q, K and V store the a, q, k, and v vectors as rows. 

### Intuition behind the Self-Attention mechanism
- Until now, we have been using fixed pre-trained word embeddings
- This enabled our model to capture similarities and differences between words
- For example, the embeddings for the words "banana" and "apple" would have a relatively high cosine similarity score, since both "banana" and "apple" are fruit. 
- However, assigning a fixed embedding vector to each word ignores the fact that every word can have a different meaning depending on its context
- The same word can take on different connotations depending on the sentence it is used in
- The Self-Attention mechanism offers a way to capture the context-dependent meanings of words
- The sentence "pays attention to itself" – and creates a new vector representation for each word, depending on the words around it. 


### Attention mechanism in the Transformer model

- We incorporate the Attention mechanism in the Transformer in three places: 

- 1) Self-attention in the Encoder, where the source sequence pays attention to itself. The embeddings of the words in the source sequence are used as the "q", "k", and "v" vectors. 
- 2) Self-attention in the Decoder, where the target sequence pays attention to itself. The embeddings of the words in the target sequence are used as the "q", "k", and "v" vectors.
- 3) Encoder-Decoder-attention in the Decoder, where the target sequence pays attention to the source sequence. The outputs of the Decoder self-attention layer is used as the "q" vectors, and the outputs of the Encoder are used as the "k" and "v" vectors. 



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
