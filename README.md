# Neural Machine Translation with a Transformer 

- This is Part 3 of my three-part Neural Machine Translation project. 

The overview of the project is as follows:
- Part 1: LSTMs 
- Part 2: Attention
- Part 3: Transformer

### Deployed model
- I deployed my trained model as a simple web app using Flask and heroku. 
- Please follow this [link](https://transformer-translator-3.herokuapp.com/) to try out the translation model.
- The model translates from German into English. You can either enter an original sentence in German, or enter "random" to translate a random sentence from the training data.
- Click "plot" to see a heatmap of the attention scores.

### Data
- Data set 1: transcriptions from the proceedings of the European Parliament. Link [here](https://www.statmt.org/europarl/).
- Data set 2: bi-lingual sentence pairs written by volunteers of the Tatoeba Project. Link [here](http://www.manythings.org/bilingual/). 
- The combined data set includes around 2.1 million sentences. I selected only the sentences with fewer than 30 words per sentence, which amounted to around 1.5 million sentences.



## 1) Some remarks on key ideas

- Our Transformer model is based on the architecture laid out by Vaswani et al in the paper ['Attention Is All You Need'](https://arxiv.org/abs/1706.03762). 
- Below, I offer a high-level overview of some key features and ideas. 

### 1.1) Overview of the Self-Attention mechanism

- We have a sequence of words, with each word represented by an embedding vector
- The goal is to create a new vector representation for each word, based on the other words in the sentence 
- For each word $x_i$ in the sentence, create a query vector ( $q_i$ ) , key vector ( $k_i$ ) , and value vector ( $v_i$ )
- Then we create a new vector representation $a_i$ for each word, where $a_i$ is a linear combination of all the value vectors: 

$$ a_i = \alpha_1 * v_1 + \alpha_2 * v_2 + ... + \alpha_N * v_N $$

- Each coefficient $\alpha_j$ is computed by projecting $q_i$ onto $k_j$ : $$\alpha_j = q_i \cdot k_j$$
- The coefficients are normalised using the soft-max function 
- We vectorise these operations in the Transformer architecture for higher performance: 

$$ A = \frac{1}{\sqrt{d_k}}softmax(Q * K^T)V $$

- where A, Q, K and V store the a, q, k, and v vectors as rows, and $d_k$ is the length of the k vector. 

### 1.2) Intuition behind the Self-Attention mechanism
- In the earlier NMT models in Part 1 and Part 2, we used fixed pre-trained embeddings to represent every word
- This enabled our model to capture similarities and differences between words (for example when measured using the cosine similarity). 
- However, representing each word with a fixed embedding vector ignores the fact that any word can take on different meanings depending on the words around it. 
- The Self-Attention mechanism offers a way to capture the context-dependent meanings of words. The sentence "pays attention to itself", creating a new vector representation for each word as is appropriate **for that particular sentence**.

### 1.3) Attention mechanism in the Transformer model

- We incorporate the Attention mechanism in the Transformer in three places: 

- 1) Self-attention in the Encoder, where the source sequence pays attention to itself. The embeddings of the words in the source sequence are used as the "q", "k", and "v" vectors. 
- 2) Self-attention in the Decoder, where the target sequence pays attention to itself. The embeddings of the words in the target sequence are used as the "q", "k", and "v" vectors.
- 3) Encoder-Decoder-attention in the Decoder, where the target sequence pays attention to the source sequence. The outputs of the Decoder self-attention layer is used as the "q" vectors, and the outputs of the Encoder are used as the "k" and "v" vectors. 

### 1.4) Positional Encodings
- With earlier RNN networks, we fed the inputs into the network one word at a time in the correct order. However, when we train the Transformer model, we feed the data all at once. As such, we need an additional step to encode the order in which the words appear in each sentence. We encode the positions of the inputs using the following formulas: 

$$
PE_{(pos, 2i)}= sin\left(\frac{pos}{{10000}^{\frac{2i}{d}}}\right)
\tag{1}$$

$$
PE_{(pos, 2i+1)}= cos\left(\frac{pos}{{10000}^{\frac{2i}{d}}}\right)
\tag{2}$$

* $d$ is the dimension of the word embedding and positional encoding
* $pos$ is the position of the word.
* $k$ refers to each of the different dimensions in the positional encodings, with $i$ equal to $k$ $//$ $2$.
  

## 2) Model architecture

### Encoder
- The Encoder embeds the source sentence, adds positional encodings, and passes the encoded embeddings into a stack of Encoder layers.

### Encoder Layer
- The Encoder layer passes the input through a multi-head attention layer, whose outputs are then passed into a feed-forward network. We also include residual connections and layer normalization to speed up training. 

### Decoder
- The Decoder embeds the target sentence, adds positional encodings, and passes the encoded embeddings into a stack of Decoder layers. 

### Decoder Layer
- The Decoder layer consists of two multi-head attention layers. First, it passes the input through a multi-head attention layer for Self-Attention. The output is used as the Query matrix for the second multi-head attention layer, while the output from the Encoder is used as the Key and Value matrices. The output of the second multi-head attention layer is passed through a feed-forward network. Once again, we include residual connections and layer normalization to speed up training. 


## 3) Tokenization
- Byte-pair encoding (BPE) was used to tokenize the sentences. This offers a number of advantages over the word-based tokenization I used earlier in Part 1 and 2. When you tokenize each word, your Embedding matrix grows larger with every new word you add into the vocabulary. This in turn means there are more and more parameters in your Embedding layer that need to be optimised, leading to longer training times. Byte-pair encoding sets a limit (in our case 30,000) on the number of tokens in your vocabulary. Moreover, it enables your model to also translate new words that aren't in the training set. 

## 4) Navigating the directory

### 4.1) Data wrangling
- Please see the notebook **data_wrangling.ipynb** and the custom module **tokenizer_helpers.py**

### 4.2) Training
- Please see the notebook **train_v3.ipynb** and the custom modules **tokenizer_helpers.py**, **model_components.py** and **training_helper_functions.py**

### 4.3) Translating
- Please see the notebook **translate_v3.ipynb** and the custom modules **tokenizer_helpers.py**, **model_components.py** and **translate_helper_functions.py**

### 4.4) Flask app
- Please see the directory **flask**

