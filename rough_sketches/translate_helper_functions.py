import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Concatenate, LSTM, Embedding, Dense, MultiHeadAttention, LayerNormalization, Dropout
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.initializers import Constant
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import re
import os
import io
import time

from model_components import preprocess_sentence


def translate (source_sentence, transformer, max_target_length, de_tokenizer, en_tokenizer):
    """
    Inputs: 
    source_sentence -- string
    transformer -- trained transformer with loaded weights
    max_target_length -- int
    de_tokenizer, en_tokenizer -- loaded tokenizers

    Returns:
    pred_sentence, source_sentence, attention_weights
    """

    source_sentence = preprocess_sentence(source_sentence)

    # convert words in sentence to indices; convert to tensor with correct shape
    source_sequence = de_tokenizer.encode(source_sentence).ids
    source_sequence = tf.convert_to_tensor(source_sequence)
    source_sequence = tf.expand_dims(source_sequence,0)         # source_sequence = (1, Tx)

    start = en_tokenizer.get_vocab()['start_']
    end = en_tokenizer.get_vocab()['_end']

    # Initialise output_array with "start_" token
    output_array = tf.Variable([start], dtype = tf.int64 )
    output_array = tf.expand_dims(output_array, 0)

    pred_sentence = ''
    for t in range (max_target_length):
        
        output = output_array
        
        predictions, _ = transformer([source_sequence, output], training = False)       # (batch_size, tar_seq_len, target_vocab_size)
        
        # select final time-step of predictions
        predictions = predictions[:, -1:, :]                                            
        
        predicted_id = tf.argmax(predictions, axis=-1)
        predicted_id_int = int(predicted_id)
    
        pred_sentence += en_tokenizer.decode([predicted_id_int])
        output_array = tf.concat([output_array, predicted_id], axis = 1)

        if predicted_id == end:
            break

    
    # get attention_weights for final time-step (just before "_end" is predicted)
    _, attention_weights = transformer([source_sequence, output_array[:, :-1]], training = False)
    
    # return pred_sentence, source_sentence_proc, attention_weights         
    return pred_sentence, source_sentence, attention_weights


def random_translate_from_tensor(transformer, de_tokenizer, en_tokenizer, source_tensor, target_tensor, max_target_length):

    """
    Picks random sentence from the training data and translates it.

    Inputs: 
    transformer -- trained transformer with loaded weights
    de_tokenizer, en_tokenizer -- loaded tokenizers

    Returns:
    pred_sentence, original_sentence, attention_weights
    """
    
    k = np.random.randint(len(source_tensor))
    random_input = source_tensor[k]               # random_input = (Tx,)
    random_output = target_tensor[k]

    random_input = de_tokenizer.decode(random_input.astype(int))        # random_input does not include "start_" and "_end"
    
    pred_sentence, original_sentence, attention_weights = translate(random_input, transformer, 
                                                                    max_target_length,
                                                                    de_tokenizer, en_tokenizer)

    print(f'Input: {original_sentence}')
    print(f'Predicted translation: {pred_sentence}')
    
    true_translation = en_tokenizer.decode(random_output.astype(int))
    

    print(f'Actual translation: {true_translation}')
    return pred_sentence, original_sentence, attention_weights


def plot_attention_weights(pred_sentence, input_sentence, attention_weights, de_tokenizer, en_tokenizer):
    """
    Plots the scores from the attention block for final decoder layer, second block (decoder-encoder MHA)

    Inputs: 
    pred_sentence - string
    input_sentence - string
    attention_weights - dictionary; there are X decoder layers, and 2 blocks per layer, so there are 2*X keys
    """
    
    input_encoding = de_tokenizer.encode(input_sentence)
    input_tokens = input_encoding.tokens
    len_input = len(input_tokens)

    output_encoding = en_tokenizer.encode(pred_sentence)
    output_tokens = output_encoding.tokens
    # Want to ignore "start_" token, since it is not included in the attention plot
    output_tokens = output_tokens[1:]

    # retrieve final key of dictionary – this corresponds to the final decoder layer, block 2 (decoder-encoder mha)
    final_block_decenc = list(attention_weights.keys())[-1]
    
    # retrieve final attention block
    attention_block = attention_weights[final_block_decenc]     # attention_weights is a dictionary
                                                                # attention_block = (1, num_heads, len(pred_sentence), Tx)
    attention_block_squeezed = tf.squeeze(attention_block, 0)   # attention_block_squeezed = (num_heads, len(pred_sentence), Tx)
    num_heads = attention_block_squeezed.shape[0]

    fig, ax = plt.subplots(num_heads, 1, figsize = (14,14))
    
    for head in range (num_heads):
        temp_attn_block = attention_block_squeezed[head]
        temp_attn_block = np.around(temp_attn_block, 3)
        ax[head].set_title(f'Head {head + 1}')
        sns.heatmap(temp_attn_block[:,:len_input], ax = ax[head], cmap = 'hot', annot = True, xticklabels = input_tokens, yticklabels = output_tokens)
    

    plt.tight_layout()
    plt.show()



