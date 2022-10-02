import numpy as np
import tensorflow as tf
# When running Flask app, set the back-end to a non-interactive one so that the server
# doesn't try to create GUI windows that will never be seen:
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import seaborn as sns

from tokenizer_helpers import preprocess_sentence


def translate (source_sentence, transformer, max_target_length, de_tokenizer, en_tokenizer):
    """
    Translates input sentence (German) into English

    Arguments: 
        source_sentence -- string
        transformer -- trained transformer with loaded weights
        max_target_length -- int
        de_tokenizer, en_tokenizer -- loaded tokenizers

    Returns:
        pred_sentence, source_sentence, attention_weights
    """

    # preprocess sentence
    source_sentence = preprocess_sentence(source_sentence)

    # convert words in sentence to tokens; convert to 2D tensor 
    source_sequence = de_tokenizer.encode(source_sentence).ids
    source_sequence = tf.convert_to_tensor(source_sequence)
    source_sequence = tf.expand_dims(source_sequence,0)         # source_sequence = (1, Tx)

    start = en_tokenizer.get_vocab()['start_']
    end = en_tokenizer.get_vocab()['_end']

    # Initialise output_array with "start_" token
    output_array = tf.Variable([start], dtype = tf.int64 )
    output_array = tf.expand_dims(output_array, 0)

    # Initialise predicted transltion as empty string
    pred_sentence = ''
    for t in range (max_target_length):

        output = output_array
        
        # pass through source_sequence and current output through transformer
        predictions, _ = transformer([source_sequence, output], training = False)       # (batch_size, tar_seq_len, target_vocab_size)
        
        # select final time-step of predictions
        predictions = predictions[:, -1:, :]                                            
        
        # fetch index of the final time-step 
        predicted_id = tf.argmax(predictions, axis=-1)
        predicted_id_int = int(predicted_id)
    
        # decode the index and append to pred_sentence
        pred_sentence += en_tokenizer.decode([predicted_id_int])
        # concatenate to output_array
        output_array = tf.concat([output_array, predicted_id], axis = 1)

        # if the predicted_id is the token for "_end", then break loop
        if predicted_id == end:
            break

    
    # get attention_weights for final time-step (just before "_end" is predicted)
    _, attention_weights = transformer([source_sequence, output_array[:, :-1]], training = False)
    
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
    
    # pick random integer from range corresponding to length of source_tensor
    k = np.random.randint(len(source_tensor))
    # select random input and output
    random_input = source_tensor[k]               # random_input = (Tx,)
    random_output = target_tensor[k]

    # convert integers into sentence
    random_input = de_tokenizer.decode(random_input.astype(int))        # random_input does not include "start_" and "_end"
    
    # pass random_input (as string/sentence) through "translate" function
    pred_sentence, original_sentence, attention_weights = translate(random_input, transformer, 
                                                                    max_target_length,
                                                                    de_tokenizer, en_tokenizer)

    print(f'Input: {original_sentence}')
    print(f'Predicted translation: {pred_sentence}')
    
    # ground-truth translation
    true_translation = en_tokenizer.decode(random_output.astype(int))
    
    print(f'Actual translation: {true_translation}')
    return pred_sentence, original_sentence, true_translation, attention_weights


def plot_attention_weights(pred_sentence, input_sentence, attention_weights, de_tokenizer, en_tokenizer):
    """
    Plots the scores from the final decoder layer's Attention Block 2 (decoder-encoder MHA)

    Inputs: 
    pred_sentence - string
    input_sentence - string
    attention_weights - dictionary; there are 2 blocks per layer, so there are (2 * num_layers) keys
    """
    
    input_encoding = de_tokenizer.encode(input_sentence)
    input_tokens = input_encoding.tokens
    len_input = len(input_tokens)

    output_encoding = en_tokenizer.encode(pred_sentence)
    output_tokens = output_encoding.tokens
    # Want to ignore "start_" token, since it is not included in the attention plot
    output_tokens = output_tokens[1:]
    len_output = len(output_tokens)

    # retrieve final key of dictionary – this corresponds to the final decoder layer, block 2 (decoder-encoder mha)
    final_block_decenc = list(attention_weights.keys())[-1]
    
    # retrieve final attention block
    attention_block = attention_weights[final_block_decenc]     # attention_weights is a dictionary
                                                                # attention_block = (1, num_heads, len(pred_sentence), Tx)
    attention_block_squeezed = tf.squeeze(attention_block, 0)   # attention_block_squeezed = (num_heads, len(pred_sentence), Tx)
    num_heads = attention_block_squeezed.shape[0]

    fig, ax = plt.subplots(num_heads, 1, figsize = (len_input, len_output*2))
    
    for head in range (num_heads):
        temp_attn_block = attention_block_squeezed[head]
        temp_attn_block = np.around(temp_attn_block, 3)
        ax[head].set_title(f'Head {head + 1}')
        sns.heatmap(temp_attn_block[:,:len_input], ax = ax[head], cmap = 'hot', annot = True, xticklabels = input_tokens, yticklabels = output_tokens)
    

    plt.tight_layout()
    plt.savefig('./static/attention_plots.png')
    plt.show()



