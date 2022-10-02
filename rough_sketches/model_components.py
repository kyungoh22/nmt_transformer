import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Concatenate, LSTM, Embedding, Dense, MultiHeadAttention, LayerNormalization, Dropout
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.initializers import Constant

from sklearn.model_selection import train_test_split
import re
import os
import io
import time

from tokenizers import Tokenizer, ByteLevelBPETokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# def preprocess_sentence(sentence):
#     sentence = sentence.lower()
#     sentence = re.sub("'", '', sentence)
#     sentence = sentence.replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe').replace('ß', 'ss')
#     exclude = set(string.punctuation)
#     sentence = ''.join(ch for ch in sentence if ch not in exclude)
#     sentence = 'start_ ' + sentence + ' _end'
#     sentence = sentence.encode("ascii", "ignore")
#     sentence = sentence.decode()
#     return sentence

def preprocess_sentence(sentence):
    
    # replace umlauts and sharp s
    sentence = sentence.replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe').replace('ß', 'ss')
    sentence = sentence.replace('Ü', 'Ue').replace('Ä', 'ae').replace('Ö', 'Oe')
    # add space before and after punctuation
    sentence = re.sub(r"\b([.,!?()':;])\b", ' \g<0> ', sentence)

    # collapse multiple spaces into single space
    sentence = re.sub('\s{2,}', ' ', sentence)
    sentence = ''.join(ch for ch in sentence)
    # encode ascii and ignore unencodable characters; this removes non-english characters
    sentence = sentence.encode("ascii", "ignore")
    # decode back to utf-8 (which is default of the "encoding" parameter)
    sentence = sentence.decode()
    return sentence

def get_angles(pos, k, d):
    """
    Get the angles for the positional encoding
    
    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d(integer) -- Encoding size
    
    Returns:
        angles -- (pos, d) numpy array 
    """

    # Get i from dimension span k
    i = k // 2
    # Calculate the angles using pos, i and d
    angles = pos / (10000 ** (2*i/d))
    
    return angles

def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int) -- Maximum number of positions to be encoded 
        d (int) -- Encoding size 
    
    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """

    # initialize a matrix angle_rads of all the angles 
    angle_rads = get_angles(np.array(list(range(positions))).reshape((positions,1)),
                            np.array(list(range(d))).reshape((1,d)),
                            d
                            )
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    print(pos_encoding.shape)
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        seq -- (batch_size, seq_len) 
    
    Returns:
        mask -- (batch_size, 1, seq_len) binary tensor
    """    
    seq = 1 - tf.cast(tf.math.equal(seq, 0), tf.float32)
  
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, :] 

def create_look_ahead_mask(sequence_length):
    """
    Returns a lower triangular matrix filled with ones
    
    Arguments:
        sequence_length -- matrix size
    
    Returns:
        mask -- (size, size) tensor
    """
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    
    #K: band_part: tf.linalg.bandPart (a, numLower, numUpper)
    #a: it is tf.tensor to be passed.
    #numLower: Number of subdiagonals to keep. If negative, keep entire lower triangle
    #numUpper: Number of subdiagonals to keep. If negative, keep entire upper triangle
    
    return mask 

def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
    ])



class EncoderLayer(tf.keras.layers.Layer):
    """
    The encoder layer is composed of a multi-head self-attention mechanism,
    followed by a simple, positionwise fully connected feed-forward network. 
    This archirecture includes a residual connection around each of the two 
    sub-layers, followed by layer normalization.
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()
        #K: Note: key_dim refers to the final dimensiosn AFTER concatenating the multiple heads.
        # And we want this to be the same as embedding_dim, since we are stacking multiple MHA layers
        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim,
                                      dropout=dropout_rate)

        self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = Dropout(dropout_rate)
    
    def call(self, x, training, mask):


        """
        Forward pass for the Encoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """
        
        # calculate self-attention using mha(~1 line).
        # Dropout is added by Keras automatically if the dropout parameter is non-zero during training

        attn_output = self.mha(x,x,x,mask) # Self attention (batch_size, input_seq_len, embedding_dim)
        
        # apply layer normalization on sum of the input and the attention output to get the  
        # output of the multi-head attention layer (~1 line)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, embedding_dim)

        # pass the output of the multi-head attention layer through a ffn (~1 line)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embedding_dim)
        
        # apply dropout layer to ffn output during training (~1 line)
        # K: this wasn't in assignment, but add training parameter 
        ffn_output =  self.dropout_ffn(ffn_output, training = training)
        
        # apply layer normalization on sum of the output from multi-head attention and ffn output to get the
        # output of the encoder layer (~1 line)
        encoder_layer_out = self.layernorm2(ffn_output + out1)  # (batch_size, input_seq_len, embedding_dim)
        
        
        return encoder_layer_out


class Encoder(tf.keras.layers.Layer):
    """
    The entire Encoder starts by passing the input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    encoder Layers
        
    """  
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.embedding_dim)


        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self.num_layers)]

        self.dropout = Dropout(dropout_rate)
        
    def call(self, x, training, mask):
        """
        Forward pass for the Encoder
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            out2 -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """
        #mask = create_padding_mask(x)
        seq_len = tf.shape(x)[1]
        
        # Pass input through the Embedding layer
        x = self.embedding(x)  # (batch_size, input_seq_len, embedding_dim)
        # Scale embedding by multiplying it by the square root of the embedding dimension
        x *= (tf.math.sqrt(tf.cast(self.embedding_dim, dtype=tf.float32)))
        # Add the position encoding to embedding
        x += self.pos_encoding[:, :seq_len, :]
        # Pass the encoded embedding through a dropout layer
        x = self.dropout(x, training = training) # K: Note: training = training, not True
        # Pass the output through the stack of encoding layers 
        # K: this wasn't in assignment, but add training parameter
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask= mask, training = training)

        return x  # (batch_size, input_seq_len, embedding_dim)


class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer is composed of two multi-head attention blocks, 
    one that takes the new input and uses self-attention, and the other 
    one that combines it with the output of the encoder. This is followed by a
    fully connected block. 

    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim,
                                      dropout=dropout_rate)

        self.mha2 = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim,
                                      dropout=dropout_rate)

        self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = Dropout(dropout_rate)
    
    # Note: pass in "look_ahead_mask" and "padding_mask" as parameters
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the Decoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            enc_output --  Tensor of shape(batch_size, input_seq_len, embedding_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            out3 -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            attn_weights_block1 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
            attn_weights_block2 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
        """

        # enc_output.shape == (batch_size, input_seq_len, embedding_dim)
        
        # BLOCK 1
        # calculate self-attention and return attention scores as attn_weights_block1.
        # Dropout will be applied during training (~1 line).
        mult_attn_out1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, training = training, return_attention_scores=True)  # (batch_size, target_seq_len, embedding_dim)
        
        # apply layer normalization (layernorm1) to the sum of the attention output and the input (~1 line)
        Q1 = self.layernorm1(mult_attn_out1 + x)

        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output. 
        # Dropout will be applied during training
        # Return attention scores as attn_weights_block2 (~1 line) 
        mult_attn_out2, attn_weights_block2 = self.mha2(Q1, enc_output, enc_output, padding_mask, training = training, return_attention_scores=True)  # (batch_size, target_seq_len, embedding_dim)
        
        # apply layer normalization (layernorm2) to the sum of the attention output and the output of the first block (~1 line)
        mult_attn_out2 = self.layernorm2(mult_attn_out2 + Q1)  # (batch_size, target_seq_len, embedding_dim)
                
        #BLOCK 3
        # pass the output of the second block through a ffn
        ffn_output = self.ffn(mult_attn_out2)  # (batch_size, target_seq_len, embedding_dim)
        
        # apply a dropout layer to the ffn output
        # K: don't forget training parameter
        ffn_output = self.dropout_ffn(ffn_output, training = training)
        
        # apply layer normalization (layernorm3) to the sum of the ffn output and the output of the second block
        out3 = self.layernorm3(ffn_output + mult_attn_out2)  # (batch_size, target_seq_len, embedding_dim)

        return out3, attn_weights_block1, attn_weights_block2



class Decoder(tf.keras.layers.Layer):
    """
    The entire Decoder starts by passing the target input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    decoder Layers

        
    """ 
    
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)

        self.dec_layers = [DecoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self.num_layers)]
        self.dropout = Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
        """
        Forward  pass for the Decoder
        
        Arguments:
            x -- Tensor of shape (batch_size, target_seq_len)
            enc_output --  Tensor of shape(batch_size, input_seq_len, embedding_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            x -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            attention_weights - Dictionary of tensors containing all the attention weights
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        # create word embeddings 
        x = self.embedding(x)  # (batch_size, target_seq_len, embedding_dim)
        
        # scale embeddings by multiplying by the square root of their dimension
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, dtype = tf.float32 ))

        # calculate positional encodings and add to word embedding
        x += self.pos_encoding[:, :seq_len, :]
             
        # apply a dropout layer to x
        # don't forget training parameter
        x = self.dropout(x, training = training)

        # use a for loop to pass x through a stack of decoder layers and update attention_weights (~4 lines total)
        for i in range(self.num_layers):
            # pass x and the encoder output through a stack of decoder layers and save the attention weights
            # of block 1 and 2 (~1 line)
            

            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)

            #update attention_weights dictionary with the attention weights of block 1 and block 2
            attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, embedding_dim)
        return x, attention_weights


class Transformer(tf.keras.Model):
    """
    Complete transformer with an Encoder and a Decoder


    """
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, 
               target_vocab_size, max_positional_encoding_input,
               max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               input_vocab_size=input_vocab_size,
                               maximum_position_encoding=max_positional_encoding_input,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers, 
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               target_vocab_size=target_vocab_size, 
                               maximum_position_encoding=max_positional_encoding_target,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.final_layer = Dense(target_vocab_size, activation='softmax')
    
    def create_masks(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tar.shape[1])
        #dec_padding_mask = create_padding_mask(tar)
        return enc_padding_mask, look_ahead_mask#, dec_padding_mask


    def call(self, inputs, training):
        # K:
        # Check dimensions of input_sentence and output_sentence
        # Should be shape 2, no? We're passing in 2D array of integers
        """
        Forward pass for the entire Transformer
       
        Returns:
            final_output -- # (batch_size, tar_seq_len, target_vocab_size)
            attention_weights - Dictionary of tensors containing all the attention weights for the decoder
                                each of type Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        
        """
        
        # input_sentence -- Tensor of shape (batch_size, input_seq_len)
        # output_sentence -- Tensor of shape (batch_size, target_seq_len)
        input_sentence, output_sentence = inputs
        
        # enc_padding_mask -- Boolean mask to ensure that the padding is not treated as part of the input
        # look_ahead_mask -- Boolean mask for the target_input
        # dec_padding_mask -- Boolean mask for the second multihead attention layer
        enc_padding_mask, look_ahead_mask = self.create_masks(input_sentence, output_sentence)
        #print(f'enc_padding_mask shape: {enc_padding_mask.shape}, look_ahead_mask shape: {look_ahead_mask.shape}, dec_padding_mask shape: {dec_padding_mask.shape}')

        # call self.encoder with the appropriate arguments to get the encoder output
        enc_output = self.encoder(input_sentence, training, enc_padding_mask)  # (batch_size, inp_seq_len, embedding_dim)
        
        # call self.decoder with the appropriate arguments to get the decoder output
        # dec_output.shape == (batch_size, tar_seq_len, embedding_dim)
        dec_output, attention_weights = self.decoder(output_sentence, enc_output, training, look_ahead_mask, enc_padding_mask)
        
        # pass decoder output through a linear layer and softmax (~2 lines)
        final_output = self.final_layer(dec_output) # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_train_tokenizer(file):

    """
    Creates token-index mapping as a json file

    Argument:
    file = list with single element – must end with .txt
    """
    # initialise tokenizer
    tokenizer = ByteLevelBPETokenizer()
    # train on file, adding two tokens
    tokenizer.train(files = file, 
                    special_tokens= ['_____', 'start_', '_end'])        # '_____' added as special token to ensure 'start_' has index 1
                                                                        # will later remove '_____' so that no token has index 0
    tokenizer.save(f'tokenizer_{file[0][:-4]}.json')                    # save as json instead of txt
    print(f'json file saved at: tokenizer_{file[0][:-4]}.json')


def load_tokenizer(json_file):
    """
    Loads pre-trained tokenizer from json file

    Argument: json_file path
    Returns: tokenizer, dict_word_index
    """

    tokenizer = Tokenizer.from_file(json_file)                          # load tokenizer

    # add "start_" and "_end" tokens to every tokenized sequence
    tokenizer.post_processor = TemplateProcessing(
    single="start_ $A _end",                                        # $A represents input sentence
    special_tokens=[
        ("start_", tokenizer.token_to_id("start_")), ("_end", tokenizer.token_to_id("_end")),],)
    
    dict_word_index = tokenizer.get_vocab()
    dict_word_index.pop('_____')            # remove '_____' from dictionary – now indexing starts from 1

    return tokenizer, dict_word_index