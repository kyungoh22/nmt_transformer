
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Embedding, Dense


def get_angles(pos, k, d):
    """
    Get the angles for the positional encoding
    
    Arguments:
        pos        -- Column vector containing the positions [[0], [1], ...,[N-1]]
        k          -- Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d (int)    -- Encoding size
    
    Returns:
        angles     -- numpy array of shape (pos, d) 
    """

    # Get i from dimension span k
    i = k // 2
    # Calculate the angles using pos, i and d
    angles = pos / (10000 ** (2*i/d))
    
    return angles

def positional_encoding(positions, d):
    """
    Computes an array with all the positional encodings 
    
    Arguments:
        positions (int) -- Maximum number of positions to be encoded 
        d (int)         -- Encoding size 
    
    Returns:
        pos_encoding    -- (1, position, d_model) A matrix with the positional encodings
    """

    # initialize a matrix angle_rads of all the angles 
    angle_rads = get_angles(np.array(list(range(positions))).reshape((positions,1)),
                            np.array(list(range(d))).reshape((1,d)),
                            d)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    # Add new dimension to enable broadcasting later in the Encoder
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        seq -- (batch_size, seq_len) 
    
    Returns:
        mask -- binary tensor with shape (batch_size, 1, seq_len)
    """    
    seq = 1 - tf.cast(tf.math.equal(seq, 0), tf.float32)
  
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, :] 

def create_look_ahead_mask(sequence_length):
    """
    Returns a lower triangular matrix filled with ones
    
    Arguments:
        sequence_length -- matrix size
    
    Returns:
        mask -- (sequence_length, sequence_length) tensor
    """
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    
    #syntax for band_part: tf.linalg.band_part (a, numLower, numUpper)
    #a: tf.tensor to be passed.
    #numLower: Number of subdiagonals to keep. If negative, keep entire lower triangle
    #numUpper: Number of subdiagonals to keep. If negative, keep entire upper triangle
    
    return mask 

def FullyConnected(embedding_dim, fully_connected_dim):
    """
    Returns Dense layer with relu activation (fully_connected_dim) followed by another Dense layer (embedding_dim)
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    """
    The encoder layer: 
    1) multi-head self-attention 
    2) residual connection + normalisation
    3) fully connected feed-forward network
    4) residual connection + normalisation
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()
        
        # MultiHeadAttention
        # Note: "key_dim" = size of each attention head for Q and K.
        # We want this to be the same as "embedding_dim", since we are stacking multiple MHA layers
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
            training -- Boolean, set to True to activate the training mode for the dropout layers
            mask -- Boolean mask to ignore the padding in the input

        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """

        # calculate self-attention using mha
        # Keras automatically adds Dropout if the Dropout parameter is non-zero during training
        attn_output = self.mha(x,x,x,mask)          # (batch_size, input_seq_len, embedding_dim)
        
        # residual connection + layer normalization to get 
        # the output of the multi-head attention layer 
        out1 = self.layernorm1(x + attn_output)     # (batch_size, input_seq_len, embedding_dim)

        # pass through ffn / Dense layer
        ffn_output = self.ffn(out1)                 # (batch_size, input_seq_len, embedding_dim)
        
        # Dropout layer; don't forget training parameter
        ffn_output =  self.dropout_ffn(ffn_output, training = training)
        
        # residual connection + layer normalization to get 
        # the output of the encoder layer (~1 line)
        encoder_layer_out = self.layernorm2(ffn_output + out1)  # (batch_size, input_seq_len, embedding_dim)
                
        return encoder_layer_out


class Encoder(tf.keras.layers.Layer):
    """
    Pass the input through an embedding layer, and then add positional encodings. 
    Then pass the output through a stack of encoder layers
        
    """  
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)

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
            x               -- Tensor of shape (batch_size, input_seq_len)
            training        -- Boolean, set to true to activate the training mode for dropout layers
            mask            -- Boolean mask to ignore the padding in the input

        Returns:
            encoder_out            -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """
        
        seq_len = tf.shape(x)[1]
        
        # Pass input through the Embedding layer
        x = self.embedding(x)                       # (batch_size, input_seq_len, embedding_dim)
        # Scale embedding by the square root of the encoding dimensions (embedding_dim)
        x *= (tf.math.sqrt(tf.cast(self.embedding_dim, dtype=tf.float32)))
        # Add the position encoding to embedding
        x += self.pos_encoding[:, :seq_len, :]
        # Pass the encoded embedding through a dropout layer; note the "training" parameter
        x = self.dropout(x, training = training)
        # Pass the output through the stack of encoding layers; note the "training" parameter
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask= mask, training = training)

        encoder_out = x
        return encoder_out                                    # (batch_size, input_seq_len, embedding_dim)


class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer: 
    1) MHA Block 1 -- uses self-attention on input for target language
    2) MHA Block 2 -- uses output from Block 1 with output from Encoder 
    3) Fully connected layer 
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
            x               -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            enc_output      -- Tensor of shape(batch_size, input_seq_len, embedding_dim)
            training        -- Boolean, set to True to activate the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask    -- Boolean mask for the second multi-head attention layer
        Returns:
            decoder_layer_out   -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            attn_weights_block1 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
            attn_weights_block2 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
        """

        # enc_output.shape == (batch_size, input_seq_len, embedding_dim)
        
        # BLOCK 1
        # calculate self-attention and return attention scores as attn_weights_block1.
        # Dropout will be applied during training.
        mult_attn_out1, attn_weights_block1 = self.mha1(x, x, x, 
                                                        look_ahead_mask, training = training, 
                                                        return_attention_scores=True)  # (batch_size, target_seq_len, embedding_dim)
        
        # residual connection + layer normalization 
        Q1 = self.layernorm1(mult_attn_out1 + x)

        # BLOCK 2
        # Calculate self-attention using the output from Block 1 as Q, 
        # and the encoder output as K and V. 
        # Dropout will be applied during training
        # Return attention scores as attn_weights_block2 
        mult_attn_out2, attn_weights_block2 = self.mha2(Q1, enc_output, enc_output, 
                                                        padding_mask, training = training, 
                                                        return_attention_scores=True)  # (batch_size, target_seq_len, embedding_dim)
        
        # residual connection + layer normalization 
        mult_attn_out2 = self.layernorm2(mult_attn_out2 + Q1)  # (batch_size, target_seq_len, embedding_dim)
                
        # BLOCK 3
        # pass the output of BLOCK 2 through a ffn
        ffn_output = self.ffn(mult_attn_out2)  # (batch_size, target_seq_len, embedding_dim)
        
        # apply a dropout layer to the ffn output; note the "training" parameter
        ffn_output = self.dropout_ffn(ffn_output, training = training)
        
        # residual connection + layer normalization 
        decoder_layer_out = self.layernorm3(ffn_output + mult_attn_out2)  # (batch_size, target_seq_len, embedding_dim)
        
        return decoder_layer_out, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    """
    Pass the target input through an embedding layer, and then add positional encodings. 
    Then pass the output through a stack of decoder layers   
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
            x               -- Tensor of shape (batch_size, target_seq_len)
            enc_output      -- Tensor of shape(batch_size, input_seq_len, embedding_dim)
            training        -- Boolean, set to True to activate the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask    -- Boolean mask for the second multihead attention layer
        Returns:
            decoder_out       -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            attention_weights -- Dictionary of tensors containing all the attention weights
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        # create word embeddings 
        x = self.embedding(x)  # (batch_size, target_seq_len, embedding_dim)
        
        # scale embeddings by the square root of the embedding dimension
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, dtype = tf.float32 ))

        # calculate positional encodings and add to word embedding
        x += self.pos_encoding[:, :seq_len, :]
             
        # apply a dropout layer to x; note the "training" parameter
        x = self.dropout(x, training = training)

        # pass x through a stack of decoder layers and update attention_weights
        for i in range(self.num_layers):
            # pass x and the encoder output through a stack of decoder layers and 
            # save the attention weights of block 1 and 2 
            
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)         # x = (batch_size, target_seq_len, embedding_dim)

            # update attention_weights dictionary with the attention weights of block 1 and block 2
            attention_weights[f'decoder_layer{i+1}_block1_self_att'] = block1
            attention_weights[f'decoder_layer{i+1}_block2_decenc_att'] = block2
        
        decoder_out = x
        return decoder_out, attention_weights


class Transformer(tf.keras.Model):
    """
    Complete Transformer with an Encoder and a Decoder
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

        """
        Arguments:
            inp -- (batch_size, input_seq_len)
            tar -- (batch_size, target_seq_len)
        
        Returns: 
            enc_padding_mask -- (batch_size, 1, input_seq_len)
            look_ahead_mask -- (target_seq_len, target_seq_len)
        """
        enc_padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tar.shape[1])
        return enc_padding_mask, look_ahead_mask

    def call(self, inputs, training):
        # K:
        # Check dimensions of input_sentence and output_sentence
        # Should be shape 2, no? We're passing in 2D array of integers
        """
        Forward pass for the entire Transformer

        Arguments: 
            inputs = (input_sentence, output_sentence)
                        input_sentence = (batch_size, input_seq_len)
                        output_sentence = (batch_size, target_seq_len)

        Returns:
            final_output        -- (batch_size, tar_seq_len, target_vocab_size)
            attention_weights   -- Dictionary of tensors containing all the attention weights for the decoder.
                                    Each tensor is of shape: (1, num_heads, target_seq_len, input_seq_len)
        """
        
        input_sentence, output_sentence = inputs
        enc_padding_mask, look_ahead_mask = self.create_masks(input_sentence, output_sentence)
        
        # call self.encoder to get the encoder output
        enc_output = self.encoder(input_sentence, training, enc_padding_mask)  # (batch_size, inp_seq_len, embedding_dim)
        
        # call self.decoder to get the decoder output
        # note that "enc_padding_mask" is used in decoder as well, since the encoder output is used as K and V.
        dec_output, attention_weights = self.decoder(output_sentence, 
                                                        enc_output, 
                                                        training, 
                                                        look_ahead_mask, 
                                                        enc_padding_mask)  # dec_output = (batch_size, tar_seq_len, embedding_dim)
        
        # pass decoder output through a linear layer and softmax
        final_output = self.final_layer(dec_output)                         # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


