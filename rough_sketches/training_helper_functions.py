import pandas as pd
import numpy as np
import string

import matplotlib.pyplot as plt
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

def loss_function(real, pred, loss_object):
                                                            # real = (m, Ty)
                                                            # pred = (m, Ty, num_tokens_target)

  mask = tf.math.logical_not(tf.math.equal(real, 0))        # want to select only non-zero values
                                                            # mask = (m, Ty), and is "True" for non-zero values

  loss_ = loss_object(real, pred)                           # compute loss for each time-step
                                                            # loss = (m, Ty)

  mask = tf.cast(mask, dtype=loss_.dtype)                   
  loss_ *= mask                                             # only count loss from non-zero values

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)           # divide sum(loss) by number of non-zero values



def accuracy_function(real, pred):                          # pred = (m, Ty, num_tokens_target)
  
  accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), tf.int32))      # accuracies = (m, Ty) -- binary values

  mask = tf.math.logical_not(tf.math.equal(real, 0))        # mask = (m, Ty) -- boolean values
  accuracies = tf.math.logical_and(mask, accuracies)        # suppress values where real value is 0

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)      # divide sum of 1s in "accuracies" by sum of 1s in "mask"


def compute_test_metrics(inp, tar, transformer, loss_object):
    # inp = (m, Tx)
    # tar = (m, Ty)

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _ = transformer (inputs = (inp, tar_inp), training = False)
    test_loss = loss_function(tar_real, predictions, loss_object)
    test_acc = accuracy_function(tar_real, predictions)

    return test_loss, test_acc