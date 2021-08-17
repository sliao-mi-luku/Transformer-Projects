"""
This file was inspired and modified from the following resources:

1. The original transformer paper "Attention is all you need" https://arxiv.org/abs/1706.03762

2. Tensorflow tutorial https://www.tensorflow.org/text/tutorials/transformer#positional_encoding
"""

import numpy as np
import tensorflow as tf


def Positional_encoding(num_positions, d_model):
    """
    Inputs
        num_positions: length of the data to infuse with positional information
        d_model: size of the input embedding (ex. vocabulary embedding size)
    Output
        PE: a (1 x num_position x d_model) tensor
    """
    # create an array for pos
    pos_vector = np.arange(num_positions)[:, np.newaxis] # (num_positions, 1)
    # create an array for i
    i_vector = np.arange(d_model)[np.newaxis, :] # (1, d_model)
    # positional encoding angle
    PE_matrix = pos_vector / np.power(10000, 2*(i_vector//2)/np.float32(d_model))
    # convert all even columns to sine
    PE_matrix[:, 0::2] = np.sin(PE_matrix[:, 0::2])
    # convert all odd columns to cosine
    PE_matrix[:, 1::2] = np.cos(PE_matrix[:, 1::2])
    # create a new dimension along axis 0
    PE_matrix = PE_matrix[np.newaxis, ...]
    # cast to float32
    PE_matrix = tf.cast(PE_matrix, dtype=tf.float32)

    return PE_matrix