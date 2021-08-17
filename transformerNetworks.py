"""
This file was inspired and modified from the following resources:

1. The original transformer paper "Attention is all you need" https://arxiv.org/abs/1706.03762

2. Tensorflow tutorial https://www.tensorflow.org/text/tutorials/transformer#positional_encoding
"""

import numpy as np
import tensorflow as tf


class MHA(tf.keras.layers.Layer):
    """
    Multi-Head Attention
    """

    def __init__(self, d_model, num_heads):
        """
        Arguments:
            d_model - the dimesion of the input embedding. default: 512
            num_heads - the number of the heads. default: 8
        """
        super(MHA, self).__init__()

        assert (d_model % num_heads == 0)

        self.num_heads = num_heads
        self.d_model = d_model

        # dimension of the head (i.e. d_v, d_q)
        self.d_head = d_model // num_heads 
        # dense layers for Q, K, and V
        self.Q_dense = tf.keras.layers.Dense(d_model)
        self.K_dense = tf.keras.layers.Dense(d_model)
        self.V_dense = tf.keras.layers.Dense(d_model)
        # dense layer
        self.final_dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        """
        Split the tensor x into num_heads heads
        input
            x - input tensor with shape (batch_size, seq_len, d_model)
            batch_size
        """
        # find batch_size and seq_len
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        # reshape the tensor
        # from (batch_size, seq_len, d_model) to (batch_size, seq_len, num_heads, d_head)
        x = tf.reshape(x, [batch_size, seq_len, self.num_heads, self.d_head])
        # transpose to (batch_size, num_heads, seq_len, d_head)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x
    
    def scaled_dot_product_attention(self, Q, K, V, mask):
        """
        Scaled dot-product attention
        Inputs
            Q - query tensor. {batch_size, num_heads, seq_len, d_head}
            K - key tensor. {batch_size, num_heads, seq_len, d_head}
            V - value tensor. {batch_size, num_heads, seq_len, d_head}
            mask - mask. {batch_size, num_heads, seq_len, seq_len}
        Outputs
            attn - attention
            V_weights - weights of the values
        """
        # d_k
        d_k = tf.cast(self.d_head, tf.float32)
        # multiply Q and K' and divided by sqrt(d_k)
        QK = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)
        # apply the mask
        if mask is not None:
            QK += (-1e9*mask)
        # apply softmax
        V_weights = tf.nn.softmax(QK, axis=-1)
        # multiply with V
        attn = tf.matmul(V_weights, V)
        return attn, V_weights

    def call(self, Q, K, V, mask):
        """
        Multi-head attention
        """
        # batch size
        batch_size = tf.shape(Q)[0]
        # pass Q/K/V into their own dense layers
        Q = self.Q_dense(Q)
        K = self.K_dense(K)
        V = self.V_dense(V)
        # split Q/K/V into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        # scaled dot-product attention
        attn, V_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        # transpose attn: (batch_size, num_heads, seqlen, d_head) -> (batch_size, seqlen, num_heads, d_head)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        # concatenate attentions from all heads (combining the last 2 dimemsions)
        concat_attn = tf.reshape(attn, (batch_size, -1, self.d_model))
        # pass through the final dense layer
        output = self.final_dense(concat_attn)
        return output, V_weights
		


class EncoderLayer(tf.keras.layers.Layer):
    """
    A single encoder layer
    """
    def __init__(self, d_model=512, num_heads=8, d_pwff=2048, dropout=0.1):
        """
        Arguments
            d_model - the dimension of input embedding (default=512)
            num_heads - number of the heads (default=8)
            d_pwff - dimension of position-wise feed-forward inner layer (default=2048)
            dropout - dropout rate (default=0.1)
        """
        super(EncoderLayer, self).__init__()
        # multi-head attention network
        self.MHA = MHA(d_model=d_model, num_heads=num_heads)
        # position-wise feed forward
        self.PWFF = tf.keras.Sequential([tf.keras.layers.Dense(d_pwff, activation='relu'),
                                         tf.keras.layers.Dense(d_model)])
        # norm layer 1 (after MHA)
        self.Norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # norm layer 2 (after PWFF)
        self.Norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # dropout 1
        self.Dropout1 = tf.keras.layers.Dropout(dropout)
        # dropout 2
        self.Dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        """
        Inputs
            x - input tensor
            training - True during training, False during eval
            mask - look ahead mask
        """
        # multi-head attention stage
        attn, _ = self.MHA(x, x, x, mask)
        # dropout 1
        attn = self.Dropout1(attn, training=training)
        # norm 1
        normed_attn = self.Norm1(x + attn)
        # position-wise feed-forward
        pwff = self.PWFF(normed_attn)
        # dropout 2
        pwff = self.Dropout2(pwff, training=training)
        # norm 2
        output = self.Norm2(normed_attn + pwff)
        return output


class EncoderBlock(tf.keras.layers.Layer):
    """
    The encoder block of Transformer
    """
    def __init__(self, num_layers=6, d_model=512, num_heads=8, d_pwff=2048, input_vocab_size=32000, max_seq_len=1000, dropout=0.1):
        """
        Arguments
            num_layers - number of encoder layers (default=6)
            d_model - dimension of input embedding (default=512)
            num_heads - number of heads (default=8)
            d_pwff - dimension of pwff inner layers (default=2048)
            input_vocab_size - size of the input vocabulary (default=32000)
            max_seq_len - length of each input sequence (default=1000)
            dropout - dropout rate (default=0.1)
        """
        super(EncoderBlock, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        # input embedding layer
        self.Embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # positional encoding
        self.PE = Positional_encoding(max_seq_len, d_model)
        # encoder layers
        self.EncoderLayers = [EncoderLayer(d_model, num_heads, d_pwff, dropout) for _ in range(num_layers)]
        # dropout (after positional encoding)
        self.Dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        # seqlen
        seqlen = tf.shape(x)[1]
        # apply embedding, multiply by sqrt(d_model)
        x = self.Embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # add positional encoding
        x += self.PE[:, :seqlen, :]
        # dropout
        x = self.Dropout(x, training=training)
        # loop all encoder layers
        for i in range(self.num_layers):
            x = self.EncoderLayers[i](x, training, mask)
        return x



class DecoderLayer(tf.keras.layers.Layer):
    """
    A single decoder layer
    """
    def __init__(self, d_model=512, num_heads=8, d_pwff=2048, dropout=0.1):
        """
        Arguments
            d_model - the dimension of input embedding (default=512)
            num_heads - number of the heads (default=8)
            d_pwff - dimension of position-wise feed-forward inner layer (default=2048)
            dropout - dropout rate (default=0.1)
        """
        super(DecoderLayer, self).__init__()
        # first multi-head attention network
        self.MHA1 = MHA(d_model, num_heads)
        # second multi-head attention network
        self.MHA2 = MHA(d_model, num_heads)
        # position-wise feed forward network
        self.PWFF = tf.keras.Sequential([tf.keras.layers.Dense(d_pwff, activation='relu'),
                                         tf.keras.layers.Dense(d_model)])
        # norm layer 1 (after 1st MHA)
        self.Norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # norm layer 2 (after 2nd MHA)
        self.Norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # norm layer 3 (after PWFF)
        self.Norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # dropout 1
        self.Dropout1 = tf.keras.layers.Dropout(dropout)
        # dropout 2
        self.Dropout2 = tf.keras.layers.Dropout(dropout)
        # dropout 3
        self.Dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, encoder_output, training, padding_mask, lookahead_mask):
        """
        Inputs
            x - input tensor
            encoder_output - output of the encoder block
            training - True during training, False during eval
            padding_mask - padding mask
            lookahead_mask - look ahead mask
        Outputs
            output - output of the decoder layer
        """
        # 1st multi-head attention -> dropout -> residual & norm
        attn1, _ = self.MHA1(x, x, x, lookahead_mask)
        attn1 = self.Dropout1(attn1, training=training)
        normed_attn1 = self.Norm1(x + attn1)

        # 2nd multi-head attention -> dropout -> residual & norm
        attn2, _ = self.MHA2(normed_attn1, encoder_output, encoder_output, padding_mask)
        attn2 = self.Dropout2(attn2, training=training)
        normed_attn2 = self.Norm2(normed_attn1 + attn2)

        # pwff -> dropout -> residual & norm
        pwff = self.PWFF(normed_attn2)
        pwff = self.Dropout3(pwff, training=training)
        output = self.Norm3(normed_attn2 + pwff)

        return output


class DecoderBlock(tf.keras.layers.Layer):
    """
    The decoder block of Transformer
    """
    def __init__(self, num_layers=6, d_model=512, num_heads=8, d_pwff=2048, target_vocab_size=32000, max_seq_len=1000, dropout=0.1):
        """
        Arguments
            num_layers - number of encoder layers (default=6)
            d_model - dimension of target embedding (default=512)
            num_heads - number of heads (default=8)
            d_pwff - dimension of pwff inner layers (default=2048)
            target_vocab_size - size of the target vocabulary (default=32000)
            max_seq_len - length of each target sequence (default=1000)
            dropout - dropout rate (default=0.1)
        """
        super(DecoderBlock, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        # target embedding layer
        self.Embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        # positional encoding
        self.PE = Positional_encoding(max_seq_len, d_model)
        # decoder layers
        self.DecoderLayers = [DecoderLayer(d_model, num_heads, d_pwff, dropout) for _ in range(num_layers)]
        # initial dropout (after positional encoding)
        self.Dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, encoder_output, training, padding_mask, lookahead_mask):
        # seqlen
        seqlen = tf.shape(x)[1]
        # apply embedding, multiply by sqrt(d_model)
        x = self.Embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # add positional encoding
        x += self.PE[:, :seqlen, :]
        # dropout
        x = self.Dropout(x, training=training)
        # loop all decoder layers
        for i in range(self.num_layers):
            x = self.DecoderLayers[i](x, encoder_output, training, padding_mask, lookahead_mask)
        return x
		
		
class Transformer(tf.keras.Model):

    def __init__(self, num_layers=6, d_model=512, num_heads=8, d_pwff=2048,
                 input_vocab_size=32000, target_vocab_size=32000,
                 input_maxlen=1000, target_maxlen=1000, dropout=0.1):
        
        super(Transformer, self).__init__()

        self.EncoderBlock = EncoderBlock(num_layers, d_model, num_heads, d_pwff, input_vocab_size, input_maxlen, dropout)
        self.DecoderBlock = DecoderBlock(num_layers, d_model, num_heads, d_pwff, target_vocab_size, target_maxlen, dropout)
        self.FinalDense = tf.keras.layers.Dense(target_vocab_size)

    def create_padding_mask(self, padded_seq):
        """
        Create the padding mask
        The value is 1 if there's padding. The value is 0 if it's not padding
        
        Inputs
            padded_seq - padded input sequence. shape = (batch_size, seq_len)
        Output
            padding_mask - the padding mask. shape = (batch_size, 1, 1, seq_len)
        """
        # set the mask to True if it's the padding
        padding_mask = tf.math.equal(padded_seq, 0)
        # convert booleans to float32
        padding_mask = tf.cast(padding_mask, tf.float32)
        # add extra dimensions -> (batch_size, 1, 1, seq_len)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

        return padding_mask

    def create_lookahead_mask(self, seq_len):
        """
        Create the lookhead mask. The mask value is 1 on future inputs
        Input
            seq_len: the length of the input sequence
        Ouput
            lookahead_mask: the mask with shape (seq_len, seq_len), whose upper triangular part are all zeros
        """
        # set the upper triangular part is 1 and subtract it from 1
        lookahead_mask = 1- tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        return lookahead_mask

    def call(self, inputs_targets, training):

        inputs, targets = inputs_targets

        # mask for encoder
        encoder_mask = self.create_padding_mask(inputs)
        # 1st mask for decoder
        decoder_mask1 = tf.maximum(self.create_padding_mask(targets), self.create_lookahead_mask(tf.shape(targets)[1]))
        # 2nd mask for decoder
        decoder_mask2 = self.create_padding_mask(inputs)

        # encoder block
        encoder_output = self.EncoderBlock(inputs, training, encoder_mask)
        # decoder block
        decoder_output = self.DecoderBlock(targets, encoder_output, training, decoder_mask2, decoder_mask1)
        # final dense
        outputs = self.FinalDense(decoder_output)

        return outputs