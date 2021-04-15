import tensorflow as tf
import numpy as np
import pdb

from datetime import datetime
from .conv_layer import ConvBaseLayer


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, d_model=model_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(model_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output, _ = self.att(inputs, inputs, inputs, None)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class CNN_SA_CTC(tf.keras.Model):
    def __init__(self, hparams):
        super(CNN_SA_CTC, self).__init__()
        self.hparams = hparams
        self.conv_layer = ConvBaseLayer(hparams)
        self.input_embedding = tf.keras.layers.Dense(self.hparams.model_dim)
        self.pos_encoding = positional_encoding(self.conv_layer.conv_out_shape[1]*self.conv_layer.conv_out_shape[2], \
                                                self.hparams.model_dim)
        self.SA_blocks = [TransformerBlock(self.hparams.model_dim, self.hparams.num_heads, self.hparams.ff_dim, \
                          self.hparams.dropout_rate) for i in range(self.hparams.num_layers)]
        self.ff = tf.keras.layers.Dense(self.hparams.charset_size)
        self.softmax = tf.keras.layers.Softmax()

    def forward(self, logits, logits_length, targets, targets_length):
        targets_length = tf.expand_dims(targets_length, 1)
        loss = tf.keras.backend.ctc_batch_cost(y_true=targets, y_pred=logits, label_length=targets_length, input_length=logits_length)
        loss = tf.math.reduce_sum(loss)
        return loss

    def infer(self, logits, logits_length):
        logits_length = tf.squeeze(logits_length, axis=1)
        decoded = tf.keras.backend.ctc_decode(logits, input_length=logits_length, greedy=True)
        return decoded

    def call(self, inputs, targets=None, targets_length=None):
        training = True
        if targets == None:
            training = False

        # extract feature map
        start = datetime.now()
        conv_out = self.conv_layer(inputs)
        # print(datetime.now() - start)

        # add embedding and positional encoding
        start = datetime.now()
        seq_len = tf.shape(conv_out)[1]
        input_embedding = self.input_embedding(conv_out)  # (batch_size, input_seq_len, model_dim)
        input_embedding *= tf.math.sqrt(tf.cast(self.hparams.model_dim, tf.float32))
        input_embedding += self.pos_encoding[:, :seq_len, :]

        # pass self attention blocks
        sa_out = input_embedding
        for SA_block in self.SA_blocks:
            sa_out = SA_block(sa_out, training=training)
        # print(datetime.now() - start)
        # embed output of self attention blocks
        output_embedding = self.ff(sa_out)
        logits = self.softmax(output_embedding)
        logits_length = tf.shape(logits)[1] * tf.ones((tf.shape(logits)[0], 1), tf.int32)

        # run alignment
        if targets == None:
            return self.infer(logits, logits_length)
        else:
            return self.forward(logits, logits_length, targets, targets_length)