import tensorflow as tf
import numpy as np
import pdb

from .conv_layer import ConvBaseLayer

class CNN_CTC(tf.keras.Model):
    def __init__(self, hparams):
        super(CNN_CTC, self).__init__()
        self.hparams = hparams
        self.conv_layer = ConvBaseLayer(hparams)
        self.embedding = tf.keras.layers.Dense(self.hparams.charset_size)
        self.softmax = tf.keras.layers.Softmax()


    def forward(self, logits, logits_length, targets, targets_length):
        # loss = tf.nn.ctc_loss(labels=targets, logits=logits, label_length=targets_length, logit_length=logits_length, \
        #                       logits_time_major=False)
        targets_length = tf.expand_dims(targets_length, 1)
        loss = tf.keras.backend.ctc_batch_cost(y_true=targets, y_pred=logits, label_length=targets_length, input_length=logits_length)
        loss = tf.math.reduce_sum(loss)
        return loss
        

    def infer(self, logits, logits_length):
        # logits = tf.transpose(logits, [1, 0, 2])
        # decoded = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=logits_length)
        logits_length = tf.squeeze(logits_length, axis=1)
        decoded = tf.keras.backend.ctc_decode(logits, input_length=logits_length, greedy=True)
        return decoded


    def call(self, inputs, targets=None, targets_length=None):
        conv_out = self.conv_layer(inputs)
        embedding = self.embedding(conv_out)
        logits = self.softmax(embedding)
        logits_length = tf.shape(logits)[1] * tf.ones((tf.shape(logits)[0], 1), tf.int32)
        if targets == None:
            return self.infer(logits, logits_length)
        else:
            return self.forward(logits, logits_length, targets, targets_length)