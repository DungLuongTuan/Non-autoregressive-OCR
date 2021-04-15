import efficientnet.tfkeras as efn
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import collections
import logging
import pdb
logging.basicConfig(level=logging.DEBUG)

class EncodeCordinate(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super(EncodeCordinate, self).__init__()
        _, self.h, self.w, _ = input_shape

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x, y = tf.meshgrid(tf.range(self.w), tf.range(self.h))
        w_loc = tf.one_hot(indices=x, depth=self.w)
        h_loc = tf.one_hot(indices=y, depth=self.h)
        loc = tf.concat([h_loc, w_loc], 2)
        loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])
        return tf.concat([inputs, loc], 3)


class SliceRNNInput(tf.keras.layers.Layer):
    def __init__(self):
        super(SliceRNNInput, self).__init__()

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        feature_size = tf.shape(inputs)[-1]
        return tf.reshape(inputs, [batch_size, -1, feature_size])

class ConvBaseLayer(tf.keras.layers.Layer):
    def __init__(self, hparams):
        super(ConvBaseLayer, self).__init__()
        self.hparams = hparams
        if hparams.base_model_name == 'InceptionV3':
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        if hparams.base_model_name == 'ResNet50V2':
            base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        if hparams.base_model_name == 'Xception':
            base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'InceptionResNetV2':
            base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB0':
            base_model = efn.EfficientNetB0(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB1':
            base_model = efn.EfficientNetB1(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB2':
            base_model = efn.EfficientNetB2(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB3':
            base_model = efn.EfficientNetB3(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB4':
            base_model = efn.EfficientNetB4(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB5':
            base_model = efn.EfficientNetB5(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB6':
            base_model = efn.EfficientNetB6(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB7':
            base_model = efn.EfficientNetB7(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]

        assert hparams.end_point in base_model_layers, "no {} layer in {}".format(hparams.end_point, hparams.base_model_name)
        conv_tower_output = base_model.get_layer(hparams.end_point).output
        self.conv_model = tf.keras.models.Model(inputs=base_model.input, outputs=conv_tower_output)
        self.conv_out_shape = self.conv_model.predict(np.array([np.zeros(hparams.image_shape)])).shape
        self.encode_cordinate = EncodeCordinate(input_shape=self.conv_out_shape)
        self.slice_rnn_input = SliceRNNInput()
        # self.base_model = base_model

    def call(self, inputs):
        conv_out  = self.conv_model(inputs)
        loc_out   = self.encode_cordinate(conv_out)
        if self.hparams.use_encode_cordinate:
            input_rnn = self.slice_rnn_input(loc_out)
        else:
            input_rnn = self.slice_rnn_input(conv_out)
        return input_rnn
