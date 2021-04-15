from augment import distort, stretch, perspective
from models import CNN_CTC, CNN_SA_CTC
from datetime import datetime
from dataset import Dataset
from hparams import hparams
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import logging
import random
logging.basicConfig(level=logging.DEBUG)
import pdb
import os


class Model(object):
    def __init__(self):
        self.char_mapping = {}
        with open(hparams.charset_path, 'r') as f:
            for row in f:
                index, label = row[:-1].split('\t')
                self.char_mapping[int(index)] = label


    def loss_function(self, real, pred):
        loss_ = self.loss_object(real, pred)
        return tf.reduce_mean(loss_)


    def create_model(self):
        ### create model
        self.best_val_acc = 0.0
        # dataset
        self.train_dataset = Dataset(hparams, hparams.train_record_path)
        self.valid_dataset = Dataset(hparams, hparams.valid_record_path)
        self.train_dataset.load_tfrecord()
        self.valid_dataset.load_tfrecord()
        # create model
        if hparams.model_name == 'cnn_ctc':
            self.model = CNN_CTC(hparams)
        elif hparams.model_name == 'cnn_sa_ctc':
            self.model = CNN_SA_CTC(hparams)
        
        ### define training ops and params
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.lr)

        self.last_epoch = 0
        self.train_summary_writer = tf.summary.create_file_writer(hparams.save_path + '/logs/train')
        self.valid_summary_writer = tf.summary.create_file_writer(hparams.save_path + '/logs/valid')
        self.checkpoint_dir = os.path.join(hparams.save_path, 'train')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              model=self.model)

    def load_model(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest != None:
            logging.info('load model from {}'.format(latest))
            self.last_epoch = int(latest.split('-')[-1])
            self.checkpoint.restore(latest)


    def evaluate(self, batch_input, batch_target, batch_target_length):
        output = self.model(batch_input)[0][0].numpy()
        batch_true_char = 0
        batch_true_str = 0
        for i in range(len(batch_target_length)):
            true_char = 0
            for j in range(batch_target_length[i]):
                if output[i][j] == batch_target[i][j]:
                    true_char += 1
            if true_char == batch_target_length[i]:
                batch_true_str += 1
            batch_true_char += true_char
        return batch_true_char, batch_true_str


    def inference(self, inputs):
        outputs = self.model(np.array([inputs]))
        output = outputs[0][0].numpy()
        probs = outputs[1][0].numpy()
        result = []
        for indexes in output:
            res = ''
            for index in indexes:
                if index == -1:
                    break
                res += self.char_mapping[index]
            result.append(res)
        return result, probs


    def train_step(self, batch_input, batch_target, batch_target_length):
        current_batch_size = batch_input.shape[0]
        with tf.GradientTape() as tape:
            loss = self.model(batch_input, batch_target, batch_target_length)
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return (loss / int(batch_target.shape[1]))


    def augment(self, batch_input):
        batch = []
        for sample in batch_input:
            aug_func = random.choice([0, 1, 1, 2, 3, 3])
            if aug_func == 0:
                aug_sample = sample.numpy()
            elif aug_func == 1:
                aug_sample = distort(sample.numpy(), 4)
            elif aug_func == 2:
                aug_sample = stretch(sample.numpy(), 4)
            elif aug_func == 3:
                aug_sample = perspective(sample.numpy())
            batch.append(aug_sample)
        return np.array(batch)


    def train(self):
        self.load_model()
        for epoch in range(self.last_epoch, hparams.max_epochs):
            start = datetime.now()
            total_loss = 0
            #train each batch in dataset
            for batch, (batch_input, batch_target, batch_target_length) in enumerate(self.train_dataset.dataset):
                if hparams.augment:
                    batch_input = self.augment(batch_input)
                    batch_input = tf.cast(batch_input, tf.float32)/255
                else:
                    batch_input = tf.cast(batch_input, tf.float32)/255
                batch_loss = self.train_step(batch_input, batch_target, batch_target_length)
                total_loss += batch_loss
                if batch % 1 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

            self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            # evaluate on valid set
            cnt_true_char = 0
            cnt_true_str = 0
            sum_char = 1
            sum_str = 1
            for batch, (batch_input, batch_target, batch_target_length) in enumerate(self.valid_dataset.dataset):
                batch_input = tf.cast(batch_input, tf.float32)/255
                batch_true_char, batch_true_str = self.evaluate(batch_input, batch_target, batch_target_length)
                cnt_true_char += batch_true_char
                cnt_true_str  += batch_true_str
                sum_char += np.sum(batch_target_length)
                sum_str  += batch_input.shape[0]
            valid_char_acc = cnt_true_char/sum_char
            valid_str_acc  = cnt_true_str/sum_str

            # write log
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', total_loss/batch, step=epoch)

            with self.valid_summary_writer.as_default():
                tf.summary.scalar('character accuracy', valid_char_acc, step=epoch)
                tf.summary.scalar('sequence accuracy', valid_str_acc, step=epoch)

            # log traing result of each epoch
            logging.info('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / batch))
            logging.info('Accuracy on valid set:')
            logging.info('character accuracy: {:.6f}'.format(valid_char_acc))
            logging.info('sequence accuracy : {:.6f}'.format(valid_str_acc))
            logging.info('Time taken for 1 epoch {} sec\n'.format(datetime.now() - start))
