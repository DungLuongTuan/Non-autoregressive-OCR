import os
import cv2
import pdb
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tqdm import tqdm
from model import Model
from dataset import Dataset
from hparams import hparams
from datetime import datetime, timedelta

model = Model()
model.create_model()
latest = tf.train.latest_checkpoint(hparams.save_path)
model.checkpoint.restore('/data2/users/dunglt/fast_ocr/training_checkpoints_50_50_IR_mixed_6a_4x128/train/ckpt-573')

model.inference(np.random.uniform(1, 255, (50, 500, 3))/255.)

tf.saved_model.save(model.model, 'saved_model')