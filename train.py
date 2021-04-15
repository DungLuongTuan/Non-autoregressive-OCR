from model import Model

import tensorflow as tf
import numpy as np
import pdb
import os

#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
    ### create model
    model = Model()
    model.create_model()
    ### train new model
    model.train()

if __name__ == '__main__':
    main()
