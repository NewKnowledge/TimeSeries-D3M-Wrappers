from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.keras.layers import Input, Dense, concatenate, Activation, LSTM, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import math
import numpy as np
from TimeSeriesD3MWrappers.models.layer_utils import AttentionLSTM
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_lstmfcn(MAX_SEQUENCE_LENGTH, 
    NB_CLASS, 
    lstm_dim = 128, 
    attention = True, 
    dropout = 0.2
    ):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))
    if attention:
        x = AttentionLSTM(lstm_dim, implementation=2)(ip)
    else:
        x = LSTM(lstm_dim)(ip)
    x = Dropout(dropout)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    return Model(inputs = ip, outputs = out)

class LSTMSequence(Sequence):
    """ custom Sequence for LSTM_FCN input data """

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.X.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return tf.constant(batch_x), tf.constant(batch_y)

class LSTMSequenceTest(Sequence):
    """ custom Sequence for LSTM_FCN input data """

    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.X.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        return tf.constant(batch_x)
