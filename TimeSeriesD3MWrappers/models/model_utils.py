from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.keras.layers import Input, Dense, concatenate, Activation, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
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
        x = AttentionLSTM(lstm_dim)(ip)
    else:
        x = LSTM(lstm_dim)(ip)
    x = Dropout(dropout)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
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

    model = Model(ip, out)
    return model

class LSTMSequence(Sequence):
    """ custom Sequence for LSTM_FCN input data """

    def __init__(self, X, y, batch_size):
        self.X = np.float32(X)
        self.y = np.float32(y)

        # make sure batch_size is not bigger than array
        if batch_size > X.shape[0]:
            batch_size = X.shape[0]
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.X.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

