import sys
import argparse
import numpy as np
import pickle as pkl
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, RepeatVector, Flatten
from keras.layers import Activation, Dense, TimeDistributed
from keras.callbacks import ModelCheckpoint
from layers import AttentionRNN
import utils

# Attention model
input_ = Input(shape=(512, 196))
# x = Lambda(lambda x: x.swapaxes(-1, -2))(input_)
x = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(input_)
x = Flatten()(x)
x = RepeatVector(10)(x)
x = AttentionRNN(512, fv_dim=512, consume_less='gpu', return_sequences=True)(x)
x = TimeDistributed(Dense(20+2))(x)
output_ = TimeDistributed(Activation('softmax'))(x)
model = Model(input=input_, output=output_)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              sample_weight_mode='temporal',
              metrics=['accuracy'])
