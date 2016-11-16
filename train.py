import os
import numpy as np
import pickle as pkl
from vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.callbacks import TensorBoard
from layers import AttentionRNN
import utils


print('Loading data...')
fnames_train2014 = np.array(pkl.load(open('data/fnames_train2014.pkl', 'rb')))
fnames_val2014 = np.array(pkl.load(open('data/fnames_val2014.pkl', 'rb')))
images_train2014 = np.memmap('data/images_train2014.mmap', dtype='float32', mode='readonly', shape=(len(fnames_train2014), 512, 196))
images_val2014 = np.memmap('data/images_val2014.mmap', dtype='float32', mode='readonly', shape=(len(fnames_val2014), 512, 196))
id2captions = pkl.load(open('data/id2captions.pkl', 'rb'))
print('done!')

SENTENCE_LEN = 1 + max(len(caps) for img_captions in id2captions.values() for caps in img_captions)
VOCABULARY_SIZE = 10
BATCH_SIZE = 1
NB_EPOCH = 5
HIDDEN_SIZE = 30

print('#'*15, 'Parameters', '#'*15)
print('SENTENCE_LEN:', SENTENCE_LEN)
print('VOCABULARY_SIZE:', VOCABULARY_SIZE)
print('BATCH_SIZE:', BATCH_SIZE)
print('NB_EPOCH:', NB_EPOCH)
print('HIDDEN_SIZE:', HIDDEN_SIZE)
print('#'*40)

print('#'*15, 'Building model', '#'*15)
input_ = Input(shape=(512, 196))
x = Lambda(lambda x: x.swapaxes(-1, -2))(input_)
x = Flatten()(x)
x = RepeatVector(SENTENCE_LEN)(x)
x = AttentionRNN(HIDDEN_SIZE, 512, consume_less='gpu', return_sequences=True)(x)
x = TimeDistributed(Dense(VOCABULARY_SIZE+1))(x)
output_ = TimeDistributed(Activation('softmax'))(x)
model = Model(input=input_, output=output_)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              sample_weight_mode='temporal')


def captions_from_fnames(files, it):
    captions = []
    sample_weights = []
    for f in files:
        img_captions = id2captions[utils.get_id(f)]
        if it > len(img_captions):
            it = it%len(img_captions)

        caption_labels = np.array(img_captions[it])
        caption_labels[caption_labels >= VOCABULARY_SIZE] = VOCABULARY_SIZE
        caption = np.zeros([SENTENCE_LEN, VOCABULARY_SIZE+1], dtype='float32')
        caption[np.arange(len(caption_labels)), caption_labels] = 1.0
        caption[len(caption_labels), VOCABULARY_SIZE] = 1.0

        weights = np.zeros(SENTENCE_LEN)
        weights[:len(caption_labels)+1] = 1.0

        captions.append(caption)
        sample_weights.append(weights)

    return np.array(captions), np.array(sample_weights)


def data_generator(dataset):
    if dataset == 'train':
        fnames = fnames_train2014
        images = images_train2014
    elif dataset == 'val':
        fnames = fnames_val2014
        images = images_val2014
    else:
        raise Exception('Unknown dataset {0}'.format(dataset))

    n = len(fnames)
    it_caption = 0
    while True:
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        for i in range(0, n - n%BATCH_SIZE, BATCH_SIZE):
            X = images[idxs[i:i+BATCH_SIZE]]
            Y, sample_weights = captions_from_fnames(fnames[idxs[i:i+BATCH_SIZE]], it_caption)
            print(X.shape, Y.shape, sample_weights.shape)
            yield (X, Y, sample_weights)
        it_caption = (it_caption + 1)%5


model.fit_generator(generator=data_generator('train'),
                    samples_per_epoch=len(fnames_train2014),
                    nb_epoch=NB_EPOCH,
                    validation_data=data_generator('val'),
                    nb_val_samples=len(fnames_val2014))
