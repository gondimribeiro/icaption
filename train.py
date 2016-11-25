import sys
import argparse
import numpy as np
import pickle as pkl
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, RepeatVector, Flatten
from keras.layers import Activation, Dense, TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.objectives import categorical_crossentropy
from layers import AttentionRNN
from keras import regularizers
import utils

def custom_loss(lmbda, alphas):
    def loss_regularized(y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred) + \
        lmbda*K.sum(K.square(1 - K.sum(alphas, axis=-1)), axis=-1)
    return loss_regularized

parser = argparse.ArgumentParser(description='Image captioning model.')
parser.add_argument('--sentence-size', dest='sentence_size', type=int, default=60)
parser.add_argument('--vocabulary-size', dest='vocabulary_size', type=int, default=10000)
parser.add_argument('--batch-size', dest='batch_size', type=int, default=64)
parser.add_argument('--nb-epoch', dest='nb_epoch', type=int, default=100)
args = parser.parse_args(sys.argv[1:])

# Loading datasets
fnames_train2014 = np.array(pkl.load(open('data/fnames_train2014.pkl', 'rb')))
fnames_val2014 = np.array(pkl.load(open('data/fnames_val2014.pkl', 'rb')))
images_train2014 = np.memmap('data/images_train2014.mmap', dtype='float32', mode='readonly', shape=(len(fnames_train2014), 512, 196))
images_val2014 = np.memmap('data/images_val2014.mmap', dtype='float32', mode='readonly', shape=(len(fnames_val2014), 512, 196))
id2captions = pkl.load(open('data/id2captions.pkl', 'rb'))

# check is param sentence_size is grater than the biggest sentence caption
max_len = max(len(caps) for img_captions in id2captions.values() for caps in img_captions)
assert args.sentence_size > max_len

# Attention model
input_ = Input(shape=(512, 196))
x = Lambda(lambda x: x.swapaxes(-1, -2))(input_)
x = Flatten()(x)
x = RepeatVector(args.sentence_size)(x)
x = AttentionRNN(512, fv_dim=512, consume_less='gpu', return_sequences=True, l2=1.)(x)
x = TimeDistributed(Dense(args.vocabulary_size+2))(x)
output_ = TimeDistributed(Activation('softmax'))(x)
model = Model(input=input_, output=[output_])
optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              sample_weight_mode='temporal',
              metrics=['accuracy'])

def captions_from_fnames(files, it):
    captions = []
    sample_weights = []
    for f in files:
        img_captions = id2captions[utils.get_id(f)]
        if it > len(img_captions):
            it = it%len(img_captions)

        caption_labels = np.array(img_captions[it])
        caption_labels[caption_labels >= args.vocabulary_size] = args.vocabulary_size
        caption = np.zeros([args.sentence_size, args.vocabulary_size+2], dtype='float32')
        caption[np.arange(len(caption_labels)), caption_labels] = 1.0
        caption[len(caption_labels), args.vocabulary_size+1] = 1.0

        weights = np.zeros(args.sentence_size)
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
        for i in range(0, n, args.batch_size):
            X = images[idxs[i:i+args.batch_size]]
            Y, sample_weights = captions_from_fnames(fnames[idxs[i:i+args.batch_size]], it_caption)
            yield (X, Y, sample_weights)
        it_caption = (it_caption + 1)%5

mc = ModelCheckpoint('weights/epoch_{epoch:02d}-loss_{val_loss:.2f}.hdf5',
                     save_best_only=True, save_weights_only=True)
model.fit_generator(generator=data_generator('train'),
                    samples_per_epoch=len(fnames_train2014),
                    nb_epoch=args.nb_epoch,
                    validation_data=data_generator('val'),
                    nb_val_samples=len(fnames_val2014),
                    callbacks=[mc])
