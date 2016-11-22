import sys
import argparse
import numpy as np
import pickle as pkl
from vgg19 import VGG19
from keras import preprocessing
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, RepeatVector, Flatten
from keras.layers import Activation, Dense, TimeDistributed
from keras.callbacks import ModelCheckpoint
from layers import AttentionRNN


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

parser = argparse.ArgumentParser(description='Image captioning.')
parser.add_argument('--image-file', dest='image_file', required=True)
parser.add_argument('--weights-file', dest='weights_file', required=True)
parser.add_argument('--sentence-size', dest='sentence_size', type=int, default=60)
parser.add_argument('--vocabulary-size', dest='vocabulary_size', type=int, default=10000)
args = parser.parse_args(sys.argv[1:])

model, block4 = VGG19(include_top=False)
block4_features = K.function([model.input], [block4])


input_ = Input(shape=(512, 196))
x = Lambda(lambda x: x.swapaxes(-1, -2))(input_)
x = Flatten()(x)
x = RepeatVector(args.sentence_size)(x)
x = AttentionRNN(512, fv_dim=512, consume_less='gpu', return_sequences=True)(x)
x = TimeDistributed(Dense(args.vocabulary_size+2))(x)
output_ = TimeDistributed(Activation('softmax'))(x)
model = Model(input=input_, output=output_)
model.load_weights(args.weights_file)

img = preprocessing.image.load_img(args.image_file, target_size=(224, 224))
img = preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
img_features = block4_features([img])[0][0]
img_features = img_features.reshape([512, 196])

y_pred = model.predict(np.array([img_features]))[0]
idxs = y_pred.argmax(axis=-1)
print(idxs)

idx2word = pkl.load(open('data/idx2word.pkl', 'rb'))
for idx in idxs:
    if idx == args.vocabulary_size:
        print('OOV', end=' ')
    elif idx == args.vocabulary_size + 1:
        print('END')
        break
    else:
        print(idx2word[idx], end=' ')

print('')

