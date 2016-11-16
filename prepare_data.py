import os
import sys
import tqdm
import json
import numpy as np
import pickle as pkl
from collections import defaultdict
from keras import preprocessing
from keras import backend as K


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


def tokenize(sentence):
    chars = ['\n', '!', '.', '#', '@', '?', ')', '(', '"', "'", '$',
             ':', ',', '/', '[', ']', '`', '=', '=', ';', '\\', '-',
             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    sentence = sentence.lower()
    for c in chars:
        sentence = sentence.replace(c, ' '+c+' ').strip()
    return [w for w in sentence.split(' ') if w]


def prepare_captions():
    print('Preparing captions... ')
    assert os.path.exists('data/captions_train2014.json')
    assert os.path.exists('data/captions_val2014.json')

    captions_train2014 = json.loads(open('data/captions_train2014.json').read())
    captions_val2014 = json.loads(open('data/captions_val2014.json').read())

    print('train2014 images:', len(captions_train2014['images']))
    print('val2014 images:', len(captions_val2014['images']))

    word_counts = {}
    for a in captions_train2014['annotations']:
        for w in tokenize(a['caption']):
            word_counts[w] = 1 if w not in word_counts else word_counts[w] + 1
    for a in captions_val2014['annotations']:
        for w in tokenize(a['caption']):
            word_counts[w] = 1 if w not in word_counts else word_counts[w] + 1

    word_counts = sorted([(c, w) for w, c in word_counts.items()], reverse=True)
    word2idx = {w: i for i, (c, w) in enumerate(word_counts)}
    idx2word = {i: w for i, (c, w) in enumerate(word_counts)}
    print('words:', len(word_counts))

    print('Top 10 word counts:')
    for c, w in word_counts[:10]:
        print(w, ' -> ', c)

    id2captions = defaultdict(list)
    for annotation in captions_train2014['annotations']:
        image_id = annotation['image_id']
        caption = [word2idx[w] for w in tokenize(annotation['caption'])]
        id2captions[image_id].append(caption)
    for annotation in captions_val2014['annotations']:
        image_id = annotation['image_id']
        caption = [word2idx[w] for w in tokenize(annotation['caption'])]
        id2captions[image_id].append(caption)

    pkl.dump(word_counts, open('data/word_counts.pkl', 'wb'))
    pkl.dump(word2idx, open('data/word2idx.pkl', 'wb'))
    pkl.dump(idx2word, open('data/idx2word.pkl', 'wb'))
    pkl.dump(id2captions, open('data/id2captions.pkl', 'wb'))
    print('done!')


def prepare_images():
    print('Preparing images...')
    assert os.path.exists('data/train2014')
    assert os.path.exists('data/val2014')

    fnames_train2014 = os.listdir('data/train2014')
    fnames_val2014 = os.listdir('data/val2014')

    print('train2014 images:', len(fnames_train2014))
    print('val2014 images:', len(fnames_val2014))

    def get_id(fname):
        fname = fname.replace('COCO_train2014_', '')
        fname = fname.replace('COCO_val2014_', '')
        fname = fname.replace('.jpg', '')
        return int(fname)


    print('Loading VGG')
    from vgg19 import VGG19
    model, block4 = VGG19(include_top=False)
    block4_features = K.function([model.input], [block4])


    images_train2014 = np.memmap('data/images_train2014.mmap', dtype='float32', mode='write',
                                 shape=(len(fnames_train2014), 512, 196))
    for i, fname in tqdm.tqdm(list(enumerate(fnames_train2014)), desc='train2014'):
        img = preprocessing.image.load_img('data/train2014/'+fname, target_size=(224, 224))
        img = preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img_features = block4_features([img])[0][0]
        images_train2014[i] = img_features.reshape([512, 196])
        if (i % 1000) == 0: images_train2014.flush()
    images_train2014.flush()

    images_val2014 = np.memmap('data/images_val2014.mmap', dtype='float32', mode='write',
                                 shape=(len(fnames_val2014), 512, 196))
    for i, fname in tqdm.tqdm(list(enumerate(fnames_val2014)), desc='val2014'):
        img = preprocessing.image.load_img('data/val2014/'+fname, target_size=(224, 224))
        img = preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img_features = block4_features([img])[0][0]
        images_val2014[i] = img_features.reshape([512, 196])
        if (i % 1000) == 0: images_val2014.flush()
    images_val2014.flush()

    pkl.dump(fnames_train2014, open('data/fnames_train2014.pkl', 'wb'))
    pkl.dump(fnames_val2014, open('data/fnames_val2014.pkl', 'wb'))


if __name__ == '__main__':
    prepare_captions()
    prepare_images()

