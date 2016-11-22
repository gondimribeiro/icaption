import numpy as np
from keras import backend as K
from keras.layers import LSTM

# comment the line 494 (self.assert_input_compatibility(x)) from:
#/Users/roliveira/anaconda3/lib/python3.5/site-packages/keras/engine/topology.py


class AttentionRNN(LSTM):

    """AttentionRNN mechanism for LSTM layer"""

    def __init__(self, output_dim, fv_dim, *args, **kwargs):
        self.fv_dim = fv_dim
        super(AttentionRNN, self).__init__(output_dim, *args, **kwargs)

    def build(self, input_shape):
        fv_dim = self.fv_dim
        out_dim = self.output_dim
        rnn_input_shape = list(input_shape[:-1]) + [fv_dim]

        super(AttentionRNN, self).build(rnn_input_shape)
        self.Wah = self.init([out_dim, fv_dim], name='{}_Wah'.format(self.name))
        self.Waa = self.init([fv_dim, fv_dim], name='{}_Waa'.format(self.name))
        self.B = self.init([fv_dim], name='{}_B'.format(self.name))
        self.trainable_weights.extend([self.Wah, self.Waa, self.B])

    def step(self, x, states):
        # x = K.reshape(x, (-1, 196, 512))
        h = states[0]

        p1 = K.dot(h, self.Wah)
        p2 = K.dot(x, self.Waa)
        e = K.tanh(K.expand_dims(p1, dim=1) + p2) + self.B
        sums = K.sum(K.exp(e), axis=-1, keepdims=True)
        alphas = K.exp(e)/sums
        z = K.sum(x*alphas, axis=1)

        return super(AttentionRNN, self).step(z, states)

    def get_config(self):
        config = {'fv_dim': self.fv_dim}
        base_config = super(AttentionRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
