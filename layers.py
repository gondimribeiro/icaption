import numpy as np
from keras import backend as K
from keras.layers import LSTM
from keras import regularizers

# comment the line 494 (self.assert_input_compatibility(x)) from:
#/Users/roliveira/anaconda3/lib/python3.5/site-packages/keras/engine/topology.py


class AttentionRNN(LSTM):

    """AttentionRNN mechanism for LSTM layer"""

    def __init__(self, output_dim, fv_dim, l2=0., *args, **kwargs):
        self.fv_dim = fv_dim
        self.attention_regularizer = regularizers.l2(l2)
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
        self.csum_alphas = K.zeros([196, 512], name='{}_csum_alphas'.format(self.name)) # verificar
        self.attention_regularizer.set_param(self.csum_alphas)
        self.regularizers = [self.attention_regularizer]
        self.non_trainable_weights = [self.csum_alphas]

    def step(self, x, states):
        x = K.reshape(x, (-1, 196, 512))
        h = states[0]

        p1 = K.dot(h, self.Wah)
        p2 = K.dot(x, self.Waa)
        e = K.tanh(K.expand_dims(p1, dim=1) + p2) + self.B
        tmp = K.exp(e)
        sums = K.sum(tmp, axis=-1, keepdims=True)
        alphas = tmp / sums
        csum_alphas = 1 - K.sum(alphas, axis=0)
        self.updates = [(self.csum_alphas, csum_alphas)]

        z = K.sum(x*alphas, axis=1)

        return super(AttentionRNN, self).step(z, states)

    def get_config(self):
        config = {'fv_dim': self.fv_dim}
        base_config = super(AttentionRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
