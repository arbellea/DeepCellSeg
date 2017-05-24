import Layers
import tensorflow as tf


def layer(op):

    def layer_decorated(self, name, *args, **kwargs):

        out = op(self, name, *args, **kwargs)
        self.layers[name] = out
        return out
    return layer_decorated


class Network(object):

    def __init__(self):
        self.layers = {}
        self.weights = {}

    @layer
    def conv(self, name,
             in_tensor,
             kx,
             ky,
             kout,
             stride=None,
             biased=True,
             kernel_initializer=None,
             biase_initializer=None,
             padding='SAME',
             ):
        out, w, b = Layers.conv(in_tensor,
                                name,
                                kx,
                                ky,
                                kout,
                                stride,
                                biased,
                                kernel_initializer,
                                biase_initializer,
                                padding,
                                )
        self.weights.update({w.op.name: w})
        if biased:
            self.weights.update({b.op.name: b})
        return out

    @layer
    def conv2d_transpose(self, name,
                         in_tensor,
                         kx,
                         ky,
                         kout,
                         outshape,
                         stride=None,
                         biased=True,
                         kernel_initializer=None,
                         biase_initializer=None,
                         padding='SAME',
                         ):
        out, w, b = Layers.conv2d_transpose(in_tensor,
                                            name,
                                            kx,
                                            ky,
                                            kout,
                                            outshape,
                                            stride,
                                            biased,
                                            kernel_initializer,
                                            biase_initializer,
                                            padding,
                                            )
        self.weights.update({w.op.name: w})
        if biased:
            self.weights.update({b.op.name: b})
        return out

    @layer
    def fc(self, name,
           in_tensor,
           kout,
           biased=True,
           weights_initializer=None,
           biase_initializer=None,
           ):

        out, w, b = Layers.fc(in_tensor, name, kout, biased, weights_initializer, biase_initializer)
        if biased:
            self.weights.update({b.op.name: b})
        self.weights.update({w.op.name: w})
        return out

    @layer
    def leaky_relu(self, name, in_tensor, alpha=0.1):
        return Layers.leaky_relu(in_tensor, name, alpha)

    @layer
    def max_pool(self, name, in_tensor, ksize=None, strides=None, padding='VALID'):
        return Layers.max_pool(in_tensor, name, ksize, strides, padding)

    @layer
    def batch_norm(self, name, in_tensor, phase_train, reuse=None):

        return Layers.batch_norm(in_tensor, phase_train, name, reuse)

    @layer
    def concat(self, name, in_tensor_list, dim=3):
        return tf.concat(dim, in_tensor_list, name)

    @layer
    def sigmoid(self, name, in_tensor):
        return tf.sigmoid(in_tensor, name)

    @layer
    def argmax(self, name, in_tensor, axis):
        return tf.argmax(in_tensor, axis=axis, name=name)

    @layer
    def softmax(self, name, in_tensor, dim=-1):
        return tf.nn.softmax(in_tensor, name=name, dim=dim)

    @layer
    def ge(self, name, in_tensor, thr):
        return tf.greater_equal(in_tensor, thr, name=name)
__author__ = 'assafarbelle'
