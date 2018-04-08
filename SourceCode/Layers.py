import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import layer_norm

__author__ = 'assafarbelle'


def conv(in_tensor,
         name,
         kx,
         ky,
         kout,
         stride=None,
         biased=True,
         kernel_initializer=None,
         biase_initializer=None,
         padding='VALID',
         data_format='NHWC',
         reuse=False
         ):
    with tf.variable_scope(name, reuse=reuse):
        channel = 1 if data_format == 'NCHW' else 3
        in_shape = in_tensor.get_shape().as_list()[channel]
        kernel_shape = [kx, ky, in_shape, kout]
        if not stride:
            stride = [1, 1, 1, 1]
        elif isinstance(stride, int):
            if channel == 3:
                stride = [1, stride, stride, 1]
            else:
                stride = [1, 1, stride, stride]
        elif isinstance(stride, list) and len(stride) == 2:

            if channel == 3:
                stride = [1] + stride + [1]
            else:
                stride = [1, 1] + stride

        kernel = tf.get_variable('weights', shape=kernel_shape, initializer=kernel_initializer)
        conv = tf.nn.conv2d(in_tensor, kernel, strides=stride, padding=padding, data_format=data_format)
        if biased:
            b = tf.get_variable('bias', kout, initializer=biase_initializer)
            out = tf.nn.bias_add(conv, b, data_format=data_format, name=name)
        else:
            out = conv
            b = None

    return out, kernel, b


def conv2d_transpose(in_tensor,
                     name,
                     kx,
                     ky,
                     kout,
                     outshape,
                     stride=None,
                     biased=True,
                     kernel_initializer=None,
                     biase_initializer=None,
                     padding='VALID',
                     data_format='NHWC'
                     ):
    with tf.variable_scope(name):
        channel = 1 if data_format == 'NCHW' else 3
        in_shape = in_tensor.get_shape().as_list()[channel]
        kernel_shape = [kx, ky, kout, in_shape]
        if not stride:
            stride = [1, 1, 1, 1]
        elif isinstance(stride, int):
            stride = [1, stride, stride, 1]
        elif isinstance(stride, list) and len(stride) == 2:
            stride = [1] + stride + [1]

        kernel = tf.get_variable('weights', shape=kernel_shape, initializer=kernel_initializer)
        conv_t = tf.nn.conv2d_transpose(in_tensor, kernel, output_shape=outshape, strides=stride, padding=padding,
                                        data_format=data_format)
        if biased:
            b = tf.get_variable('bias', kout, initializer=biase_initializer)
            out = tf.nn.bias_add(conv_t, b, name=name)
        else:
            out = conv_t
            b = None

    return out, kernel, b


def fc(in_tensor, name, kout,
       biased=True,
       weights_initializer=None,
       biase_initializer=None,
       ):
    in_shape = in_tensor.get_shape().as_list()
    if len(in_shape) > 2:
        in_tensor = tf.reshape(in_tensor, [in_shape[0], -1])

    in_shape = in_tensor.get_shape().as_list()[1]

    with tf.variable_scope(name):
        weights_shape = [in_shape, kout]

        weights = tf.get_variable('weights', weights_shape, initializer=weights_initializer)
        matmul = tf.matmul(in_tensor, weights, name=name)
        if biased:
            b = tf.get_variable('bias', kout, initializer=biase_initializer)
            out = tf.add(matmul, b, name=name)
        else:
            out = matmul
            b = None

    return out, weights, b


def leaky_relu(in_tensor, name, alpha=0.1):
    return tf.maximum(in_tensor, tf.multiply(tf.constant(alpha), in_tensor), name=name)


def max_pool(in_tensor, name, ksize=None, strides=None, padding='VALID', data_format='NHWC'):
    channel = 1 if data_format == 'NCHW' else 3

    if not ksize:
        if channel == 3:
            ksize = [1, 2, 2, 1]
        else:
            ksize = [1, 1, 2, 2]
    elif isinstance(ksize, int):
        if channel == 3:
            ksize = [1, ksize, ksize, 1]
        else:
            ksize = [1, 1, ksize, ksize]
    elif isinstance(strides, list) and len(ksize) == 2:

        if channel == 3:
            ksize = [1] + ksize + [1]
        else:
            ksize = [1, 1] + ksize

    if not strides:
        if channel == 3:
            strides = [1, 2, 2, 1]
        else:
            strides = [1, 1, 2, 2]
    elif isinstance(strides, int):
        if channel == 3:
            strides = [1, strides, strides, 1]
        else:
            strides = [1, 1, strides, strides]
    elif isinstance(strides, list) and len(strides) == 2:

        if channel == 3:
            strides = [1] + strides + [1]
        else:
            strides = [1, 1] + strides

    return tf.nn.max_pool(in_tensor, ksize, strides, padding, name=name, data_format=data_format)


def batch_norm(in_tensor, phase_train, name, reuse=None, data_format='NHWC', center=True, scale=True):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    axis = -1 if data_format == 'NHWC' else 1
    with tf.variable_scope(name):
        # return tf.contrib.layers.batch_norm(in_tensor, is_training=phase_train, scope=scope, reuse=reuse)
        return tf.layers.batch_normalization(in_tensor, axis=axis, center=center, scale=scale, training=phase_train,
                                             reuse=reuse, fused=True, momentum=0.99, epsilon=1e-1)


def layer_norm(in_tensor, data_format='NHWC'):
    if data_format == 'NCHW':
        in_tensor = tf.transpose(in_tensor, (0, 2, 3, 1))
    out_tensor = layer_norm(in_tensor)
    if data_format == 'NCHW':
        out_tensor = tf.transpose(out_tensor, (0, 3, 1, 2))
    return out_tensor
