__author__ = 'assafarbelle'
import tensorflow as tf

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
         ):


    with tf.variable_scope(name) as scope:

        in_shape = in_tensor.get_shape().as_list()[-1]
        kernel_shape = [kx, ky, in_shape, kout]
        if not stride:
            stride = [1, 1, 1, 1]
        elif isinstance(stride,int):
            stride = [1,stride,stride,1]
        elif isinstance(stride,list) and len(stride)==2:
            stride = [1]+stride+[1]



        kernel = tf.get_variable('weights', shape=kernel_shape, initializer=kernel_initializer)

        if biased:
            b = tf.get_variable('bias', kout, initializer=biase_initializer)
            out = tf.add(tf.nn.conv2d(in_tensor, kernel, strides=stride, padding=padding), b, name=name)
        else:
            out = tf.nn.conv2d(in_tensor, kernel, strides=stride, padding=padding, name=name)
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
                     ):


    with tf.variable_scope(name) as scope:

        in_shape = in_tensor.get_shape().as_list()[-1]
        kernel_shape = [kx, ky, kout,in_shape]
        if not stride:
            stride = [1, 1, 1, 1]
        elif isinstance(stride,int):
            stride = [1,stride,stride,1]
        elif isinstance(stride,list) and len(stride)==2:
            stride = [1]+stride+[1]



        kernel = tf.get_variable('weights', shape=kernel_shape, initializer=kernel_initializer)

        if biased:
            b = tf.get_variable('bias', kout, initializer=biase_initializer)
            out = tf.add(tf.nn.conv2d_transpose(in_tensor, kernel, output_shape=outshape, strides=stride, padding=padding), b, name=name)
        else:
            out = tf.nn.conv2d(in_tensor, kernel, strides=stride, padding=padding, name=name)
            b = None

    return out, kernel, b


def fc(in_tensor,
         name,
         kout,
         biased=True,
         weights_initializer=None,
         biase_initializer=None,
         ):

    in_shape = in_tensor.get_shape().as_list()
    if len(in_shape)>2:
        in_tensor = tf.reshape(in_tensor,[in_shape[0], -1])

    in_shape = in_tensor.get_shape().as_list()[1]

    with tf.variable_scope(name) as scope:
        weights_shape = [in_shape, kout]

        weights = tf.get_variable('weights', weights_shape, initializer=weights_initializer)

        if biased:
            b = tf.get_variable('bias', kout, initializer=biase_initializer)
            out = tf.add(tf.matmul(in_tensor, weights), b, name=name)
        else:
            out = tf.matmul(in_tensor, weights, name=name)
            b = None

    return out, weights, b


def leaky_relu(in_tensor, name, alpha=0.1):

    return tf.maximum(in_tensor,tf.mul(tf.constant(alpha),in_tensor),name=name)


def max_pool(in_tensor, name, ksize=None, strides=None, padding='VALID'):
    if not ksize:
        ksize = [1, 2, 2, 1]
    if not strides:
        strides = [1, 2, 2, 1]

    return tf.nn.max_pool(in_tensor, ksize, strides, padding, name=name)


def batch_norm(in_tensor, phase_train, name, reuse=None):
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
    with tf.variable_scope(name) as scope:
       return tf.contrib.layers.batch_norm(in_tensor, is_training=phase_train, scope=scope, reuse=reuse)


#