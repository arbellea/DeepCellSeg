
import tensorflow as tf



class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """
    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """

        shape = self.shape
        num_features = self.num_features
        if self._state_is_tuple:
            if self._data_format == 'NHWC':
                zeros = (tf.zeros([batch_size, shape[0], shape[1], num_features]),
                       tf.zeros([batch_size, shape[0], shape[1], num_features]))
            else:
                zeros = (tf.zeros([batch_size, num_features, shape[0], shape[1]]),
                       tf.zeros([batch_size, num_features, shape[0], shape[1]]))
        else:
            if self._data_format == 'NHWC':
                zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2])
            else:
                zeros = tf.zeros([batch_size, num_features * 2, shape[0], shape[1]])

        return zeros


class BasicConvLSTMCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, filter_size, num_features, forget_bias=1.0,
                 state_is_tuple=True, activation=tf.nn.tanh, data_format='NHWC'):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell 
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """

        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._data_format = data_format

    @property
    def state_size(self):
        return (self.num_features, self.num_features) if self._state_is_tuple else 2 * self.num_features

    @property
    def output_size(self):
        return self.num_features

    def __call__(self, inputs, state, scope=None, reuse=None, clip_cell=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            channel_axis = 3 if self._data_format == 'NHWC' else 1
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=channel_axis, num_or_size_splits=2, value=state)
            concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True,
                                  data_format=self._data_format)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=channel_axis, num_or_size_splits=4, value=concat)
            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(j))
            clip_cell = False
            if clip_cell:
                new_c = tf.clip_by_value(new_c, -clip_cell, clip_cell)
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = (new_c, new_h)
            else:
                new_state = tf.concat(axis=channel_axis, values=[new_c, new_h])
        return new_h, new_state

class LayerNormConvLSTMCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, filter_size, num_features, forget_bias=1.0,
                 state_is_tuple=True, activation=tf.nn.tanh, data_format='NHWC'):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell 
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """

        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._data_format = data_format

    @property
    def state_size(self):
        return (self.num_features, self.num_features) if self._state_is_tuple else 2 * self.num_features

    @property
    def output_size(self):
        return self.num_features

    def __call__(self, inputs, state, scope=None, reuse=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            channel_axis = 3 if self._data_format == 'NHWC' else 1
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=channel_axis, num_or_size_splits=2, value=state)
            concat_i = _conv_linear([inputs], self.filter_size, self.num_features * 4, False,
                                  data_format=self._data_format, scope='Input')
            concat_h = _conv_linear([inputs], self.filter_size, self.num_features * 4, False,
                                  data_format=self._data_format, scope='H')
            bias_term = tf.get_variable("Bias", [self.num_features * 4], dtype=inputs.dtype,
                                        initializer=tf.constant_initializer(0.0, dtype=inputs.dtype))
            concat = tf.nn.bias_add(tf.contrib.layers.layer_norm(concat_i) + tf.contrib.layers.layer_norm(concat_h),
                                    bias_term, data_format=self._data_format)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=channel_axis, num_or_size_splits=4, value=concat)
            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(j))
            new_c_norm = tf.contrib.layers.layer_norm(new_c)
            new_h = self._activation(new_c_norm) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = (new_c, new_h)
            else:
                new_state = tf.concat(axis=channel_axis, values=[new_c, new_h])
        return new_h, new_state


def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None, data_format='NHWC'):
    """convolution:
    Args:
    args: a 4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
    A 4D Tensor with shape [batch h w num_features]
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    channel_axis = 3 if data_format == 'NHWC' else 1
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[channel_axis]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        weights = tf.get_variable("Weights", [filter_size[0], filter_size[1], total_arg_size_depth, num_features],
                                  dtype=dtype)
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], weights, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
        else:
            res = tf.nn.conv2d(tf.concat(axis=channel_axis, values=args), weights, strides=[1, 1, 1, 1], padding='SAME',
                               data_format=data_format)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [num_features], dtype=dtype,
                                    initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return tf.nn.bias_add(res, bias_term, data_format=data_format)


class BasicConvGRUCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, filter_size, num_features, forget_bias=1.0,
                 state_is_tuple=True, activation=tf.nn.tanh, data_format='NHWC'):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell 
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """

        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._data_format = data_format

    @property
    def state_size(self):
        return (self.num_features, self.num_features) if self._state_is_tuple else 2 * self.num_features

    @property
    def output_size(self):
        return self.num_features

    def __call__(self, inputs, h, scope=None, reuse=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            channel_axis = 3 if self._data_format == 'NHWC' else 1

            concat = tf.nn.sigmoid(_conv_linear([inputs, h], self.filter_size, self.num_features * 2, True,
                                  data_format=self._data_format, scope='gates'))
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            z, r = tf.split(axis=channel_axis, num_or_size_splits=2, value=concat)
            i = tf.nn.tanh(_conv_linear([inputs, tf.multiply(r, h)], self.filter_size, self.num_features, True,
                                        data_format=self._data_format, scope='input'))
            new_h = tf.add(tf.multiply(z, h), tf.multiply(1-z, i))

        return new_h


