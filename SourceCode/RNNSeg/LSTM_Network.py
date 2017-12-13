import tensorflow as tf
from ConvLSTM.BasicConvLSTMCell import BasicConvLSTMCell, LayerNormConvLSTMCell, BasicConvGRUCell
import Layers

DEFAULT_NET_PARAMS = {
                'conv_kxy': 3,
                'kout1': 32,
                'kout2': 64,
                'kout3': 128,
                'kout4': 256,
                'lstm_kxy': [7, 7],
                'lstm_kout1': 32,
                'lstm_kout2': 64,
                'lstm_kout3': 128,
                'lstm_kout4': 256,
}


class LSTMNetwork(object):
    def __init__(self):
        self.layers_dict = []
        self.states = []
        pass

    def build(self, input_sequence, phase_train=True, net_params=DEFAULT_NET_PARAMS):
        data_format = 'NCHW'

        def conv_bn_relu_pool(_input_image, name, kxy, kout, stride=None, biased=True, reuse=None):

            conv, _, _ = Layers.conv(_input_image, name=name+'/conv', kx=kxy, ky=kxy, kout=kout, stride=stride, biased=biased,
                               padding='SAME', data_format=data_format, reuse=reuse)
            bn = Layers.batch_norm(conv, phase_train, name+'/bn', reuse=reuse, data_format=data_format)
            relu = Layers.leaky_relu(bn, name+'/relu')
            pool = Layers.max_pool(relu, name=name+'/pool', padding='SAME', data_format=data_format)

            return pool, relu

        for t, input_image in enumerate(input_sequence):

            with tf.name_scope('time_step-{}'.format(t)):

                # conv_kxy = 3
                # kout1 = 32
                # kout2 = 32
                # kout3 = 64
                # kout4 = 92
                conv_kxy = net_params['conv_kxy']
                kout1 = net_params['kout1']
                kout2 = net_params['kout2']
                kout3 = net_params['kout3']
                kout4 = net_params['kout4']

                # lstm_kxy = [7, 7]
                # lstm_kout1 = 32
                # lstm_kout2 = 32
                # lstm_kout3 = 64
                # lstm_kout4 = 92
                lstm_kxy = net_params['lstm_kxy']
                lstm_kout1 = net_params['lstm_kout1']
                lstm_kout2 = net_params['lstm_kout2']
                lstm_kout3 = net_params['lstm_kout3']
                lstm_kout4 = net_params['lstm_kout4']

                layer1p, layer1 = conv_bn_relu_pool(input_image, 'layer1', conv_kxy, kout1, reuse=t > 0)
                layer2p, layer2 = conv_bn_relu_pool(layer1p, 'layer2', conv_kxy, kout2, reuse=t > 0)
                layer3p, layer3 = conv_bn_relu_pool(layer2p, 'layer3', conv_kxy, kout3, reuse=t > 0)
                _, layer4 = conv_bn_relu_pool(layer3p, 'layer4', conv_kxy, kout4, reuse=t > 0)
                input_shape = input_image.get_shape().as_list()
                layer2_shape = layer2.get_shape().as_list()
                layer3_shape = layer3.get_shape().as_list()
                layer4_shape = layer4.get_shape().as_list()
                lstm4 = LayerNormConvLSTMCell(shape=layer4_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout4, data_format=data_format)
                lstm3 = LayerNormConvLSTMCell(shape=layer3_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout3, data_format=data_format)
                lstm2 = LayerNormConvLSTMCell(shape=layer2_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout2, data_format=data_format)
                lstm1 = LayerNormConvLSTMCell(shape=input_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout1, data_format=data_format)

                if t == 0:
                    z1 = lstm1.zero_state(batch_size=input_shape[0])
                    init_state = [(tf.placeholder_with_default(z1[0], z1[0].get_shape(), 'init_state_ph_1'),
                                  tf.placeholder_with_default(z1[1], z1[1].get_shape(), 'init_state_ph_1_c'))]
                    z2 = lstm2.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z2[0], z2[0].get_shape(), 'init_state_ph_2'),
                                       tf.placeholder_with_default(z2[1], z2[1].get_shape(), 'init_state_ph_2_c')))
                    z3 = lstm3.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z3[0], z3[0].get_shape(), 'init_state_ph_3'),
                                       tf.placeholder_with_default(z3[1], z3[1].get_shape(), 'init_state_ph_3_c')))
                    z4 = lstm4.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z4[0], z4[0].get_shape(), 'init_state_ph_4'),
                                       tf.placeholder_with_default(z4[1], z4[1].get_shape(), 'init_state_ph_4_c')))
                    self.states.append(init_state)

                lstm4_out, lstm4_state = lstm4(layer4, self.states[t][3], scope='lstm4', reuse=t > 0)
                lstm4_up = tf.image.resize_bilinear(tf.transpose(lstm4_out, [0, 2, 3, 1]), layer3_shape[2:])
                lstm4_up = tf.transpose(lstm4_up, [0, 3, 1, 2])
                lstm3_input = tf.concat(values=[lstm4_up, layer3], axis=1)
                lstm3_out, lstm3_state = lstm3(lstm3_input, self.states[t][2], scope='lstm3', reuse=t > 0)
                lstm3_up = tf.image.resize_bilinear(tf.transpose(lstm3_out, [0, 2, 3, 1]), layer2_shape[2:])
                lstm3_up = tf.transpose(lstm3_up, [0, 3, 1, 2])
                lstm2_input = tf.concat(values=[lstm3_up, layer2], axis=1)
                lstm2_out, lstm2_state = lstm2(lstm2_input, self.states[t][1], scope='lstm2', reuse=t > 0)
                lstm2_up = tf.image.resize_bilinear(tf.transpose(lstm2_out, [0, 2, 3, 1]), input_shape[2:])
                lstm2_up = tf.transpose(lstm2_up, [0, 3, 1, 2])
                lstm1_input = tf.concat(values=[lstm2_up, layer1], axis=1)
                lstm1_out, lstm1_state = lstm1(lstm1_input, self.states[t][0], scope='lstm1', reuse=t > 0)

                self.states.append([lstm1_state, lstm2_state, lstm3_state, lstm4_state])

                out_conv, _, _ = Layers.conv(lstm1_out, name='out_conv', kx=1, ky=1, kout=3, padding='SAME',
                                       data_format=data_format, reuse=t > 0)
                local_layers = {'input_image': input_image, 'layer1': layer1, 'layer1p': layer1p, 'layer2': layer2,
                                'layer2p': layer2p, 'layer3': layer3, 'layer3p': layer3p, 'layer4': layer4,
                                'lstm4_out': lstm4_out, 'lstm4_up': lstm4_up, 'lstm3_out': lstm3_out,
                                'lstm3_up': lstm3_up, 'lstm2_out': lstm2_out, 'lstm2_up': lstm2_up,
                                'lstm1_out': lstm4_out, 'out_conv': out_conv
                                }
                self.layers_dict.append(local_layers)

        return [ld['out_conv'] for ld in self.layers_dict]


class NormBiLSTMNetwork(object):
    def __init__(self):
        self.layers_dict = []
        self.states = []
        self.states_back = []
        self.layers_dict_back = []
        self.outputs = []
        self.fw_outputs = []
        self.bw_outputs = []
        pass

    def build(self, input_sequence, phase_train=True, net_params=DEFAULT_NET_PARAMS):
        data_format = 'NCHW'

        def conv_bn_relu_pool(_input_image, name, kxy, kout, stride=None, biased=True, reuse=None):

            conv, _, _ = Layers.conv(_input_image, name=name+'/conv', kx=kxy, ky=kxy, kout=kout, stride=stride, biased=biased,
                               padding='SAME', data_format=data_format, reuse=reuse)
            bn = Layers.batch_norm(conv, phase_train, name+'/bn', reuse=reuse, data_format=data_format)
            relu = Layers.leaky_relu(bn, name+'/relu')
            pool = Layers.max_pool(relu, name=name+'/pool', padding='SAME', data_format=data_format)

            return pool, relu

        for t, input_image in enumerate(input_sequence):

            with tf.name_scope('time_step-{}_forward'.format(t)):


                conv_kxy = net_params['conv_kxy']
                kout1 = net_params['kout1']
                kout2 = net_params['kout2']
                kout3 = net_params['kout3']
                kout4 = net_params['kout4']

                lstm_kxy = net_params['lstm_kxy']
                lstm_kout1 = net_params['lstm_kout1']
                lstm_kout2 = net_params['lstm_kout2']
                lstm_kout3 = net_params['lstm_kout3']
                lstm_kout4 = net_params['lstm_kout4']

                layer1p, layer1 = conv_bn_relu_pool(input_image, 'layer1', conv_kxy, kout1, reuse=t > 0)
                layer2p, layer2 = conv_bn_relu_pool(layer1p, 'layer2', conv_kxy, kout2, reuse=t > 0)
                layer3p, layer3 = conv_bn_relu_pool(layer2p, 'layer3', conv_kxy, kout3, reuse=t > 0)
                _, layer4 = conv_bn_relu_pool(layer3p, 'layer4', conv_kxy, kout4, reuse=t > 0)
                input_shape = input_image.get_shape().as_list()
                layer2_shape = layer2.get_shape().as_list()
                layer3_shape = layer3.get_shape().as_list()
                layer4_shape = layer4.get_shape().as_list()
                lstm4 = LayerNormConvLSTMCell(shape=layer4_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout4, data_format=data_format)
                lstm3 = LayerNormConvLSTMCell(shape=layer3_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout3, data_format=data_format)
                lstm2 = LayerNormConvLSTMCell(shape=layer2_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout2, data_format=data_format)
                lstm1 = LayerNormConvLSTMCell(shape=input_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout1, data_format=data_format)


                if t == 0:
                    z1 = lstm1.zero_state(batch_size=input_shape[0])
                    init_state = [(tf.placeholder_with_default(z1[0], z1[0].get_shape(), 'init_state_ph_1'),
                                  tf.placeholder_with_default(z1[1], z1[1].get_shape(), 'init_state_ph_1_c'))]
                    z2 = lstm2.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z2[0], z2[0].get_shape(), 'init_state_ph_2'),
                                       tf.placeholder_with_default(z2[1], z2[1].get_shape(), 'init_state_ph_2_c')))
                    z3 = lstm3.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z3[0], z3[0].get_shape(), 'init_state_ph_3'),
                                       tf.placeholder_with_default(z3[1], z3[1].get_shape(), 'init_state_ph_3_c')))
                    z4 = lstm4.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z4[0], z4[0].get_shape(), 'init_state_ph_4'),
                                       tf.placeholder_with_default(z4[1], z4[1].get_shape(), 'init_state_ph_4_c')))

                    self.states.append(init_state)

                lstm4_out, lstm4_state = lstm4(layer4, self.states[t][3], scope='lstm4', reuse=t > 0)
                lstm4_up = tf.image.resize_bilinear(tf.transpose(lstm4_out, [0, 2, 3, 1]), layer3_shape[2:])
                lstm4_up = tf.transpose(lstm4_up, [0, 3, 1, 2])
                lstm3_input = tf.concat(values=[lstm4_up, layer3], axis=1)
                lstm3_out, lstm3_state = lstm3(lstm3_input, self.states[t][2], scope='lstm3', reuse=t > 0)
                lstm3_up = tf.image.resize_bilinear(tf.transpose(lstm3_out, [0, 2, 3, 1]), layer2_shape[2:])
                lstm3_up = tf.transpose(lstm3_up, [0, 3, 1, 2])
                lstm2_input = tf.concat(values=[lstm3_up, layer2], axis=1)
                lstm2_out, lstm2_state = lstm2(lstm2_input, self.states[t][1], scope='lstm2', reuse=t > 0)
                lstm2_up = tf.image.resize_bilinear(tf.transpose(lstm2_out, [0, 2, 3, 1]), input_shape[2:])
                lstm2_up = tf.transpose(lstm2_up, [0, 3, 1, 2])
                lstm1_input = tf.concat(values=[lstm2_up, layer1], axis=1)
                lstm1_out, lstm1_state = lstm1(lstm1_input, self.states[t][0], scope='lstm1', reuse=t > 0)

                self.states.append([lstm1_state, lstm2_state, lstm3_state, lstm4_state])

                local_layers = {'input_image': input_image, 'layer1': layer1, 'layer1p': layer1p, 'layer2': layer2,
                                'layer2p': layer2p, 'layer3': layer3, 'layer3p': layer3p, 'layer4': layer4,
                                'lstm4_out': lstm4_out, 'lstm4_up': lstm4_up, 'lstm3_out': lstm3_out,
                                'lstm3_up': lstm3_up, 'lstm2_out': lstm2_out, 'lstm2_up': lstm2_up,
                                'lstm1_out': lstm1_out
                                }
                self.layers_dict.append(local_layers)

        for t, input_image in enumerate(input_sequence[::-1]):

            with tf.name_scope('time_step-{}_back'.format(t)):

                lstm4_back = LayerNormConvLSTMCell(shape=layer4_shape[2:], filter_size=lstm_kxy,
                                               num_features=lstm_kout4, data_format=data_format)
                lstm3_back = LayerNormConvLSTMCell(shape=layer3_shape[2:], filter_size=lstm_kxy,
                                               num_features=lstm_kout3, data_format=data_format)
                lstm2_back = LayerNormConvLSTMCell(shape=layer2_shape[2:], filter_size=lstm_kxy,
                                               num_features=lstm_kout2, data_format=data_format)
                lstm1_back = LayerNormConvLSTMCell(shape=input_shape[2:], filter_size=lstm_kxy,
                                               num_features=lstm_kout1, data_format=data_format)

                if t == 0:
                    z1 = lstm1.zero_state(batch_size=input_shape[0])
                    init_state = [(tf.placeholder_with_default(z1[0], z1[0].get_shape(), 'init_state_ph_1_back'),
                                   tf.placeholder_with_default(z1[1], z1[1].get_shape(), 'init_state_ph_1_c_back'))]
                    z2 = lstm2.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z2[0], z2[0].get_shape(), 'init_state_ph_2_back'),
                                       tf.placeholder_with_default(z2[1], z2[1].get_shape(), 'init_state_ph_2_c_back')))
                    z3 = lstm3.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z3[0], z3[0].get_shape(), 'init_state_ph_3_back'),
                                       tf.placeholder_with_default(z3[1], z3[1].get_shape(), 'init_state_ph_3_c_back')))
                    z4 = lstm4.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z4[0], z4[0].get_shape(), 'init_state_ph_4_back'),
                                       tf.placeholder_with_default(z4[1], z4[1].get_shape(), 'init_state_ph_4_c_back')))

                    self.states_back.append(init_state)
                local_layers = self.layers_dict[-1-t]

                lstm4_out, lstm4_state = lstm4_back(local_layers['layer4'], self.states_back[t][3], scope='lstm4',
                                                    reuse=True)
                lstm4_up = tf.image.resize_bilinear(tf.transpose(lstm4_out, [0, 2, 3, 1]), layer3_shape[2:])
                lstm4_up = tf.transpose(lstm4_up, [0, 3, 1, 2])
                lstm3_input = tf.concat(values=[lstm4_up, local_layers['layer3']], axis=1)
                lstm3_out, lstm3_state = lstm3_back(lstm3_input, self.states_back[t][2], scope='lstm3', reuse=True)
                lstm3_up = tf.image.resize_bilinear(tf.transpose(lstm3_out, [0, 2, 3, 1]), layer2_shape[2:])
                lstm3_up = tf.transpose(lstm3_up, [0, 3, 1, 2])
                lstm2_input = tf.concat(values=[lstm3_up, local_layers['layer2']], axis=1)
                lstm2_out, lstm2_state = lstm2_back(lstm2_input, self.states_back[t][1], scope='lstm2',reuse=True)
                lstm2_up = tf.image.resize_bilinear(tf.transpose(lstm2_out, [0, 2, 3, 1]), input_shape[2:])
                lstm2_up = tf.transpose(lstm2_up, [0, 3, 1, 2])
                lstm1_input = tf.concat(values=[lstm2_up, local_layers['layer1']], axis=1)
                lstm1_out, lstm1_state = lstm1_back(lstm1_input, self.states_back[t][0], scope='lstm1', reuse=True)

                self.states_back.append([lstm1_state, lstm2_state, lstm3_state, lstm4_state])

                local_layers = {'input_image': input_image, 'layer1': layer1, 'layer1p': layer1p, 'layer2': layer2,
                                'layer2p': layer2p, 'layer3': layer3, 'layer3p': layer3p, 'layer4': layer4,
                                'lstm4_out': lstm4_out, 'lstm4_up': lstm4_up, 'lstm3_out': lstm3_out,
                                'lstm3_up': lstm3_up, 'lstm2_out': lstm2_out, 'lstm2_up': lstm2_up,
                                'lstm1_out': lstm1_out
                                }
                self.layers_dict_back.append(local_layers)
        reuse = False
        for fw, bw in zip(self.layers_dict, self.layers_dict_back[::-1]):
            lstm1_out_fw = fw['lstm1_out']
            lstm1_out_bw = bw['lstm1_out']
            con_input = tf.concat([lstm1_out_fw, lstm1_out_bw], axis=1)
            out_conv, w, b = Layers.conv(con_input, name='out_conv', kx=1, ky=1, kout=3, padding='SAME',
                                         data_format=data_format, reuse=reuse)

            w_fw, w_bw = tf.split(w, 2, 2)

            b_half = tf.div(b, 2.)
            conv_fw = tf.nn.conv2d(lstm1_out_fw, w_fw, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
            conv_bw = tf.nn.conv2d(lstm1_out_bw, w_bw, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
            out_conv_fw = tf.nn.bias_add(conv_fw, b_half, data_format=data_format)
            out_conv_bw = tf.nn.bias_add(conv_bw, b_half, data_format=data_format)

            self.outputs.append(out_conv)
            self.fw_outputs.append(out_conv_fw)
            self.bw_outputs.append(out_conv_bw)

            reuse = True

        return self.outputs


class BiLSTMNetwork(object):
    def __init__(self):
        self.layers_dict = []
        self.states = []
        self.states_back = []
        self.layers_dict_back = []
        self.outputs = []
        self.fw_outputs = []
        self.bw_outputs = []
        pass

    def build(self, input_sequence, phase_train=True, net_params=DEFAULT_NET_PARAMS, max_len=None):
        data_format = 'NCHW'

        def conv_bn_relu_pool(_input_image, name, kxy, kout, stride=None, biased=True, reuse=None):

            conv, _, _ = Layers.conv(_input_image, name=name+'/conv', kx=kxy, ky=kxy, kout=kout, stride=stride, biased=biased,
                               padding='SAME', data_format=data_format, reuse=reuse)
            bn = Layers.batch_norm(conv, phase_train, name+'/bn', reuse=reuse, data_format=data_format)
            relu = Layers.leaky_relu(bn, name+'/relu')
            pool = Layers.max_pool(relu, name=name+'/pool', padding='SAME', data_format=data_format)

            return pool, relu
        if not max_len:
            max_len = len(input_sequence)
        for t, input_image in enumerate(input_sequence):

            with tf.name_scope('time_step-{}_forward'.format(t)):


                conv_kxy = net_params['conv_kxy']
                kout1 = net_params['kout1']
                kout2 = net_params['kout2']
                kout3 = net_params['kout3']
                kout4 = net_params['kout4']

                lstm_kxy = net_params['lstm_kxy']
                lstm_kout1 = net_params['lstm_kout1']
                lstm_kout2 = net_params['lstm_kout2']
                lstm_kout3 = net_params['lstm_kout3']
                lstm_kout4 = net_params['lstm_kout4']

                layer1p, layer1 = conv_bn_relu_pool(input_image, 'layer1', conv_kxy, kout1, reuse=t > 0)
                layer2p, layer2 = conv_bn_relu_pool(layer1p, 'layer2', conv_kxy, kout2, reuse=t > 0)
                layer3p, layer3 = conv_bn_relu_pool(layer2p, 'layer3', conv_kxy, kout3, reuse=t > 0)
                _, layer4 = conv_bn_relu_pool(layer3p, 'layer4', conv_kxy, kout4, reuse=t > 0)
                input_shape = input_image.get_shape().as_list()
                layer2_shape = layer2.get_shape().as_list()
                layer3_shape = layer3.get_shape().as_list()
                layer4_shape = layer4.get_shape().as_list()
                lstm4 = BasicConvLSTMCell(shape=layer4_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout4, data_format=data_format)
                lstm3 = BasicConvLSTMCell(shape=layer3_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout3, data_format=data_format)
                lstm2 = BasicConvLSTMCell(shape=layer2_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout2, data_format=data_format)
                lstm1 = BasicConvLSTMCell(shape=input_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout1, data_format=data_format)


                if t == 0:
                    z1 = lstm1.zero_state(batch_size=input_shape[0])
                    init_state = [(tf.placeholder_with_default(z1[0], z1[0].get_shape(), 'init_state_ph_1'),
                                  tf.placeholder_with_default(z1[1], z1[1].get_shape(), 'init_state_ph_1_c'))]
                    z2 = lstm2.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z2[0], z2[0].get_shape(), 'init_state_ph_2'),
                                       tf.placeholder_with_default(z2[1], z2[1].get_shape(), 'init_state_ph_2_c')))
                    z3 = lstm3.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z3[0], z3[0].get_shape(), 'init_state_ph_3'),
                                       tf.placeholder_with_default(z3[1], z3[1].get_shape(), 'init_state_ph_3_c')))
                    z4 = lstm4.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z4[0], z4[0].get_shape(), 'init_state_ph_4'),
                                       tf.placeholder_with_default(z4[1], z4[1].get_shape(), 'init_state_ph_4_c')))

                    self.states.append(init_state)

                lstm4_out, lstm4_state = lstm4(layer4, self.states[t][3], scope='lstm4', reuse=t > 0, clip_cell=max_len)
                lstm4_up = tf.image.resize_bilinear(tf.transpose(lstm4_out, [0, 2, 3, 1]), layer3_shape[2:])
                lstm4_up = tf.transpose(lstm4_up, [0, 3, 1, 2])
                lstm3_input = tf.concat(values=[lstm4_up, layer3], axis=1)
                lstm3_out, lstm3_state = lstm3(lstm3_input, self.states[t][2], scope='lstm3', reuse=t > 0, clip_cell=max_len)
                lstm3_up = tf.image.resize_bilinear(tf.transpose(lstm3_out, [0, 2, 3, 1]), layer2_shape[2:])
                lstm3_up = tf.transpose(lstm3_up, [0, 3, 1, 2])
                lstm2_input = tf.concat(values=[lstm3_up, layer2], axis=1)
                lstm2_out, lstm2_state = lstm2(lstm2_input, self.states[t][1], scope='lstm2', reuse=t > 0, clip_cell=max_len)
                lstm2_up = tf.image.resize_bilinear(tf.transpose(lstm2_out, [0, 2, 3, 1]), input_shape[2:])
                lstm2_up = tf.transpose(lstm2_up, [0, 3, 1, 2])
                lstm1_input = tf.concat(values=[lstm2_up, layer1], axis=1)
                lstm1_out, lstm1_state = lstm1(lstm1_input, self.states[t][0], scope='lstm1', reuse=t > 0, clip_cell=max_len)

                self.states.append([lstm1_state, lstm2_state, lstm3_state, lstm4_state])

                local_layers = {'input_image': input_image, 'layer1': layer1, 'layer1p': layer1p, 'layer2': layer2,
                                'layer2p': layer2p, 'layer3': layer3, 'layer3p': layer3p, 'layer4': layer4,
                                'lstm4_out': lstm4_out, 'lstm4_up': lstm4_up, 'lstm3_out': lstm3_out,
                                'lstm3_up': lstm3_up, 'lstm2_out': lstm2_out, 'lstm2_up': lstm2_up,
                                'lstm1_out': lstm1_out
                                }
                self.layers_dict.append(local_layers)

        for t, input_image in enumerate(input_sequence[::-1]):

            with tf.name_scope('time_step-{}_back'.format(t)):

                lstm4_back = BasicConvLSTMCell(shape=layer4_shape[2:], filter_size=lstm_kxy,
                                               num_features=lstm_kout4, data_format=data_format)
                lstm3_back = BasicConvLSTMCell(shape=layer3_shape[2:], filter_size=lstm_kxy,
                                               num_features=lstm_kout3, data_format=data_format)
                lstm2_back = BasicConvLSTMCell(shape=layer2_shape[2:], filter_size=lstm_kxy,
                                               num_features=lstm_kout2, data_format=data_format)
                lstm1_back = BasicConvLSTMCell(shape=input_shape[2:], filter_size=lstm_kxy,
                                               num_features=lstm_kout1, data_format=data_format)

                if t == 0:
                    z1 = lstm1.zero_state(batch_size=input_shape[0])
                    init_state = [(tf.placeholder_with_default(z1[0], z1[0].get_shape(), 'init_state_ph_1_back'),
                                   tf.placeholder_with_default(z1[1], z1[1].get_shape(), 'init_state_ph_1_c_back'))]
                    z2 = lstm2.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z2[0], z2[0].get_shape(), 'init_state_ph_2_back'),
                                       tf.placeholder_with_default(z2[1], z2[1].get_shape(), 'init_state_ph_2_c_back')))
                    z3 = lstm3.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z3[0], z3[0].get_shape(), 'init_state_ph_3_back'),
                                       tf.placeholder_with_default(z3[1], z3[1].get_shape(), 'init_state_ph_3_c_back')))
                    z4 = lstm4.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z4[0], z4[0].get_shape(), 'init_state_ph_4_back'),
                                       tf.placeholder_with_default(z4[1], z4[1].get_shape(), 'init_state_ph_4_c_back')))

                    self.states_back.append(init_state)
                local_layers = self.layers_dict[-1-t]

                lstm4_out, lstm4_state = lstm4_back(local_layers['layer4'], self.states_back[t][3], scope='lstm4',
                                                    reuse=True, clip_cell=max_len)
                lstm4_up = tf.image.resize_bilinear(tf.transpose(lstm4_out, [0, 2, 3, 1]), layer3_shape[2:])
                lstm4_up = tf.transpose(lstm4_up, [0, 3, 1, 2])
                lstm3_input = tf.concat(values=[lstm4_up, local_layers['layer3']], axis=1)
                lstm3_out, lstm3_state = lstm3_back(lstm3_input, self.states_back[t][2], scope='lstm3', reuse=True, clip_cell=max_len)
                lstm3_up = tf.image.resize_bilinear(tf.transpose(lstm3_out, [0, 2, 3, 1]), layer2_shape[2:])
                lstm3_up = tf.transpose(lstm3_up, [0, 3, 1, 2])
                lstm2_input = tf.concat(values=[lstm3_up, local_layers['layer2']], axis=1)
                lstm2_out, lstm2_state = lstm2_back(lstm2_input, self.states_back[t][1], scope='lstm2',reuse=True, clip_cell=max_len)
                lstm2_up = tf.image.resize_bilinear(tf.transpose(lstm2_out, [0, 2, 3, 1]), input_shape[2:])
                lstm2_up = tf.transpose(lstm2_up, [0, 3, 1, 2])
                lstm1_input = tf.concat(values=[lstm2_up, local_layers['layer1']], axis=1)
                lstm1_out, lstm1_state = lstm1_back(lstm1_input, self.states_back[t][0], scope='lstm1', reuse=True, clip_cell=max_len)

                self.states_back.append([lstm1_state, lstm2_state, lstm3_state, lstm4_state])

                local_layers = {'input_image': input_image, 'layer1': layer1, 'layer1p': layer1p, 'layer2': layer2,
                                'layer2p': layer2p, 'layer3': layer3, 'layer3p': layer3p, 'layer4': layer4,
                                'lstm4_out': lstm4_out, 'lstm4_up': lstm4_up, 'lstm3_out': lstm3_out,
                                'lstm3_up': lstm3_up, 'lstm2_out': lstm2_out, 'lstm2_up': lstm2_up,
                                'lstm1_out': lstm1_out
                                }
                self.layers_dict_back.append(local_layers)
        reuse = False
        for fw, bw in zip(self.layers_dict, self.layers_dict_back[::-1]):
            lstm1_out_fw = fw['lstm1_out']
            lstm1_out_bw = bw['lstm1_out']
            con_input = tf.concat([lstm1_out_fw, lstm1_out_bw], axis=1)
            out_conv, w, b = Layers.conv(con_input, name='out_conv', kx=1, ky=1, kout=3, padding='SAME',
                                         data_format=data_format, reuse=reuse)

            w_fw, w_bw = tf.split(w, 2, 2)

            b_half = tf.div(b, 2.)
            conv_fw = tf.nn.conv2d(lstm1_out_fw, w_fw, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
            conv_bw = tf.nn.conv2d(lstm1_out_bw, w_bw, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
            out_conv_fw = tf.nn.bias_add(conv_fw, b_half, data_format=data_format)
            out_conv_bw = tf.nn.bias_add(conv_bw, b_half, data_format=data_format)

            self.outputs.append(out_conv)
            self.fw_outputs.append(out_conv_fw)
            self.bw_outputs.append(out_conv_bw)

            reuse = True

        return self.outputs


class BiGRUNetwork(object):
    def __init__(self):
        self.layers_dict = []
        self.states = []
        self.states_back = []
        self.layers_dict_back = []
        self.outputs = []
        self.fw_outputs = []
        self.bw_outputs = []
        pass

    def build(self, input_sequence, phase_train=True, net_params=DEFAULT_NET_PARAMS):
        data_format = 'NCHW'

        def conv_bn_relu_pool(_input_image, name, kxy, kout, stride=None, biased=True, reuse=None):

            conv, _, _ = Layers.conv(_input_image, name=name+'/conv', kx=kxy, ky=kxy, kout=kout, stride=stride, biased=biased,
                               padding='SAME', data_format=data_format, reuse=reuse)
            bn = Layers.batch_norm(conv, phase_train, name+'/bn', reuse=reuse, data_format=data_format)
            relu = Layers.leaky_relu(bn, name+'/relu')
            pool = Layers.max_pool(relu, name=name+'/pool', padding='SAME', data_format=data_format)

            return pool, relu

        for t, input_image in enumerate(input_sequence):

            with tf.name_scope('time_step-{}_forward'.format(t)):


                conv_kxy = net_params['conv_kxy']
                kout1 = net_params['kout1']
                kout2 = net_params['kout2']
                kout3 = net_params['kout3']
                kout4 = net_params['kout4']

                lstm_kxy = net_params['lstm_kxy']
                lstm_kout1 = net_params['lstm_kout1']
                lstm_kout2 = net_params['lstm_kout2']
                lstm_kout3 = net_params['lstm_kout3']
                lstm_kout4 = net_params['lstm_kout4']

                layer1p, layer1 = conv_bn_relu_pool(input_image, 'layer1', conv_kxy, kout1, reuse=t > 0)
                layer2p, layer2 = conv_bn_relu_pool(layer1p, 'layer2', conv_kxy, kout2, reuse=t > 0)
                layer3p, layer3 = conv_bn_relu_pool(layer2p, 'layer3', conv_kxy, kout3, reuse=t > 0)
                _, layer4 = conv_bn_relu_pool(layer3p, 'layer4', conv_kxy, kout4, reuse=t > 0)
                input_shape = input_image.get_shape().as_list()
                layer2_shape = layer2.get_shape().as_list()
                layer3_shape = layer3.get_shape().as_list()
                layer4_shape = layer4.get_shape().as_list()
                lstm4 = BasicConvGRUCell(shape=layer4_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout4, data_format=data_format)
                lstm3 = BasicConvGRUCell(shape=layer3_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout3, data_format=data_format)
                lstm2 = BasicConvGRUCell(shape=layer2_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout2, data_format=data_format)
                lstm1 = BasicConvGRUCell(shape=input_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout1, data_format=data_format)

                if t == 0:
                    z1 = lstm1.zero_state(batch_size=input_shape[0])
                    init_state = [tf.placeholder_with_default(z1[0], z1[0].get_shape(), 'init_state_ph_1')]
                    z2 = lstm2.zero_state(batch_size=input_shape[0])
                    init_state.append(tf.placeholder_with_default(z2[0], z2[0].get_shape(), 'init_state_ph_2'))
                    z3 = lstm3.zero_state(batch_size=input_shape[0])
                    init_state.append(tf.placeholder_with_default(z3[0], z3[0].get_shape(), 'init_state_ph_3'))
                    z4 = lstm4.zero_state(batch_size=input_shape[0])
                    init_state.append(tf.placeholder_with_default(z4[0], z4[0].get_shape(), 'init_state_ph_4'))

                    self.states.append(init_state)

                lstm4_out = lstm4(layer4, self.states[t][3], scope='lstm4', reuse=t > 0)
                lstm4_up = tf.image.resize_bilinear(tf.transpose(lstm4_out, [0, 2, 3, 1]), layer3_shape[2:])
                lstm4_up = tf.transpose(lstm4_up, [0, 3, 1, 2])
                lstm3_input = tf.concat(values=[lstm4_up, layer3], axis=1)
                lstm3_out = lstm3(lstm3_input, self.states[t][2], scope='lstm3', reuse=t > 0)
                lstm3_up = tf.image.resize_bilinear(tf.transpose(lstm3_out, [0, 2, 3, 1]), layer2_shape[2:])
                lstm3_up = tf.transpose(lstm3_up, [0, 3, 1, 2])
                lstm2_input = tf.concat(values=[lstm3_up, layer2], axis=1)
                lstm2_out = lstm2(lstm2_input, self.states[t][1], scope='lstm2', reuse=t > 0)
                lstm2_up = tf.image.resize_bilinear(tf.transpose(lstm2_out, [0, 2, 3, 1]), input_shape[2:])
                lstm2_up = tf.transpose(lstm2_up, [0, 3, 1, 2])
                lstm1_input = tf.concat(values=[lstm2_up, layer1], axis=1)
                lstm1_out = lstm1(lstm1_input, self.states[t][0], scope='lstm1', reuse=t > 0)

                self.states.append([lstm1_out, lstm2_out, lstm3_out, lstm4_out])

                local_layers = {'input_image': input_image, 'layer1': layer1, 'layer1p': layer1p, 'layer2': layer2,
                                'layer2p': layer2p, 'layer3': layer3, 'layer3p': layer3p, 'layer4': layer4,
                                'lstm4_out': lstm4_out, 'lstm4_up': lstm4_up, 'lstm3_out': lstm3_out,
                                'lstm3_up': lstm3_up, 'lstm2_out': lstm2_out, 'lstm2_up': lstm2_up,
                                'lstm1_out': lstm1_out
                                }
                self.layers_dict.append(local_layers)

        for t, input_image in enumerate(input_sequence[::-1]):

            with tf.name_scope('time_step-{}_back'.format(t)):

                lstm4_back = BasicConvGRUCell(shape=layer4_shape[2:], filter_size=lstm_kxy,
                                              num_features=lstm_kout4, data_format=data_format)
                lstm3_back = BasicConvGRUCell(shape=layer3_shape[2:], filter_size=lstm_kxy,
                                              num_features=lstm_kout3, data_format=data_format)
                lstm2_back = BasicConvGRUCell(shape=layer2_shape[2:], filter_size=lstm_kxy,
                                              num_features=lstm_kout2, data_format=data_format)
                lstm1_back = BasicConvGRUCell(shape=input_shape[2:], filter_size=lstm_kxy,
                                              num_features=lstm_kout1, data_format=data_format)

                if t == 0:
                    z1 = lstm1.zero_state(batch_size=input_shape[0])
                    init_state = [tf.placeholder_with_default(z1[0], z1[0].get_shape(), 'init_state_ph_1')]
                    z2 = lstm2.zero_state(batch_size=input_shape[0])
                    init_state.append(tf.placeholder_with_default(z2[0], z2[0].get_shape(), 'init_state_ph_2'))
                    z3 = lstm3.zero_state(batch_size=input_shape[0])
                    init_state.append(tf.placeholder_with_default(z3[0], z3[0].get_shape(), 'init_state_ph_3'))
                    z4 = lstm4.zero_state(batch_size=input_shape[0])
                    init_state.append(tf.placeholder_with_default(z4[0], z4[0].get_shape(), 'init_state_ph_4'))
                    self.states_back.append(init_state)
                local_layers = self.layers_dict[-1-t]

                lstm4_out = lstm4_back(local_layers['layer4'], self.states_back[t][3], scope='lstm4',
                                                    reuse=True)
                lstm4_up = tf.image.resize_bilinear(tf.transpose(lstm4_out, [0, 2, 3, 1]), layer3_shape[2:])
                lstm4_up = tf.transpose(lstm4_up, [0, 3, 1, 2])
                lstm3_input = tf.concat(values=[lstm4_up, local_layers['layer3']], axis=1)
                lstm3_out = lstm3_back(lstm3_input, self.states_back[t][2], scope='lstm3', reuse=True)
                lstm3_up = tf.image.resize_bilinear(tf.transpose(lstm3_out, [0, 2, 3, 1]), layer2_shape[2:])
                lstm3_up = tf.transpose(lstm3_up, [0, 3, 1, 2])
                lstm2_input = tf.concat(values=[lstm3_up, local_layers['layer2']], axis=1)
                lstm2_out = lstm2_back(lstm2_input, self.states_back[t][1], scope='lstm2',reuse=True)
                lstm2_up = tf.image.resize_bilinear(tf.transpose(lstm2_out, [0, 2, 3, 1]), input_shape[2:])
                lstm2_up = tf.transpose(lstm2_up, [0, 3, 1, 2])
                lstm1_input = tf.concat(values=[lstm2_up, local_layers['layer1']], axis=1)
                lstm1_out = lstm1_back(lstm1_input, self.states_back[t][0], scope='lstm1', reuse=True)

                self.states_back.append([lstm1_out, lstm2_out, lstm3_out, lstm4_out])

                local_layers = {'input_image': input_image, 'layer1': layer1, 'layer1p': layer1p, 'layer2': layer2,
                                'layer2p': layer2p, 'layer3': layer3, 'layer3p': layer3p, 'layer4': layer4,
                                'lstm4_out': lstm4_out, 'lstm4_up': lstm4_up, 'lstm3_out': lstm3_out,
                                'lstm3_up': lstm3_up, 'lstm2_out': lstm2_out, 'lstm2_up': lstm2_up,
                                'lstm1_out': lstm1_out
                                }
                self.layers_dict_back.append(local_layers)
        reuse = False
        for fw, bw in zip(self.layers_dict, self.layers_dict_back[::-1]):
            lstm1_out_fw = fw['lstm1_out']
            lstm1_out_bw = bw['lstm1_out']
            con_input = tf.concat([lstm1_out_fw, lstm1_out_bw], axis=1)
            out_conv, w, b = Layers.conv(con_input, name='out_conv', kx=1, ky=1, kout=3, padding='SAME',
                                         data_format=data_format, reuse=reuse)

            w_fw, w_bw = tf.split(w, 2, 2)

            b_half = tf.div(b, 2.)
            conv_fw = tf.nn.conv2d(lstm1_out_fw, w_fw, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
            conv_bw = tf.nn.conv2d(lstm1_out_bw, w_bw, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
            out_conv_fw = tf.nn.bias_add(conv_fw, b_half, data_format=data_format)
            out_conv_bw = tf.nn.bias_add(conv_bw, b_half, data_format=data_format)

            self.outputs.append(out_conv)
            self.fw_outputs.append(out_conv_fw)
            self.bw_outputs.append(out_conv_bw)

            reuse = True

        return self.outputs


class LSTMNetwork_Trans(object):
    def __init__(self):
        self.layers_dict = []
        self.states = []
        pass

    def build(self, input_sequence, phase_train=True, net_params=DEFAULT_NET_PARAMS):
        data_format = 'NCHW'

        def conv_bn_relu_pool(_input_image, name, kxy, kout, stride=None, biased=True, reuse=None):

            conv, _, _ = Layers.conv(_input_image, name=name+'/conv', kx=kxy, ky=kxy, kout=kout, stride=stride, biased=biased,
                               padding='SAME', data_format=data_format, reuse=reuse)
            bn = Layers.batch_norm(conv, phase_train, name+'/bn', reuse=reuse, data_format=data_format)
            relu = Layers.leaky_relu(bn, name+'/relu')
            pool = Layers.max_pool(relu, name=name+'/pool', padding='SAME', data_format=data_format)

            return pool, relu

        for t, input_image in enumerate(input_sequence):

            with tf.name_scope('time_step-{}'.format(t)):

                # conv_kxy = 3
                # kout1 = 32
                # kout2 = 32
                # kout3 = 64
                # kout4 = 92
                conv_kxy = net_params['conv_kxy']
                kout1 = net_params['kout1']
                kout2 = net_params['kout2']
                kout3 = net_params['kout3']
                kout4 = net_params['kout4']

                # lstm_kxy = [7, 7]
                # lstm_kout1 = 32
                # lstm_kout2 = 32
                # lstm_kout3 = 64
                # lstm_kout4 = 92
                lstm_kxy = net_params['lstm_kxy']
                lstm_kout1 = net_params['lstm_kout1']
                lstm_kout2 = net_params['lstm_kout2']
                lstm_kout3 = net_params['lstm_kout3']
                lstm_kout4 = net_params['lstm_kout4']

                layer1p, layer1 = conv_bn_relu_pool(input_image, 'layer1', conv_kxy, kout1, reuse=t > 0)
                layer2p, layer2 = conv_bn_relu_pool(layer1p, 'layer2', conv_kxy, kout2, reuse=t > 0)
                layer3p, layer3 = conv_bn_relu_pool(layer2p, 'layer3', conv_kxy, kout3, reuse=t > 0)
                _, layer4 = conv_bn_relu_pool(layer3p, 'layer4', conv_kxy, kout4, reuse=t > 0)
                input_shape = input_image.get_shape().as_list()
                layer2_shape = layer2.get_shape().as_list()
                layer3_shape = layer3.get_shape().as_list()
                layer4_shape = layer4.get_shape().as_list()
                lstm4 = LayerNormConvLSTMCell(shape=layer4_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout4, data_format=data_format)
                lstm3 = LayerNormConvLSTMCell(shape=layer3_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout3, data_format=data_format)
                lstm2 = LayerNormConvLSTMCell(shape=layer2_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout2, data_format=data_format)
                lstm1 = LayerNormConvLSTMCell(shape=input_shape[2:], filter_size=lstm_kxy,
                                          num_features=lstm_kout1, data_format=data_format)

                if t == 0:
                    z1 = lstm1.zero_state(batch_size=input_shape[0])
                    init_state = [(tf.placeholder_with_default(z1[0], z1[0].get_shape(), 'init_state_ph_1'),
                                  tf.placeholder_with_default(z1[1], z1[1].get_shape(), 'init_state_ph_1_c'))]
                    z2 = lstm2.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z2[0], z2[0].get_shape(), 'init_state_ph_2'),
                                       tf.placeholder_with_default(z2[1], z2[1].get_shape(), 'init_state_ph_2_c')))
                    z3 = lstm3.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z3[0], z3[0].get_shape(), 'init_state_ph_3'),
                                       tf.placeholder_with_default(z3[1], z3[1].get_shape(), 'init_state_ph_3_c')))
                    z4 = lstm4.zero_state(batch_size=input_shape[0])
                    init_state.append((tf.placeholder_with_default(z4[0], z4[0].get_shape(), 'init_state_ph_4'),
                                       tf.placeholder_with_default(z4[1], z4[1].get_shape(), 'init_state_ph_4_c')))
                    self.states.append(init_state)

                lstm4_out, lstm4_state = lstm4(layer4, self.states[t][3], scope='lstm4', reuse=t > 0)

                lstm4_up = tf.image.resize_bilinear(tf.transpose(lstm4_out, [0, 2, 3, 1]), layer3_shape[2:])
                lstm4_up = tf.transpose(lstm4_up, [0, 3, 1, 2])
                lstm3_input = tf.concat(values=[lstm4_up, layer3], axis=1)
                lstm3_out, lstm3_state = lstm3(lstm3_input, self.states[t][2], scope='lstm3', reuse=t > 0)
                lstm3_up = tf.image.resize_bilinear(tf.transpose(lstm3_out, [0, 2, 3, 1]), layer2_shape[2:])
                lstm3_up = tf.transpose(lstm3_up, [0, 3, 1, 2])
                lstm2_input = tf.concat(values=[lstm3_up, layer2], axis=1)
                lstm2_out, lstm2_state = lstm2(lstm2_input, self.states[t][1], scope='lstm2', reuse=t > 0)
                lstm2_up = tf.image.resize_bilinear(tf.transpose(lstm2_out, [0, 2, 3, 1]), input_shape[2:])
                lstm2_up = tf.transpose(lstm2_up, [0, 3, 1, 2])
                lstm1_input = tf.concat(values=[lstm2_up, layer1], axis=1)
                lstm1_out, lstm1_state = lstm1(lstm1_input, self.states[t][0], scope='lstm1', reuse=t > 0)

                self.states.append([lstm1_state, lstm2_state, lstm3_state, lstm4_state])

                out_conv, _, _ = Layers.conv(lstm1_out, name='out_conv', kx=1, ky=1, kout=3, padding='SAME',
                                       data_format=data_format, reuse=t > 0)
                local_layers = {'input_image': input_image, 'layer1': layer1, 'layer1p': layer1p, 'layer2': layer2,
                                'layer2p': layer2p, 'layer3': layer3, 'layer3p': layer3p, 'layer4': layer4,
                                'lstm4_out': lstm4_out, 'lstm4_up': lstm4_up, 'lstm3_out': lstm3_out,
                                'lstm3_up': lstm3_up, 'lstm2_out': lstm2_out, 'lstm2_up': lstm2_up,
                                'lstm1_out': lstm4_out, 'out_conv': out_conv
                                }
                self.layers_dict.append(local_layers)

        return [ld['out_conv'] for ld in self.layers_dict]
