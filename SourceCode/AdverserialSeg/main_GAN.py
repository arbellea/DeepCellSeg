import tensorflow as tf
from tensorflow.python.client import timeline
from Network import Network
from DataHandeling import CSVSegReader2, CSVSegReaderRandom2
import utils
import os
import re
import time
import numpy as np
import scipy.misc
import argparse
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
__author__ = 'arbellea@post.bgu.ac.il'


class SegUNetG(Network):
    def __init__(self, image_batch):
        self.image_batch = image_batch
        self.data_format = 'NCHW'
        super(SegUNetG, self).__init__()

    def build(self, phase_train, reuse=None, use_edges=False):

        # Layer 1
        kxy = 7
        kout = 16

        conv = self.conv('conv1', self.image_batch, kxy, kxy, kout)
        bn = self.batch_norm('bn1', conv, phase_train, reuse)
        relu = self.leaky_relu('relu1', bn)
        pool = self.max_pool('pool1', relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 2
        kxy = 3
        kout = 32

        conv = self.conv('conv2', pool, kxy, kxy, kout)
        bn = self.batch_norm('bn2', conv, phase_train, reuse)
        relu = self.leaky_relu('relu2', bn)
        pool = self.max_pool('pool2', relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 3
        kxy = 3
        kout = 32

        conv = self.conv('conv3', pool, kxy, kxy, kout)
        bn = self.batch_norm('bn3', conv, phase_train, reuse)
        relu = self.leaky_relu('relu3', bn)
        pool = self.max_pool('pool3', relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 4
        kxy = 1
        kout = 64

        conv = self.conv('conv4', pool, kxy, kxy, kout)
        bn = self.batch_norm('bn4', conv, phase_train, reuse)
        relu = self.leaky_relu('relu4', bn)

        # Start Upsampeling

        # Layer 5
        kxy = 3
        kout = 32
        in_shape = self.layers['relu3'].get_shape().as_list()
        if self.channel_first:
            out_shape = [in_shape[0], kout] + in_shape[2:]
        else:
            out_shape = self.layers['relu3'].get_shape().as_list()[:3] + [kout]
        conv = self.conv2d_transpose('conv5', relu, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])
        s = conv.get_shape().is_fully_defined()
        bn = self.batch_norm('bn5', conv, phase_train, reuse)
        relu = self.leaky_relu('relu5', bn)

        # Layer 6
        kxy = 3
        kout = 32
        concat = self.concat('concat1', [relu, self.layers['relu3']])

        in_shape = self.layers['relu2'].get_shape().as_list()
        if self.channel_first:
            out_shape = [in_shape[0], kout] + in_shape[2:]
        else:
            out_shape = self.layers['relu2'].get_shape().as_list()[:3] + [kout]
        conv = self.conv2d_transpose('conv6', concat, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])
        bn = self.batch_norm('bn6', conv, phase_train, reuse)
        relu = self.leaky_relu('relu6', bn)

        # Layer 7
        kxy = 7
        kout = 16
        concat = self.concat('concat7', [relu, self.layers['relu2']])

        in_shape = self.layers['relu1'].get_shape().as_list()
        if self.channel_first:
            out_shape = [in_shape[0], kout] + in_shape[2:]
        else:
            out_shape = self.layers['relu1'].get_shape().as_list()[:3] + [kout]
        conv = self.conv2d_transpose('conv7', concat, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])
        bn = self.batch_norm('bn7', conv, phase_train, reuse)
        relu = self.leaky_relu('relu7', bn)

        # Layer 8
        kxy = 1
        if use_edges:
            kout = 3
        else:
            kout = 1
        concat = self.concat('concat8', [relu, self.layers['relu1']])

        conv = self.conv('conv8', concat, kxy, kxy, kout)
        bn = self.batch_norm('bn8', conv, phase_train, reuse)
        relu = self.leaky_relu('relu8', bn)
        kxy = 3

        in_shape = self.image_batch.get_shape().as_list()
        if self.channel_first:
            out_shape = [in_shape[0], kout] + in_shape[2:]
        else:
            out_shape = self.image_batch.get_shape().as_list()[:3] + [kout]
        conv = self.conv2d_transpose('conv9', relu, kxy, kxy, kout, outshape=out_shape, stride=[1, 1, 1, 1])

        if use_edges:
            if self.channel_first:
                softmax = self.softmax('out', conv, 1)
                bg, fg, edge = tf.unstack(softmax, num=3, axis=1)
            else:
                softmax = self.softmax('out', conv, 3)
                bg, fg, edge = tf.unstack(softmax, num=3, axis=3)
            out = softmax  # tf.expand_dims(tf.add_n([fg, 2*edge]), 3)
            self.ge('prediction', fg, tf.constant(0.5))
            self.layers['bg'] = bg
            self.layers['fg'] = fg
            self.layers['edge'] = edge
        else:
            out = tf.sigmoid(conv, 'out')
            self.ge('prediction', out, tf.constant(0.5))

        return out, 0


class SegUNetG2(Network):
    def __init__(self, image_batch):
        self.image_batch = image_batch
        self.data_format = 'NCHW'
        super(SegUNetG2, self).__init__()

    def build(self, phase_train, reuse=None, use_edges=False):

        def conv_bn_rel(ten_in, kxy, kout, idx):
            conv_ = self.conv('conv%d' % idx, ten_in, kxy, kxy, kout, padding='SAME')
            bn = self.batch_norm('bn%d' % idx, conv_, phase_train, reuse)
            relu = self.leaky_relu('relu%d' % idx, bn)
            return relu

        # Layer 1 Left
        kxy = 3
        kout = 64
        idx = 1
        relu1_1 = conv_bn_rel(self.image_batch, kxy, kout, idx)
        idx += 1
        relu1_2 = conv_bn_rel(relu1_1, kxy, kout, idx)
        idx += 1
        pool1_2 = self.max_pool('pool1', relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 2 Left

        kout = 128
        relu2_1 = conv_bn_rel(pool1_2, kxy, kout, idx)
        idx += 1
        relu2_2 = conv_bn_rel(relu2_1, kxy, kout, idx)
        idx += 1
        pool2_2 = self.max_pool('pool2', relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 3 Left
        kout = 256
        relu3_1 = conv_bn_rel(pool2_2, kxy, kout, idx)
        idx += 1
        relu3_2 = conv_bn_rel(relu3_1, kxy, kout, idx)
        idx += 1
        pool3_2 = self.max_pool('pool3', relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 4 Left
        kout = 512
        relu4_1 = conv_bn_rel(pool3_2, kxy, kout, idx)
        idx += 1
        relu4_2 = conv_bn_rel(relu4_1, kxy, kout, idx)
        idx += 1
        pool4_2 = self.max_pool('pool4', relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 5 Left
        kout = 1024
        relu5_1 = conv_bn_rel(pool4_2, kxy, kout, idx)
        idx += 1
        relu5_2 = conv_bn_rel(relu5_1, kxy, kout, idx)
        idx += 1

        kout = 512
        in_shape = relu4_2.get_shape().as_list()
        if self.channel_first:
            out_shape = [in_shape[0], kout] + in_shape[2:]
        else:
            out_shape = in_shape[:3] + [kout]

        up_5_2 = self.conv2d_transpose('conv_up5', relu5_2, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])
        concat_4 = self.concat('concat4', [relu4_2, up_5_2], dim=3)
        relu_4_3 = conv_bn_rel(concat_4, kxy, kout, idx)
        idx += 1

        kout = 256
        relu4_4 = conv_bn_rel(relu_4_3, kxy, kout, idx)
        idx += 1
        in_shape = relu3_2.get_shape().as_list()
        if self.channel_first:
            out_shape = [in_shape[0], kout] + in_shape[2:]
        else:
            out_shape = in_shape[:3] + [kout]

        up_4_4 = self.conv2d_transpose('conv_up4', relu4_4, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])
        concat_3 = self.concat('concat3', [relu3_2, up_4_4], dim=3)
        relu3_3 = conv_bn_rel(concat_3, kxy, kout, idx)
        idx += 1
        kout = 128
        relu3_4 = conv_bn_rel(relu3_3, kxy, kout, idx)
        idx += 1
        in_shape = relu2_2.get_shape().as_list()
        if self.channel_first:
            out_shape = [in_shape[0], kout] + in_shape[2:]
        else:
            out_shape = in_shape[:3] + [kout]

        up_3_4 = self.conv2d_transpose('conv_up3', relu3_4, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])

        concat_2 = self.concat('concat2', [relu2_2, up_3_4], dim=3)
        relu2_3 = conv_bn_rel(concat_2, kxy, kout, idx)
        idx += 1
        kout = 64
        relu2_4 = conv_bn_rel(relu2_3, kxy, kout, idx)
        idx += 1
        in_shape = relu1_2.get_shape().as_list()
        if self.channel_first:
            out_shape = [in_shape[0], kout] + in_shape[2:]
        else:
            out_shape = in_shape[:3] + [kout]

        up_2_4 = self.conv2d_transpose('conv_up2', relu2_4, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])
        concat_1 = self.concat('concat1', [relu1_2, up_2_4], dim=3)
        relu1_3 = conv_bn_rel(concat_1, kxy, kout, idx)
        idx += 1
        relu1_4 = conv_bn_rel(relu1_3, kxy, kout, idx)
        idx += 1

        # Layer 8
        kxy = 1
        if use_edges:
            kout = 3
        else:
            kout = 1

        conv = self.conv('conv_out', relu1_4, kxy, kxy, kout)

        if use_edges:
            if self.channel_first:
                softmax = self.softmax('out', conv, 1)
                bg, fg, edge = tf.unstack(softmax, num=3, axis=1)
            else:
                softmax = self.softmax('out', conv, 3)
                bg, fg, edge = tf.unstack(softmax, num=3, axis=3)
            out = softmax  # tf.expand_dims(tf.add_n([fg, 2*edge]), 3)
            self.ge('prediction', fg, tf.constant(0.5))
            self.layers['bg'] = bg
            self.layers['fg'] = fg
            self.layers['edge'] = edge
        else:
            out = tf.sigmoid(conv, 'out')
            self.ge('prediction', out, tf.constant(0.5))
        return out, 0


class SegUNetG3(Network):
    def __init__(self, image_batch):
        self.image_batch = self.fix_image_size(image_batch)
        self.data_format = 'NCHW'
        super(SegUNetG3, self).__init__()

    @staticmethod
    def fix_image_size(image_batch):
        im_size = image_batch.get_shape().as_list()[1:3]
        imy = im_size[0]
        imx = im_size[1]
        if not ((np.floor(imx / (2.0 ** 4))) * 2 ** 4 == imx):
            new_w = ((np.floor(imx / (2.0 ** 4))) * 2 ** 4).astype(np.int32)
            start_x = np.floor((imx - new_w) / 2.0).astype(np.int32)
        else:
            new_w = imx
            start_x = 0
        if not ((np.floor(imy / (2.0 ** 4))) * 2 ** 4 == imy):
            new_h = ((np.floor(imx / (2.0 ** 4))) * 2 ** 4).astype(np.int32)
            start_y = np.floor((imy - new_h) / 2.0).astype(np.int32)
        else:
            new_h = imy
            start_y = 0
        fixed_image_batch = tf.slice(image_batch, [0, start_y, start_x, 0], [-1, new_h, new_w, -1])
        return fixed_image_batch

    def build(self, phase_train, reuse=None, use_edges=False):

        def conv_bn_rel(ten_in, kxy, kout, idx):
            conv_ = self.conv('conv%d' % idx, ten_in, kxy, kxy, kout, padding='SAME')
            bn = self.batch_norm('bn%d' % idx, conv_, phase_train, reuse)
            relu = self.leaky_relu('relu%d' % idx, bn)
            return relu

        # Layer 1 Left
        kxy = 3
        kout = 64
        idx = 1
        relu1_1 = conv_bn_rel(self.image_batch, kxy, kout, idx)
        idx += 1
        relu1_2 = conv_bn_rel(relu1_1, kxy, kout, idx)
        idx += 1
        pool1_2 = self.max_pool('pool1', relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 2 Left
        kout = 128
        relu2_1 = conv_bn_rel(pool1_2, kxy, kout, idx)
        idx += 1
        relu2_2 = conv_bn_rel(relu2_1, kxy, kout, idx)
        idx += 1
        pool2_2 = self.max_pool('pool2', relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 3 Left
        kout = 256
        relu3_1 = conv_bn_rel(pool2_2, kxy, kout, idx)
        idx += 1
        relu3_2 = conv_bn_rel(relu3_1, kxy, kout, idx)
        idx += 1
        pool3_2 = self.max_pool('pool3', relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 4 Left
        kout = 512
        relu4_1 = conv_bn_rel(pool3_2, kxy, kout, idx)
        idx += 1
        relu4_2 = conv_bn_rel(relu4_1, kxy, kout, idx)
        idx += 1
        pool4_2 = self.max_pool('pool4', relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Layer 5 Left
        kout = 1024
        relu5_1 = conv_bn_rel(pool4_2, kxy, kout, idx)
        idx += 1
        relu5_2 = conv_bn_rel(relu5_1, kxy, kout, idx)
        idx += 1

        kout = 512
        in_shape = relu4_2.get_shape().as_list()
        if self.channel_first:
            out_shape = [tf.shape(relu4_2)[0], kout] + in_shape[2:]
        else:
            out_shape = [tf.shape(relu4_2)[0]] + in_shape[1:3] + [kout]

        up_5_2 = self.conv2d_transpose('conv_up5', relu5_2, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])
        # relu4_2_c = tf.slice(relu4_2, [0,1,1,0],[-1, int(out_shape[1]), int(out_shape[2]), -1])
        concat_4 = self.concat('concat4', [relu4_2, up_5_2], dim=3)
        relu_4_3 = conv_bn_rel(concat_4, kxy, kout, idx)
        idx += 1

        kout = 256
        relu4_4 = conv_bn_rel(relu_4_3, kxy, kout, idx)
        idx += 1
        in_shape = relu3_2.get_shape().as_list()
        if self.channel_first:
            out_shape = [tf.shape(relu3_2)[0], kout] + in_shape[2:]
        else:
            out_shape = [tf.shape(relu3_2)[0]] + in_shape[1:3] + [kout]

        up_4_4 = self.conv2d_transpose('conv_up4', relu4_4, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])
        # relu3_2_c = tf.slice(relu3_2, [0, 1, 1, 0], [-1, int(out_shape[1]), int(out_shape[2]), -1])
        concat_3 = self.concat('concat3', [relu3_2, up_4_4], dim=3)
        relu3_3 = conv_bn_rel(concat_3, kxy, kout, idx)
        idx += 1

        kout = 128
        relu3_4 = conv_bn_rel(relu3_3, kxy, kout, idx)
        idx += 1
        in_shape = relu2_2.get_shape().as_list()
        if self.channel_first:
            out_shape = [tf.shape(relu2_2)[0], kout] + in_shape[2:]
        else:
            out_shape = [tf.shape(relu2_2)[0]] + in_shape[1:3] + [kout]
        up_3_4 = self.conv2d_transpose('conv_up3', relu3_4, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])
        # relu2_2_c = tf.slice(relu2_2, [0, 1, 1, 0], [-1, int(out_shape[1]), int(out_shape[2]), -1])
        concat_2 = self.concat('concat2', [relu2_2, up_3_4], dim=3)
        relu2_3 = conv_bn_rel(concat_2, kxy, kout, idx)
        idx += 1

        kout = 64
        relu2_4 = conv_bn_rel(relu2_3, kxy, kout, idx)
        idx += 1
        in_shape = relu1_2.get_shape().as_list()
        if self.channel_first:
            out_shape = [tf.shape(relu1_2)[0], kout] + in_shape[2:]
        else:
            out_shape = [tf.shape(relu1_2)[0]] + in_shape[1:3] + [kout]
        up_2_4 = self.conv2d_transpose('conv_up2', relu2_4, kxy, kxy, kout, outshape=out_shape, stride=[1, 2, 2, 1])
        # relu1_2_c = tf.slice(relu1_2, [0, 1, 1, 0], [-1, int(out_shape[1]), int(out_shape[2]), -1])
        concat_1 = self.concat('concat1', [relu1_2, up_2_4], dim=3)
        relu1_3 = conv_bn_rel(concat_1, kxy, kout, idx)
        idx += 1
        relu1_4 = conv_bn_rel(relu1_3, kxy, kout, idx)
        idx += 1

        # Layer 8
        kxy = 1
        if use_edges:
            kout = 3
        else:
            kout = 1

        conv = self.conv('conv_out', relu1_4, kxy, kxy, kout)

        if use_edges:
            if self.channel_first:
                softmax = self.softmax('out', conv, 1)
                bg, fg, edge = tf.unstack(softmax, num=3, axis=1)
            else:
                softmax = self.softmax('out', conv, 3)
                bg, fg, edge = tf.unstack(softmax, num=3, axis=3)
            out = softmax  # tf.expand_dims(tf.add_n([fg, 2*edge]), 3)
            self.ge('prediction', fg, tf.constant(0.5))
            self.layers['bg'] = bg
            self.layers['fg'] = fg
            self.layers['edge'] = edge
        else:
            out = tf.sigmoid(conv, 'out')
            self.ge('prediction', out, tf.constant(0.5))
        return out, 0


class SegNetG(Network):
    def __init__(self, image_batch):
        self.image_batch = image_batch
        self.data_format = 'NCHW'
        super(SegNetG, self).__init__()

    def build(self, phase_train, reuse=None, use_edges=False):

        crop_size = 0
        # Layer 1
        kxy = 9
        kout = 16
        conv = self.conv('conv1', self.image_batch, kxy, kxy, kout, padding='VALID')
        bn = self.batch_norm('bn1', conv, phase_train, reuse)
        relu = self.leaky_relu('relu1', bn)
        crop_size += (kxy - 1) / 2

        # Layer 2
        kxy = 7
        kout = 32
        conv = self.conv('conv2', relu, kxy, kxy, kout, padding='VALID')
        bn = self.batch_norm('bn2', conv, phase_train, reuse)
        relu = self.leaky_relu('relu2', bn)
        crop_size += (kxy - 1) / 2

        # Layer 3
        kxy = 5
        kout = 64
        conv = self.conv('conv3', relu, kxy, kxy, kout, padding='VALID')
        bn = self.batch_norm('bn3', conv, phase_train, reuse)
        relu = self.leaky_relu('relu3', bn)
        crop_size += (kxy - 1) / 2

        # Layer 4
        kxy = 4
        kout = 64
        conv = self.conv('conv4', relu, kxy, kxy, kout, padding='VALID')
        bn = self.batch_norm('bn4', conv, phase_train, reuse)
        relu = self.leaky_relu('relu4', bn)
        crop_size += (kxy - 1) / 2

        # Layer 5
        kxy = 1
        if use_edges:
            kout = 3
        else:
            kout = 1
        bn = self.batch_norm('bn5', relu, phase_train, reuse)
        conv = self.conv('conv5', bn, kxy, kxy, kout)
        crop_size += (kxy - 1) / 2
        if use_edges:

            if self.channel_first:
                softmax = self.softmax('out', conv, 1)
                bg, fg, edge = tf.unstack(softmax, num=3, axis=1)
            else:
                softmax = self.softmax('out', conv, 3)
                bg, fg, edge = tf.unstack(softmax, num=3, axis=3)
            out = softmax  # tf.expand_dims(tf.add_n([fg, 2*edge]), 3)
            self.ge('prediction', fg, tf.constant(0.5))
            self.layers['bg'] = bg
            self.layers['fg'] = fg
            self.layers['edge'] = edge
        else:
            out = tf.sigmoid(conv, 'out')
            self.ge('prediction', out, tf.constant(0.5))

        return out, int(crop_size)


class SegNetG2(Network):
    def __init__(self, image_batch):
        self.image_batch = image_batch
        super(SegNetG2, self).__init__()

    def build(self, phase_train, reuse=None, use_edges=False):
        crop_size = 0
        # Layer 1
        kxy = 3
        kout = 64
        conv = self.conv('conv1', self.image_batch, kxy, kxy, kout, padding='VALID')
        bn = self.batch_norm('bn1', conv, phase_train, reuse)
        relu = self.leaky_relu('relu1', bn)
        crop_size += (kxy - 1) / 2

        # Layer 2
        kxy = 3
        kout = 128
        conv = self.conv('conv2', relu, kxy, kxy, kout, padding='VALID')
        bn = self.batch_norm('bn2', conv, phase_train, reuse)
        relu = self.leaky_relu('relu2', bn)
        crop_size += (kxy - 1) / 2

        # Layer 3
        kxy = 3
        kout = 256
        conv = self.conv('conv3', relu, kxy, kxy, kout, padding='VALID')
        bn = self.batch_norm('bn3', conv, phase_train, reuse)
        relu = self.leaky_relu('relu3', bn)
        crop_size += (kxy - 1) / 2

        # Layer 4
        kxy = 3
        kout = 512
        conv = self.conv('conv4', relu, kxy, kxy, kout, padding='VALID')
        bn = self.batch_norm('bn4', conv, phase_train, reuse)
        relu = self.leaky_relu('relu4', bn)
        crop_size += (kxy - 1) / 2

        # Layer 5
        kxy = 1
        if use_edges:
            kout = 3
        else:
            kout = 1
        bn = self.batch_norm('bn5', relu, phase_train, reuse)
        conv = self.conv('conv5', bn, kxy, kxy, kout)
        crop_size += (kxy - 1) / 2
        if use_edges:
            if self.channel_first:

                softmax = self.softmax('out', conv, 1)
                bg, fg, edge = tf.unstack(softmax, num=3, axis=1)
            else:
                softmax = self.softmax('out', conv, 3)
                bg, fg, edge = tf.unstack(softmax, num=3, axis=3)
            out = softmax  # tf.expand_dims(tf.add_n([fg, 2*edge]), 3)
            self.ge('prediction', fg, tf.constant(0.5))
            self.layers['bg'] = bg
            self.layers['fg'] = fg
            self.layers['edge'] = edge
        else:
            out = tf.sigmoid(conv, 'out')
            self.ge('prediction', out, tf.constant(0.5))

        return out, int(crop_size)


class RibSegNet(Network):
    def __init__(self, image_batch, seg_batch):
        self.image_batch = image_batch
        self.seg_batch = seg_batch
        self.data_format = 'NCHW'
        super(RibSegNet, self).__init__()

    def build(self, phase_train, reuse=None):
        def rib(name, left, right, center, kxy, kout, stride=None):
            # Left

            conv_left = self.conv('left_' + name, left, kxy, kxy, kout, stride, biased=False)
            bn_left = self.batch_norm('bn_left_' + name, conv_left, phase_train, reuse)
            relu_left = self.leaky_relu('relu_left_' + name, bn_left)
            out_left = tf.nn.max_pool(relu_left, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

            # Right

            conv_right = self.conv('right_' + name, right, kxy, kxy, kout, stride, biased=False)
            bn_right = self.batch_norm('bn_right_' + name, conv_right, phase_train, reuse)
            relu_right = self.leaky_relu('relu_right_' + name, bn_right)
            out_right = tf.nn.max_pool(relu_right, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
            # Center

            conv_center = self.conv('center' + name, center, kxy, kxy, kout / 2, stride, biased=False)
            bn_center = self.batch_norm('bn_center_' + name, conv_center, phase_train, reuse)
            relu_center = self.leaky_relu('relu_center_' + name, bn_center)
            pool_center = tf.nn.max_pool(relu_center, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
            out_center = self.concat('center_out_' + name, [out_left, out_right, pool_center], dim=3)

            return out_left, out_right, out_center

        center0 = self.concat('center0', [self.image_batch, self.seg_batch], dim=3)

        # Layer 1
        k1 = 9
        k1out = 8
        left1, right1, center1 = rib('rib1', self.image_batch, self.seg_batch, center0, k1, k1out)

        # Layer 2
        k2 = 7
        k2out = 16
        left2, right2, center2 = rib('rib2', left1, right1, center1, k2, k2out)

        # Layer 3
        k3 = 5
        k3out = 32
        left3, right3, center3 = rib('rib3', left2, right2, center2, k3, k3out)

        # Concat

        concat3 = self.concat('concat_out', [left3, right3, center3])

        # FC 1

        fc1 = self.fc('fc1', concat3, 64, biased=False)
        relu_fc1 = self.leaky_relu('relu_fc1', fc1)
        fc2 = self.fc('fc2', relu_fc1, 64, biased=False)
        relu_fc2 = self.leaky_relu('relu_fc2', fc2)
        fc_out = self.fc('fc_out', relu_fc2, 1, biased=False)
        out = self.sigmoid('out', fc_out)

        return out


class RibSegNet2(Network):
    def __init__(self, image_batch, seg_batch):
        self.image_batch = image_batch
        self.seg_batch = seg_batch
        self.data_format = 'NCHW'
        super(RibSegNet2, self).__init__()

    def build(self, phase_train, reuse=None):
        concat_dim = 1 if self.channel_first else 3

        def rib(name, left, right, center, kxy, kout, stride=None):
            # Left

            conv_left = self.conv('left_' + name, left, kxy, kxy, kout, stride, biased=False)
            bn_left = self.batch_norm('bn_left_' + name, conv_left, phase_train, reuse)
            relu_left = self.leaky_relu('relu_left_' + name, bn_left)
            out_left = tf.nn.max_pool(relu_left, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

            # Right

            conv_right = self.conv('right_' + name, right, kxy, kxy, kout, stride, biased=False)
            bn_right = self.batch_norm('bn_right_' + name, conv_right, phase_train, reuse)
            relu_right = self.leaky_relu('relu_right_' + name, bn_right)
            out_right = tf.nn.max_pool(relu_right, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
            # Center

            conv_center = self.conv('center' + name, center, kxy, kxy, kout / 2, stride, biased=False)
            bn_center = self.batch_norm('bn_center_' + name, conv_center, phase_train, reuse)
            relu_center = self.leaky_relu('relu_center_' + name, bn_center)
            pool_center = tf.nn.max_pool(relu_center, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

            out_center = self.concat('center_out_' + name, [out_left, out_right, pool_center], dim=concat_dim)

            return out_left, out_right, out_center

        center0 = self.concat('center0', [self.image_batch, self.seg_batch], dim=concat_dim)

        # Layer 1
        k1 = 3
        k1out = 32
        left1, right1, center1 = rib('rib1', self.image_batch, self.seg_batch, center0, k1, k1out)

        # Layer 2
        k2 = 3
        k2out = 64
        left2, right2, center2 = rib('rib2', left1, right1, center1, k2, k2out)

        # Layer 3
        k3 = 3
        k3out = 128
        left3, right3, center3 = rib('rib3', left2, right2, center2, k3, k3out)

        # Concat

        concat3 = self.concat('concat_out', [left3, right3, center3], dim=concat_dim)

        # FC 1

        fc1 = self.fc('fc1', concat3, 64, biased=False)
        relu_fc1 = self.leaky_relu('relu_fc1', fc1)
        fc2 = self.fc('fc2', relu_fc1, 64, biased=False)
        relu_fc2 = self.leaky_relu('relu_fc2', fc2)
        fc_out = self.fc('fc_out', relu_fc2, 1, biased=False)
        out = self.sigmoid('out', fc_out)

        return out


class VGGNet(Network):
    def __init__(self, image_batch, seg_batch):
        self.image_batch = image_batch
        self.seg_batch = seg_batch
        self.data_format = 'NCHW'
        super(VGGNet, self).__init__()

    def build(self, phase_train, reuse=None):
        def conv_bn_relu(name, in_tensor, kxy, kout, stride=(1, 1, 1, 1)):
            conv = self.conv('conv_' + name, in_tensor, kxy, kxy, kout, stride, biased=True)
            bn = self.batch_norm('bn_' + name, conv, phase_train, reuse)
            relu = self.leaky_relu('relu' + name, bn)
            return relu

        in_concat = self.concat('concat_input', [self.image_batch, self.seg_batch], dim=3)
        conv1 = conv_bn_relu('1', in_concat, 3, 64)
        conv2 = conv_bn_relu('2', conv1, 3, 64)
        pool2 = self.max_pool('pool_2', conv2, [1, 2, 2, 1], [1, 2, 2, 1])

        conv3 = conv_bn_relu('3', pool2, 3, 128)
        conv4 = conv_bn_relu('4', conv3, 3, 128)
        pool4 = self.max_pool('pool_4', conv4, [1, 2, 2, 1], [1, 2, 2, 1])

        conv5 = conv_bn_relu('5', pool4, 3, 256)
        conv6 = conv_bn_relu('6', conv5, 3, 256)
        conv7 = conv_bn_relu('7', conv6, 3, 256)
        pool7 = self.max_pool('pool_7', conv7, [1, 2, 2, 1], [1, 2, 2, 1])

        conv8 = conv_bn_relu('8', pool7, 3, 256)
        conv9 = conv_bn_relu('9', conv8, 3, 256)
        conv10 = conv_bn_relu('10', conv9, 3, 256)
        pool10 = self.max_pool('pool_10', conv10, [1, 2, 2, 1], [1, 2, 2, 1])

        fc1 = self.fc('fc1', pool10, 4096, biased=False)
        relu_fc1 = self.leaky_relu('relu_fc1', fc1)
        fc2 = self.fc('fc2', relu_fc1, 4096, biased=False)
        relu_fc2 = self.leaky_relu('relu_fc2', fc2)
        fc_out = self.fc('fc_out', relu_fc2, 1, biased=False)
        out = self.sigmoid('out', fc_out)

        return out


class GANTrainer(object):
    def __init__(self, train_filenames, val_filenames, test_filenames, summaries_dir, num_examples=None, Unet=False,
                 RibD=True, crop_size=(128, 128)):

        self.train_filenames = train_filenames if isinstance(train_filenames, list) else [train_filenames]
        self.val_filenames = val_filenames if isinstance(val_filenames, list) else [val_filenames]
        self.test_filenames = test_filenames if isinstance(test_filenames, list) else [test_filenames]
        self.summaries_dir = summaries_dir
        self.train_csv_reader = CSVSegReaderRandom2(self.train_filenames, base_folder=base_folder,
                                                    image_size=image_size, capacity=100000, min_after_dequeue=10000,
                                                    num_threads=10, num_examples=num_examples, crop_size=crop_size)
        self.val_csv_reader = CSVSegReaderRandom2(self.val_filenames, base_folder=base_folder, image_size=image_size,
                                                  capacity=1000, min_after_dequeue=10, num_threads=2,
                                                  crop_size=crop_size)
        self.test_csv_reader = CSVSegReader2(self.test_filenames, base_folder=test_base_folder, image_size=image_size,
                                             capacity=100, min_after_dequeue=10, random=False, num_threads=2)
        # Set variable for net and losses
        self.net = None
        self.batch_loss_d = None
        self.batch_loss_g = None
        self.total_loss_d = None
        self.total_loss_g = None
        # Set validation variable for net and losses
        self.val_batch_loss_d = None
        self.val_batch_loss_g = None
        self.val_dice = None

        # Set placeholders for training parameters
        self.train_step_g = None
        self.train_step_d = None
        self.LR_g = tf.placeholder(tf.float32, [], 'learning_rate')
        self.LR_d = tf.placeholder(tf.float32, [], 'learning_rate')
        self.L2_coeff = tf.placeholder(tf.float32, [], 'L2_coeff')
        self.L1_coeff = tf.placeholder(tf.float32, [], 'L1_coeff')
        self.val_fetch = []
        # Set variables for tensorboard summaries
        self.loss_summary = None
        self.val_loss_summary = None
        self.objective_summary_d = None
        self.objective_summary_g = None
        self.val_objective_summary = None
        self.val_image_summary = None
        self.hist_summaries_d = []
        self.hist_summaries_g = []
        self.image_summaries = []
        if Unet:
            self.netG = SegUNetG3
        else:
            self.netG = SegNetG
        if RibD:
            self.netD = RibSegNet2
        else:
            self.netD = VGGNet
        self.t = tf.placeholder(tf.int64, (), 'iteration')

    @staticmethod
    def cross_entropy_loss(image, label):
        im_reshape = tf.reshape(image, (-1, 3))
        label_reshape = tf.reshape(label, (-1, 3))
        pix_loss = tf.nn.softmax_cross_entropy_with_logits(logits=im_reshape, labels=label_reshape)
        return tf.reduce_mean(pix_loss)

    def build(self, batch_size=1, use_edges=False, ce_percent=0, adv_ascent_temprature=0.):

        train_image_batch_gan, train_seg_batch_gan, _ = self.train_csv_reader.get_batch(batch_size)
        train_image_batch, train_seg_batch, _ = self.train_csv_reader.get_batch(batch_size)

        val_image_batch_gan, val_seg_batch_gan, _ = self.val_csv_reader.get_batch(batch_size)
        val_image_batch, val_seg_batch, _ = self.val_csv_reader.get_batch(batch_size)
        device = '/gpu:0' if (gpu_num > -1) else '/cpu:0'
        with tf.device(device):
            with tf.name_scope('tower0'):

                net_g = self.netG(train_image_batch_gan)
                with tf.variable_scope('net_g'):
                    gan_seg_batch, crop_size = net_g.build(True, use_edges=use_edges)
                target_hw = gan_seg_batch.get_shape().as_list()[2:]
                target_hw = [int(hw) for hw in target_hw]
                cropped_image = tf.slice(train_image_batch, [0, 0, crop_size, crop_size],
                                         [-1, -1, target_hw[0], target_hw[1]])
                if use_edges:

                    cropped_seg = tf.slice(tf.to_int32(train_seg_batch), [0, 0, crop_size, crop_size],
                                           [-1, -1, target_hw[0], target_hw[1]])
                    cropped_seg = tf.squeeze(tf.one_hot(indices=cropped_seg, depth=3, axis=1), axis=2)

                    cropped_seg_gan = tf.to_int32(tf.slice(train_seg_batch_gan, [0, 0, crop_size, crop_size],
                                                           [-1, -1, target_hw[0], target_hw[1]]))
                    cropped_seg_gan = tf.squeeze(tf.one_hot(indices=cropped_seg_gan, depth=3, axis=1), axis=2)

                else:
                    cropped_seg = tf.to_float(tf.equal(tf.slice(train_seg_batch, [0, 0, crop_size, crop_size],
                                                                [-1, -1, target_hw[0], target_hw[1]]), tf.constant(1.)))
                cropped_image_gan = tf.slice(train_image_batch_gan, [0, 0, crop_size, crop_size],
                                             [-1, -1, target_hw[0], target_hw[1]])

                full_batch_im = tf.concat(axis=0, values=[cropped_image, cropped_image_gan])
                full_batch_seg = tf.concat(axis=0, values=[cropped_seg, gan_seg_batch])
                full_batch_label = tf.concat(axis=0, values=[tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])])

                net_d = self.netD(full_batch_im, full_batch_seg)
                with tf.variable_scope('net_d'):
                    net_d.build(True)
                loss_d = tf.nn.sigmoid_cross_entropy_with_logits(logits=net_d.layers['fc_out'], labels=full_batch_label)
                log2_const = tf.constant(0.6931)
                # loss_g = tf.div(1., tf.maximum(loss_d, 0.01))
                loss_g = tf.nn.sigmoid_cross_entropy_with_logits(logits=net_d.layers['fc_out'],
                                                                 labels=1 - full_batch_label)
                loss_g_crossentropy = self.cross_entropy_loss(gan_seg_batch, cropped_seg_gan)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='net_g')
                updates_g = update_ops if update_ops else [tf.no_op()]
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='net_d')
                updates_d = update_ops if update_ops else [tf.no_op()]

                self.batch_loss_d = tf.reduce_mean(loss_d)

                adv_percent = 1 - ce_percent
                if not adv_ascent_temprature:
                    adv_ascent = 1
                else:
                    adv_ascent = 1 - tf.exp(-tf.div(tf.to_float(self.t),
                                                    tf.to_float(adv_ascent_temprature)))
                adv_percent_ascent = adv_ascent * adv_percent
                self.batch_loss_g = (tf.reduce_mean(loss_g) * adv_percent_ascent +
                                     loss_g_crossentropy * (1 - adv_percent_ascent))

                # self.batch_loss_g = tf.reduce_mean(loss_g) + loss_g_crossentropy

                # tf.get_variable_scope().reuse_variables()

                self.total_loss_d = self.batch_loss_d
                self.total_loss_g = self.batch_loss_g
        with tf.device(device):
            with tf.name_scope('val_tower0'):

                val_net_g = self.netG(val_image_batch_gan)
                val_cropped_image = tf.slice(val_image_batch, [0, 0, crop_size, crop_size],
                                             [-1, -1, target_hw[0], target_hw[1]])
                if use_edges:
                    val_cropped_seg = tf.to_int32(tf.slice(val_seg_batch, [0, 0, crop_size, crop_size],
                                                           [-1, -1, target_hw[0], target_hw[1]]))
                    val_cropped_seg = tf.squeeze(tf.one_hot(indices=val_cropped_seg, depth=3, axis=1), axis=2)
                    val_cropped_seg_gan = tf.to_int32(tf.slice(val_seg_batch_gan, [0, 0, crop_size, crop_size],
                                                               [-1, -1, target_hw[0], target_hw[1]]))
                    val_cropped_seg_gan = tf.squeeze(tf.one_hot(indices=val_cropped_seg_gan, depth=3, axis=1), axis=2)

                else:
                    val_cropped_seg = tf.to_float(tf.equal(tf.slice(val_seg_batch, [0, 0, crop_size, crop_size],
                                                                    [-1, -1, target_hw[0], target_hw[1]]),
                                                           tf.constant(1.)))

                val_cropped_image_gan = tf.slice(val_image_batch_gan, [0, 0, crop_size, crop_size],
                                                 [-1, -1, target_hw[0], target_hw[1]])

                with tf.variable_scope('net_g', reuse=True):
                    val_gan_seg_batch, _ = val_net_g.build(True, reuse=True, use_edges=use_edges)
                val_full_batch_im = tf.concat(axis=0, values=[val_cropped_image, val_cropped_image_gan])
                val_full_batch_seg = tf.concat(axis=0, values=[val_cropped_seg, val_gan_seg_batch])
                val_full_batch_label = tf.concat(axis=0, values=[tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])])
                val_net_d = self.netD(val_full_batch_im, val_full_batch_seg)
                with tf.variable_scope('net_d', reuse=True):
                    val_net_d.build(phase_train=True)

                val_loss_d = tf.nn.sigmoid_cross_entropy_with_logits(logits=val_net_d.layers['fc_out'],
                                                                     labels=val_full_batch_label)

                val_loss_g = tf.nn.sigmoid_cross_entropy_with_logits(logits=val_net_d.layers['fc_out'],
                                                                     labels=1 - val_full_batch_label)
                eps = tf.constant(np.finfo(np.float32).eps)
                if use_edges:
                    val_hard_seg = tf.expand_dims(tf.greater(tf.to_float(val_net_g.layers['fg']), tf.constant(0.5)), 1)
                    gt_hard_seg = tf.expand_dims(tf.greater(val_cropped_seg_gan[:, 1, :, :], tf.constant(0.5)), 1)
                else:
                    val_hard_seg = tf.greater(tf.to_float(val_net_g.layers['prediction']), tf.constant(0.5))
                    gt_hard_seg = val_cropped_seg_gan
                val_intersection = tf.to_float(tf.logical_and(gt_hard_seg, val_hard_seg))
                val_union = tf.to_float(tf.logical_or(gt_hard_seg, val_hard_seg))

                val_dice = tf.reduce_mean(tf.div(tf.add(tf.reduce_sum(val_intersection, [2, 3]), eps),
                                                 tf.add(tf.reduce_sum(val_union, [2, 3]), eps)))

                self.val_batch_loss_d = tf.reduce_mean(val_loss_d)
                self.val_batch_loss_g = tf.reduce_mean(val_loss_g)
                self.val_dice = val_dice
                self.val_fetch = [val_cropped_image_gan, gt_hard_seg, val_gan_seg_batch, val_hard_seg, val_intersection,
                                  val_union]
        with tf.device(device):
            opt_d = tf.train.RMSPropOptimizer(self.LR_d)
            opt_g = tf.train.RMSPropOptimizer(self.LR_g)
            grads_vars_d = opt_d.compute_gradients(self.total_loss_d, var_list=list(net_d.weights.values()))
            grads_vars_g = opt_g.compute_gradients(self.total_loss_g, var_list=list(net_g.weights.values()))

            if ce_percent == 1:
                self.train_step_d = tf.no_op()
            else:
                with tf.control_dependencies(updates_d):
                    self.train_step_d = opt_d.apply_gradients(grads_vars_d)
            with tf.control_dependencies(updates_g):
                self.train_step_g = opt_g.apply_gradients(grads_vars_g)

        self.objective_summary_d = [tf.summary.scalar('train/objective_d', self.total_loss_d)]
        self.objective_summary_g = [tf.summary.scalar('train/objective_g', self.total_loss_g)]
        self.val_objective_summary = [tf.summary.scalar('val/objective_d', self.val_batch_loss_d),
                                      tf.summary.scalar('val/objective_g', self.val_batch_loss_g),
                                      tf.summary.scalar('val/dice', val_dice)]
        val_cropped_image_gan_nhwc = tf.transpose(val_cropped_image_gan, perm=[0, 2, 3, 1])
        val_cropped_seg_gan_nhwc = tf.transpose(val_cropped_seg_gan, perm=[0, 2, 3, 1])
        val_gan_seg_batch_nhwc = tf.transpose(val_gan_seg_batch, perm=[0, 2, 3, 1])
        self.val_image_summary = [tf.summary.image('Raw', val_cropped_image_gan_nhwc),
                                  tf.summary.image('GT', val_cropped_seg_gan_nhwc),
                                  tf.summary.image('GAN', val_gan_seg_batch_nhwc)]

        for g, v in grads_vars_d:
            self.hist_summaries_d.append(tf.summary.histogram(v.op.name + '/value', v))
            self.hist_summaries_d.append(tf.summary.histogram(v.op.name + '/grad', g))
        for g, v in grads_vars_g:
            self.hist_summaries_g.append(tf.summary.histogram(v.op.name + '/value', v))
            self.hist_summaries_g.append(tf.summary.histogram(v.op.name + '/grad', g))

    def train(self, lr_g=0.1, lr_d=0.1, g_steps=1, d_steps=3, max_itr=100000,
              summaries=True, validation_interval=10,
              save_checkpoint_interval=200, plot_examples_interval=100, use_crossentropy=0):

        if summaries:
            train_merged_summaries_d = tf.summary.merge(self.objective_summary_d)
            train_merged_summaries_g = tf.summary.merge(self.objective_summary_g)
            val_merged_summaries = tf.summary.merge(self.val_objective_summary)
            val_merged_image_summaries = tf.summary.merge(self.val_image_summary)
            train_writer = tf.summary.FileWriter(os.path.join(self.summaries_dir, 'train'),
                                                 graph=tf.get_default_graph())
            val_writer = tf.summary.FileWriter(os.path.join(self.summaries_dir, 'val'))
        else:
            train_merged_summaries_g = tf.no_op()
            train_merged_summaries_d = tf.no_op()
            val_merged_summaries = tf.no_op()
            val_merged_image_summaries = tf.no_op()

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1000)

        coord = tf.train.Coordinator()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            sess.run(init_op)
            t = 0
            if restore:
                chkpt_info = tf.train.get_checkpoint_state(save_dir)
                if chkpt_info:
                    chkpt_filename = chkpt_info.model_checkpoint_path
                    t = int(re.findall(r'\d+', os.path.basename(chkpt_filename))[0]) + 1
                    saver.restore(sess, os.path.join(save_dir, os.path.basename(chkpt_filename)))

            threads = tf.train.start_queue_runners(sess, coord=coord)
            feed_dict = {self.LR_g: lr_g, self.LR_d: lr_d, self.t: t}

            train_fetch_d = [self.train_step_d, self.batch_loss_d, self.total_loss_d, train_merged_summaries_d]
            train_fetch_g = [self.train_step_g, self.batch_loss_g, self.total_loss_g, train_merged_summaries_g]

            train_d = True
            PROFILE = False
            if PROFILE:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                try:
                    os.makedirs(os.path.join(self.summaries_dir, 'profile'))
                except:
                    pass
            else:
                options = tf.RunOptions()
            run_metadata = tf.RunMetadata()

            for i in range(t, max_itr):
                feed_dict[self.t] = i
                if not i % (d_steps + g_steps):
                    train_d = True
                if i % (d_steps + g_steps) == d_steps:
                    train_d = False

                if use_crossentropy == 1:
                    train_d = False
                try:
                    if train_d:

                        start = time.time()
                        _, loss, objective, summaries_string = sess.run(train_fetch_d, feed_dict=feed_dict,
                                                                        options=options, run_metadata=run_metadata)
                        elapsed = time.time() - start
                        if PROFILE:
                            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            with open(os.path.join(self.summaries_dir, 'profile', 'timeline_d{}.json'.format(i)),
                                      'w') as f:
                                f.write(chrome_trace)
                        if not i % 10:
                            print("Train Step D: %d Elapsed Time: %g Objective: %g \n" % (i, elapsed, objective))
                        if summaries and not i % 100:
                            train_writer.add_summary(summaries_string, i)
                            train_writer.flush()
                    else:

                        start = time.time()
                        _, loss, objective, summaries_string = sess.run(train_fetch_g, feed_dict=feed_dict,
                                                                        options=options, run_metadata=run_metadata)
                        elapsed = time.time() - start
                        if PROFILE:
                            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            with open(os.path.join(self.summaries_dir, 'profile', 'timeline_g{}.json'.format(i)),
                                      'w') as f:
                                f.write(chrome_trace)
                        if not i % 10:
                            print("Train Step G: %d Elapsed Time: %g Objective: %g \n" % (i, elapsed, objective))
                        if summaries and not i % 100:
                            train_writer.add_summary(summaries_string, i)
                            train_writer.flush()
                    v_dice = 0
                    if not i % validation_interval:
                        start = time.time()
                        v_dice, summaries_string = sess.run([self.val_dice, val_merged_summaries])
                        elapsed = time.time() - start
                        print("Validation Step: %d Elapsed Time: %g Dice: %g\n" % (i, elapsed, v_dice))
                        if summaries:
                            val_writer.add_summary(summaries_string, i)
                            val_writer.flush()
                    if (not i % save_checkpoint_interval) or (i == max_itr - 1) or v_dice > 0.9:
                        save_path = saver.save(sess, os.path.join(save_dir, "model_%d.ckpt") % i)
                        print("Model saved in file: %s" % save_path)
                    if not i % plot_examples_interval:  # or (i < plot_examples_interval and not i % (d_steps+g_steps)):
                        fetch = sess.run(val_merged_image_summaries)
                        val_writer.add_summary(fetch, i)
                        val_writer.flush()
                except (ValueError, RuntimeError, KeyboardInterrupt):
                    coord.request_stop()
                    coord.join(threads)
                    save_path = saver.save(sess, os.path.join(save_dir, "model_%d.ckpt") % i)
                    print("Model saved in file: %s Because of error" % save_path)

                    return False
            coord.request_stop()
            coord.join(threads)
            return True

    def validate_checkpoint(self, chekpoint_path, batch_size, use_edges):

        test_image_batch_gan, test_seg_batch_gan, filename_batch = self.test_csv_reader.get_batch(batch_size)
        net_g = self.netG(test_image_batch_gan)
        with tf.variable_scope('net_g'):
            gan_seg_batch, crop_size = net_g.build(False, use_edges)
        target_hw = gan_seg_batch.get_shape().as_list()[1:3]
        cropped_image = tf.slice(test_image_batch_gan, [0, 0, crop_size, crop_size],
                                 [-1, -1, target_hw[0], target_hw[1]])
        cropped_seg = tf.slice(test_seg_batch_gan, [0, 0, crop_size, crop_size], [-1, -1, target_hw[0], target_hw[1]])
        eps = tf.constant(np.finfo(np.float32).eps)
        test_hard_seg = tf.round(gan_seg_batch)
        test_intersection = tf.multiply(cropped_seg, test_hard_seg)
        test_union = tf.subtract(tf.add(cropped_seg, test_hard_seg), test_intersection)
        test_dice = tf.reduce_mean(tf.div(tf.add(tf.reduce_sum(test_intersection, [2, 3]), eps),
                                          tf.add(tf.reduce_sum(test_union, [2, 3]), eps)))
        saver = tf.train.Saver(var_list=tf.global_variables(), allow_empty=True)
        coord = tf.train.Coordinator()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:

            sess.run(init_op)
            threads = tf.train.start_queue_runners(sess, coord=coord)
            saver.restore(sess, chekpoint_path)
            print('Showing Images (Close figures to continue to next example)')
            for _ in range(3):
                dice, image, seg, gan_seg = sess.run([test_dice, cropped_image, cropped_seg, gan_seg_batch])
                for i in range(image.shape[0]):
                    img = np.squeeze(image[i])
                    seg_img = np.squeeze(seg[i])
                    gan_seg_img = np.squeeze(gan_seg[i])
                    if plt:
                        plt.figure(1)
                        plt.imshow(img)
                        plt.figure(2)
                        plt.imshow(seg_img)
                        plt.figure(3)
                        plt.imshow(gan_seg_img)
                        plt.show()
            coord.request_stop()
            coord.join(threads)

    def write_full_output_from_checkpoint(self, chekpoint_path, batch_size, use_edges, reuse=True):

        test_image_batch_gan, test_seg_batch_gan, filename_batch = self.test_csv_reader.get_batch(batch_size)
        net_g = self.netG(test_image_batch_gan)
        with tf.variable_scope('net_g'):
            gan_seg_batch, crop_size = net_g.build(True, reuse, use_edges)
        # target_hw = gan_seg_batch.get_shape().as_list()[1:3]
        # cropped_image = tf.slice(test_image_batch_gan, [0, crop_size, crop_size, 0],
        #                                                [-1, target_hw[0], target_hw[1], -1])
        # cropped_seg = tf.slice(test_seg_batch_gan, [0, crop_size, crop_size, 0], [-1, target_hw[0], target_hw[1], -1])
        # eps = tf.constant(np.finfo(np.float32).eps)
        # test_hard_seg = tf.round(gan_seg_batch)
        # test_intersection = tf.multiply(cropped_seg, test_hard_seg)
        # test_union = tf.subtract(tf.add(cropped_seg, test_hard_seg), test_intersection)
        # test_dice = tf.reduce_mean(tf.div(tf.add(tf.reduce_sum(test_intersection, [1,2]), eps),
        #                                 tf.add(tf.reduce_sum(test_union, [1,2]), eps)))
        saver = tf.train.Saver(var_list=tf.global_variables(), allow_empty=True)
        coord = tf.train.Coordinator()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
            sess.run(init_op)
            threads = tf.train.start_queue_runners(sess, coord=coord)
            saver.restore(sess, chekpoint_path)
            try:
                while True:
                    gan_seg, file_names = sess.run([gan_seg_batch, filename_batch])

                    for i in range(gan_seg.shape[0]):
                        file_name = file_names[i].decode('utf-8')
                        gan_seg_squeeze = np.squeeze(gan_seg[i])
                        if not os.path.exists(os.path.dirname(os.path.join(out_dir,
                                                                           os.path.basename(chkpt_full_filename),
                                                                           file_name[2:]))):
                            os.makedirs(os.path.dirname(os.path.join(out_dir, os.path.basename(chkpt_full_filename),
                                                                     file_name[2:])))
                            print("made dir")
                        scipy.misc.toimage(gan_seg_squeeze, cmin=0.0,
                                           cmax=2.).save(os.path.join(out_dir, os.path.basename(chkpt_full_filename),
                                                                      file_name[2:]))
                        print("Saved File: {}".format(file_name[2:]))
                        # coord.request_stop()
                        # coord.join(threads)
            except (ValueError, RuntimeError, KeyboardInterrupt, tf.errors.OutOfRangeError):
                coord.request_stop()
                coord.join(threads)
                print("Stopped Saving Files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GAN Segmentation')
    parser.add_argument('-n', '--example_num', default=0.0, type=float,
                        help="Number of examples from train set, can be less than one for a franction of the frame")
    parser.add_argument('-b', '--batch_size', type=int, default=200, help="Number of examples per batch")
    parser.add_argument('-g', '--gpu_num', type=int, default=0, help="Number of examples from train set")
    parser.add_argument('-c', '--checkpoint', help="Load Specific checkpint for test")
    parser.add_argument('-r', '--restore', help="Restore from last checkpoint", action="store_true")
    parser.add_argument('-t', '--test_only', help="Skip training phase and only run test", action="store_true")
    parser.add_argument('-N', '--run_name', default='default_run', help="Name of the run")
    parser.add_argument('-e', '--use_edges', help="segment to foregorund, background and edge", action="store_false")
    parser.add_argument('-u', '--unet', help=" Use Unet instead of regular CNN", action="store_true")
    parser.add_argument('-o', '--out_to_file', help="Write console output to file ", action="store_true")
    parser.add_argument('-C', '--use_crossentropy', type=float, default=0.0, help="Percentage of cross-entropy lossin "
                                                                                  "total loss")
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help="Learning Rate for training")
    parser.add_argument('-m', '--max_iter', type=int, default=1000000, help="Maximum number of iterations")
    parser.add_argument('-d', '--data', default='', help="Path to data set")
    parser.add_argument('-i', '--image_size', help="Image Size Y,X"
                                                   "ex. -i 512,640")
    parser.add_argument('--crop_size', help="Crop Size Y,X ex. -i 128,128", default='128,128')
    parser.add_argument('-s', '--switch_rate', default='20,20',
                        help="Number of steps for Generator and Discriminator. "
                             "ex. -s 20,30 20 for Generator and 30 for Discriminator")
    parser.add_argument('-a', '--adversarial_ascent', type=int, default=0, help="Coefficiant for adversarial ascent")
    parser.add_argument('--vgg_disc', action="store_false",
                        help="Use VGG architecutre instead of RibCage Architecture for discriminator")
    parser.add_argument('--classic_disc', action="store_false", help="Use Clasic instead of RibCage Architecture for discriminator")
    parser.add_argument('--output_dir', default='~/DeepCellSegOutput',
                        help="Directory to save outputs")
    args = parser.parse_args()

    print(args)
    example_num = args.example_num
    if example_num:
        print("Examples set to: {}".format(example_num))

    batch_size = args.batch_size
    print("Batch Size set to: {}".format(batch_size))

    gpu_num = args.gpu_num
    if gpu_num > -1:
        print("GPU set to: {}".format(gpu_num))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    checkpoint = args.checkpoint
    restore = args.restore
    test_only = args.test_only
    run_name = args.run_name
    use_edges_flag = args.use_edges
    Unet = args.unet
    use_crossentropy_flag = args.use_crossentropy
    learning_rate = args.learning_rate
    max_iter = args.max_iter
    data_set_name = args.data

    gsteps, dsteps = args.switch_rate.split(',')
    gsteps = int(gsteps)
    dsteps = int(dsteps)

    output_to_file = args.out_to_file

    SNAPSHOT_DIR = os.path.join(args.output_dir, 'Snapshots')
    LOG_DIR = os.path.join(args.output_dir, 'Logs')
    OUT_DIR = os.path.join(args.output_dir, 'Outputs')

    base_folder = data_set_name if data_set_name[-1] == '/' else data_set_name + '/'
    train_filename = os.path.join(base_folder, 'train.csv')
    val_filename = os.path.join(base_folder, 'val.csv')
    test_filename = os.path.join(base_folder, 'test.csv')
    test_base_folder = base_folder
    if args.image_size:
        image_y, image_x = args.image_size.split(',')
        image_size = (int(image_y), int(image_x), 1)
    else:
        image_size = (512, 640, 1)
    crop_y, crop_x = args.crop_size.split(',')
    crop_size = (int(crop_y), int(crop_x), 1)
    # image_size = (256,160, 1)
    # image_size = (64, 64, 1)
    save_dir = os.path.join(SNAPSHOT_DIR, run_name)
    out_dir = os.path.join(OUT_DIR, run_name)
    summaries_dir_name = os.path.join(LOG_DIR, run_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(summaries_dir_name):
        os.makedirs(summaries_dir_name)
    if output_to_file:
        f = open("{}.txt".format(run_name), 'w')
        orig_stdout = sys.stdout
        sys.stdout = f
    print("Start")
    trainer = GANTrainer(train_filename, val_filename, test_filename, summaries_dir_name, num_examples=example_num,
                         Unet=Unet, RibD=args.classic_disc, crop_size=crop_size)
    success_flag = False
    if not test_only:
        print("Build Trainer")
        trainer.build(batch_size=batch_size, use_edges=use_edges_flag, ce_percent=use_crossentropy_flag,
                      adv_ascent_temprature=args.adversarial_ascent)
        print("Start Training")

        success_flag = trainer.train(lr_g=learning_rate, lr_d=learning_rate, g_steps=gsteps, d_steps=dsteps,
                                     max_itr=max_iter,
                                     summaries=True, validation_interval=100,
                                     save_checkpoint_interval=50000, plot_examples_interval=1000,
                                     use_crossentropy=use_crossentropy_flag)
    if success_flag or test_only:
        print("Writing Output")
        output_chkpnt_info = tf.train.get_checkpoint_state(save_dir)
        if output_chkpnt_info:
            if not checkpoint:
                chkpt_full_filename = output_chkpnt_info.model_checkpoint_path

            print("Loading Checkpoint: {}".format(os.path.basename(chkpt_full_filename)))
            trainer.write_full_output_from_checkpoint(os.path.join(save_dir, os.path.basename(chkpt_full_filename)), 10,
                                                      use_edges_flag, reuse=(not test_only))
        else:
            print("Could not load any checkpoint")
    print("Done!")
    if output_to_file:
        f.close()
        sys.stdout = orig_stdout
