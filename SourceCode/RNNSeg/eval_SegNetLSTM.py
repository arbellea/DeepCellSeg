import os
import time

import numpy as np
import scipy.misc
import tensorflow as tf

from RNNSeg.LSTM_Network import LSTMNetwork
from RNNSeg.Params import ParamsEvalLSTM


def run_net():

    # Data input
    data_provider = params.data_provider

    with tf.name_scope('Data'):
        image_seq, filename_seq = data_provider.get_sequence(params.seq_length)

    # Build Network Graph
    net = LSTMNetwork()

    with tf.device('/gpu:0'):
        with tf.name_scope('run_tower'):
            with tf.variable_scope('net'):
                image_seq_norm = [tf.div(tf.subtract(im, params.norm),
                                                     params.norm) for im in image_seq]

                net_segs_logits = net.build(image_seq_norm, phase_train=True, net_params=params.net_params)
                net_segs = [tf.nn.softmax(logits=logits, dim=1) for logits in net_segs_logits]

    saver = tf.train.Saver(var_list=tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    coord = tf.train.Coordinator()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if params.load_checkpoint:
            saver.restore(sess, params.load_checkpoint_path)

        threads = tf.train.start_queue_runners(sess, coord=coord)
        elapsed_time = 0.
        end_time = 0.
        other_time = 0.
        options = tf.RunOptions()
        feed_dict = {}
        loop = True
        while loop:
            try:

                start_time = time.time()
                other_time += start_time - end_time
                seg_seq_out, states, file_names = sess.run([net_segs, net.states[-1], filename_seq], options=options,
                                                           feed_dict=feed_dict)
                end_time = time.time()
                elapsed_time += end_time-start_time

                for state_ph, last_state in zip(net.states[0], states):
                    feed_dict[state_ph[0]] = last_state[0]
                    feed_dict[state_ph[1]] = last_state[1]

                if not params.dry_run:
                    for file_name, image_seg in zip(file_names, seg_seq_out):
                        file_name = file_name.decode('utf-8')
                        seg_squeeze = np.squeeze(image_seg)
                        seg_squeeze = (seg_squeeze*255).astype(np.uint8)
                        seg_squeeze = seg_squeeze.transpose([1, 2, 0])
                        out_dir = params.experiment_out_dir
                        scipy.misc.toimage(seg_squeeze, cmin=0.0,
                                           cmax=255.).save(os.path.join(out_dir, os.path.basename(file_name)))
                        print("Saved File: {}".format(os.path.join(out_dir, os.path.basename(file_name))))

            except (ValueError, RuntimeError, KeyboardInterrupt):

                coord.request_stop()
                coord.join(threads)
                loop = False

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    params = ParamsEvalLSTM()
    run_net()
