import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from RNNSeg.LSTM_Network import LSTMNetwork
from RNNSeg.Params import ParamsLSTM
from utils import summary_tag_replace


def log_print(string):
    now_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S: ')
    print(now_string + string)


tf_eps = tf.constant(1E-8)


def train():
    # Data input
    train_data_provider = params.train_data_provider
    val_data_provider = params.val_data_provider
    with tf.device('/cpu:0'):
        with tf.name_scope('Data'):
            (train_image_seq_batch, train_seg_seq_batch,
             train_is_last_batch) = train_data_provider.get_batch()
            val_image_seq_batch, val_seg_seq_batch, val_is_last_batch = val_data_provider.get_batch()
    # Build Network Graph
    net = LSTMNetwork()
    val_net = LSTMNetwork()

    def calc_losses(out_sequence, gt_sequence):
        t_loss = []
        t_jaccard = []
        eps = tf.constant(np.finfo(np.float32).eps)
        with tf.name_scope('loss_calc'):
            for time_step, (out_seg, gt_seg) in enumerate(zip(out_sequence, gt_sequence)):
                with tf.name_scope('timestep-{}'.format(time_step)):
                    if not params.one_seg and (time_step in params.skip_t_loss):
                        continue
                    gt_labels = tf.to_int32(tf.squeeze(gt_seg, axis=1))
                    gt_valid = tf.to_float(tf.greater(gt_labels, -1))
                    gt_fg = tf.equal(gt_labels, 1)
                    gt_bg = tf.to_float(tf.equal(gt_labels, 0))
                    gt_edge = tf.to_float(tf.equal(gt_labels, 2))
                    out_seg = tf.transpose(out_seg, [0, 2, 3, 1])
                    t_pixel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.maximum(gt_labels, 0),
                                                                                  logits=out_seg)
                    valid_pixel_loss = tf.multiply(t_pixel_loss, gt_valid)
                    cw = tf.constant(params.class_weights)
                    t_pixel_loss_weighted = valid_pixel_loss * (
                        gt_bg * cw[0] + tf.to_float(gt_fg) * cw[1] + gt_edge * cw[2])
                    t_loss.append(tf.divide(tf.reduce_sum(t_pixel_loss_weighted), tf.reduce_sum(gt_valid) + tf_eps))
                    out_fg = tf.equal(tf.argmax(out_seg, 3), 1)
                    intersection = tf.reduce_sum(tf.to_float(tf.logical_and(out_fg, gt_fg)),
                                                 axis=(1, 2), name='intersection')
                    union = tf.reduce_sum(tf.to_float(tf.logical_or(out_fg, gt_fg)), axis=(1, 2), name='union')

                    t_jaccard.append(tf.reduce_mean(tf.divide(intersection + eps, union + eps, name='jaccard')))

            jaccard = tf.divide(tf.add_n(t_jaccard), len(t_loss))
            loss = tf.divide(tf.add_n(t_loss), len(t_loss))
            local_dict = {k: v for k, v in locals().items() if type(v) is tf.Tensor}
        return loss, jaccard, t_pixel_loss_weighted, local_dict

    with tf.device('/gpu:0'):
        with tf.name_scope('train_tower'):
            with tf.variable_scope('net'):
                norm_train_image_seq_batch = [tf.div(tf.subtract(ti, params.norm),
                                                     params.norm) for ti in train_image_seq_batch]
                net_segs = net.build(norm_train_image_seq_batch, phase_train=True, net_params=params.net_params)
            log_print('Done building train network')
            net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net')
            not_last_batch = tf.expand_dims(tf.expand_dims(tf.expand_dims(1 - train_is_last_batch, axis=1), axis=2),
                                            axis=3)
            train_gated_states = [[s * not_last_batch for s in state] for state in net.states[-1]]
            if params.one_seg:
                train_loss, train_jaccard, _, _ = calc_losses([net_segs[-1]], train_seg_seq_batch)

            else:
                train_loss, train_jaccard, tmp, tmp2 = calc_losses(net_segs, train_seg_seq_batch)
            opt = tf.train.RMSPropOptimizer(params.learning_rate)
            grads_and_vars = opt.compute_gradients(train_loss, net_vars)
            global_step = tf.Variable(0, trainable=False)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                train_step = opt.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

        with tf.name_scope('val_tower'):
            with tf.variable_scope('net', reuse=True):
                norm_val_image_seq_batch = [tf.div(tf.subtract(vi, params.norm),
                                                   params.norm) for vi in val_image_seq_batch]
                valnet_segs = val_net.build(norm_val_image_seq_batch, phase_train=True, net_params=params.net_params)
                log_print('Done building validation network')
                val_not_last_batch = tf.expand_dims(
                    tf.expand_dims(tf.expand_dims(1 - val_is_last_batch, axis=1), axis=2),
                    axis=3)
                val_gated_states = [[s * val_not_last_batch for s in state] for state in val_net.states[-1]]
            if params.one_seg:
                val_loss, val_jaccard, _, _ = calc_losses([valnet_segs[-1]], val_seg_seq_batch)
            else:
                val_loss, val_jaccard, _, _ = calc_losses(valnet_segs, val_seg_seq_batch)

    # Tensorboard

    # Train Summaries
    tb_image = tf.transpose(train_image_seq_batch[-1], [0, 2, 3, 1])
    tf.summary.image('Image', tb_image, max_outputs=1, collections=['train_summaries'])

    tb_image = tf.transpose(net_segs[-1], [0, 2, 3, 1])
    train_out_img = tf.nn.softmax(tb_image)
    tf.summary.image('Segmentation', train_out_img, max_outputs=1, collections=['train_summaries'])

    tb_image = tf.transpose(train_seg_seq_batch[-1], [0, 2, 3, 1])
    tf.summary.image('Ground Truth', tb_image, max_outputs=1, collections=['train_summaries'])
    tf.summary.scalar('Loss', train_loss, collections=['train_summaries'])
    tf.summary.scalar('Jaccard', train_jaccard, collections=['train_summaries'])
    tf.summary.scalar('Learning Rate', params.learning_rate, collections=['train_summaries'])

    # Val Summaries
    with tf.name_scope('tb_val'):
        tb_image = tf.transpose(val_image_seq_batch[-1], [0, 2, 3, 1])
        tf.summary.image('Image', tb_image, max_outputs=1, collections=['val_summaries'])

        tb_image = tf.transpose(valnet_segs[-1], [0, 2, 3, 1])
        val_out_img = tf.nn.softmax(tb_image)
        tf.summary.image('Segmentation', val_out_img, max_outputs=1, collections=['val_summaries'])

        tb_image = tf.transpose(val_seg_seq_batch[-1], [0, 2, 3, 1])
        tf.summary.image('Ground Truth', tb_image, max_outputs=1, collections=['val_summaries'])
        tf.summary.scalar('Loss', val_loss, collections=['val_summaries'])
        tf.summary.scalar('Jaccard', val_jaccard, collections=['val_summaries'])

    q_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='Data')
    train_summaries = tf.summary.merge(tf.get_collection('train_summaries') + q_summaries)
    val_summaries = tf.summary.merge(tf.get_collection('val_summaries'))
    summaries_dir = params.experiment_log_dir
    train_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train'),
                                         graph=tf.get_default_graph())
    val_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'val'))

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=params.save_checkpoint_max_to_keep,
                           keep_checkpoint_every_n_hours=params.save_checkpoint_every_N_hours)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    coord = tf.train.Coordinator()
    with tf.Session(config=config) as sess:
        log_print('Started TF Session')
        sess.run(init_op)
        log_print('Variables Initialize')
        if params.load_checkpoint:
            saver.restore(sess, params.load_checkpoint_path)
            log_print('Params loaded from checkpoint: {}'.format(params.load_checkpoint_path))

        t = sess.run(global_step)
        threads = params.train_data_provider.start_queues(sess, coord)
        threads += params.val_data_provider.start_queues(sess, coord)
        log_print('Started {} data threads'.format(len(threads)))
        elapsed_time = 0.
        end_time = 0.
        other_time = 0.
        if params.profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            try:
                os.makedirs(os.path.join(summaries_dir, 'profile'))
            except OSError:
                pass
        else:
            options = tf.RunOptions()

        run_metadata = tf.RunMetadata()
        feed_dict = {}
        val_feed_dict = {}
        while t < params.num_iterations:
            try:
                start_time = time.time()
                other_time += start_time - end_time
                _, t, train_loss_eval, train_jaccard_eval, train_summaries_eval, last_state, gts, imin = sess.run(
                    [train_step, global_step, train_loss, train_jaccard, train_summaries,
                     train_gated_states] + [train_seg_seq_batch[-1], norm_train_image_seq_batch[-1]],
                    feed_dict=feed_dict,
                    options=options,
                    run_metadata=run_metadata)

                for state_pls, states in zip(net.states[0], last_state):
                    for pl, s in zip(state_pls, states):
                        feed_dict[pl] = s

                end_time = time.time()
                elapsed_time += end_time - start_time
                if params.profile:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(os.path.join(summaries_dir, 'profile', 'timeline_d{}.json'.format(t)), 'w') as f:
                        f.write(chrome_trace)

                if not t % 10:
                    log_print('Iteration {0}: Loss: {1:.3f}, Jaccard: {2:.3f}, Time: {3:.1f} seconds. '.format(t,
                                                                                                               train_loss_eval,
                                                                                                               train_jaccard_eval,
                                                                                                               elapsed_time / 10.0, ))
                    elapsed_time = 0.
                    other_time = 0.

                if not t % params.validation_interval:
                    vout, val_loss_eval, val_jaccard_eval, val_summaries_eval, val_last_state = sess.run(
                        [val_out_img, val_loss, val_jaccard,
                         val_summaries, val_gated_states],
                        feed_dict=val_feed_dict)

                    for state_pls, states in zip(val_net.states[0], val_last_state):
                        for pl, s in zip(state_pls, states):
                            val_feed_dict[pl] = s
                    if not params.dry_run:
                        val_summaries_eval = summary_tag_replace(val_summaries_eval, 'tb_val/', '')
                        val_writer.add_summary(val_summaries_eval, t)

                if not params.dry_run:
                    if not t % params.save_checkpoint_iteration:
                        save_path = saver.save(sess, os.path.join(params.experiment_save_dir, "model_%d.ckpt") % t)
                        log_print('Model saved to path: {}'.format(save_path))
                    if not t % params.write_to_tb_interval:
                        train_writer.add_summary(train_summaries_eval, t)
                        log_print('Printed iteration {} to Tensorboard'.format(t))
            except (ValueError, RuntimeError, KeyboardInterrupt):

                if not params.dry_run:
                    save_path = saver.save(sess, os.path.join(params.experiment_save_dir, "model_%d.ckpt") % t)
                    log_print('Model saved to path: {}'.format(save_path))

                coord.request_stop()
                coord.join(threads)

                return
        coord.request_stop()
        coord.join(threads)
        if not params.dry_run:
            save_path = saver.save(sess, os.path.join(params.experiment_save_dir, "model_final.ckpt") % t)
            log_print('Model saved to path: {}'.format(save_path))


if __name__ == '__main__':
    params = ParamsLSTM()

    train()
