import os
import time
import csv
import cv2
import numpy as np
import tensorflow as tf
import argparse
import scipy.ndimage
from RNNSeg.LSTM_Network import LSTMNetwork
from RNNSeg.Isbi_Params import ParamsEvalIsbiLSTM


def run_net():
    # Data input
    data_provider = params.data_provider

    with tf.name_scope('Data'):
        image_seq, filename_seq = data_provider.get_sequence(params.seq_length)

    # Build Network Graph
    net = LSTMNetwork()
    device = '/gpu:0' if params.useGPU else '/cpu:0'
    with tf.device(device):
        with tf.name_scope('run_tower'):
            image_seq_norm = [tf.div(tf.subtract(tf.multiply(im, params.renorm_factor), params.norm),
                                     params.norm) for im in image_seq]

            with tf.variable_scope('net'):
                outputs = net.build(image_seq_norm, phase_train=True, net_params=params.net_params,
                                    data_format=params.data_format)

    saver = tf.train.Saver(var_list=tf.global_variables())
    init_op = tf.group(tf.local_variables_initializer())

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
        if params.data_format == 'NCHW':
            sigout = [tf.nn.softmax(tf.transpose(o, (0, 2, 3, 1))) for o in outputs]
        else:

            sigout = [tf.nn.softmax(o) for o in outputs]
        t = -1
        max_id = 0
        labels_prev = None
        base_out_fname = os.path.join(params.experiment_isbi_out, 'mask{time:03d}.tif')
        base_out_vis_fname = os.path.join(params.experiment_tmp_vis_dir, 'mask{time:03d}.tif')
        base_out_overlay_fname = os.path.join(params.experiment_tmp_overlay_dir, 'mask{time:03d}.tif')
        while loop:
            try:
                t += 1
                start_time = time.time()
                other_time += start_time - end_time
                fetch_out = sess.run([net.states[-1],
                                      filename_seq, sigout[0][0], image_seq], options=options,
                                     feed_dict=feed_dict)
                (states_out, file_names, sigoutnp, imin) = fetch_out

                end_time = time.time()
                if params.data_format == 'NCHW':
                    if params.pad_y:
                        sigoutnp = sigoutnp[:-params.pad_y, :, :]
                        imin[0] = imin[0][:, :, :-params.pad_y, :]
                    if params.pad_x:
                        sigoutnp = sigoutnp[:, :-params.pad_x, :]
                        imin[0] = imin[0][:, :, :, :-params.pad_x]
                    sigoutnp = sigoutnp[16:-16, 16:-16, :]
                    imin[0] = imin[0][:, :, 16:-16, 16:-16]

                else:
                    if params.pad_y:
                        sigoutnp = sigoutnp[:-params.pad_y, :, :]
                        imin[0] = imin[0][:, :-params.pad_y, :, :]
                    if params.pad_x:
                        sigoutnp = sigoutnp[:, :-params.pad_x, :]
                        imin[0] = imin[0][:, :, :-params.pad_x, :]
                    sigoutnp = sigoutnp[16:-16, 16:-16, :]
                    imin[0] = imin[0][:, 16:-16, 16:-16, :]

                elapsed_time += end_time - start_time

                for state_ph, last_state in zip(net.states[0], states_out):
                    feed_dict[state_ph] = last_state

                if not params.dry_run:
                    seg_cell = np.equal(np.argmax(sigoutnp, 2), 1).astype(np.float32)
                    seg_edge = np.equal(np.argmax(sigoutnp, 2), 2).astype(np.float32)
                    seg_cell = scipy.ndimage.morphology.binary_fill_holes(seg_cell).astype(np.float32)
                    seg_edge = np.maximum((seg_edge - seg_cell), 0)
                    cc_out = cv2.connectedComponentsWithStats(seg_cell.astype(np.uint8), 8, cv2.CV_32S)
                    num_cells = cc_out[0]
                    labels = cc_out[1]
                    stats = cc_out[2]

                    dist, ind = scipy.ndimage.morphology.distance_transform_edt(1 - seg_cell, return_indices=True)
                    labels = labels[ind[0, :], ind[1, :]] * seg_edge * (dist < 20) + labels
                    for n in range(1, num_cells):
                        fill = scipy.ndimage.morphology.binary_fill_holes(labels == n).astype(np.float32)
                        labels = labels + (fill - (labels == n)) * n
                    sigoutnp_vis = np.flip(np.round(sigoutnp * (2 ** 16 - 1)).astype(np.uint16), 2)
                    cv2.imwrite(filename=base_out_vis_fname.format(time=t), img=sigoutnp_vis)
                    imrgb = np.stack([imin[0].squeeze()] * 3, 2)
                    imrgb = imrgb / imrgb.max()
                    overlay = np.round(imrgb * sigoutnp_vis)
                    cv2.imwrite(filename=base_out_overlay_fname.format(time=t), img=overlay.astype(np.uint16))

                    if t == 0:

                        labels_out = np.zeros_like(labels)
                        isbi_out_dict = {}
                        p = 0
                        for n in range(1, num_cells):
                            area = stats[n, cv2.CC_STAT_AREA]
                            if params.min_cell_size <= area <= params.max_cell_size:
                                p += 1
                                isbi_out_dict[p] = [p, 0, 0, 0]
                                labels_out[labels == n] = p

                            else:
                                labels[labels == n] = 0
                        max_id = labels_out.max()
                        labels_prev = labels_out
                        out_fname = base_out_fname.format(time=t)
                        cv2.imwrite(filename=out_fname, img=labels_out.astype(np.uint16))

                        continue
                    matched_labels = np.zeros_like(labels, dtype=np.uint16)
                    unmatched_indexes = list(range(1, num_cells))
                    continued_ids = []
                    ids_to_split = set()
                    split_dict = {}
                    for p in np.arange(1, num_cells):
                        area = stats[p, cv2.CC_STAT_AREA]
                        if not (params.min_cell_size <= area <= params.max_cell_size):
                            unmatched_indexes.remove(p)
                            continue

                        matching_mask = labels_prev[labels == p]
                        matching_candidates = np.unique(matching_mask)
                        for candidate in matching_candidates:
                            if candidate == 0:
                                continue
                            intersection = np.count_nonzero(matching_mask.flatten() == candidate)
                            area = np.count_nonzero(labels.flatten() == p)
                            if intersection / area > 0.5:
                                # Keep score of matched_indexes labels in order to know which are new cells
                                unmatched_indexes.remove(p)
                                if candidate not in continued_ids:
                                    # Keep score of matched_indexes labels in order to know which tracks to stop
                                    continued_ids.append(candidate)
                                    split_dict[candidate] = [p]  # Keep for mitosis detection
                                else:
                                    # Keep score of matched_indexes labels in order to know which tracks to stop
                                    split_dict[candidate].append(p)
                                    ids_to_split.add(candidate)

                                matched_labels[labels == p] = candidate
                                continue

                    out_labels = matched_labels.copy()
                    for parent_id, idxs in split_dict.items():
                        if len(idxs) == 2:
                            continued_ids.remove(parent_id)
                            for idx in idxs:
                                max_id += 1
                                out_labels[labels == idx] = max_id
                                isbi_out_dict[max_id] = [max_id, t, t, parent_id]

                    for cont_id in continued_ids:
                        isbi_out_dict[cont_id][2] += 1

                    for unmatched in unmatched_indexes:
                        max_id += 1
                        out_labels[labels == unmatched] = max_id
                        isbi_out_dict[max_id] = [max_id, t, t, 0]
                    out_fname = base_out_fname.format(time=t)
                    cv2.imwrite(filename=out_fname, img=out_labels.astype(np.uint16))
                    print("Saved File: {}".format(out_fname))

                    labels_prev = out_labels

            except (KeyboardInterrupt, tf.errors.OutOfRangeError):

                coord.request_stop()
                coord.join(threads)
                loop = False

        if not params.dry_run:
            csv_file_name = os.path.join(params.experiment_isbi_out, 'res_track.txt')
            with open(csv_file_name, 'w') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerows(isbi_out_dict.values())
            print("Saved File: {}".format(csv_file_name))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GAN Segmentation')
    parser.add_argument('-d', '--data_set', dest='selected_data_set', type=str,
                        help="ISBI Data Set Name")
    parser.add_argument('-s', '--sequence', type=int, dest='selected_seq',
                        help="1 or 2")
    parser.add_argument('--data_root_dir', dest='data_root_dir', type=str,
                        help="root directory for data sets")
    parser.add_argument('-m', '--model_path', dest='load_checkpoint_path', type=str,
                        help="Path to net model *.ckpt file")
    parser.add_argument('--output_dir', dest='final_out_dir', type=str,
                        help="Directory to save final outputs")
    parser.add_argument('--tmp_output_dir', dest='save_out_dir', type=str,
                        help="Directory to save temporary outputs")
    parser.add_argument('--cpu', dest='useGPU', action="store_false",
                        help="Use CPU instead of GPU")
    parser.add_argument('-r', '--renorm', dest='renorm_factor', type=float)

    parser.add_argument('--min_size', dest='min_cell_size', type=float)
    parser.add_argument('--max_size', dest='max_cell_size', type=float)

    args = parser.parse_args()
    args_dict = {key: val for key, val in vars(args).items() if val}
    params = ParamsEvalIsbiLSTM(args_dict)
    run_net()
