import os
import time
import csv
import cv2
import numpy as np
import tensorflow as tf
import argparse

from RNNSeg.LSTM_Network import BiGRUNetwork
from RNNSeg.Isbi_Params import ParamsEvalIsbiBiGRU


def run_net():
    # Data input
    data_provider = params.data_provider

    with tf.name_scope('Data'):
        image_seq_fw, filename_seq_fw, image_seq_bw, filename_seq_bw = data_provider.get_sequence(params.seq_length)
    filename_seq_bw = filename_seq_bw[::-1]
    # Build Network Graph
    net_fw = BiGRUNetwork()
    net_bw = BiGRUNetwork()
    device = '/gpu:0' if params.useGPU else '/cpu:0'
    with tf.device(device):
        with tf.name_scope('run_tower'):
            image_seq_norm_fw = [tf.div(tf.subtract(im, params.norm),
                                        params.norm) for im in image_seq_fw]
            image_seq_norm_bw = [tf.div(tf.subtract(im, params.norm),
                                        params.norm) for im in image_seq_bw[::-1]]
            with tf.variable_scope('net'):
                _ = net_fw.build(image_seq_norm_fw, phase_train=True, net_params=params.net_params)
            with tf.variable_scope('net', reuse=True):
                _ = net_bw.build(image_seq_norm_bw, phase_train=True, net_params=params.net_params)
            fw_outputs = net_fw.fw_outputs
            bw_outputs = net_bw.bw_outputs

        fw_ph = tf.placeholder(tf.float32, params.image_size + (3,), 'fw_placeholder')
        bw_ph = tf.placeholder(tf.float32, params.image_size + (3,), 'bw_placeholder')
        merged = tf.add(fw_ph, bw_ph)
        final_out = tf.nn.softmax(merged)

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
        all_filenames = []
        sigout = [tf.nn.softmax(tf.transpose(o, (0, 2, 3, 1))) for o in net_fw.outputs]
        t = 0
        while loop:
            try:
                t += 1
                start_time = time.time()
                other_time += start_time - end_time
                fetch_out = sess.run([fw_outputs, bw_outputs, net_fw.states[-1], net_bw.states_back[-1],
                                      filename_seq_fw, filename_seq_bw, sigout, image_seq_fw], options=options,
                                     feed_dict=feed_dict)
                (seg_seq_out_fw, seg_seq_out_bw, states_fw, states_bw, file_names_fw, file_names_bw, sigoutnp,
                 imin) = fetch_out
                end_time = time.time()
                elapsed_time += end_time - start_time

                for state_ph, last_state in zip(net_fw.states[0], states_fw):
                    feed_dict[state_ph] = last_state

                for state_ph, last_state in zip(net_bw.states_back[0], states_bw):
                    feed_dict[state_ph] = last_state

                if not params.dry_run:
                    out_dir = params.experiment_tmp_fw_dir
                    for file_name, image_seg in zip(file_names_fw, seg_seq_out_fw):
                        file_name = file_name.decode('utf-8')
                        fw_squeeze = np.squeeze(image_seg)
                        fw_squeeze = fw_squeeze.transpose([1, 2, 0])
                        fw_filename = os.path.join(out_dir, os.path.basename(file_name))
                        np.save(fw_filename, fw_squeeze)
                        all_filenames.append(os.path.basename(file_name) + '.npy')
                        print("Saved File: {}".format(os.path.join(out_dir, os.path.basename(file_name))))
                    out_dir = params.experiment_tmp_bw_dir
                    for file_name, image_seg in zip(file_names_bw, seg_seq_out_bw):
                        file_name = file_name.decode('utf-8')
                        bw_squeeze = np.squeeze(image_seg)
                        bw_squeeze = bw_squeeze.transpose([1, 2, 0])
                        bw_filename = os.path.join(out_dir, os.path.basename(file_name))
                        np.save(bw_filename, bw_squeeze)
                        print("Saved File: {}".format(os.path.join(out_dir, os.path.basename(file_name))))

            except (ValueError, RuntimeError, KeyboardInterrupt, tf.errors.OutOfRangeError):

                coord.request_stop()
                coord.join(threads)
                loop = False

        if not params.dry_run:

            max_id = 0
            base_out_fname = os.path.join(params.experiment_isbi_out, 'mask{time:03d}.tif')
            for t, file_name in enumerate(all_filenames):
                fw_logits = np.load(os.path.join(params.experiment_tmp_fw_dir, file_name))
                bw_logits = np.load(os.path.join(params.experiment_tmp_bw_dir, file_name))
                feed_dict = {bw_ph: bw_logits, fw_ph: fw_logits}
                seg_out = sess.run(final_out, feed_dict)
                seg_cell = (seg_out[:, :, 1] > 0.3).astype(
                    np.float32)  # (np.argmax(seg_out, axis=2) == 1).astype(np.float16)
                cc_out = cv2.connectedComponentsWithStats(seg_cell.astype(np.uint8), 8, cv2.CV_32S)
                num_cells = cc_out[0]
                labels = cc_out[1]
                stats = cc_out[2]

                if t == 0:

                    labels_prev = cc_out[1]
                    max_id = num_cells
                    isbi_out_dict = {n: [n, 0, 0, 0] for n in range(1, num_cells)}
                    out_fname = base_out_fname.format(time=t)
                    cv2.imwrite(filename=out_fname, img=labels.astype(np.uint16))
                    continue
                matched_labels = np.zeros_like(labels, dtype=np.uint16)
                unmatched_indexes = list(range(1,  num_cells))
                continued_ids = []
                ids_to_split = set()
                split_dict = {}
                for p in np.arange(1, num_cells):
                    matching_mask = labels_prev[labels == p]
                    matching_candidates = np.unique(matching_mask)
                    for candidate in matching_candidates:
                        if candidate == 0:
                            continue
                        intersection = np.count_nonzero(matching_mask.flatten() == candidate)
                        area = stats[p, cv2.CC_STAT_AREA].astype(np.float32)
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
                                continued_ids.remove(candidate)
                            matched_labels[labels == p] = candidate
                            continue
                for cont_id in continued_ids:
                    isbi_out_dict[cont_id][2] += 1

                out_labels = matched_labels.copy()
                for parent_id, idxs in split_dict.items():
                    if len(idxs) > 1:
                        for idx in idxs:
                            max_id += 1
                            out_labels[labels == idx] = max_id
                            isbi_out_dict[max_id] = [max_id, t, t, parent_id]

                for unmatched in unmatched_indexes:
                    max_id += 1
                    out_labels[labels == unmatched] = max_id
                    isbi_out_dict[max_id] = [max_id, t, t, 0]
                out_fname = base_out_fname.format(time=t)
                cv2.imwrite(filename=out_fname, img=out_labels.astype(np.uint16))
                print("Saved File: {}".format(out_fname))


                labels_prev = out_labels

            csv_file_name = os.path.join(params.experiment_isbi_out, 'res_track.txt')

            with open(csv_file_name, 'w') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerows(isbi_out_dict.values())
            print("Saved File: {}".format(csv_file_name))

                # seg_out = (seg_out*255).astype(np.uint8)
                # scipy.misc.toimage(seg_out, cmin=0.0,
                #                    cmax=255.).save(os.path.join(out_dir, os.path.basename(file_name[:-4])))
                # print("Saved File: {}".format(os.path.join(out_dir, os.path.basename(file_name[:-4]))))

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
    args = parser.parse_args()
    args_dict = {key: val for key, val in vars(args).items() if val}
    params = ParamsEvalIsbiBiGRU(args_dict)
    run_net()
