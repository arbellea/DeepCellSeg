
import tensorflow as tf
from Network import Network
from DataHandeling import CSVSegReader
# import utils
import os
import re
import time

# import matplotlib.pyplot as plt
import numpy as np
__author__ = 'assafarbelle'

DATA_DIR = os.environ['DATA_DIR']
SNAPSHOT_DIR = os.environ['SNAPSHOT_DIR']
LOG_DIR = os.environ['LOG_DIR']
restore = True
run_num = '1'
base_folder = os.path.join(DATA_DIR, 'Alon_Small/')
train_filename = os.path.join(base_folder, 'test.csv')
val_filename = os.path.join(base_folder, 'val.csv')
image_size = (64, 64, 1)
save_dir = os.path.join(SNAPSHOT_DIR, 'Alon_Small', 'GAN', run_num)
summaries_dir_name = os.path.join(LOG_DIR, 'Alon_Small', 'GAN', run_num)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(summaries_dir_name):
    os.makedirs(summaries_dir_name)


class SegNetG(Network):

    def __init__(self, image_batch):
        self.image_batch = image_batch
        super(SegNetG, self).__init__()

    def build(self, phase_train):
        crop_size = 0
        # Layer 1
        kxy = 7
        kout = 16
        bn = self.batch_norm('bn1', self.image_batch, phase_train)
        conv = self.conv('conv1', bn, kxy, kxy, kout)
        relu = self.leaky_relu('relu1', conv)
        crop_size += (kxy-1)/2

        # Layer 2
        kxy = 5
        kout = 32
        bn = self.batch_norm('bn2', relu, phase_train)
        conv = self.conv('conv2', bn, kxy, kxy, kout)
        relu = self.leaky_relu('relu2', conv)
        crop_size += (kxy-1)/2

        # Layer 3
        kxy = 3
        kout = 64
        bn = self.batch_norm('bn3', relu, phase_train)
        conv = self.conv('conv3', bn, kxy, kxy, kout)
        relu = self.leaky_relu('relu3', conv)
        crop_size += (kxy-1)/2

        # Layer 4
        kxy = 1
        kout = 64
        bn = self.batch_norm('bn4', relu, phase_train)
        conv = self.conv('conv4', bn, kxy, kxy, kout)
        relu = self.leaky_relu('relu4', conv)
        crop_size += (kxy-1)/2

        # Layer 5
        kxy = 1
        kout = 1
        bn = self.batch_norm('bn5', relu, phase_train)
        conv = self.conv('conv5', bn, kxy, kxy, kout)
        out = self.sigmoid('out', conv)
        crop_size += (kxy-1)/2

        return out, crop_size


class RibSegNet(Network):
    def __init__(self, image_batch, seg_batch):
        self.image_batch = image_batch
        self.seg_batch = seg_batch
        super(RibSegNet, self).__init__()

    def build(self, phase_train):

        def rib(name, left, right, center, kxy, kout, stride=None):
            # Left
            bn_left = self.batch_norm('bn_left_' + name, left, phase_train)
            conv_left = self.conv('left_' + name, bn_left, kxy, kxy, kout, stride, biased=False)
            out_left = self.leaky_relu('relu_left_' + name, conv_left)

            # Right
            bn_right = self.batch_norm('bn_right_' + name, right, phase_train)
            conv_right = self.conv('right_' + name, bn_right, kxy, kxy, kout, stride, biased=False)
            out_right = self.leaky_relu('relu_right_' + name, conv_right)
            # Center
            bn_center = self.batch_norm('bn_center_' + name, center, phase_train)
            conv_center = self.conv('center' + name, bn_center, kxy, kxy, kout / 2, stride, biased=False)
            relu_center = self.leaky_relu('relu_center_' + name, conv_center)
            out_center = self.concat('center_out_' + name, [out_left, out_right, relu_center], dim=3)

            return out_left, out_right, out_center

        center0 = self.concat('center0', [self.image_batch, self.seg_batch], dim=3)

        # Layer 1
        k1 = 7
        k1out = 8
        left1, right1, center1 = rib('rib1', self.image_batch, self.seg_batch, center0, k1, k1out)

        # Layer 2
        k2 = 5
        k2out = 16
        left2, right2, center2 = rib('rib2', left1, right1, center1, k2, k2out)

        # Layer 3
        k3 = 3
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


class GANTrainer(object):

    def __init__(self, train_filenames, val_filenames, summaries_dir):

        self.train_filenames = train_filenames if isinstance(train_filenames, list) else [train_filenames]
        self.val_filenames = val_filenames if isinstance(val_filenames, list) else [val_filenames]
        self.summaries_dir = summaries_dir
        self.train_csv_reader = CSVSegReader(self.train_filenames, base_folder=base_folder, image_size=image_size)
        self.val_csv_reader = CSVSegReader(self.val_filenames, base_folder=base_folder, image_size=image_size)
        # Set variable for net and losses
        self.net = None
        self.batch_loss = None
        self.total_loss = None
        # Set validation variable for net and losses
        self.val_net = None
        self.val_batch_loss = None
        # Set placeholders for training parameters
        self.train_step_g = None
        self.train_step_d = None
        self.LR_g = tf.placeholder(tf.float32, [], 'learning_rate')
        self.LR_d = tf.placeholder(tf.float32, [], 'learning_rate')
        self.L2_coeff = tf.placeholder(tf.float32, [], 'L2_coeff')
        self.L1_coeff = tf.placeholder(tf.float32, [], 'L1_coeff')

        # Set variables for tensorboard summaries
        self.loss_summary = None
        self.val_loss_summary = None
        self.objective_summary = None
        self.hist_summaries = []
        self.image_summaries = []

    def build(self, batch_size=1):

        train_image_batch_gan, train_seg_batch_gan = self.train_csv_reader.get_batch(batch_size)
        train_image_batch, train_seg_batch = self.train_csv_reader.get_batch(batch_size)

        val_image_batch_gan, val_seg_batch_gan = self.val_csv_reader.get_batch(batch_size)
        val_image_batch, val_seg_batch = self.val_csv_reader.get_batch(batch_size)


        with tf.device('/cpu:0'):
            with tf.name_scope('tower0'):

                net_g = SegNetG(train_image_batch_gan)
                with tf.variable_scope('net_g'):
                    gan_seg_batch, crop_size = net_g.build(True)
                target_hw = gan_seg_batch.get_shape().as_list()[1:3]
                croped_image = tf.slice(train_image_batch, [0, crop_size, crop_size, 0], [-1, target_hw[0], target_hw[1], -1])
                croped_seg = tf.slice(train_seg_batch, [0, crop_size, crop_size, 0], [-1, target_hw[0], target_hw[1], -1])
                croped_image_gan = tf.slice(train_image_batch_gan,  [0, crop_size, crop_size, 0], [-1, target_hw[0], target_hw[1], -1])

                full_batch_im = tf.concat(0,[croped_image, croped_image_gan])
                full_batch_seg = tf.concat(0,[croped_seg, gan_seg_batch])
                full_batch_label = tf.concat(0, [tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])])

                net_d = RibSegNet(full_batch_im, full_batch_seg)
                with tf.variable_scope('net_d'):
                    net_d.build(True)
                loss_d = tf.nn.sigmoid_cross_entropy_with_logits(net_d.layers['fc_out'], full_batch_label)
                log2_const = tf.constant(0.6931)
                loss_g = tf.div(1., tf.maximum(loss_d, 0.01)) #tf.abs(tf.sub(loss_d,log2_const)) #10*tf.exp(-loss_d)#

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                updates = tf.group(*update_ops) if update_ops else tf.no_op()
                with tf.control_dependencies([updates]):
                    self.batch_loss_d = tf.reduce_mean(loss_d)
                    self.batch_loss_g = tf.reduce_mean(loss_g)

                tf.get_variable_scope().reuse_variables()

                loss_l2_d = tf.reduce_mean([tf.nn.l2_loss(v) for v in net_d.weights.values()])
                loss_l2_g = tf.reduce_mean([tf.nn.l2_loss(v) for v in net_g.weights.values()])

                self.total_loss_d = self.batch_loss_d
                self.total_loss_g = self.batch_loss_g

            with tf.name_scope('val_tower0'):
                val_net_g = SegNetG(val_image_batch_gan)
                val_croped_image = tf.slice(val_image_batch,  [0, crop_size, crop_size, 0], [-1, target_hw[0], target_hw[1], -1])
                val_croped_seg = tf.slice(val_seg_batch,  [0, crop_size, crop_size, 0], [-1, target_hw[0], target_hw[1], -1])
                val_croped_image_gan = tf.slice(val_image_batch_gan,  [0, crop_size, crop_size, 0], [-1, target_hw[0], target_hw[1], -1])
                val_croped_seg_gan = tf.slice(val_seg_batch_gan,  [0, crop_size, crop_size, 0], [-1, target_hw[0], target_hw[1], -1])
                with tf.variable_scope('net_g'):
                    val_gan_seg_batch, _ = val_net_g.build(False)
                val_full_batch_im = tf.concat(0,[val_croped_image, val_croped_image_gan])
                val_full_batch_seg = tf.concat(0,[val_croped_seg, val_gan_seg_batch])
                val_full_batch_label = tf.concat(0, [tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])])
                val_net_d = RibSegNet(val_full_batch_im, val_full_batch_seg)
                with tf.variable_scope('net_d'):
                    val_net_d.build(False)

                val_loss_d = tf.nn.sigmoid_cross_entropy_with_logits(val_net_d.layers['fc_out'], val_full_batch_label)

                val_loss_g = tf.abs(tf.sub(val_loss_d, log2_const))
                eps = tf.constant(np.finfo(np.float32).eps)
                val_hard_seg = tf.round(val_gan_seg_batch)
                val_intersection = tf.mul(val_croped_seg_gan, val_hard_seg)
                val_union = tf.sub(tf.add(val_croped_seg_gan, val_hard_seg), val_intersection)
                val_dice = tf.reduce_mean(tf.div(tf.add(tf.reduce_sum(val_intersection, [1,2]), eps),
                                                 tf.add(tf.reduce_sum(val_union, [1,2]), eps)))

                self.val_batch_loss_d = tf.reduce_mean(val_loss_d)
                self.val_batch_loss_g = tf.reduce_mean(val_loss_g)
                self.val_dice = val_dice
                self.val_fetch = [val_croped_image_gan, val_croped_seg_gan, val_gan_seg_batch]

        opt_d = tf.train.RMSPropOptimizer(self.LR_d)
        opt_g = tf.train.RMSPropOptimizer(self.LR_g)

        grads_vars_d = opt_d.compute_gradients(self.total_loss_d, var_list=list(net_d.weights.values()))
        grads_vars_g = opt_g.compute_gradients(self.total_loss_g, var_list=list(net_g.weights.values()))

        self.train_step_d = opt_d.apply_gradients(grads_vars_d)
        self.train_step_g = opt_g.apply_gradients(grads_vars_g)

        self.objective_summary = [tf.scalar_summary('objective_d', self.total_loss_d, name='objective_summary_d'),
                                  tf.scalar_summary('objective_g', self.total_loss_g, name='objective_summary_g')]
        self.val_objective_summary = [tf.scalar_summary('objective_d', self.val_batch_loss_d, name='objective_summary_d'),
                                  tf.scalar_summary('objective_g', self.val_batch_loss_g, name='objective_summary_g'),
                                      tf.scalar_summary('dice', val_dice, name='val_dice')]

        for g, v in grads_vars_d:
            self.hist_summaries.append(tf.histogram_summary(v.op.name + '/value', v, name=v.op.name + '_summary'))
            self.hist_summaries.append(tf.histogram_summary(v.op.name + '/grad', g,
                                                            name=v.op.name + '_grad_summary'))
        for g, v in grads_vars_g:
            self.hist_summaries.append(tf.histogram_summary(v.op.name + '/value', v, name=v.op.name + '_summary'))
            self.hist_summaries.append(tf.histogram_summary(v.op.name + '/grad', g,
                                                            name=v.op.name + '_grad_summary'))

    def train(self, lr_g=0.1, lr_d=0.1, g_steps=1, d_steps=3, l2_coeff=0.0001, l1_coeff=0.5, max_itr=100000,
              summaries=True, validation_interval=10,
              save_checkpoint_interval=200, plot_examples_interval=100):

        if summaries:

            train_merged_summaries = tf.merge_summary(self.hist_summaries+self.objective_summary)
            val_merged_summaries = tf.merge_summary(self.val_objective_summary)
            train_writer = tf.train.SummaryWriter(os.path.join(self.summaries_dir, 'train'),
                                                  graph=tf.get_default_graph())
            val_writer = tf.train.SummaryWriter(os.path.join(self.summaries_dir, 'val'))
        else:
            train_merged_summaries = tf.no_op()
            val_merged_summaries = tf.no_op()

        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())
            t = 0
            if restore:
                chkpnt_info = tf.train.get_checkpoint_state(save_dir)
                if chkpnt_info:
                    fullfilename = chkpnt_info.model_checkpoint_path
                    t = int(re.findall(r'\d+', os.path.basename(fullfilename))[0])+1
                    saver.restore(sess, fullfilename)

            tf.train.start_queue_runners(sess)
            feed_dict = {self.LR_g: lr_g, self.LR_d: lr_d,}
            train_fetch_d = [self.train_step_d, self.batch_loss_d, self.total_loss_d, train_merged_summaries]
            train_fetch_g = [self.train_step_g, self.batch_loss_g, self.total_loss_g, train_merged_summaries]
            ii = t*(d_steps+g_steps)
            for i in range(t, max_itr):

                for _ in range(d_steps):
                    ii += 1
                    start = time.time()
                    _, loss, objective, summaries_string = sess.run(train_fetch_d, feed_dict=feed_dict)
                    elapsed = start - time.time()
                    print "Train Step D: %d Elapsed Time: %g Objective: %g \n" % (ii, elapsed, objective)
                    if summaries:
                        train_writer.add_summary(summaries_string, ii)
                        train_writer.flush()
                for _ in range(g_steps):
                    ii += 1
                    start = time.time()
                    _, loss, objective, summaries_string = sess.run(train_fetch_g, feed_dict=feed_dict)
                    elapsed = start - time.time()
                    print "Train Step G: %d Elapsed Time: %g Objective: %g \n" % (ii, elapsed, objective)
                    if summaries:
                        train_writer.add_summary(summaries_string, ii)
                        train_writer.flush()

                if not i % validation_interval:
                    start = time.time()
                    v_dice, summaries_string = sess.run([self.val_dice, val_merged_summaries])
                    elapsed = start - time.time()
                    print "Validation Step: %d Elapsed Time: %g Dice: %g\n" % (ii, elapsed, v_dice)
                    if summaries:
                        val_writer.add_summary(summaries_string, ii)
                        val_writer.flush()
                if not i % save_checkpoint_interval:
                    save_path = saver.save(sess, os.path.join(save_dir, "model_%d.ckpt") % i)
                    print("Model saved in file: %s" % save_path)
                if not i % plot_examples_interval:
                    fetch = sess.run(self.val_fetch)
                    """
                    plt.figure(1)
                    plt.imshow(fetch[0][0][:, :, 0])
                    plt.ion()
                    plt.show()
                    plt.figure(2)
                    plt.imshow(fetch[1][0][:, :, 0])
                    plt.ion()
                    plt.show()
                    plt.figure(3)
                    plt.clf()
                    plt.imshow(fetch[2][0][:, :, 0])
                    plt.colorbar()
                    plt.ion()
                    plt.show()
                    plt.pause(0.001)
                    """

print __name__
if __name__ == "__main__":
    print "Start"
    trainer = GANTrainer(train_filename, val_filename, summaries_dir_name)
    print "Build Trainer"
    trainer.build(batch_size=20)
    print "Start Training"
    trainer.train(lr_g=0.0001, lr_d=0.0001, g_steps=3, d_steps=1, l2_coeff=0.01, l1_coeff=0, max_itr=20000,
                  summaries=True, validation_interval=10,
                  save_checkpoint_interval=100, plot_examples_interval=10000000)



