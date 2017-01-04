import tensorflow as tf
import Layers
import os
import utils
import re
from DataHandeling import CSVSegReader

restore = True
run_num = '6'
base_folder = '/Users/assafarbelle/Google Drive/PhD/Courses/DeepLearning/CellSegmentation/'
train_filename = '/Users/assafarbelle/Google Drive/PhD/Courses/DeepLearning/CellSegmentation/train.csv'
val_filename = '/Users/assafarbelle/Google Drive/PhD/Courses/DeepLearning/CellSegmentation/val.csv'

save_dir = '/Users/assafarbelle/Google Drive/PhD/DeepSegmentation/Snapshots/AlonLab/RibSeg/'+run_num
summaries_dir_name = '/Users/assafarbelle/Google Drive/PhD/DeepSegmentation/Logs/AlonLab/RibSeg/'+run_num
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(summaries_dir_name):
    os.makedirs(summaries_dir_name)


def layer(op):

    def layer_decorated(self, name, *args, **kwargs):

        out = op(self, name, *args, **kwargs)
        self.layers[name] = out
        return out
    return layer_decorated



class Network(object):

    def __init__(self):
        self.layers = {}
        self.weights = {}

    @layer
    def conv(self, name,
             in_tensor,
             kx,
             ky,
             kout,
             stride=None,
             biased=True,
             kernel_initializer=None,
             biase_initializer=None,
             padding='VALID',
             ):
        out, w, b = Layers.conv(in_tensor,
                                name,
                                kx,
                                ky,
                                kout,
                                stride,
                                biased,
                                kernel_initializer,
                                biase_initializer,
                                padding,
                                )
        self.weights.update({w.op.name: w})
        if biased:
            self.weights.update({b.op.name: b})
        return out

    @layer
    def fc(self, name,
           in_tensor,
           kout,
           biased=True,
           weights_initializer=None,
           biase_initializer=None,
           ):

        out, w, b = Layers.fc(in_tensor, name, kout, biased, weights_initializer, biase_initializer)
        if biased:
            self.weights.update({b.op.name: b})
        self.weights.update({w.op.name: w})
        return out

    @layer
    def leaky_relu(self, name, in_tensor, alpha=0.1):
        return Layers.leaky_relu(in_tensor, name, alpha)

    @layer
    def max_pool(self, name, in_tensor, ksize=None, strides=None, padding='VALID'):
        return Layers.max_pool(in_tensor, name, ksize, strides, padding)

    @layer
    def batch_norm(self, name, in_tensor, phase_train):

        return Layers.batch_norm(in_tensor, phase_train, name)

    @layer
    def concat(self, name, in_tensor_list, dim=3):
        return tf.concat(dim, in_tensor_list, name)

    @layer
    def sigmoid(self, name, in_tensor):
        return tf.sigmoid(in_tensor, name)


class SegNet(object):
    def __init__(self, image_batch):
        self.image_batch = image_batch
        self.layers = {}
        self.kernels = {}

    def build(self, phase_train):
        # Layer 1
        conv1, k1, _ = Layers.conv(self.image_batch,
                                   'conv1',
                                   9,
                                   9,
                                   32,
                                   stride=1,
                                   biased=True,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   biase_initializer=tf.zeros_initializer,
                                   padding='VALID',
                                   )

        bn1 = Layers.batch_norm(conv1, phase_train, 'bn1')
        relu1 = Layers.leaky_relu(bn1, 'relu1')
        self.layers.update({'conv1': conv1, 'bn1': bn1, 'relu1': relu1})
        self.kernels.update({k1.op.name: k1})
        # Layer 2
        conv2, k2, _ = Layers.conv(relu1,
                                   'conv2',
                                   7,
                                   7,
                                   64,
                                   stride=1,
                                   biased=True,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   biase_initializer=tf.zeros_initializer,
                                   padding='VALID',
                                   )

        bn2 = Layers.batch_norm(conv2, phase_train, 'bn2')
        relu2 = Layers.leaky_relu(bn2, 'relu2')
        self.layers.update({'conv2': conv2, 'bn2': bn2, 'relu2': relu2})
        self.kernels.update({k2.op.name: k2})
        # Layer 3
        conv3, k3, _ = Layers.conv(relu2,
                                   'conv3',
                                   3,
                                   3,
                                   128,
                                   stride=1,
                                   biased=True,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   biase_initializer=tf.zeros_initializer,
                                   padding='VALID',
                                   )

        bn3 = Layers.batch_norm(conv3, phase_train, 'bn3')
        relu3 = Layers.leaky_relu(bn3, 'relu3')
        self.layers.update({'conv3': conv3, 'bn3': bn3, 'relu3': relu3})
        self.kernels.update({k3.op.name: k3})
        # Layer 4
        conv4, k4, _ = Layers.conv(relu3,
                                   'conv4',
                                   7,
                                   7,
                                   16,
                                   stride=1,
                                   biased=True,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   biase_initializer=tf.zeros_initializer,
                                   padding='VALID',
                                   )

        bn4 = Layers.batch_norm(conv4, phase_train, 'bn4')
        relu4 = Layers.leaky_relu(bn4, 'relu4')
        self.layers.update({'conv4': conv4, 'bn4': bn4, 'relu4': relu4})
        self.kernels.update({k4.op.name: k4})
        # Layer 5
        conv5, k5, _ = Layers.conv(relu4,
                                   'conv5',
                                   3,
                                   3,
                                   32,
                                   stride=1,
                                   biased=True,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   biase_initializer=tf.zeros_initializer,
                                   padding='VALID',
                                   )

        bn5 = Layers.batch_norm(conv5, phase_train, 'bn5')
        relu5 = Layers.leaky_relu(bn5, 'relu5')
        self.layers.update({'conv5': conv5, 'bn5': bn5, 'relu5': relu5})
        self.kernels.update({k5.op.name: k5})
        # Layer 6
        conv6, k6, _ = Layers.conv(relu5,
                                   'conv6',
                                   1,
                                   1,
                                   64,
                                   stride=1,
                                   biased=True,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   biase_initializer=tf.zeros_initializer,
                                   padding='VALID',
                                   )

        bn6 = Layers.batch_norm(conv6, phase_train, 'bn6')
        relu6 = Layers.leaky_relu(bn6, 'relu6')
        self.layers.update({'conv6': conv6, 'bn6': bn6, 'relu6': relu6})
        self.kernels.update({k6.op.name: k6})
        # Layer 7
        conv7, k7, _ = Layers.conv(relu6,
                                   'conv7',
                                   1,
                                   1,
                                   1,
                                   stride=1,
                                   biased=True,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                   biase_initializer=tf.zeros_initializer,
                                   padding='VALID',
                                   )

        # bn7 = Layers.batch_norm(conv7, phase_train, 'bn7')
        out = tf.sigmoid(conv7, 'out')
        self.layers.update({'conv7': conv7,  'out': out})
        self.kernels.update({k7.op.name: k7})
        return out


class RibSegNet(Network):
    def __init__(self, image_batch1, image_batch2):
        self.image_batch1 = image_batch1
        self.image_batch2 = image_batch2
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

        center0 = self.concat('center0', [self.image_batch1, self.image_batch2], dim=3)

        # Layer 1
        k1 = 7
        k1out = 8
        left1, right1, center1 = rib('rib1', self.image_batch1, self.image_batch2, center0, k1, k1out)

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
        fcout = self.fc('fc_out', relu_fc2, 1, biased=False)
        out = self.sigmoid('out', fcout)
        self.layers['out'] = tf.sub(tf.constant(1.), out)
        return 1-out


class SegTrainer(object):

    def __init__(self, train_filenames, val_filenames, summaries_dir):

        self.train_filenames = train_filenames if isinstance(train_filenames, list) else [train_filenames]
        self.val_filenames = val_filenames if isinstance(val_filenames, list) else [val_filenames]
        self.summaries_dir = summaries_dir
        self.train_csv_reader = CSVSegReader(self.train_filenames, base_folder=base_folder)
        self.val_csv_reader = CSVSegReader(self.val_filenames, base_folder=base_folder)
        self.net = None
        self.train_step = None
        self.LR = tf.placeholder(tf.float32, [], 'learning_rate')
        self.batch_loss = None
        self.val_batch_loss = None
        self.net = None
        self.val_net = None
        self.train_image_summaries = None
        self.val_summaries = None
        self.hist_summaries = []
        self.k1_im = None
        self.train_kernel_summaries = None
        self.loss_summary = None
        self.val_loss_summary = None
        self.val_image_summaries = None

    def build(self, batch_size=1):

        train_image_batch, train_seg_batch = self.train_csv_reader.get_batch(batch_size)
        val_image_batch, val_seg_batch = self.val_csv_reader.get_batch(batch_size)

        with tf.device('/cpu:0'):
            with tf.name_scope('tower0'):
                net = SegNet(train_image_batch)
                net.build(True)
                tf.get_variable_scope().reuse_variables()
                val_net = SegNet(val_image_batch)
                val_net.build(False)

        k1 = net.kernels['conv1/weights']
        k1pad = tf.pad(k1, [[0, 0], [0, 0], [0, 0], [0, 4]])
        self.k1_im = utils.put_kernels_on_grid(k1pad, (6, 6), 2)
        self.train_kernel_summaries = tf.image_summary('kernel1', self.k1_im, max_images=1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        updates = tf.group(*update_ops) if update_ops else tf.no_op()
        with tf.control_dependencies([updates]):
            pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(net.layers['conv7'], train_seg_batch)
        image_loss = tf.reduce_mean(pixel_loss, [1, 2])
        self.batch_loss = tf.reduce_mean(image_loss)
        self.loss_summary = tf.scalar_summary('loss', self.batch_loss, name='loss_summary')

        self.train_image_summaries = [tf.image_summary('input image', train_image_batch, max_images=1),
                                      tf.image_summary('output image', net.layers['conv7'], max_images=1)]

        with tf.control_dependencies([updates]):
            val_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(val_net.layers['conv7'], val_seg_batch)
        val_image_loss = tf.reduce_mean(val_pixel_loss, [1, 2])
        self.val_batch_loss = tf.reduce_mean(val_image_loss)
        self.val_loss_summary = tf.scalar_summary('loss', self.val_batch_loss, name='val_loss_summary')
        self.val_image_summaries = [tf.image_summary('input image', val_image_batch, max_images=1),
                                    tf.image_summary('output image', val_net.layers['conv7'], max_images=1)]

        opt = tf.train.RMSPropOptimizer(self.LR)
        grads_vars = opt.compute_gradients(self.batch_loss)

        for g, v in grads_vars:
                self.hist_summaries.append(tf.histogram_summary(v.op.name + '/value', v, name=v.op.name + '_summary'))
                self.hist_summaries.append(tf.histogram_summary(v.op.name + '/grad',
                                                                g, name=v.op.name + '_grad_summary'))

        self.train_step = opt.apply_gradients(grads_vars)
        self.net = net

    def train(self, lr=0.001, max_itr=100000, summaries=True):

        if summaries:

            loss_summary = self.loss_summary
            train_merged_summaries = tf.merge_summary(self.hist_summaries+[loss_summary, self.train_kernel_summaries] +
                                                      self.train_image_summaries)
            val_loss_summary = self.val_loss_summary
            val_merged_summaries = tf.merge_summary([val_loss_summary] + self.val_image_summaries)
            train_writer = tf.train.SummaryWriter(os.path.join(self.summaries_dir, 'train'))
            val_writer = tf.train.SummaryWriter(os.path.join(self.summaries_dir, 'val'))
        else:
            train_merged_summaries = tf.no_op()
            val_merged_summaries = tf.no_op()

        saver = tf.train.Saver(tf.all_variables())
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            tf.train.start_queue_runners(sess)
            for i in range(max_itr):
                _, loss, summaries_string = sess.run([self.train_step, self.batch_loss, train_merged_summaries],
                                                     feed_dict={self.LR: lr})
                print "Train Step: %d Loss: %g \n" % (i, loss)
                if summaries:
                    train_writer.add_summary(summaries_string, i)
                    train_writer.flush()
                if not i % 10:
                    save_path = saver.save(sess, os.path.join(save_dir, "model_%d.ckpt") % i)
                    print("Model saved in file: %s" % save_path)
                    loss, summaries_string = sess.run([self.val_batch_loss, val_merged_summaries])
                    print "Validation Step: %d Loss: %g \n" % (i, loss)
                    if summaries:
                        val_writer.add_summary(summaries_string, i)
                        val_writer.flush()


class SegTrainer2(object):

    def __init__(self, train_filenames, val_filenames, summaries_dir):

        self.train_filenames = train_filenames if isinstance(train_filenames, list) else [train_filenames]
        self.val_filenames = val_filenames if isinstance(val_filenames, list) else [val_filenames]
        self.summaries_dir = summaries_dir
        self.train_csv_reader = CSVSegReader(self.train_filenames, base_folder=base_folder)
        self.val_csv_reader = CSVSegReader(self.val_filenames, base_folder=base_folder)
        self.net = None
        self.train_step = None
        self.LR = tf.placeholder(tf.float32, [], 'learning_rate')
        self.batch_loss = None
        self.val_batch_loss = None
        self.net = None
        self.val_net = None
        self.train_image_summaries = None
        self.val_summaries = None
        self.val_image_summaries = None
        self.hist_summaries = []
        self.loss_summary = None
        self.val_loss_summary = None
        self.train_kernel_summaries = None
        self.k1_im = None

    def build(self, batch_size=1):

        train_image_batch, train_seg_batch = self.train_csv_reader.get_batch(batch_size)
        val_image_batch, val_seg_batch = self.val_csv_reader.get_batch(batch_size)

        with tf.device('/cpu:0'):
            with tf.name_scope('tower0'):
                net = SegNet(train_image_batch)
                net.build(True)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                updates = tf.group(*update_ops) if update_ops else tf.no_op()
                with tf.control_dependencies([updates]):
                    self.batch_loss = utils.my_clustering_loss(net.layers['out'], net.layers['conv7'])
                tf.get_variable_scope().reuse_variables()
                val_net = SegNet(val_image_batch)
                val_net.build(False)

        k1 = net.kernels['conv1/weights']
        k1pad = tf.pad(k1, [[0, 0], [0, 0], [0, 0], [0, 4]])
        self.k1_im = utils.put_kernels_on_grid(k1pad, (6, 6), 2)
        self.train_kernel_summaries = tf.image_summary('kernel1', self.k1_im, max_images=1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        updates = tf.group(*update_ops) if update_ops else tf.no_op()
        with tf.control_dependencies([updates]):
            self.batch_loss = tf.squeeze(utils.my_clustering_loss(net.layers['out'], net.layers['conv7']))

        self.loss_summary = tf.scalar_summary('loss', self.batch_loss, name='loss_summary')

        self.train_image_summaries = [tf.image_summary('input image', train_image_batch, max_images=1),
                                      tf.image_summary('output image', net.layers['conv7'], max_images=1)]

        self.val_batch_loss = tf.squeeze(utils.my_clustering_loss(val_net.layers['out'], val_net.layers['conv7']))
        self.val_loss_summary = tf.scalar_summary('loss', self.val_batch_loss, name='val_loss_summary')
        self.val_image_summaries = [tf.image_summary('input image', val_image_batch, max_images=1),
                                    tf.image_summary('output image', val_net.layers['conv7'], max_images=1)]

        opt = tf.train.RMSPropOptimizer(self.LR)
        grads_vars = opt.compute_gradients(self.batch_loss)

        for g, v in grads_vars:
                self.hist_summaries.append(tf.histogram_summary(v.op.name + '/value', v, name=v.op.name + '_summary'))
                self.hist_summaries.append(tf.histogram_summary(v.op.name + '/grad', g,
                                                                name=v.op.name + '_grad_summary'))

        self.train_step = opt.apply_gradients(grads_vars)
        self.net = net

    def train(self, lr=0.001, max_itr=100000, summaries=True):

        if summaries:

            loss_summary = self.loss_summary
            train_merged_summaries = tf.merge_summary(self.hist_summaries+[loss_summary, self.train_kernel_summaries] +
                                                      self.train_image_summaries)
            val_loss_summary = self.val_loss_summary
            val_merged_summaries = tf.merge_summary([val_loss_summary] + self.val_image_summaries)
            train_writer = tf.train.SummaryWriter(os.path.join(self.summaries_dir, 'train'))
            val_writer = tf.train.SummaryWriter(os.path.join(self.summaries_dir, 'val'))
        else:
            train_merged_summaries = tf.no_op()
            val_merged_summaries = tf.no_op()

        saver = tf.train.Saver(tf.all_variables())
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            tf.train.start_queue_runners(sess)
            for i in range(max_itr):
                _, loss, summaries_string = sess.run([self.train_step, self.batch_loss, train_merged_summaries],
                                                     feed_dict={self.LR: lr})
                print "Train Step: %d Loss: %g \n" % (i, loss)
                if summaries:
                    train_writer.add_summary(summaries_string, i)
                    train_writer.flush()
                if not i % 10:
                    save_path = saver.save(sess, os.path.join(save_dir, "model_%d.ckpt") % i)
                    print("Model saved in file: %s" % save_path)
                    loss, summaries_string = sess.run([self.val_batch_loss, val_merged_summaries])
                    print "Validation Step: %d Loss: %g \n" % (i, loss)
                    if summaries:
                        val_writer.add_summary(summaries_string, i)
                        val_writer.flush()


class RibSegTrainer(object):

    def __init__(self, train_filenames, val_filenames, summaries_dir):

        self.train_filenames = train_filenames if isinstance(train_filenames, list) else [train_filenames]
        self.val_filenames = val_filenames if isinstance(val_filenames, list) else [val_filenames]
        self.summaries_dir = summaries_dir
        self.train_csv_reader = CSVSegReader(self.train_filenames, base_folder=base_folder)
        self.val_csv_reader = CSVSegReader(self.val_filenames, base_folder=base_folder)
        # Set variable for net and losses
        self.net = None
        self.batch_loss = None
        self.total_loss = None
        # Set validation variable for net and losses
        self.val_net = None
        self.val_batch_loss = None
        # Set placeholders for training parameters
        self.train_step = None
        self.LR = tf.placeholder(tf.float32, [], 'learning_rate')
        self.L2_coeff = tf.placeholder(tf.float32, [], 'L2_coeff')
        self.L1_coeff = tf.placeholder(tf.float32, [], 'L1_coeff')

        # Set variables for tensorboard summaries
        self.loss_summary = None
        self.val_loss_summary = None
        self.objective_summary = None
        self.hist_summaries = []
        self.image_summaries = []

    def build(self, batch_size=1):

        train_image_batch, train_seg_batch = self.train_csv_reader.get_batch(batch_size)

        val_image_batch, val_seg_batch = self.val_csv_reader.get_batch(batch_size)

        with tf.device('/cpu:0'):
            with tf.name_scope('tower0'):
                net = RibSegNet(train_image_batch, train_seg_batch)
                net.build(True)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                updates = tf.group(*update_ops) if update_ops else tf.no_op()
                with tf.control_dependencies([updates]):
                    self.batch_loss = tf.reduce_mean(net.layers['out'])
                tf.get_variable_scope().reuse_variables()
                const = tf.constant(1.)
                loss_l2 = tf.add_n([tf.abs(tf.sub(tf.nn.l2_loss(v), const)) for v in net.weights.values()])
                loss_l1 = tf.add_n([tf.reduce_sum(tf.abs(v)) for v in net.weights.values()])
                self.total_loss = tf.add_n([self.batch_loss,
                                            tf.mul(loss_l2, self.L2_coeff), tf.div(self.L1_coeff, loss_l1)])
            with tf.name_scope('val_tower0'):
                val_net = RibSegNet(val_image_batch, val_seg_batch)
                val_net.build(False)
                self.val_batch_loss = tf.reduce_mean(net.layers['out'])

        opt = tf.train.RMSPropOptimizer(self.LR)
        grads_vars = opt.compute_gradients(self.total_loss)

        self.objective_summary = tf.scalar_summary('objective', self.total_loss, name='objective_summary')
        self.loss_summary = tf.scalar_summary('loss', self.batch_loss, name='loss_summary')
        self.val_loss_summary = tf.scalar_summary('loss', self.val_batch_loss, name='loss_summary')
        conv1_l = utils.put_kernels_on_grid(tf.pad(net.weights['left_rib1/weights'],
                                                   [[0, 0], [0, 0], [0, 0], [0, 1]]), (3, 3), pad=1)
        conv1_r = utils.put_kernels_on_grid(tf.pad(net.weights['right_rib1/weights'],
                                                   [[0, 0], [0, 0], [0, 0], [0, 1]]), (3, 3), pad=1)
        c1_w = net.weights['centerrib1/weights']
        c1_wp = tf.pad(c1_w, [[0, 0], [0, 0], [0, 1], [0, 0]])
        conv1_c = utils.put_kernels_on_grid(c1_wp, (2, 2), pad=1)
        self.image_summaries.append(tf.image_summary('left_1_weights', conv1_l, max_images=1))
        self.image_summaries.append(tf.image_summary('rigtht_1_weights', conv1_r, max_images=1))
        self.image_summaries.append(tf.image_summary('center_1_weights', conv1_c, max_images=1))

        for g, v in grads_vars:
                self.hist_summaries.append(tf.histogram_summary(v.op.name + '/value', v, name=v.op.name + '_summary'))
                self.hist_summaries.append(tf.histogram_summary(v.op.name + '/grad', g,
                                                                name=v.op.name + '_grad_summary'))
        self.train_step = opt.apply_gradients(grads_vars)
        self.net = net
        self.val_net = val_net

    def train(self, lr=0.1, l2_coeff=0.0001, l1_coeff=0.5, max_itr=100000, summaries=True, validation_interval=10,
              save_checkpoint_interval=200):

        if summaries:

            train_merged_summaries = tf.merge_summary(self.hist_summaries+[self.objective_summary, self.loss_summary] +
                                                      self.image_summaries)
            val_merged_summaries = self.val_loss_summary
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
            feed_dict = {self.LR: lr, self.L2_coeff: l2_coeff, self.L1_coeff: l1_coeff}
            train_fetch = [self.train_step, self.batch_loss, self.total_loss, train_merged_summaries]

            for i in range(t, max_itr):
                _, loss, objective, summaries_string = sess.run(train_fetch, feed_dict=feed_dict)
                print "Train Step: %d Loss: %g Objective: %g \n" % (i, loss, objective)
                if summaries:
                    train_writer.add_summary(summaries_string, i)
                    train_writer.flush()
                if not i % validation_interval:
                    loss, summaries_string = sess.run([self.val_batch_loss, val_merged_summaries])
                    print "Validation Step: %d Loss: %g \n" % (i, loss)
                    if summaries:
                        val_writer.add_summary(summaries_string, i)
                        val_writer.flush()
                if not i % save_checkpoint_interval:
                    save_path = saver.save(sess, os.path.join(save_dir, "model_%d.ckpt") % i)
                    print("Model saved in file: %s" % save_path)

class RibSegTrainer2(object):

    def __init__(self, train_filenames, val_filenames, summaries_dir):

        self.train_filenames = train_filenames if isinstance(train_filenames, list) else [train_filenames]
        self.val_filenames = val_filenames if isinstance(val_filenames, list) else [val_filenames]
        self.summaries_dir = summaries_dir
        self.train_csv_reader = CSVSegReader(self.train_filenames, base_folder=base_folder)
        self.val_csv_reader = CSVSegReader(self.val_filenames, base_folder=base_folder)
        # Set variable for net and losses
        self.net = None
        self.batch_loss = None
        self.total_loss = None
        # Set validation variable for net and losses
        self.val_net = None
        self.val_batch_loss = None
        # Set placeholders for training parameters
        self.train_step = None
        self.LR = tf.placeholder(tf.float32, [], 'learning_rate')
        self.L2_coeff = tf.placeholder(tf.float32, [], 'L2_coeff')
        self.L1_coeff = tf.placeholder(tf.float32, [], 'L1_coeff')

        # Set variables for tensorboard summaries
        self.loss_summary = None
        self.val_loss_summary = None
        self.objective_summary = None
        self.hist_summaries = []
        self.image_summaries = []

    def get_batch(self,batch_size=1):

        train_image_batch, train_seg_batch = self.train_csv_reader.get_batch(batch_size)


    def build(self, batch_size=1):

        train_image_batch, train_seg_batch = self.train_csv_reader.get_batch(batch_size)


        val_image_batch, val_seg_batch = self.val_csv_reader.get_batch(batch_size)

        with tf.device('/cpu:0'):
            with tf.name_scope('tower0'):
                net = RibSegNet(train_image_batch, train_seg_batch)
                net.build(True)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                updates = tf.group(*update_ops) if update_ops else tf.no_op()
                with tf.control_dependencies([updates]):
                    self.batch_loss = tf.reduce_mean(net.layers['out'])
                tf.get_variable_scope().reuse_variables()
                const = tf.constant(1.)
                loss_l2 = tf.add_n([tf.abs(tf.sub(tf.nn.l2_loss(v), const)) for v in net.weights.values()])
                loss_l1 = tf.add_n([tf.reduce_sum(tf.abs(v)) for v in net.weights.values()])
                self.total_loss = tf.add_n([self.batch_loss,
                                            tf.mul(loss_l2, self.L2_coeff), tf.div(self.L1_coeff, loss_l1)])
            with tf.name_scope('val_tower0'):
                val_net = RibSegNet(val_image_batch, val_seg_batch)
                val_net.build(False)
                self.val_batch_loss = tf.reduce_mean(net.layers['out'])

        opt = tf.train.RMSPropOptimizer(self.LR)
        grads_vars = opt.compute_gradients(self.total_loss)

        self.objective_summary = tf.scalar_summary('objective', self.total_loss, name='objective_summary')
        self.loss_summary = tf.scalar_summary('loss', self.batch_loss, name='loss_summary')
        self.val_loss_summary = tf.scalar_summary('loss', self.val_batch_loss, name='loss_summary')
        conv1_l = utils.put_kernels_on_grid(tf.pad(net.weights['left_rib1/weights'],
                                                   [[0, 0], [0, 0], [0, 0], [0, 1]]), (3, 3), pad=1)
        conv1_r = utils.put_kernels_on_grid(tf.pad(net.weights['right_rib1/weights'],
                                                   [[0, 0], [0, 0], [0, 0], [0, 1]]), (3, 3), pad=1)
        c1_w = net.weights['centerrib1/weights']
        c1_wp = tf.pad(c1_w, [[0, 0], [0, 0], [0, 1], [0, 0]])
        conv1_c = utils.put_kernels_on_grid(c1_wp, (2, 2), pad=1)
        self.image_summaries.append(tf.image_summary('left_1_weights', conv1_l, max_images=1))
        self.image_summaries.append(tf.image_summary('rigtht_1_weights', conv1_r, max_images=1))
        self.image_summaries.append(tf.image_summary('center_1_weights', conv1_c, max_images=1))

        for g, v in grads_vars:
                self.hist_summaries.append(tf.histogram_summary(v.op.name + '/value', v, name=v.op.name + '_summary'))
                self.hist_summaries.append(tf.histogram_summary(v.op.name + '/grad', g,
                                                                name=v.op.name + '_grad_summary'))
        self.train_step = opt.apply_gradients(grads_vars)
        self.net = net
        self.val_net = val_net

    def train(self, lr=0.1, l2_coeff=0.0001, l1_coeff=0.5, max_itr=100000, summaries=True, validation_interval=10,
              save_checkpoint_interval=200):

        if summaries:

            train_merged_summaries = tf.merge_summary(self.hist_summaries+[self.objective_summary, self.loss_summary] +
                                                      self.image_summaries)
            val_merged_summaries = self.val_loss_summary
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
            feed_dict = {self.LR: lr, self.L2_coeff: l2_coeff, self.L1_coeff: l1_coeff}
            train_fetch = [self.train_step, self.batch_loss, self.total_loss, train_merged_summaries]

            for i in range(t, max_itr):
                _, loss, objective, summaries_string = sess.run(train_fetch, feed_dict=feed_dict)
                print "Train Step: %d Loss: %g Objective: %g \n" % (i, loss, objective)
                if summaries:
                    train_writer.add_summary(summaries_string, i)
                    train_writer.flush()
                if not i % validation_interval:
                    loss, summaries_string = sess.run([self.val_batch_loss, val_merged_summaries])
                    print "Validation Step: %d Loss: %g \n" % (i, loss)
                    if summaries:
                        val_writer.add_summary(summaries_string, i)
                        val_writer.flush()
                if not i % save_checkpoint_interval:
                    save_path = saver.save(sess, os.path.join(save_dir, "model_%d.ckpt") % i)
                    print("Model saved in file: %s" % save_path)

__author__ = 'assafarbelle'
if __name__ == "__main__":
    trainer = RibSegTrainer(train_filename, val_filename, summaries_dir_name)
    trainer.build(batch_size=10)
    trainer.train(lr=0.1, l2_coeff=0.01, l1_coeff=0, max_itr=100000, summaries=True, validation_interval=10,
                  save_checkpoint_interval=200)
