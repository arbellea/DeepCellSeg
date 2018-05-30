import csv
import tensorflow as tf
import os
import glob
import cv2
import queue
import threading
import numpy as np
import pandas as pd

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimage
# import utils


__author__ = 'assafarbelle'


class CSVSegReader(object):
    def __init__(self, filenames, base_folder='.', image_size=(64, 64, 1), num_threads=4,
                 capacity=20, min_after_dequeue=10, random=True, data_format='NCHW'):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:

            filenames: a list of csv files filename
            batch_size: the number of examples in each patch
            image_size: a tuple containing the image size in Y and X dimensions
            num_threads: number of threads for prefetch
            capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            min_after_dequeue: the minimum example in the queue after a dequeue op. ensures good mixing
        """
        self.reader = tf.TextLineReader()
        num_epochs = None if random else 1
        self.input_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, seed=0)
        self.image_size = image_size
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = None
        self.base_folder = base_folder
        self.random = random
        self.data_format = data_format

    def _get_image(self):

        _, records = self.reader.read(self.input_queue)
        file_names = tf.decode_csv(records, [tf.constant([], tf.string), tf.constant([], tf.string)], field_delim=None,
                                   name=None)

        im_raw = tf.read_file(self.base_folder + file_names[0])
        seg_raw = tf.read_file(self.base_folder + file_names[1])
        image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                           self.image_size, name='input_image')
        seg = tf.reshape(tf.cast(tf.image.decode_png(seg_raw, channels=1, dtype=tf.uint8), tf.float32),
                         self.image_size, name='input_seg')

        return image, seg, file_names[0]

    def get_batch(self, batch_size=1):

        self.batch_size = batch_size

        image, seg, file_name = self._get_image()
        if self.random:
            image_batch, seg_batch, filename_batch = tf.train.shuffle_batch([image, seg, file_name],
                                                                            batch_size=self.batch_size,
                                                                            num_threads=self.num_threads,
                                                                            capacity=self.capacity,
                                                                            min_after_dequeue=self.min_after_dequeue)
        else:
            image_batch, seg_batch, filename_batch = tf.train.batch_join([(image, seg, file_name)],
                                                                         batch_size=self.batch_size,
                                                                         capacity=self.capacity,
                                                                         allow_smaller_final_batch=True)
        if self.data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, perm=[0, 3, 1, 2])
            seg_batch = tf.transpose(seg_batch, perm=[0, 3, 1, 2])

        return image_batch, seg_batch, filename_batch


class CSVSegReaderRandom(object):
    def __init__(self, filenames, base_folder='.', image_size=(), crop_size=(64, 64), num_threads=4,
                 capacity=20, min_after_dequeue=10, data_format='NCHW',
                 random_rotate=tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:

            filenames: a list of csv files filename
            batch_size: the number of examples in each patch
            image_size: a tuple containing the image size in Y and X dimensions
            num_threads: number of threads for prefetch
            capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            min_after_dequeue: the minimum example in the queue after a dequeue op. ensures good mixing
        """
        self.reader = tf.TextLineReader()
        self.input_queue = tf.train.string_input_producer(filenames)
        self.image_size = image_size
        self.crop_size = crop_size
        self.random_rotate = random_rotate
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = None
        self.base_folder = base_folder
        self.data_format = data_format

    def _get_image(self):
        _, records = self.reader.read(self.input_queue)
        file_names = tf.decode_csv(records, [tf.constant([], tf.string), tf.constant([], tf.string)],
                                   field_delim=None, name=None)

        im_raw = tf.read_file(self.base_folder + file_names[0])
        seg_raw = tf.read_file(self.base_folder + file_names[1])
        image = tf.reshape(
            tf.cast(tf.image.decode_png(
                im_raw,
                channels=1, dtype=tf.uint16),
                tf.float32), self.image_size, name='input_image')
        seg = tf.reshape(
            tf.cast(tf.image.decode_png(
                seg_raw,
                channels=1, dtype=tf.uint8),
                tf.float32), self.image_size, name='input_seg')

        return image, seg, file_names[0]

    def get_batch(self, batch_size=1):
        self.batch_size = batch_size

        image, seg, file_name = self._get_image()
        concat = tf.concat(axis=2, values=[image, seg])

        concat = tf.random_crop(concat, [self.crop_size[0], self.crop_size[1], 2])
        shape = concat.get_shape()
        concat = tf.image.random_flip_left_right(concat)
        concat = tf.image.random_flip_up_down(concat)
        concat = tf.image.rot90(concat, k=self.random_rotate)
        concat.set_shape(shape)
        image, seg = tf.unstack(concat, 2, 2)
        image = tf.expand_dims(image, 2)
        seg = tf.expand_dims(seg, 2)

        image_batch, seg_batch, filename_batch = tf.train.shuffle_batch([image, seg, file_name],
                                                                        batch_size=self.batch_size,
                                                                        num_threads=self.num_threads,
                                                                        capacity=self.capacity,
                                                                        min_after_dequeue=self.min_after_dequeue)
        if self.data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, perm=[0, 3, 1, 2])
            seg_batch = tf.transpose(seg_batch, perm=[0, 3, 1, 2])

        return image_batch, seg_batch, filename_batch


class CSVSegReaderRandom2(object):
    def __init__(self, filenames, base_folder='.', image_size=(), crop_size=(128, 128), crops_per_image=50,
                 num_threads=4, capacity=20, min_after_dequeue=10, num_examples=None, data_format='NCHW',
                 random_rotate=tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:

            filenames: a list of csv files filename
            batch_size: the number of examples in each patch
            image_size: a tuple containing the image size in Y and X dimensions
            num_threads: number of threads for prefetch
            capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            min_after_dequeue: the minimum example in the queue after a dequeue op. ensures good mixing
        """
        raw_filenames = []

        for filename in filenames:
            with open(filename, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
                for row in csv_reader:
                    raw_filenames.append(row[0] + ':' + row[1])

        self.partial_frame = 0

        if not num_examples:
            pass
        elif isinstance(num_examples, int):
            num_examples = min(num_examples, len(raw_filenames))
            raw_filenames = raw_filenames[-num_examples:]
        elif isinstance(num_examples, float) and num_examples < 1:
            self.partial_frame = num_examples
            if num_examples <= 0:
                ValueError('number of examples has to be positive')
            raw_filenames = raw_filenames[-1:]

        elif isinstance(num_examples, list):
            raw_filenames = [f_name for n, f_name in enumerate(raw_filenames) if n in num_examples]

        self.raw_queue = tf.train.string_input_producer(raw_filenames, seed=0)
        self.image_size = image_size
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.random_rotate = random_rotate
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = None
        self.base_folder = base_folder
        self.data_format = data_format

    def _get_image(self):

        im_filename = tf.sparse_tensor_to_dense(tf.string_split(tf.expand_dims(self.raw_queue.dequeue(), 0), ':'), '')
        im_filename.set_shape([1, 2])
        im_raw = tf.read_file(self.base_folder + im_filename[0][0])
        seg_raw = tf.read_file(self.base_folder + im_filename[0][1])

        image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                           self.image_size, name='input_image')
        seg = tf.reshape(tf.cast(tf.image.decode_png(seg_raw, channels=1, dtype=tf.uint8), tf.float32), self.image_size,
                         name='input_seg')
        if self.partial_frame:
            crop_y_start = int(((1 - self.partial_frame) * self.image_size[0]) / 2)
            crop_y_end = int(((1 + self.partial_frame) * self.image_size[0]) / 2)
            crop_x_start = int(((1 - self.partial_frame) * self.image_size[1]) / 2)
            crop_x_end = int(((1 + self.partial_frame) * self.image_size[1]) / 2)
            image = tf.slice(image, [crop_y_start, crop_x_start, 0], [crop_y_end, crop_x_end, -1])
            seg = tf.slice(seg, [crop_y_start, crop_x_start, 0], [crop_y_end, crop_x_end, -1])

        return image, seg, im_filename[0][0], im_filename[0][1]

    def get_batch(self, batch_size=1):
        self.batch_size = batch_size
        image_in, seg_in, file_name, seg_filename = self._get_image()
        concat = tf.concat(axis=2, values=[image_in, seg_in])
        image_list = []
        seg_list = []
        filename_list = []
        seg_filename_list = []
        for _ in range(self.crops_per_image):
            cropped = tf.random_crop(concat, [self.crop_size[0], self.crop_size[1], 2])
            shape = cropped.get_shape()
            fliplr = tf.image.random_flip_left_right(cropped)
            flipud = tf.image.random_flip_up_down(fliplr)
            rot = tf.image.rot90(flipud, k=self.random_rotate)
            rot.set_shape(shape)
            image, seg = tf.unstack(rot, 2, 2)
            image = tf.expand_dims(image, 2)
            seg = tf.expand_dims(seg, 2)
            image_list.append(image)
            seg_list.append(seg)
            filename_list.append(file_name)
            seg_filename_list.append(seg_filename)
        image_many = tf.stack(values=image_list, axis=0)
        seg_many = tf.stack(values=seg_list, axis=0)
        filename_many = tf.stack(values=filename_list, axis=0)
        seg_filename_many = tf.stack(values=seg_filename_list, axis=0)

        (image_batch, seg_batch, filename_batch,
         seg_filename_batch) = tf.train.shuffle_batch([image_many, seg_many, filename_many, seg_filename_many],
                                                      batch_size=self.batch_size,
                                                      num_threads=self.num_threads,
                                                      capacity=self.capacity,
                                                      min_after_dequeue=self.min_after_dequeue,
                                                      enqueue_many=True
                                                      )
        if self.data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, perm=[0, 3, 1, 2])
            seg_batch = tf.transpose(seg_batch, perm=[0, 3, 1, 2])

        return image_batch, seg_batch, filename_batch


class CSVSegReader2(object):
    def __init__(self, filenames, base_folder='.', image_size=(64, 64, 1), num_threads=4,
                 capacity=20, min_after_dequeue=10, num_examples=None, random=True, data_format='NCHW'):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:

            filenames: a list of csv files filename
            batch_size: the number of examples in each patch
            image_size: a tuple containing the image size in Y and X dimensions
            num_threads: number of threads for prefetch
            capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            min_after_dequeue: the minimum example in the queue after a dequeue op. ensures good mixing
        """
        num_epochs = None if random else 1
        raw_filenames = []

        for filename in filenames:
            with open(filename, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
                for row in csv_reader:
                    raw_filenames.append(row[0] + ':' + row[1])

        if not num_examples:
            pass
        elif isinstance(num_examples, int):
            raw_filenames = raw_filenames[:num_examples]
            # seg_filenames = seg_filenames[:num_examples]
        elif isinstance(num_examples, list):
            raw_filenames = [f_name for n, f_name in enumerate(raw_filenames) if n in num_examples]
            # seg_filenames = [f_name for n, f_name in enumerate(seg_filenames) if n in num_examples]

        self.raw_queue = tf.train.string_input_producer(raw_filenames, num_epochs=num_epochs, shuffle=random, seed=0)
        # self.seg_queue = tf.train.string_input_producer(seg_filenames, num_epochs=num_epochs, shuffle=random, seed=0)

        self.image_size = image_size
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = None
        self.base_folder = base_folder
        self.random = random
        self.data_format = data_format

    def _get_image(self):

        filename = tf.sparse_tensor_to_dense(tf.string_split(tf.expand_dims(self.raw_queue.dequeue(), 0), ':'), '')
        filename.set_shape([1, 2])
        # seg_filename = self.seg_queue.dequeue()

        im_raw = tf.read_file(self.base_folder + filename[0][0])
        seg_raw = tf.read_file(self.base_folder + filename[0][1])
        image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                           self.image_size, name='input_image')
        seg = tf.reshape(tf.cast(tf.image.decode_png(seg_raw, channels=1, dtype=tf.uint8), tf.float32), self.image_size,
                         name='input_seg')

        return image, seg, filename[0][0], filename[0][1]

    def get_batch(self, batch_size=1):

        self.batch_size = batch_size

        image, seg, file_name, seg_file_name = self._get_image()
        if self.random:
            image_batch, seg_batch, filename_batch, seg_filename_batch = tf.train.shuffle_batch(
                [image, seg, file_name, seg_file_name], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.capacity, min_after_dequeue=self.min_after_dequeue)
        else:

            image_batch, seg_batch, filename_batch = tf.train.batch_join([(image, seg, file_name)],
                                                                         batch_size=self.batch_size,
                                                                         capacity=self.capacity,
                                                                         allow_smaller_final_batch=True)

        if self.data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, perm=[0, 3, 1, 2])
            seg_batch = tf.transpose(seg_batch, perm=[0, 3, 1, 2])
        return image_batch, seg_batch, filename_batch


class CSVSegReaderRandomLSTM(object):
    def __init__(self, filenames, base_folder='.', image_size=(), crop_size=(128, 128), crops_per_image=50,
                 num_threads=4, capacity=20, min_after_dequeue=10, num_examples=None, data_format='NCHW', one_seg=False,
                 random_rotate=tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:

            filenames: a list of csv files filename
            batch_size: the number of examples in each patch
            image_size: a tuple containing the image size in Y and X dimensions
            num_threads: number of threads for prefetch
            capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            min_after_dequeue: the minimum example in the queue after a dequeue op. ensures good mixing
        """
        raw_filenames = []

        for filename in filenames:
            with open(os.path.join(base_folder, filename), 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
                for row in csv_reader:
                    raw_filenames.append(row)

        self.partial_frame = 0

        if not num_examples:
            pass
        elif isinstance(num_examples, int):
            num_examples = min(num_examples, len(raw_filenames))
            raw_filenames = raw_filenames[-num_examples:]
        elif isinstance(num_examples, float) and num_examples < 1:
            self.partial_frame = num_examples
            if num_examples <= 0:
                ValueError('number of examples has to be positive')
            raw_filenames = raw_filenames[-1:]

        elif isinstance(num_examples, list):
            raw_filenames = [f_name for n, f_name in enumerate(raw_filenames) if n in num_examples]
        raw_filenames = list(map(list, zip(*raw_filenames)))
        self.raw_queue = tf.train.slice_input_producer(raw_filenames, seed=0)
        self.image_size = image_size
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.random_rotate = random_rotate
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = None
        self.base_folder = base_folder
        self.data_format = data_format
        self.one_seg = one_seg

    def _get_image_sequence(self):
        filenames = self.raw_queue
        im_list = []
        seg_list = []
        for i in range(0, len(filenames), 2):
            im_filename, seg_filename = filenames[i], filenames[i + 1]
            im_raw = tf.read_file(self.base_folder + im_filename)
            seg_raw = tf.read_file(self.base_folder + seg_filename)

            image_size = self.image_size + (1,)
            image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                               image_size)
            seg = tf.reshape(tf.cast(tf.image.decode_png(seg_raw, channels=1, dtype=tf.uint8), tf.float32),
                             image_size)
            if self.partial_frame:
                crop_y_start = int(((1 - self.partial_frame) * image_size[0]) / 2)
                crop_y_end = int(((1 + self.partial_frame) * image_size[0]) / 2)
                crop_x_start = int(((1 - self.partial_frame) * image_size[1]) / 2)
                crop_x_end = int(((1 + self.partial_frame) * image_size[1]) / 2)
                image = tf.slice(image, [crop_y_start, crop_x_start, 0], [crop_y_end, crop_x_end, -1])
                seg = tf.slice(seg, [crop_y_start, crop_x_start, 0], [crop_y_end, crop_x_end, -1])
            im_list.append(image)
            seg_list.append(seg)

        return im_list, seg_list, filenames

    def _get_image_sequence_one_seg(self):
        filenames = self.raw_queue
        image_size = self.image_size + (1,)
        im_list = []

        seg_raw = tf.read_file(self.base_folder + filenames[-1])
        seg = tf.reshape(tf.cast(tf.image.decode_png(seg_raw, channels=1, dtype=tf.uint8), tf.float32),
                         image_size)
        if self.partial_frame:
            crop_y_start = int(((1 - self.partial_frame) * image_size[0]) / 2)
            crop_y_end = int(((1 + self.partial_frame) * image_size[0]) / 2)
            crop_x_start = int(((1 - self.partial_frame) * image_size[1]) / 2)
            crop_x_end = int(((1 + self.partial_frame) * image_size[1]) / 2)
            seg = tf.slice(seg, [crop_y_start, crop_x_start, 0], [crop_y_end, crop_x_end, -1])

        for i in range(len(filenames) - 1):
            im_filename = filenames[i]
            im_raw = tf.read_file(self.base_folder + im_filename)
            image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                               image_size)

            if self.partial_frame:
                image = tf.slice(image, [crop_y_start, crop_x_start, 0], [crop_y_end, crop_x_end, -1])

            im_list.append(image)

        return im_list, [seg], filenames

    def get_batch(self, batch_size=1):
        self.batch_size = batch_size
        if self.one_seg:
            image_list_in, seg_list_in, file_names = self._get_image_sequence_one_seg()
        else:
            image_list_in, seg_list_in, file_names = self._get_image_sequence()
        seq_length = len(image_list_in)
        seg_length = len(seg_list_in)
        concat = tf.concat(axis=2, values=(image_list_in + seg_list_in))
        image_seq_list = []
        seg_seq_list = []
        filename_list = []

        for _ in range(self.crops_per_image):
            cropped = tf.random_crop(concat, [self.crop_size[0], self.crop_size[1], seq_length + seg_length])
            shape = cropped.get_shape()
            flip_lr = tf.image.random_flip_left_right(cropped)
            flip_ud = tf.image.random_flip_up_down(flip_lr)
            rot = tf.image.rot90(flip_ud, k=self.random_rotate)
            rot.set_shape(shape)
            rot = tf.expand_dims(rot, 3)
            rot = tf.transpose(rot, [2, 0, 1, 3])
            image_seq, seg_seq = tf.split(rot, num_or_size_splits=[seq_length, seg_length], axis=0)

            image_seq_list.append(image_seq)
            seg_seq_list.append(seg_seq)
            filename_list.append(file_names)

        image_many = tf.stack(values=image_seq_list, axis=0)
        seg_many = tf.stack(values=seg_seq_list, axis=0)
        filename_many = tf.stack(values=filename_list, axis=0)

        (image_seq_batch, seg_seq_batch,
         filename_batch) = tf.train.shuffle_batch([image_many, seg_many, filename_many], batch_size=self.batch_size,
                                                  num_threads=self.num_threads, capacity=self.capacity,
                                                  min_after_dequeue=self.min_after_dequeue, enqueue_many=True)

        if self.data_format == 'NCHW':
            image_seq_batch = tf.transpose(image_seq_batch, perm=[0, 1, 4, 2, 3])
            seg_seq_batch = tf.transpose(seg_seq_batch, perm=[0, 1, 4, 2, 3])

        image_seq_batch_list = tf.unstack(image_seq_batch, seq_length, axis=1)
        if self.one_seg:
            seg_seq_batch_list = [tf.squeeze(seg_seq_batch, axis=1)]
        else:
            seg_seq_batch_list = tf.unstack(seg_seq_batch, seq_length, axis=1)

        return image_seq_batch_list, seg_seq_batch_list, filename_batch


class CSVSegReaderEvalLSTM(object):
    def __init__(self, filenames, base_folder='.', image_size=(), num_threads=1, capacity=20, min_after_dequeue=10,
                 data_format='NCHW'):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:

            filenames: a list of csv files filename
            batch_size: the number of examples in each patch
            image_size: a tuple containing the image size in Y and X dimensions
            num_threads: number of threads for prefetch
            capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            min_after_dequeue: the minimum example in the queue after a dequeue op. ensures good mixing
        """
        raw_filenames = []

        for filename in filenames:
            with open(os.path.join(base_folder, filename), 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')

                for row in csv_reader:
                    raw_filenames.append(row[0])
                    # if len(raw_filenames) < 2:
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])
                    #     raw_filenames.append(row[0])

        self.raw_queue = tf.train.string_input_producer(
            raw_filenames + raw_filenames[::-1] + raw_filenames + raw_filenames[::-1], shuffle=False, num_epochs=1)
        self.image_size = image_size
        self.seq_length = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.base_folder = base_folder
        self.data_format = data_format

    def _get_image(self):
        filename = self.raw_queue.dequeue()
        im_raw = tf.read_file(self.base_folder + filename)
        image_size = self.image_size + (1,)
        image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                           image_size)

        return image, filename

    def get_sequence(self, seq_length=7):
        self.seq_length = seq_length
        image, filename = self._get_image()

        (image_seq,
         filename_seq) = tf.train.batch([image, filename], batch_size=self.seq_length, num_threads=1,
                                        capacity=self.capacity)

        if self.data_format == 'NCHW':
            image_seq = tf.transpose(image_seq, perm=[0, 3, 1, 2])

        image_seq_list = tf.split(image_seq, seq_length, axis=0)

        return image_seq_list, filename_seq


class CSVSegReaderEvalBiLSTM(object):
    def __init__(self, filenames, base_folder='.', image_size=(), num_threads=1, capacity=20, min_after_dequeue=10,
                 data_format='NCHW'):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:

            filenames: a list of csv files filename
            batch_size: the number of examples in each patch
            image_size: a tuple containing the image size in Y and X dimensions
            num_threads: number of threads for prefetch
            capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            min_after_dequeue: the minimum example in the queue after a dequeue op. ensures good mixing
        """
        raw_filenames = []

        for filename in filenames:
            with open(os.path.join(base_folder, filename), 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')

                for row in csv_reader:
                    raw_filenames.append(row[0])
        # raw_filenames = raw_filenames[:9]
        self.raw_queue_fw = tf.train.string_input_producer(raw_filenames, shuffle=False, num_epochs=1)
        self.raw_queue_bw = tf.train.string_input_producer(raw_filenames[::-1], shuffle=False, num_epochs=1)
        self.image_size = image_size
        self.seq_length = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.base_folder = base_folder
        self.data_format = data_format

    def _get_image(self):
        image_size = self.image_size + (1,)
        filename_fw = self.raw_queue_fw.dequeue()
        im_raw = tf.read_file(self.base_folder + filename_fw)
        image_fw = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                              image_size)
        filename_bw = self.raw_queue_bw.dequeue()
        im_raw = tf.read_file(self.base_folder + filename_bw)

        image_bw = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                              image_size)

        return image_fw, filename_fw, image_bw, filename_bw

    def get_sequence(self, seq_length=7):
        self.seq_length = seq_length
        image_fw, filename_fw, image_bw, filename_bw = self._get_image()

        (image_seq_fw,
         filename_seq_fw) = tf.train.batch([image_fw, filename_fw], batch_size=self.seq_length, num_threads=1,
                                           capacity=self.capacity)
        (image_seq_bw,
         filename_seq_bw) = tf.train.batch([image_bw, filename_bw], batch_size=self.seq_length, num_threads=1,
                                           capacity=self.capacity)

        if self.data_format == 'NCHW':
            image_seq_fw = tf.transpose(image_seq_fw, perm=[0, 3, 1, 2])
            image_seq_bw = tf.transpose(image_seq_bw, perm=[0, 3, 1, 2])

        image_seq_list_fw = tf.split(image_seq_fw, seq_length, axis=0)
        image_seq_list_bw = tf.split(image_seq_bw, seq_length, axis=0)

        return image_seq_list_fw, filename_seq_fw, image_seq_list_bw, filename_seq_bw

    def _load_and_enqueue(self, sess, enqueue_op, coord):
        unroll_len = self.sub_seq_size[0]
        while not coord.should_stop():
            filename_list = self._read_csv_files()
            image_pl, seg_pl, is_last_pl = self.placeholders
            random_sub_sample = np.random.randint(0, 5) if self.randomize else 0
            random_reverse = np.random.randint(0, 2) if self.randomize else 0
            if random_reverse:
                filename_list.reverse()
            if random_sub_sample:
                filename_list = filename_list[::random_sub_sample]
            seq_len = len(filename_list)
            remainder = seq_len % unroll_len

            if remainder:
                if self.deal_with_end == 0:
                    filename_list = filename_list[:-remainder]
                elif self.deal_with_end == 1:
                    filename_list += filename_list[-2:-unroll_len + remainder - 2:-1]
                elif self.deal_with_end == 2:
                    filename_list += filename_list[-1:] * (unroll_len - remainder)

            sub_seq_img = np.zeros(self.sub_seq_size)
            sub_seq_seg = np.zeros(self.sub_seq_size)
            for t, filename in enumerate(filename_list):
                img = cv2.imread(filename[0], -1)
                if filename[1].endswith('NA'):
                    seg = np.ones(img.shape[:2]) * (-1)
                elif filename[1].endswith('_not_valid.png'):
                    seg = cv2.imread(filename[1], -1)
                    if seg is None:
                        seg = np.ones(img.shape[:2]) * (-1)
                    else:
                        seg = seg.astype(np.float32)
                    seg[seg == 0] = -1
                else:
                    seg = cv2.imread(filename[1], -1)

                if t == 0:
                    img_size = img.shape
                    crop_y = np.random.randint(0, img_size[0] - self.sub_seq_size[1]) if self.randomize else 0
                    crop_x = np.random.randint(0, img_size[1] - self.sub_seq_size[2]) if self.randomize else 0
                    flip = np.random.randint(0, 2, 2) if self.randomize else 0
                    rotate = np.random.randint(0, 4) * 90 if self.randomize else 0

                img_crop = img[crop_y:crop_y + self.sub_seq_size[1], crop_x:crop_x + self.sub_seq_size[2]]
                seg_crop = seg[crop_y:crop_y + self.sub_seq_size[1], crop_x:crop_x + self.sub_seq_size[2]]

                if flip[0]:
                    img_crop = cv2.flip(img_crop, 0)
                    seg_crop = cv2.flip(seg_crop, 0)
                if flip[1]:
                    img_crop = cv2.flip(img_crop, 1)
                    seg_crop = cv2.flip(seg_crop, 1)
                if rotate == 1:
                    img_crop = img_crop.T
                    seg_crop = seg_crop.T
                elif rotate == 2:
                    img_crop = cv2.flip(img_crop, -1)
                    seg_crop = cv2.flip(seg_crop, -1)
                elif rotate == 3:
                    img_crop = cv2.flip(img_crop, -1)
                    seg_crop = cv2.flip(seg_crop, -1)
                    img_crop = img_crop.T
                    seg_crop = seg_crop.T

                sub_seq_img[t % unroll_len] = img_crop
                sub_seq_seg[t % unroll_len] = seg_crop
                if not ((t + 1) % unroll_len):
                    is_last = 0. if (t + 1) < len(filename_list) else 1.
                    sess.run(enqueue_op, {image_pl: sub_seq_img, seg_pl: sub_seq_seg, is_last_pl: is_last})

    def _create_queues(self):
        image_pl = tf.placeholder(tf.float32, self.sub_seq_size)
        seg_pl = tf.placeholder(tf.float32, self.sub_seq_size)
        is_last_pl = tf.placeholder(tf.float32, ())
        placeholders = (image_pl, seg_pl, is_last_pl)
        q_list = []
        enqueue_op_list = []
        for _ in range(self.batch_size):
            q = tf.FIFOQueue(self.queue_capacity, dtypes=[tf.float32, tf.float32, tf.float32],
                             shapes=[self.sub_seq_size, self.sub_seq_size, ()])
            q_list.append(q)
            enqueue_op_list.append(q.enqueue((image_pl, seg_pl, is_last_pl)))
        return q_list, enqueue_op_list, placeholders

    def _batch_queues(self):
        img_list = []
        seg_list = []
        is_last_list = []
        for q in self.q_list:
            img, seg, is_last = q.dequeue()
            img_list.append(img)
            seg_list.append(seg)
            is_last_list.append(is_last)
        image_batch = tf.stack(img_list, axis=1)
        seg_batch = tf.stack(seg_list, axis=1)
        is_last_batch = tf.stack(is_last_list, axis=0)
        if self.data_format == 'NHWC':
            image_batch = tf.expand_dims(image_batch, 4)
            seg_batch = tf.expand_dims(seg_batch, 4)
        elif self.data_format == 'NCHW':
            image_batch = tf.expand_dims(image_batch, 2)
            seg_batch = tf.expand_dims(seg_batch, 2)
        else:
            raise ValueError()
        image_batch_list = tf.unstack(image_batch, num=self.sub_seq_size[0], axis=0)
        seg_batch_list = tf.unstack(seg_batch, num=self.sub_seq_size[0], axis=0)
        return image_batch_list, seg_batch_list, is_last_batch

    def _create_csv_queue(self):
        csv_queue = queue.Queue(maxsize=len(self.csv_file_list))
        for csv_file in self.csv_file_list:
            csv_queue.put(csv_file)
        return csv_queue

    def start_queues(self, sess, coord=tf.train.Coordinator()):
        threads = []
        for enqueue_op in self.enqueue_op_list:
            t = threading.Thread(target=self._load_and_enqueue, args=(sess, enqueue_op, coord))
            t.daemon = True
            t.start()
            threads.append(t)
        threads += tf.train.start_queue_runners(sess, coord)
        return threads

    def get_batch(self):
        return self.batch


class DIRSegReaderEvalBiLSTM(object):
    def __init__(self, data_dir: str, filename_format='t*.tif', image_size=(0, 0), capacity=20, data_format='NCHW'):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:
            :param data_dir: directory including all image files
            :type data_dir: str
            :param filename_format: the format of the files in the directory. use * as a wildcard
            :type filename_format: str
            :param image_size: a tuple containing the image size in Y and X dimensions
            :type image_size: Tuple[int,int]
            :param capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            :type capacity: int
            :param data_format: the format of the data, either NHWC (Batch, Height, Width, Channel) of NCHW 
            :type data_format: str
            
        """

        raw_filenames = glob.glob(os.path.join(data_dir, filename_format))
        raw_filenames.sort()

        self.raw_queue_fw = tf.train.string_input_producer(raw_filenames, shuffle=False, num_epochs=1)
        self.raw_queue_bw = tf.train.string_input_producer(raw_filenames[::-1], shuffle=False, num_epochs=1)
        self.image_size = image_size
        self.seq_length = None
        self.capacity = capacity
        self.data_format = data_format

    def _get_image(self):
        image_size = self.image_size + (1,)
        filename_fw = self.raw_queue_fw.dequeue()
        im_raw = tf.read_file(filename_fw)
        image_fw = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                              image_size)
        filename_bw = self.raw_queue_bw.dequeue()
        im_raw = tf.read_file(filename_bw)

        image_bw = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                              image_size)

        return image_fw, filename_fw, image_bw, filename_bw

    def get_sequence(self, seq_length=7):
        self.seq_length = seq_length
        image_fw, filename_fw, image_bw, filename_bw = self._get_image()

        (image_seq_fw,
         filename_seq_fw) = tf.train.batch([image_fw, filename_fw], batch_size=self.seq_length, num_threads=1,
                                           capacity=self.capacity)
        (image_seq_bw,
         filename_seq_bw) = tf.train.batch([image_bw, filename_bw], batch_size=self.seq_length, num_threads=1,
                                           capacity=self.capacity)

        if self.data_format == 'NCHW':
            image_seq_fw = tf.transpose(image_seq_fw, perm=[0, 3, 1, 2])
            image_seq_bw = tf.transpose(image_seq_bw, perm=[0, 3, 1, 2])

        image_seq_list_fw = tf.split(image_seq_fw, seq_length, axis=0)
        image_seq_list_bw = tf.split(image_seq_bw, seq_length, axis=0)

        return image_seq_list_fw, filename_seq_fw, image_seq_list_bw, filename_seq_bw


class DIRSegReaderEvalLSTM(object):
    def __init__(self, data_dir: str, filename_format='t*.tif', image_size=(0, 0), capacity=20, data_format='NCHW'):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batches of corresponding image inputs and segmentation inputs.
         The inputs to the class are:
            :param data_dir: directory including all image files
            :type data_dir: str
            :param filename_format: the format of the files in the directory. use * as a wildcard
            :type filename_format: str
            :param image_size: a tuple containing the image size in Y and X dimensions
            :type image_size: Tuple[int,int]
            :param capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            :type capacity: int
            :param data_format: the format of the data, either NHWC (Batch, Height, Width, Channel) of NCHW 
            :type data_format: str

        """

        raw_filenames = glob.glob(os.path.join(data_dir, filename_format))
        raw_filenames.sort()

        self.raw_queue = tf.train.string_input_producer(raw_filenames, shuffle=False, num_epochs=1)
        self.image_size = image_size
        self.seq_length = None
        self.capacity = capacity
        self.data_format = data_format

    def _get_image(self):
        image_size = self.image_size + (1,)
        filename = self.raw_queue.dequeue()
        im_raw = tf.read_file(filename)
        image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                           image_size)

        return image, filename

    def get_sequence(self, seq_length=7):
        self.seq_length = seq_length
        image, filename = self._get_image()

        (image_seq,
         filename_seq) = tf.train.batch([image, filename], batch_size=self.seq_length, num_threads=1,
                                        capacity=self.capacity)

        if self.data_format == 'NCHW':
            image_seq = tf.transpose(image_seq, perm=[0, 3, 1, 2])

        image_seq_list = tf.split(image_seq, seq_length, axis=0)

        return image_seq_list, filename_seq


class CSVSegReaderSequence(object):
    def __init__(self, csv_file_list, image_crop_size=(128, 128), unroll_len=7, deal_with_end=0, batch_size=4,
                 queue_capacity=32, data_format='NCHW', randomize=True):
        self.csv_file_list = csv_file_list
        self.sub_seq_size = (unroll_len,) + image_crop_size
        self.deal_with_end = deal_with_end
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.randomize = randomize

        self.csv_queue = self._create_csv_queue()
        self.q_list, self.enqueue_op_list, self.q_not_full_list, self.placeholders = self._create_queues()
        self.batch = self._batch_queues()
        np.random.seed(1)

    def _read_csv_files(self):
        csv_filename = self.csv_queue.get()
        self.csv_queue.put(csv_filename)
        filenames = []
        csv_folder = os.path.dirname(csv_filename)
        with open(csv_filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            for row in csv_reader:
                full_path_row = [os.path.join(csv_folder, item) for item in row]
                filenames.append(full_path_row)
        return filenames

    def _load_and_enqueue(self, sess, enqueue_op, q_not_full_op, coord):
        run_options = tf.RunOptions(timeout_in_ms=1000)
        unroll_len = self.sub_seq_size[0]
        while not coord.should_stop():
            filename_list = self._read_csv_files()
            image_pl, seg_pl, is_last_pl = self.placeholders
            random_sub_sample = np.random.randint(1, 4) if self.randomize else 0
            random_reverse = np.random.randint(0, 2) if self.randomize else 0
            if random_reverse:
                filename_list.reverse()
            if random_sub_sample:
                filename_list = filename_list[::random_sub_sample]
            seq_len = len(filename_list)
            remainder = seq_len % unroll_len

            if remainder:
                if self.deal_with_end == 0:
                    filename_list = filename_list[:-remainder]
                elif self.deal_with_end == 1:
                    filename_list += filename_list[-2:-unroll_len + remainder - 2:-1]
                elif self.deal_with_end == 2:
                    filename_list += filename_list[-1:] * (unroll_len - remainder)

            sub_seq_img = np.zeros(self.sub_seq_size)
            sub_seq_seg = np.zeros(self.sub_seq_size)
            for t, filename in enumerate(filename_list):
                img = cv2.imread(filename[0], -1)
                if img is None:
                    print('Could Not Load Image: {}'.format(filename[0]))
                if filename[1].endswith('NA'):
                    seg = np.ones(img.shape[:2]) * (-1)
                elif '_not_valid' in filename[1]:
                    seg = cv2.imread(filename[1], -1)
                    if seg is None:
                        seg = np.ones(img.shape[:2]) * (-1)
                    else:
                        seg = seg.astype(np.float32)
                    seg[seg == 0] = -1
                else:
                    seg = cv2.imread(filename[1], -1)

                if t == 0:
                    img_size = img.shape
                    crop_y = np.random.randint(0, img_size[0] - self.sub_seq_size[1]) if self.randomize else 0
                    crop_x = np.random.randint(0, img_size[1] - self.sub_seq_size[2]) if self.randomize else 0
                    flip = np.random.randint(0, 2, 2) if self.randomize else [0,0]
                    rotate = np.random.randint(0, 4) * 90 if self.randomize else 0

                img_crop = img[crop_y:crop_y + self.sub_seq_size[1], crop_x:crop_x + self.sub_seq_size[2]]
                seg_crop = seg[crop_y:crop_y + self.sub_seq_size[1], crop_x:crop_x + self.sub_seq_size[2]]

                if flip[0]:
                    img_crop = cv2.flip(img_crop, 0)
                    seg_crop = cv2.flip(seg_crop, 0)
                if flip[1]:
                    img_crop = cv2.flip(img_crop, 1)
                    seg_crop = cv2.flip(seg_crop, 1)
                if rotate == 1:
                    img_crop = img_crop.T
                    seg_crop = seg_crop.T
                elif rotate == 2:
                    img_crop = cv2.flip(img_crop, -1)
                    seg_crop = cv2.flip(seg_crop, -1)
                elif rotate == 3:
                    img_crop = cv2.flip(img_crop, -1)
                    seg_crop = cv2.flip(seg_crop, -1)
                    img_crop = img_crop.T
                    seg_crop = seg_crop.T

                sub_seq_img[t % unroll_len] = img_crop
                sub_seq_seg[t % unroll_len] = seg_crop
                if not ((t + 1) % unroll_len):
                    is_last = 0. if (t + 1) < len(filename_list) else 1.
                    retry = True
                    while retry and not coord.should_stop():
                        try:
                            if sess.run(q_not_full_op):
                                sess.run(enqueue_op, {image_pl: sub_seq_img, seg_pl: sub_seq_seg, is_last_pl: is_last},
                                         options=run_options)
                                retry = False
                        except tf.errors.DeadlineExceededError:
                            retry = True

    def _create_queues(self):
        image_pl = tf.placeholder(tf.float32, self.sub_seq_size)
        seg_pl = tf.placeholder(tf.float32, self.sub_seq_size)
        is_last_pl = tf.placeholder(tf.float32, ())
        placeholders = (image_pl, seg_pl, is_last_pl)
        q_list = []
        enqueue_op_list = []
        q_not_full_list = []
        for _ in range(self.batch_size):
            q = tf.FIFOQueue(self.queue_capacity, dtypes=[tf.float32, tf.float32, tf.float32],
                             shapes=[self.sub_seq_size, self.sub_seq_size, ()])
            q_list.append(q)
            q_not_full = tf.greater(self.queue_capacity, q.size())
            q_not_full_list.append(q_not_full)
            enqueue_op_list.append(q.enqueue((image_pl, seg_pl, is_last_pl)))
        return q_list, enqueue_op_list, q_not_full_list, placeholders

    def _batch_queues(self):
        img_list = []
        seg_list = []
        is_last_list = []
        for q in self.q_list:
            img, seg, is_last = q.dequeue()
            img_list.append(img)
            seg_list.append(seg)
            is_last_list.append(is_last)
        image_batch = tf.stack(img_list, axis=1)
        seg_batch = tf.stack(seg_list, axis=1)
        is_last_batch = tf.stack(is_last_list, axis=0)
        if self.data_format == 'NHWC':
            image_batch = tf.expand_dims(image_batch, 4)
            seg_batch = tf.expand_dims(seg_batch, 4)
        elif self.data_format == 'NCHW':
            image_batch = tf.expand_dims(image_batch, 2)
            seg_batch = tf.expand_dims(seg_batch, 2)
        else:
            raise ValueError()
        image_batch_list = tf.unstack(image_batch, num=self.sub_seq_size[0], axis=0)
        seg_batch_list = tf.unstack(seg_batch, num=self.sub_seq_size[0], axis=0)
        return image_batch_list, seg_batch_list, is_last_batch

    def _create_csv_queue(self):
        csv_queue = queue.Queue(maxsize=len(self.csv_file_list))
        for csv_file in self.csv_file_list:
            csv_queue.put(csv_file)
        return csv_queue

    def start_queues(self, sess, coord=tf.train.Coordinator()):
        threads = []
        for enqueue_op, q_not_full_op in zip(self.enqueue_op_list, self.q_not_full_list):
            t = threading.Thread(target=self._load_and_enqueue, args=(sess, enqueue_op, q_not_full_op, coord))
            t.daemon = True
            t.start()
            threads.append(t)
        threads += tf.train.start_queue_runners(sess, coord)
        return threads

    def get_batch(self):
        return self.batch


class CSVSegReaderSequence_Unet(object):
    def __init__(self, csv_file_list, image_crop_size=(128, 128), unroll_len=7, deal_with_end=0, batch_size=4,
                 queue_capacity=32, data_format='NCHW', randomize=True):
        self.csv_file_list = csv_file_list
        self.sub_seq_size = (unroll_len,) + image_crop_size
        self.deal_with_end = deal_with_end
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.randomize = randomize

        self.csv_queue = self._create_csv_queue()
        self.q_list, self.enqueue_op_list, self.q_not_full_list, self.placeholders = self._create_queues()
        self.batch = self._batch_queues()
        np.random.seed(1)

    def _read_csv_files(self):
        csv_filename = self.csv_queue.get()
        self.csv_queue.put(csv_filename)
        filenames = []
        csv_folder = os.path.dirname(csv_filename)
        with open(csv_filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            for row in csv_reader:
                full_path_row = [os.path.join(csv_folder, item) for item in row]
                filenames.append(full_path_row)
        return filenames

    def _load_and_enqueue(self, sess, enqueue_op, q_not_full_op, coord):
        unroll_len = self.sub_seq_size[0]
        run_options = tf.RunOptions(timeout_in_ms=1000)
        while not coord.should_stop():
            filename_list = self._read_csv_files()
            image_pl, seg_pl, weight_pl, is_last_pl = self.placeholders
            random_sub_sample = np.random.randint(0, 5) if self.randomize else 0
            random_reverse = np.random.randint(0, 2) if self.randomize else 0
            if random_reverse:
                filename_list.reverse()
            if random_sub_sample:
                filename_list = filename_list[::random_sub_sample]
            seq_len = len(filename_list)
            remainder = seq_len % unroll_len

            if remainder:
                if self.deal_with_end == 0:
                    filename_list = filename_list[:-remainder]
                elif self.deal_with_end == 1:
                    filename_list += filename_list[-2:-unroll_len + remainder - 2:-1]
                elif self.deal_with_end == 2:
                    filename_list += filename_list[-1:] * (unroll_len - remainder)

            sub_seq_img = np.zeros(self.sub_seq_size)
            sub_seq_seg = np.zeros(self.sub_seq_size)
            sub_seq_weight = np.zeros(self.sub_seq_size)
            for t, filename in enumerate(filename_list):
                img = cv2.imread(filename[0], -1)
                if filename[1].endswith('NA'):
                    seg = np.ones(img.shape[:2]) * (-1)
                    weight = np.ones(img.shape[:2])
                elif filename[1].endswith('_not_valid.tif'):
                    seg = cv2.imread(filename[1], -1)
                    if seg is None:
                        seg = np.ones(img.shape[:2]) * (-1)
                    else:
                        seg = seg.astype(np.float32)
                    seg[seg == 0] = -1
                    weight = np.fromfile(filename[0].replace('Raw', 'UnetMask'))
                    weight = weight.reshape(seg.shape)

                else:
                    seg = cv2.imread(filename[1], -1)
                    weight = np.fromfile(filename[0].replace('Raw', 'UnetMask'))
                    weight = weight.reshape(seg.shape)

                if t == 0:
                    img_size = img.shape
                    crop_y = np.random.randint(0, img_size[0] - self.sub_seq_size[1]) if self.randomize else 0
                    crop_x = np.random.randint(0, img_size[1] - self.sub_seq_size[2]) if self.randomize else 0
                    flip = np.random.randint(0, 2, 2) if self.randomize else 0
                    rotate = np.random.randint(0, 4) * 90 if self.randomize else 0

                img_crop = img[crop_y:crop_y + self.sub_seq_size[1], crop_x:crop_x + self.sub_seq_size[2]]
                seg_crop = seg[crop_y:crop_y + self.sub_seq_size[1], crop_x:crop_x + self.sub_seq_size[2]]
                weight_crop = weight[crop_y:crop_y + self.sub_seq_size[1], crop_x:crop_x + self.sub_seq_size[2]]

                if flip[0]:
                    img_crop = cv2.flip(img_crop, 0)
                    seg_crop = cv2.flip(seg_crop, 0)
                    weight_crop = cv2.flip(weight_crop, 0)
                if flip[1]:
                    img_crop = cv2.flip(img_crop, 1)
                    seg_crop = cv2.flip(seg_crop, 1)
                    weight_crop = cv2.flip(weight_crop, 1)
                if rotate == 1:
                    img_crop = img_crop.T
                    seg_crop = seg_crop.T
                    weight_crop = weight_crop.T
                elif rotate == 2:
                    img_crop = cv2.flip(img_crop, -1)
                    seg_crop = cv2.flip(seg_crop, -1)
                    weight_crop = cv2.flip(weight_crop, -1)
                elif rotate == 3:
                    img_crop = cv2.flip(img_crop, -1)
                    seg_crop = cv2.flip(seg_crop, -1)
                    weight_crop = cv2.flip(weight_crop, -1)
                    img_crop = img_crop.T
                    seg_crop = seg_crop.T
                    weight_crop = weight_crop.T

                sub_seq_img[t % unroll_len] = img_crop
                sub_seq_seg[t % unroll_len] = seg_crop
                sub_seq_weight[t % unroll_len] = weight_crop
                if not ((t + 1) % unroll_len):
                    is_last = 0. if (t + 1) < len(filename_list) else 1.

                    retry = True
                    while retry and not coord.should_stop():

                        try:
                            if sess.run(q_not_full_op):
                                sess.run(enqueue_op, {image_pl: sub_seq_img, seg_pl: sub_seq_seg, weight_pl: sub_seq_weight,
                                                      is_last_pl: is_last}, options=run_options)
                                retry = False
                        except tf.errors.DeadlineExceededError:
                            retry = True


    def _create_queues(self):
        image_pl = tf.placeholder(tf.float32, self.sub_seq_size)
        seg_pl = tf.placeholder(tf.float32, self.sub_seq_size)
        weight_pl = tf.placeholder(tf.float32, self.sub_seq_size)
        is_last_pl = tf.placeholder(tf.float32, ())
        placeholders = (image_pl, seg_pl, weight_pl, is_last_pl)
        q_list = []
        enqueue_op_list = []
        not_full_list = []
        for _ in range(self.batch_size):
            q = tf.FIFOQueue(self.queue_capacity, dtypes=[tf.float32, tf.float32, tf.float32, tf.float32],
                             shapes=[self.sub_seq_size, self.sub_seq_size, self.sub_seq_size, ()])
            q_list.append(q)
            enqueue_op_list.append(q.enqueue((image_pl, seg_pl, weight_pl, is_last_pl)))
            not_full_list.append(tf.greater(self.queue_capacity, q.size()))
        return q_list, enqueue_op_list, not_full_list, placeholders

    def _batch_queues(self):
        img_list = []
        seg_list = []
        weight_list = []
        is_last_list = []
        for q in self.q_list:
            img, seg, weight, is_last = q.dequeue()
            img_list.append(img)
            seg_list.append(seg)
            weight_list.append(weight)
            is_last_list.append(is_last)
        image_batch = tf.stack(img_list, axis=1)
        seg_batch = tf.stack(seg_list, axis=1)
        weight_batch = tf.stack(weight_list, axis=1)
        is_last_batch = tf.stack(is_last_list, axis=0)
        if self.data_format == 'NHWC':
            image_batch = tf.expand_dims(image_batch, 4)
            seg_batch = tf.expand_dims(seg_batch, 4)
            weight_batch = tf.expand_dims(weight_batch, 4)
        elif self.data_format == 'NCHW':
            image_batch = tf.expand_dims(image_batch, 2)
            seg_batch = tf.expand_dims(seg_batch, 2)
            weight_batch = tf.expand_dims(weight_batch, 2)
        else:
            raise ValueError()
        image_batch_list = tf.unstack(image_batch, num=self.sub_seq_size[0], axis=0)
        seg_batch_list = tf.unstack(seg_batch, num=self.sub_seq_size[0], axis=0)
        weight_batch_list = tf.unstack(weight_batch, num=self.sub_seq_size[0], axis=0)
        return image_batch_list, seg_batch_list, weight_batch_list, is_last_batch

    def _create_csv_queue(self):
        csv_queue = queue.Queue(maxsize=len(self.csv_file_list))
        for csv_file in self.csv_file_list:
            csv_queue.put(csv_file)
        return csv_queue

    def start_queues(self, sess, coord=tf.train.Coordinator()):
        threads = []
        for enqueue_op, not_full_op in zip(self.enqueue_op_list, self.q_not_full_list):
            t = threading.Thread(target=self._load_and_enqueue, args=(sess, enqueue_op, not_full_op, coord))
            t.daemon = True
            t.start()
            threads.append(t)
        threads += tf.train.start_queue_runners(sess, coord)
        return threads

    def get_batch(self):
        return self.batch


class CSVSegReaderSequenceEval(object):
    def __init__(self, data_dir: str, filename_format='t*.tif', queue_capacity=20, padding=(16, 16, 16, 16),
                 image_size=(0, 0), data_format='NCHW'):
        self.data_dir = data_dir
        if image_size[0] % 8:
            pad_y = 8 - (image_size[0] % 8)
        else:
            pad_y = 0
        if image_size[1] % 8:
            pad_x = 8 - (image_size[1] % 8)
        else:
            pad_x = 0
        self.padding = padding
        self.pad_end = (pad_y, pad_x)
        self.sub_seq_size = (1, image_size[0] + pad_y + padding[0]+padding[1], image_size[1] + pad_x + padding[2]+padding[3])
        # self.deal_with_end = deal_with_end
        self.filename_format = filename_format
        self.batch_size = 1
        self.queue_capacity = queue_capacity
        self.data_format = data_format

        self.dir_filelist = self._get_dir_filelist_()
        self.q_list, self.enqueue_op_list, self.placeholders = self._create_queues()
        self.batch = self._batch_queues()
        np.random.seed(1)

    def _get_dir_filelist_(self):
        raw_filenames = glob.glob(os.path.join(self.data_dir, self.filename_format))
        raw_filenames.sort()
        return raw_filenames

    def _load_and_enqueue(self, sess, q, enqueue_op, coord):
        unroll_len = self.sub_seq_size[0]
        if not coord.should_stop():
            filename_list = self.dir_filelist
            image_pl, filename_ph = self.placeholders
            seq_len = len(filename_list)

            sub_seq_img = np.zeros(self.sub_seq_size)

            for t, filename in enumerate(filename_list):
                img = cv2.imread(filename, -1)

                img_size = img.shape
                img = cv2.copyMakeBorder(img, self.padding[0], self.padding[1], self.padding[2], self.padding[3],
                                         cv2.BORDER_REFLECT_101)
                img = cv2.copyMakeBorder(img, 0, self.pad_end[0], 0, self.pad_end[1], cv2.BORDER_REFLECT_101)

                sub_seq_img[t % unroll_len] = img

                if not ((t + 1) % unroll_len):
                    sess.run(enqueue_op, {image_pl: sub_seq_img, filename_ph: filename})
        coord.request_stop()
        q.close()


    def _create_queues(self):
        image_pl = tf.placeholder(tf.float32, self.sub_seq_size)
        filename_pl = tf.placeholder(tf.string, ())
        placeholders = (image_pl, filename_pl)
        q_list = []
        enqueue_op_list = []
        for _ in range(self.batch_size):
            q = tf.FIFOQueue(self.queue_capacity, dtypes=[tf.float32, tf.string],
                             shapes=[self.sub_seq_size, ()])
            q_list.append(q)
            enqueue_op_list.append(q.enqueue((image_pl, filename_pl)))
        return q_list, enqueue_op_list, placeholders

    def _batch_queues(self):
        img_list = []
        seg_list = []
        filename_list = []
        for q in self.q_list:
            img, filename = q.dequeue()
            img_list.append(img)
            filename_list.append(filename)
        image_batch = tf.stack(img_list, axis=1)

        filename_batch = tf.stack(filename_list, axis=0)
        if self.data_format == 'NHWC':
            image_batch = tf.expand_dims(image_batch, 4)

        elif self.data_format == 'NCHW':
            image_batch = tf.expand_dims(image_batch, 2)

        else:
            raise ValueError()
        image_batch_list = tf.unstack(image_batch, num=self.sub_seq_size[0], axis=0)

        return image_batch_list, filename_batch

    def start_queues(self, sess, coord=tf.train.Coordinator()):
        threads = []
        for enqueue_op, q in zip(self.enqueue_op_list, self.q_list):
            t = threading.Thread(target=self._load_and_enqueue, args=(sess, q, enqueue_op, coord))
            t.daemon = True
            t.start()
            threads.append(t)
        threads += tf.train.start_queue_runners(sess, coord)
        return threads

    def get_batch(self):
        return self.batch


def tif2png_dir(data_dir: str, out_dir: str, filename_format='t*.tif'):
    """
    tif2png_dir is a function that converts a directory of tif files to png files
     The inputs to the class are:
        :param data_dir: directory including all image files
        :type data_dir: str
        :param out_dir: directory to output
        :type out_dir: str
        :param filename_format: the format of the files in the directory. use * as a wildcard
        :type filename_format: str

    """

    tif_filenames = glob.glob(os.path.join(data_dir, filename_format))
    tif_filenames.sort()
    os.makedirs(out_dir, exist_ok=True)

    pad_y = 0
    pad_x = 0
    for tif_filename in tif_filenames:
        img = cv2.imread(tif_filename, -1)
        img_size = img.shape
        if img_size[0] % 8:
            pad_y = 8 - (img_size[0] % 8)
        else:
            pad_y = 0
        if img_size[1] % 8:
            pad_x = 8 - (img_size[1] % 8)
        else:
            pad_x = 0
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
        if pad_x or pad_y:
            img = cv2.copyMakeBorder(img, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT_101)
        base_name = os.path.basename(tif_filename)
        base_name = base_name.replace('.tif', '.png')
        png_filename = os.path.join(out_dir, base_name)
        cv2.imwrite(png_filename, img)
    return pad_y, pad_x
