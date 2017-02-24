import csv
import tensorflow as tf
# import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import utils


__author__ = 'assafarbelle'


class CSVSegReader(object):

    def __init__(self, filenames, base_folder='.', image_size=(64, 64, 1), num_threads=4,
                 capacity=20, min_after_dequeue=10, random=True):
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
        self.input_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, seed =0 )
        self.image_size = image_size
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = None
        self.base_folder = base_folder
        self.random = random

    def _get_image(self):

        _, records = self.reader.read(self.input_queue)
        file_names = tf.decode_csv(records, [tf.constant([],  tf.string), tf.constant([], tf.string)], field_delim=None,
                                   name=None)

        im_raw = tf.read_file(self.base_folder+file_names[0])
        seg_raw = tf.read_file(self.base_folder+file_names[1])
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
        return image_batch, seg_batch, filename_batch


class CSVSegReaderRandom(object):

    def __init__(self, filenames, base_folder='.', image_size=(), crop_size=(64, 64), num_threads=4,
                 capacity=20, min_after_dequeue=10,
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

    def _get_image(self):
        _, records = self.reader.read(self.input_queue)
        file_names = tf.decode_csv(records, [tf.constant([], tf.string), tf.constant([], tf.string)],
                                   field_delim=None, name=None)

        im_raw = tf.read_file(self.base_folder+file_names[0])
        seg_raw = tf.read_file(self.base_folder+file_names[1])
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
        concat = tf.concat(2, [image, seg])

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

        return image_batch, seg_batch, filename_batch


class CSVSegReaderRandom2(object):

    def __init__(self, filenames, base_folder='.', image_size=(), crop_size=(64, 64), num_threads=4,
                 capacity=20, min_after_dequeue=10,
                 random_rotate=tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32), num_examples=None):
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
        seg_filenames = []
        for filename in filenames:
            with open(filename, 'rb') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
                for row in csv_reader:
                   raw_filenames.append(row[0]+':'+row[1])
        if not num_examples:
            pass
        elif isinstance(num_examples, int):
            num_examples = min(num_examples, len(raw_filenames))
            raw_filenames = raw_filenames[-num_examples:]
            #seg_filenames = seg_filenames[:num_examples]
        elif isinstance(num_examples, list):
            raw_filenames = [f_name for n, f_name in enumerate(raw_filenames) if n in num_examples]
            #seg_filenames = [f_name for n, f_name in enumerate(seg_filenames) if n in num_examples]

        self.raw_queue = tf.train.string_input_producer(raw_filenames, seed=0)
        #self.seg_queue = tf.train.string_input_producer(seg_filenames, seed=0)

        self.image_size = image_size
        self.crop_size = crop_size
        self.random_rotate = random_rotate
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = None
        self.base_folder = base_folder

    def _get_image(self):

        im_filename = tf.sparse_tensor_to_dense(tf.string_split(tf.expand_dims(self.raw_queue.dequeue(), 0), ':'), '')
        #seg_filename = self.seg_queue.dequeue()
        im_filename.set_shape([1, 2])
        im_raw = tf.read_file(self.base_folder+im_filename[0][0])
        seg_raw = tf.read_file(self.base_folder+im_filename[0][1])

        image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                           self.image_size, name='input_image')
        seg = tf.reshape(tf.cast(tf.image.decode_png(seg_raw, channels=1, dtype=tf.uint8), tf.float32), self.image_size,
                         name='input_seg')

        return image, seg, im_filename[0][0], im_filename[0][1]

    def get_batch(self, batch_size=1):
        self.batch_size = batch_size
        image_in, seg_in, file_name, seg_filename = self._get_image()
        concat = tf.concat(2, [image_in, seg_in])
        cropped = tf.random_crop(concat, [self.crop_size[0], self.crop_size[1], 2])
        shape = cropped.get_shape()
        fliplr = tf.image.random_flip_left_right(cropped)
        flipud = tf.image.random_flip_up_down(fliplr)
        rot = tf.image.rot90(flipud, k=self.random_rotate)
        rot.set_shape(shape)
        image, seg = tf.unstack(rot, 2, 2)
        image = tf.expand_dims(image, 2)
        seg = tf.expand_dims(seg, 2)
        image_batch, seg_batch, filename_batch, seg_filename_batch = tf.train.shuffle_batch([image, seg, file_name, seg_filename],
                                                                        batch_size=self.batch_size,
                                                                        num_threads=self.num_threads,
                                                                        capacity=self.capacity,
                                                                        min_after_dequeue=self.min_after_dequeue)

        return image_batch, seg_batch, filename_batch


class CSVSegReader2(object):

    def __init__(self, filenames, base_folder='.', image_size=(64, 64, 1), num_threads=4,
                 capacity=20, min_after_dequeue=10, num_examples=None, random=True):
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
        seg_filenames = []
        for filename in filenames:
            with open(filename, 'rb') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
                for row in csv_reader:
                    raw_filenames.append(row[0]+':'+row[1])

        if not num_examples:
            pass
        elif isinstance(num_examples, int):
            raw_filenames = raw_filenames[:num_examples]
            #seg_filenames = seg_filenames[:num_examples]
        elif isinstance(num_examples, list):
            raw_filenames = [f_name for n, f_name in enumerate(raw_filenames) if n in num_examples]
            #seg_filenames = [f_name for n, f_name in enumerate(seg_filenames) if n in num_examples]

        self.raw_queue = tf.train.string_input_producer(raw_filenames, num_epochs=num_epochs, shuffle=random, seed=0)
        #self.seg_queue = tf.train.string_input_producer(seg_filenames, num_epochs=num_epochs, shuffle=random, seed=0)

        self.image_size = image_size
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = None
        self.base_folder = base_folder
        self.random = random

    def _get_image(self):

        filename = tf.sparse_tensor_to_dense(tf.string_split(tf.expand_dims(self.raw_queue.dequeue(), 0), ':'), '')
        filename.set_shape([1,2])
        #seg_filename = self.seg_queue.dequeue()

        im_raw = tf.read_file(self.base_folder+filename[0][0])
        seg_raw = tf.read_file(self.base_folder+filename[0][1])
        image = tf.reshape(tf.cast(tf.image.decode_png(im_raw, channels=1, dtype=tf.uint16), tf.float32),
                           self.image_size, name='input_image')
        seg = tf.reshape(tf.cast(tf.image.decode_png(seg_raw, channels=1, dtype=tf.uint8), tf.float32), self.image_size,
                         name='input_seg')

        return image, seg, filename[0][0], filename[0][1]

    def get_batch(self, batch_size=1):

        self.batch_size = batch_size

        image, seg, file_name, seg_file_name = self._get_image()
        if self.random:
            image_batch, seg_batch, filename_batch, seg_filename_batch = tf.train.shuffle_batch([image, seg, file_name,
                                                                                                 seg_file_name],
                                                                            batch_size=self.batch_size,
                                                                            num_threads=self.num_threads,
                                                                            capacity=self.capacity,
                                                                            min_after_dequeue=self.min_after_dequeue)
        else:

            image_batch, seg_batch, filename_batch = tf.train.batch_join([(image, seg, file_name)],
                                                                         batch_size=self.batch_size,
                                                                         capacity=self.capacity,
                                                                         allow_smaller_final_batch=True)
        return image_batch, seg_batch, filename_batch
