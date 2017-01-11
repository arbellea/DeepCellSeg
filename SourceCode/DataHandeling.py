
import tensorflow as tf
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimage

__author__ = 'assafarbelle'


class CSVSegReader(object):

    def __init__(self, filenames, base_folder='.', image_size=(64,64,1), num_threads=4,
                 capacity=300, min_after_dequeue=100):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batchs of correspoding image inputs and segmentation inputs.
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
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = None
        self.base_folder = base_folder

    def _get_image(self):
        _, records = self.reader.read(self.input_queue)
        file_names = tf.decode_csv(records, [tf.constant([],tf.string), tf.constant([],tf.string)], field_delim=None, name=None)

        im_raw = tf.read_file(self.base_folder+file_names[0])
        seg_raw =tf.read_file(self.base_folder+file_names[1])
        image = tf.reshape(
                        tf.cast(tf.image.decode_png(
                                                    im_raw,
                                                    channels=1, dtype=tf.uint16),
                        tf.float32,), self.image_size, name='input_image')
        seg = tf.reshape(
                        tf.cast(tf.image.decode_png(
                                                    seg_raw,
                                                    channels=1, dtype=tf.uint8),
                        tf.float32,), self.image_size, name='input_seg')



        return image, seg

    def get_batch(self, batch_size=1):

        self.batch_size = batch_size

        image, seg = self._get_image()
        #seg_crop = tf.image.crop_to_bounding_box(seg,12,12,40,40)
        image_batch, seg_batch = tf.train.shuffle_batch([image, seg], batch_size=self.batch_size,
                                                        num_threads=self.num_threads,
                                                        capacity=self.capacity,

                                                        min_after_dequeue=self.min_after_dequeue)
        return image_batch, seg_batch
