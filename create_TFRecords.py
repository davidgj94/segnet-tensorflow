import os
import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
import os.path
import matplotlib.pyplot as plt

height, width = 544, 1024


# def write_records_file(img_label_paths, record_location):
#     """
#     Fill a TFRecords file with the images found in `dataset` and include their category.

#     Parameters
#     ----------
#     dataset : dict(list)
#       Dictionary with each key being a label for the list of image filenames of its value.
#     record_location : str
#       Location to store the TFRecord output.
#     """
#     writer = None
#     #sess = tf.InteractiveSession()

#     # Enumerating the dataset because the current index is used to breakup the files if they get over 100
#     # images to avoid a slowdown in writing.
#     current_index = 0
#     for img_path, label_path in img_label_paths:
#         if current_index % 100 == 0:
#             if writer:
#                 writer.close()

#             record_filename = "{record_location}-{current_index}.tfrecords".format(
#                 record_location=record_location,
#                 current_index=current_index)

#             writer = tf.python_io.TFRecordWriter(record_filename)
#         current_index += 1

#         image_file = tf.read_file(img_path)
#         label_file = tf.read_file(label_path)

#         image = tf.image.decode_png(image_file)
#         label = tf.image.decode_png(label_file)

#         with tf.Session() as sess:

#             image_bytes = sess.run(image).tobytes()
#             label_bytes = sess.run(label).tobytes()

#             example = tf.train.Example(features=tf.train.Features(feature={
#                 'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
#                 'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
#                 'img_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_path.encode('utf-8')]))
#             }))

#             writer.write(example.SerializeToString())

#     writer.close()


def _write_records(img_label_paths, save_filename):
    
    record_filename = save_filename + ".tfrecords"
    writer = tf.python_io.TFRecordWriter(record_filename)

    for img_path, label_path in img_label_paths:

        image_file = tf.read_file(img_path)
        label_file = tf.read_file(label_path)

        image = tf.image.decode_png(image_file)
        label = tf.image.decode_png(label_file)

        with tf.Session() as sess:

            image_bytes = sess.run(image).tobytes()
            label_bytes = sess.run(label).tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))

            writer.write(example.SerializeToString())

    writer.close()


def read_records(record_dir, height=544, width=1024, batch_size=3, capacity=10, num_threads=1, min_after_dequeue=6):

    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("{}/*.tfrecords".format(record_dir)))
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string)
        })

    record_image = tf.decode_raw(features['image'], tf.uint8)
    record_label = tf.decode_raw(features['label'], tf.uint8)
    
    image = tf.reshape(record_image, [height, width, 3])
    label = tf.reshape(record_label, [height, width])

    images, labels = tf.train.shuffle_batch([image, label], 
                                            batch_size=batch_size, 
                                            capacity=capacity, 
                                            num_threads=num_threads, 
                                            min_after_dequeue=min_after_dequeue)

    return images, labels

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--save_filename', type=str, required=True)
    return parser


if __name__ == "__main__":

    args = make_parser().parse_args()

    img_label_paths = []
    for glob in Path(args.label_dir).glob("*.png"):
        name = glob.parts[-1]
        img_path = os.path.join(args.img_dir,name)
        label_path = os.path.join(args.label_dir,name)
        img_label_paths.append((img_path, label_path))

    _write_records(img_label_paths, args.save_filename)
