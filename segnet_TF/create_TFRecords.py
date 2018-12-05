import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
import os.path


def _write_records(img_label_paths, save_filename):
    
    record_filename = save_filename + ".tfrecords"
    writer = tf.python_io.TFRecordWriter(record_filename)

    for img_path, label_path in img_label_paths:

        image_file = tf.read_file(img_path)
        label_file = tf.read_file(label_path)

        image = tf.image.decode_png(image_file)
        image = tf.reverse(image, axis=[-1])
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


def read_records(record_dir, height=544, width=1024, batch_size=3, capacity=10, num_threads=1):

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
    image = tf.cast(image, tf.float32)
    label = tf.reshape(record_label, [height, width])

    images, labels = tf.train.batch([image, label], 
                                            batch_size=batch_size, 
                                            capacity=capacity, 
                                            num_threads=num_threads)

    return images, labels

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--save_filename', type=str, required=True)
    parser.add_argument('--height', type=int, default=544)
    parser.add_argument('--width', type=int, default=1024)
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
