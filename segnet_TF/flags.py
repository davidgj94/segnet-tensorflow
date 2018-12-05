import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecords_dir', '../tfrecords',
                            """ Folder tfrecords are stored """)
tf.app.flags.DEFINE_integer('num_images', "100",
                            """ Num images for evaluation  """)
tf.app.flags.DEFINE_integer('num_classes', "3",
                            """ Num classes """)