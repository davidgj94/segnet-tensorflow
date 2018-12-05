import tensorflow as tf
import flags; FLAGS = tf.app.flags.FLAGS
import numpy as np
from create_TFRecords import read_records
from model import segnet_extended
from score import compute_hist, print_hist_summary
import pdb

def _eval():

    with tf.Graph().as_default():

        images, labels = read_records(FLAGS.tfrecords_dir, batch_size=1)
        logits = segnet_extended(images)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        with tf.Session() as sess:

            sess.run(init_op)

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            hist = np.zeros((FLAGS.num_classes, FLAGS.num_classes))
            """ Starting iterations to train the network """
            for _ in range(FLAGS.num_images):
                    
                _labels, _logits = sess.run([labels, logits])

                hist += compute_hist(_logits, _labels)
            
            coord.request_stop()
            coord.join(threads)

        print_hist_summary(hist)

def main(args):
    _eval()

if __name__ == "__main__":
    tf.app.run() # wrapper that handles flags parsing.
	
