import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(predictions, labels):
    num_class = predictions.shape[3] #becomes 2 for aerial - correct
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist

def print_hist_summary(hist):

    acc = np.diag(hist).sum() / hist.sum()
    per_class_acc = np.diag(hist) / hist.sum(1)
    per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    print ">>>>>>>>>>>>>>>> Accuracy total {}".format(acc)
    print ">>>>>>>>>>>>>>>> Per-Class Accuaracy total {}".format(per_class_acc)
    print ">>>>>>>>>>>>>>>> Per-Class IU".format(per_class_iu)
    