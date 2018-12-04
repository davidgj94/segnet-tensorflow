import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
import pdb

def unpool_with_argmax(pool, ind, name = None, ksize=[1, 2, 2, 1]):

    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret


def conv_classifier(caffe_weights, input_layer, name=None):
    
    with tf.variable_scope(name) as scope: 
        
        with tf.device('/cpu:0'):

            kernel_np, biases_np = get_conv_weights(caffe_weights, name)
            kernel = tf.get_variable("weights", initializer=kernel_np)
            biases = tf.get_variable("biases", initializer=biases_np)
        
        conv = tf.nn.conv2d(input_layer, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier



def conv_layer_with_bn_caffe(caffe_weights, inputT, shape, is_training, activation=True, name=None):

    with tf.variable_scope(name) as scope:

        with tf.device('/cpu:0'):

            kernel_np, biases_np = get_conv_weights(caffe_weights, name)
            kernel = tf.get_variable("weights", initializer=kernel_np)
            biases = tf.get_variable("biases", initializer=biases_np)

            beta_np, gamma_np, mean_np, variance_np = get_batch_parameters(caffe_weights, name + '_bn')
            beta = tf.get_variable("beta", initializer=beta_np)
            gamma = tf.get_variable("gamma", initializer=gamma_np)
            mean = tf.get_variable("mean", initializer=mean_np)
            variance = tf.get_variable("variance", initializer=variance_np)

        #kernel = tf.get_variable(scope.name, shape, initializer=initializer)
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)

        if activation is True: #only use relu during encoder
            conv_out = tf.nn.relu(tf.layers.batch_normalization(bias, training=False, 
                beta_initializer=beta, 
                gamma_initializer=gamma, 
                moving_mean_initializer=mean,
                moving_variance_initializer=variance))
        else:
            conv_out = tf.layers.batch_normalization(bias, training=False, 
                beta_initializer=beta, 
                gamma_initializer=gamma, 
                moving_mean_initializer=mean,
                moving_variance_initializer=variance)
            
    return conv_out


def get_batch_parameters(caffe_weights, name):
    return beta_np, gamma_np, mean_np, variance_np


def get_conv_weights(caffe_weights, name):

    _kernel_np = caffe_weights[name][0] 
    num_filters, num_channels, _, _ = _kernel_np.shape
    kernel_np = np.zeros((3, 3, num_channels, num_filters), dtype=_kernel_np.dtype)
    for i in range(num_filters):
        for c in range(num_channels):
            kernel_np[:,:,c,i] = _kernel_np[i,c,:,:]

    print name
    print ">>>>>>>>>>>>>>>> {}".format(kernel_np.shape)
    print "++++++++++++++++ {}".format(_kernel_np.shape)
    print 

    biases_np = caffe_weights[name][1]

    return kernel_np, biases_np
