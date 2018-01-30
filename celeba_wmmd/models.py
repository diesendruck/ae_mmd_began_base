import pdb
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


deep = 1
def GeneratorCNN(z, num_filters, channels_out, repeat_num, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, num_filters]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, num_filters, data_format)
        start_num_filters = num_filters * 1.
        
        for idx in range(repeat_num):
            num_filters = start_num_filters * (repeat_num - idx)
            x = slim.conv2d(x, num_filters, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format)
            if deep:
                x = slim.conv2d(x, num_filters, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, channels_out, 3, 1, activation_fn=None,
                          data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def AutoencoderCNN(x, input_channel, z_num, repeat_num, num_filters,
        data_format, reuse):
    with tf.variable_scope("ae_enc", reuse=reuse) as vs_enc:
        # Encoder
        x = slim.conv2d(x, num_filters, 3, 1, activation_fn=tf.nn.elu,
                        data_format=data_format)

        start_num_filters = num_filters * 1.
        for idx in range(repeat_num):
            channel_num = num_filters * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format)
            if deep:
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu,
                                data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2],
                #                                 padding='VALID')

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

    with tf.variable_scope("ae_dec", reuse=reuse) as vs_dec:
        # Decoder
        num_output = int(np.prod([8, 8, num_filters]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = reshape(x, 8, 8, num_filters, data_format)
        
        for idx in range(repeat_num):
            num_filters = start_num_filters * (repeat_num - idx)
            x = slim.conv2d(x, num_filters, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format)
            if deep:
                x = slim.conv2d(x, num_filters, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)


        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None,
                          data_format=data_format)

    variables_enc = tf.contrib.framework.get_variables(vs_enc)
    variables_dec = tf.contrib.framework.get_variables(vs_dec)
    return out, z, variables_enc, variables_dec


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)

###############################################################################
# BEGIN section from Tensorflow website. For MNIST classification.
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/
#   tutorials/mnist/mnist_deep.py
def classifier_CNN(x, dropout_pr, reuse):
    """classifier_CNN builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
      dropout_pr: tf.float32 indicating the keeping rate for dropout.
    Returns:
      y_logits: Tensor of shape (N_examples, 10), with values equal to the logits
        of classifying the digit into one of 10 classes (the digits 0-9). 
      y_probs: Tensor of shape (N_examples, 10), with values
        equal to the probabilities of classifying the digit into one of 10 classes
        (the digits 0-9). 
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.variable_scope('classifier_CNN', reuse=reuse) as vs:
        with tf.name_scope('reshape'):
            x_image = tf.transpose(
                tf.reshape(x, [-1, 3, 28, 28]), [0, 2, 3, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 3, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, dropout_pr)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([1024, 2])
            b_fc2 = bias_variable([2])

        y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_probs = tf.nn.softmax(y_logits)

    variables = tf.contrib.framework.get_variables(vs)
    return y_logits, y_probs, variables


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# END section from Tensorflow website.
###############################################################################


def classifier_NN_enc(x, dropout_pr, reuse):
    """classifier_NN_enc builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, z_dim), where z_dim is the
      number of encoding dimension.
      dropout_pr: tf.float32 indicating the keeping rate for dropout.
    Returns:
      y_logits: Tensor of shape (N_examples, 2), with values equal to the logits
        of classifying the digit into zero/nonzero.
      y_probs: Tensor of shape (N_examples, 2), with values
        equal to the probabilities of classifying the digit into zero/nonzero.
    """
    z_dim = x.get_shape().as_list()[1]
    with tf.variable_scope('mnist_classifier', reuse=reuse) as vs:
        x = slim.fully_connected(x, 1024, activation_fn=tf.nn.elu, scope='fc1')
        x = slim.dropout(x, dropout_pr, scope='drop1')
        x = slim.fully_connected(x, 1024, activation_fn=tf.nn.elu, scope='fc2')
        x = slim.dropout(x, dropout_pr, scope='drop2')
        #x = slim.fully_connected(x, 512, activation_fn=tf.nn.elu, scope='fc3')
        #x = slim.dropout(x, dropout_pr, scope='drop3')
        x = slim.fully_connected(x, 32, activation_fn=tf.nn.elu, scope='fc4')
        #x = slim.dropout(x, dropout_pr, scope='drop4')
        y_logits = slim.fully_connected(x, 2, activation_fn=None, scope='fc5')
        y_probs = tf.nn.softmax(y_logits)

    variables = tf.contrib.framework.get_variables(vs)
    return y_logits, y_probs, variables
