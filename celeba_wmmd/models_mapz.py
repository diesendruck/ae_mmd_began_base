import pdb
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


def DecoderCNN(z, num_filters, channels_out, repeat_num, data_format, reuse):
    with tf.variable_scope('Decoder', reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, num_filters]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, num_filters, data_format)
        start_num_filters = num_filters * 1.
        
        for idx in range(repeat_num):
            num_filters = start_num_filters * (repeat_num - idx)
            x = slim.conv2d(x, num_filters, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format)
            x = slim.conv2d(x, num_filters, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        img = slim.conv2d(x, channels_out, 3, 1, activation_fn=None,
                          data_format=data_format)

    variables_dec = tf.contrib.framework.get_variables(vs)
    return img, variables_dec


def EncoderCNN(x, input_channel, z_num, repeat_num, num_filters,
        data_format, reuse):
    with tf.variable_scope('Encoder', reuse=reuse) as vs:
        x = slim.conv2d(x, num_filters, 3, 1, activation_fn=tf.nn.elu,
                        data_format=data_format)

        start_num_filters = num_filters * 1.
        for idx in range(repeat_num):
            channel_num = num_filters * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu,
                                data_format=data_format)

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

    variables_enc = tf.contrib.framework.get_variables(vs)
    return z, variables_enc


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


def MapToEncoding(z, reuse):
    """ Maps noise z to x_enc.
    Args:
      z: Tensor of noise inputs. Shape = (batch_size, z_dim).
    Returns:
      g_enc: Tensor of encodings, ideally will match distr of x_enc.
          Shape (batch_size, z_dim).
    """
    z_dim = z.get_shape().as_list()[1]
    with tf.variable_scope('generator_map', reuse=reuse) as vs:
        z = slim.fully_connected(z, 1024, activation_fn=tf.nn.elu, scope='fc1')
        z = slim.fully_connected(z, 1024, activation_fn=tf.nn.elu, scope='fc2')
        g_enc = slim.fully_connected(z, z_dim, activation_fn=tf.nn.elu, scope='fc3')

    variables_gen = tf.contrib.framework.get_variables(vs)
    return g_enc, variables_gen
