from __future__ import print_function

import argparse
import json
import numpy as np
import os
import pdb
import scipy.misc
from scipy import misc
import StringIO
import time
from collections import deque
from glob import glob
from itertools import chain
from PIL import Image
from scipy.misc import imsave
from tqdm import trange

from models import *
from utils import save_image


def next(loader):
    return loader.next()[0].data.numpy()


def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image


def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image


def norm_img(image, data_format=None):
    ''' Converts pixel values to range [-1, 1].'''
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image


def pixel_to_01(image):
    ''' Converts pixel values to range [0, 1].'''
    return image / 255. 


def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)


def str2bool(v):
        return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
        default='../../began/BEGAN-tensorflow/data/celeba')
parser.add_argument('--data_format', type=str, default='NCHW')
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--dataset', type=str, default='celeba')
parser.add_argument('--channel', type=int, default=3, choices=[3])
parser.add_argument('--scale_size', type=int, default=64, choices=[64])
parser.add_argument('--optimizer', type=str, default='rmsprop')
parser.add_argument('--train_ratio', type=str, default='1090')
# Begin commonly tuned.
parser.add_argument('--feature_name', type=str, default='Eyeglasses')
parser.add_argument('--on_encodings', type=str2bool, default=True)
parser.add_argument('--z_dim', type=int,                  default=64)
parser.add_argument('--batch_size', type=int,             default=16)
parser.add_argument('--num_conv_filters', type=int,       default=50)
parser.add_argument('--lambda_ae_setting', type=float,    default=1.0)  # focus: ae
parser.add_argument('--lambda_ae_fake_setting', type=float, default=0.0)  # focus: ae_loss_fake
parser.add_argument('--lambda_fm_setting', type=float,    default=0.0)  # focus: fm 
parser.add_argument('--lambda_mmd_d_setting', type=float, default=1000.)  # focus: mmd 
parser.add_argument('--lambda_mmd_g_setting', type=float, default=1000.)  # focus: mmd 
parser.add_argument('--lr', type=float,                   default=8e-5)
parser.add_argument('--lambda_glr_factor', type=float,    default=1.0)
parser.add_argument('--lambda_clr_factor', type=float,    default=1.0)
parser.add_argument('--lr_update_step', type=int,         default=7500)
parser.add_argument('--lr_lower_boundary', type=float,    default=2e-6)
parser.add_argument('--n_user', type=int,                 default=2000)
# End commonly tuned.
parser.add_argument('--use_mmd', type=str2bool, default=True)
parser.add_argument('--weighted', type=str2bool, default=True)
parser.add_argument('--log_step', type=int, default=100)
parser.add_argument('--save_step', type=int, default=1000)
parser.add_argument('--max_step', type=int, default=500000)
parser.add_argument('--use_gpu', type=str2bool, default=True)
parser.add_argument('--num_log_samples', type=int, default=7)

config = parser.parse_args()
data_path = config.data_path
data_format = config.data_format
tag = config.tag
dataset = config.dataset
channel = config.channel
scale_size = config.scale_size
optimizer = config.optimizer
train_ratio = [int(config.train_ratio[:2]), int(config.train_ratio[2:])]
n_user = config.n_user
batch_size = config.batch_size
num_conv_filters = config.num_conv_filters
on_encodings = config.on_encodings
feature_name = config.feature_name
z_dim = config.z_dim
lr = config.lr
lambda_glr_factor = config.lambda_glr_factor
lambda_clr_factor = config.lambda_clr_factor
lr_update_step = config.lr_update_step
lr_lower_boundary = config.lr_lower_boundary
lambda_ae_setting = config.lambda_ae_setting
lambda_ae_fake_setting = config.lambda_ae_fake_setting
lambda_fm_setting = config.lambda_fm_setting
lambda_mmd_d_setting = config.lambda_mmd_d_setting
lambda_mmd_g_setting = config.lambda_mmd_g_setting
use_mmd = config.use_mmd
weighted = config.weighted
log_step = config.log_step
save_step = config.save_step
max_step = config.max_step
use_gpu = config.use_gpu
num_log_samples = config.num_log_samples
if tag == 'test':
    log_dir = os.path.join('logs', time.strftime("%Y%m%d-%H%M%S"))
else:
    log_dir = os.path.join('logs', config.tag)
if not os.path.exists(log_dir):
        os.makedirs(log_dir)
with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
    f.write(str(config))
    print(str(config))

step = tf.Variable(0, name='step', trainable=False)
d_lr = tf.Variable(lr, name='d_lr', trainable=False)
g_lr = tf.Variable(lambda_glr_factor * lr, name='g_lr', trainable=False)
c_lr = tf.Variable(lambda_clr_factor * lr, name='c_lr', trainable=False)
d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, lr_lower_boundary),
        name='d_lr_update')
g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, lr_lower_boundary),
        name='g_lr_update')
c_lr_update = tf.assign(c_lr, tf.maximum(c_lr * 0.5, lr_lower_boundary),
        name='c_lr_update')
repeat_num = int(np.log2(scale_size)) - 2


# Set up all queues.
def img_and_lbl_queue_setup(filenames, labels):
    data_dir = os.path.dirname(filenames[0])
    labels_tensor = tf.constant(labels, dtype=tf.float32)
    
    filenames_tensor = tf.constant(filenames)
    fnq = tf.RandomShuffleQueue(
        capacity=6000, min_after_dequeue=5000, dtypes=tf.string)
    fnq_enq_op = fnq.enqueue_many(filenames_tensor)
    filename = fnq.dequeue()
    
    reader = tf.WholeFileReader()
    flnm, data = reader.read(fnq)
    image = tf.image.decode_jpeg(data, channels=3)
    image.set_shape([178, 218, 3])
    image = tf.image.crop_to_bounding_box(image, 50, 25, 128, 128)
    image = tf.to_float(image)
    
    # Below, added 1 to len(data_dir) to account for extra "/" at end.
    image_index = [tf.cast(tf.string_to_number(tf.substr(
        flnm, len(data_dir)+1, 6)) - 1, tf.int32)]
    n_labels = 2
    image_labels = tf.reshape(tf.gather(
        labels_tensor, indices=image_index, axis=0), [n_labels])
    imq = tf.RandomShuffleQueue(
        capacity=6000, min_after_dequeue=5000, dtypes=[tf.float32, tf.float32],
        shapes=[[128, 128, 3], [n_labels]])
    imq_enq_op = imq.enqueue([image, image_labels])
    imgs, img_lbls = imq.dequeue_many(batch_size)
    imgs = tf.image.resize_nearest_neighbor(imgs, size=[64, 64])
    #imgs = tf.subtract(1., tf.divide(imgs, 255.5))
    imgs = tf.transpose(imgs, [0, 3, 1, 2])
    qr_f = tf.train.QueueRunner(fnq, [fnq_enq_op] * 3)
    qr_i = tf.train.QueueRunner(imq, [imq_enq_op] * 3)

    return imgs, img_lbls, qr_f, qr_i

def load_labels(label_file, label_choices=None):
    lines = open(label_file, 'r').readlines()
    label_names = lines[1].strip().split(' ')
    if label_choices is None:
        label_choices = range(len(label_names))
    n_labels = len(label_choices)
    label_names = [label_names[choice] for choice in label_choices]
    
    # Here, idx will refer to index of the labels array. Therefore, the
    # index value gets a minus two, since lines has two extra rows, but
    # the labels array will not.
    file_to_idx = {lines[i][:10]: i - 2 for i in xrange(2, len(lines))}
    labels = np.array([[int(it) for it in line.strip().split(
        ' ')[1:] if it != ''] for line in lines[2:]])[:, label_choices]
    
    return labels, label_names, file_to_idx

label_file = os.path.join(data_path, 'list_attr_celeba.txt')
labels_, label_names, file_to_idx = load_labels(label_file)
idx_to_file = {v: i for i, v in file_to_idx.iteritems()}
feature_id = label_names.index(feature_name)
labels = np.array(  # Binary one-hot labels.
    [[1.0, 0.0] if lab[feature_id] == 1 else [0.0, 1.0] for lab in labels_]) 

tf.Graph().as_default()

# Begin: Set up queues. #######################################################
# Define directories for each split.
user_dir = os.path.join(data_path, 'splits', 'user')
train_dir = os.path.join(data_path, 'splits', 'train')
test_dir = os.path.join(data_path, 'splits', 'test')

# Get all paths for each split.
user_paths = glob('{}/*.jpg'.format(user_dir))
train_paths = glob('{}/*.jpg'.format(train_dir))
test_paths = glob('{}/*.jpg'.format(test_dir))
#test_paths = [pa for pa in test_paths if ('202600' not in pa and '202598' not in pa)]

# Get global indices for each split.
user_paths_and_global_indices = zip(
    user_paths, [file_to_idx[pa[-10:]] for pa in user_paths])
train_paths_and_global_indices = zip(
    train_paths, [file_to_idx[pa[-10:]] for pa in train_paths])
test_paths_and_global_indices = zip(
    test_paths, [file_to_idx[pa[-10:]] for pa in test_paths])

# Assemble 50-50 user/classifier paths.
user_paths_0 = [pa for pa, gi in user_paths_and_global_indices if \
    labels[gi][0] == 1]  # label [1.0, 0.0]
user_paths_1 = [pa for pa, gi in user_paths_and_global_indices if \
    labels[gi][0] != 1]
assert n_user / 2 < len(user_paths_0) and n_user / 2 < len(user_paths_1), \
    'Asking for more than we have in USER set.'
user_paths = np.concatenate((
    np.random.choice(user_paths_0, n_user / 2),
    np.random.choice(user_paths_1, n_user / 2)))
np.random.shuffle(user_paths)
print('\n\nBuilt USER set. Proportion class0 = {}, n = {}'.format(
    0.5, len(user_paths)))

# Assemble training paths for given train ratio.
train_paths_0 = [pa for pa, gi in train_paths_and_global_indices if \
    labels[gi][0] == 1]  # label [1.0, 0.0]
train_paths_1 = [pa for pa, gi in train_paths_and_global_indices if \
    labels[gi][0] != 1]
class1_multiplier = train_ratio[1] / float(train_ratio[0])
class0_multiplier = train_ratio[0] / float(train_ratio[1])
train_num_paths_1_desired = int(class1_multiplier * len(train_paths_0))  # Used if c1 big enough.
train_num_paths_0_desired = int(class0_multiplier * len(train_paths_1))  # Used if c1 not big enough.
if train_num_paths_1_desired < len(train_paths_1):
    # Subset class1.
    train_paths_1 = np.random.choice(train_paths_1, train_num_paths_1_desired)
    train_paths = np.concatenate((train_paths_0, train_paths_1))
else:
    # Subset class0.
    train_paths_0 = np.random.choice(train_paths_0, train_num_paths_0_desired)
    train_paths = np.concatenate((train_paths_0, train_paths_1))
np.random.shuffle(train_paths)
print('Built TRAIN set. Proportion class0 = {}, n = {}'.format(
    len(train_paths_0) / float(len(train_paths)), len(train_paths)))

# Assemble test paths with natural ratio.
test_paths_0 = [pa for pa, gi in test_paths_and_global_indices if \
    labels[gi][0] == 1]  # label [1.0, 0.0]
test_paths_1 = [pa for pa, gi in test_paths_and_global_indices if \
    labels[gi][0] != 1]
test_paths = np.concatenate((test_paths_0, test_paths_1))
np.random.shuffle(test_paths)
print('Built TEST set. Proportion class0 = {}, n = {}\n\n'.format(
    len(test_paths_0) / float(len(test_paths)), len(test_paths)))

# Set up queues for each split.
user_loader, user_label_loader, user_qr_f, user_qr_i = \
    img_and_lbl_queue_setup(list(user_paths), labels)
x_loader, x_label_loader, train_qr_f, train_qr_i =  \
    img_and_lbl_queue_setup(list(train_paths), labels)
test_loader, test_label_loader, test_qr_f, test_qr_i = \
    img_and_lbl_queue_setup(list(test_paths), labels)
# End: Set up queues. #########################################################


###############################################################################
# Begin: build_model() 
###############################################################################

x_pix = x_loader
x = norm_img(x_pix)  # Converts pixels to [-1, 1].
z = tf.random_normal([tf.shape(x)[0], z_dim])
weighted = tf.placeholder(tf.bool, name='weighted')

# Set up generator and autoencoder functions.
g, g_var = GeneratorCNN(
    z, num_conv_filters, channel,
    repeat_num, data_format, reuse=False)
d_out, d_enc, d_var_enc, d_var_dec = AutoencoderCNN(
    tf.concat([x, g], 0), channel, z_dim, repeat_num,
    num_conv_filters, data_format, reuse=False)
AE_x, AE_g = tf.split(d_out, 2)
x_enc, g_enc = tf.split(d_enc, 2)
g_pix = denorm_img(g, data_format)
AE_g_pix = denorm_img(AE_g, data_format)
AE_x_pix = denorm_img(AE_x, data_format)

# Begin: MMD ####################################################################
if on_encodings:
    # Kernel on encodings.
    xe = x_enc 
    ge = g_enc 
    sigma_list = [1., 2., 4., 8., 16.]
else:
    # Kernel on full imgs.
    xe = tf.reshape(x, [tf.shape(x)[0], -1]) 
    ge = tf.reshape(g, [tf.shape(g)[0], -1]) 
    sigma_list = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
data_num = tf.shape(xe)[0]
gen_num = tf.shape(ge)[0]
v = tf.concat([xe, ge], 0)
VVT = tf.matmul(v, tf.transpose(v))
sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
sqs_tiled_horiz = tf.tile(sqs, [1, tf.shape(sqs)[0]])
exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
K = 0.0
for sigma in sigma_list:
    gamma = 1.0 / (2 * sigma**2)
    K += tf.exp(-gamma * exp_object)
K_xx = K[:data_num, data_num:]
K_yy = K[data_num:, data_num:]
K_xy = K[:data_num, data_num:]
K_xx_upper = (tf.matrix_band_part(K_xx, 0, -1) -
              tf.matrix_band_part(K_xx, 0, 0))
K_yy_upper = (tf.matrix_band_part(K_yy, 0, -1) -
              tf.matrix_band_part(K_yy, 0, 0))
num_combos_xx = tf.to_float(data_num * (data_num - 1) / 2)
num_combos_yy = tf.to_float(gen_num * (gen_num - 1) / 2)


#### Begin: build_classifier() #################################################
# Build classifier, and get probabilities of keeping.
c_images_pix = user_loader
c_labels = user_label_loader
c_images = norm_img(c_images_pix)

# Compute the data probabilities used in WMMD.
# NOTE: Whatever is chosen here is used downstream in the model.
#       Other pred/prob/entropy/optims are just for reporting.
dropout_pr = tf.placeholder(tf.float32)
# Convert to [-1, 1] for autoencoder.
_, c_enc, _, _ = AutoencoderCNN(
    c_images, channel, z_dim, repeat_num,
    num_conv_filters, data_format, reuse=True)
label_pred, label_pred_pr, c_vars = classifier_NN_enc(
    c_enc, dropout_pr, reuse=False)
label_pred_pr0 = label_pred_pr[:, 0]

# Define classifier losses.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=c_labels, logits=label_pred)
cross_entropy = tf.reduce_mean(cross_entropy)

# Define optimization procedure.
if 1:
    c_optim = tf.train.RMSPropOptimizer(c_lr).minimize(
        cross_entropy, var_list=c_vars)
else:
    # Same as  above, but with gradient clipping.
    c_opt = tf.train.AdamOptimizer(c_lr)
    c_gvs = c_opt.compute_gradients(
        cross_entropy, var_list=c_vars)
    c_capped_gvs = (
        [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in c_gvs])
    c_optim = c_opt.apply_gradients(c_capped_gvs) 

# Compute accuracy.
correct_prediction = tf.equal(
    tf.argmax(label_pred, 1), tf.argmax(c_labels, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
classifier_accuracy = tf.reduce_mean(correct_prediction)

# Compute accuracy on test_loader, test_label_loader inputs.
test_images_pix = test_loader
test_labels = test_label_loader
test_images = norm_img(test_images_pix)
_, test_enc, _, _ = AutoencoderCNN(
    test_images, channel, z_dim, repeat_num,
    num_conv_filters, data_format, reuse=True)
test_label_pred, _, _ = classifier_NN_enc(
    test_enc, 1.0, reuse=True)
cross_entropy_test = tf.nn.softmax_cross_entropy_with_logits(
    labels=test_labels, logits=test_label_pred)
cross_entropy_test = tf.reduce_mean(cross_entropy)
correct_prediction_test = tf.equal(
    tf.argmax(test_label_pred, 1), tf.argmax(test_labels, 1))
correct_prediction_test = tf.cast(correct_prediction_test, tf.float32)
classifier_accuracy_test = tf.reduce_mean(correct_prediction_test)

#### End CLASSIFIER ############################################################


# Variables used for read-only probabilities.
x_pr0 = tf.Variable(tf.ones([batch_size, 1]),
    trainable=False, name='x_pr0')
g_pr0 = tf.Variable(tf.ones([batch_size, 1]),
    trainable=False, name='g_pr0')
x_prop0 = tf.Variable(0., trainable=False, name='x_prop0')
g_prop0 = tf.Variable(0., trainable=False, name='g_prop0')
x_normed_for_prediction = (tf.reshape(x,
    [batch_size, -1]) + 1.)/ 2.  # Maps [-1, 1] to [0, 1].
g_normed_for_prediction = (tf.reshape(g,
    [batch_size, -1]) + 1.)/ 2.  # Maps [-1, 1] to [0, 1].

thin_factor = 1. - float(min(train_ratio)) / max(train_ratio)
keeping_probs = 1. - thin_factor * tf.reshape(
    x_pr0, [-1, 1])

keeping_probs_tiled = tf.tile(keeping_probs, [1, gen_num])

# Autoencoder weights.
p1_weights_ae = 1. / keeping_probs
p1_weights_ae_normed = (
    p1_weights_ae / tf.reduce_sum(p1_weights_ae))
g_weights_ae = 1. / (
    1. - thin_factor * tf.reshape(g_pr0, [-1, 1]))
g_weights_ae_normed = (
    g_weights_ae / tf.reduce_sum(g_weights_ae))

# MMD weights.
p1_weights_xy = 1. / keeping_probs_tiled
p1_weights_xy_normed = (
    p1_weights_xy / tf.reduce_sum(p1_weights_xy))
p1p2_weights_xx = (
    p1_weights_xy * tf.transpose(p1_weights_xy))
p1p2_weights_xx_upper = (
    tf.matrix_band_part(p1p2_weights_xx, 0, -1) -
    tf.matrix_band_part(p1p2_weights_xx, 0, 0))
p1p2_weights_xx_upper_normed = (
    p1p2_weights_xx_upper /
    tf.reduce_sum(p1p2_weights_xx_upper))
Kw_xx_upper = K_xx_upper * p1p2_weights_xx_upper_normed
Kw_xy = K_xy * p1_weights_xy_normed
num_combos_xy = tf.to_float(data_num * gen_num)

# Compute and choose between MMD values.
mmd = tf.cond(
    weighted,
    lambda: (
        tf.reduce_sum(Kw_xx_upper) +
        tf.reduce_sum(K_yy_upper) / num_combos_yy -
        2 * tf.reduce_sum(Kw_xy)),
    lambda: (
        tf.reduce_sum(K_xx_upper) / num_combos_xx +
        tf.reduce_sum(K_yy_upper) / num_combos_yy -
        2 * tf.reduce_sum(K_xy) / num_combos_xy))
mmd_weighted = (
    tf.reduce_sum(Kw_xx_upper) +
    tf.reduce_sum(K_yy_upper) / num_combos_yy -
    2 * tf.reduce_sum(Kw_xy))
mmd_unweighted = (
    tf.reduce_sum(K_xx_upper) / num_combos_xx +
    tf.reduce_sum(K_yy_upper) / num_combos_yy -
    2 * tf.reduce_sum(K_xy) / num_combos_xy)
# End MMD #####################################################################


# Define losses.
lambda_mmd_d = tf.Variable(0., trainable=False, name='lambda_mmd_d')
lambda_mmd_g = tf.Variable(0., trainable=False, name='lambda_mmd_g')
lambda_ae = tf.Variable(0., trainable=False, name='lambda_ae')
lambda_ae_fake = tf.Variable(0., trainable=False, name='lambda_ae_fake')
lambda_fm = tf.Variable(0., trainable=False, name='lambda_fm')
fm1 = tf.reduce_mean(ge, axis=0) - tf.reduce_mean(xe, axis=0)
fm2 = tf.nn.relu(fm1)
fm3 = tf.reduce_mean(fm2)
first_moment_loss = -1. * fm3
ae_loss_real = tf.cond(
    weighted,
    lambda: (
        tf.reduce_sum(p1_weights_ae_normed * tf.reshape(
            tf.reduce_sum(tf.square(AE_x - x), [1, 2, 3]), [-1, 1]))),
    lambda: tf.reduce_mean(tf.square(AE_x - x)))
ae_loss_real_unweighted = tf.reduce_mean(tf.square(AE_x - x))
ae_loss_fake_unweighted = tf.reduce_mean(tf.square(AE_g - g))
ae_loss = ae_loss_real + lambda_ae_fake * ae_loss_fake_unweighted
controlled_mmd_d = tf.minimum(ae_loss, lambda_mmd_d * mmd)
d_loss = (
    lambda_ae * ae_loss -
    controlled_mmd_d -
    lambda_fm * first_moment_loss)
g_loss = (
    lambda_mmd_g * mmd +
    lambda_fm * first_moment_loss)

# Optimizer nodes.
if optimizer == 'adam':
    d_opt = tf.train.AdamOptimizer(d_lr)
    g_opt = tf.train.AdamOptimizer(g_lr)
elif optimizer == 'rmsprop':
    d_opt = tf.train.RMSPropOptimizer(d_lr)
    g_opt = tf.train.RMSPropOptimizer(g_lr)
elif optimizer == 'sgd':
    d_opt = tf.train.GradientDescentOptimizer(d_lr)
    g_opt = tf.train.GradientDescentOptimizer(g_lr)

# Set up optim nodes. Clip encoder only! 
if 1:
    enc_grads, enc_vars = zip(*d_opt.compute_gradients(
        d_loss, var_list=d_var_enc))
    dec_grads, dec_vars = zip(*d_opt.compute_gradients(
        d_loss, var_list=d_var_dec))
    enc_grads_clipped = tuple(
        [tf.clip_by_value(grad, -0.01, 0.01) for grad in enc_grads])
    d_grads = enc_grads_clipped + dec_grads
    d_vars = enc_vars + dec_vars
    d_optim = d_opt.apply_gradients(zip(d_grads, d_vars))
else:
    d_optim = d_opt.minimize(d_loss, var_list=ae_vars)
g_optim = g_opt.minimize(
    g_loss, var_list=g_var, global_step=step)

summary_op = tf.summary.merge([
    tf.summary.image("a_g", g_pix, max_outputs=num_log_samples),
    tf.summary.image("b_AE_g", AE_g_pix, max_outputs=num_log_samples),
    tf.summary.image("c_x", to_nhwc(x_pix, data_format), max_outputs=num_log_samples),
    tf.summary.image("d_AE_x", AE_x_pix, max_outputs=num_log_samples),
    tf.summary.scalar("loss/d_loss", d_loss),
    tf.summary.scalar("loss/first_moment_loss", first_moment_loss),
    tf.summary.scalar("loss/ae_loss_real_w", ae_loss_real),
    tf.summary.scalar("loss/ae_loss_real_u", ae_loss_real_unweighted),
    tf.summary.scalar("loss/ae_loss_fake_u", ae_loss_fake_unweighted),
    tf.summary.scalar("loss/mmd_u", mmd_unweighted),
    tf.summary.scalar("loss/mmd_w", mmd_weighted),
    tf.summary.scalar("misc/d_lr", d_lr),
    tf.summary.scalar("misc/g_lr", g_lr),
    tf.summary.scalar("prop/x_prop0", x_prop0),
    tf.summary.scalar("prop/g_prop0", g_prop0),
    tf.summary.scalar("prop/classifier_accuracy_user", classifier_accuracy),
    tf.summary.scalar("prop/classifier_accuracy_test", classifier_accuracy_test),
])

###############################################################################
# End: BUILD MODEL
###############################################################################


init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(log_dir)
sv = tf.train.Supervisor(logdir=log_dir,
                        is_chief=True,
                        saver=saver,
                        summary_op=None,
                        summary_writer=summary_writer,
                        save_model_secs=300,
                        global_step=step,
                        ready_for_local_init_op=None)

gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(allow_soft_placement=True,
                            gpu_options=gpu_options)

#sess = sv.prepare_or_wait_for_session(config=sess_config)
with sv.managed_session() as sess:
    sv.start_standard_services(sess)
    coord = sv.coord
    user_qr_f.create_threads(sess, coord=coord, start=True)
    user_qr_i.create_threads(sess, coord=coord, start=True)
    train_qr_f.create_threads(sess, coord=coord, start=True)
    train_qr_i.create_threads(sess, coord=coord, start=True)
    test_qr_f.create_threads(sess, coord=coord, start=True)
    test_qr_i.create_threads(sess, coord=coord, start=True)
    sess.run(init_op)

    def generate(inputs, root_path=None, path=None, idx=None, save=True):
        g_pix_ = sess.run(g_pix, {z: inputs})
        if path is None and save:
            path = os.path.join(root_path, 'G_{}.png'.format(idx))
            save_image(g_pix_, path)
            print("[*] Samples saved: {}".format(path))
        return g_pix_

    # Begin: train() ##########################################################
    z_fixed = np.random.normal(0, 1, size=(batch_size, z_dim))

    # Train generator.
    for step in trange(0, max_step):
        # TRAIN CLASSIFIER.
        sess.run(c_optim, {dropout_pr: 0.5})

        # Log classifier results.
        if step % log_step == 0:
            user_acc, test_acc = sess.run([
                classifier_accuracy, classifier_accuracy_test],
            feed_dict={
                dropout_pr: 1.0})
            print('classifier_acc user/test {:.3f}, {:.3f}'.format(
                step, user_acc, test_acc))

        # TRAIN GAN.
        # Always run optim nodes, and sometimes, some logs.
        fetch_dict = {
            'd_optim': d_optim,
            'g_optim': g_optim,
            'ae_loss_real_unweighted': ae_loss_real_unweighted,
            'ae_loss_fake_unweighted': ae_loss_fake_unweighted,
        }
        if step % log_step == 0:
            fetch_dict.update({
                'summary': summary_op,
                'ae_loss_real': ae_loss_real,
                'mmd': mmd,
                'controlled_mmd_d': controlled_mmd_d,
                'first_moment_loss': first_moment_loss,
                'keeping_probs': keeping_probs,
            })

        weighted_ = True 
        batch_train = sess.run(x_pix)
        batch_z = np.random.normal(0, 1, size=(batch_size, z_dim))
        # Pre-fetch data and simulations, and normalize for classification.
        x_for_pr0 = batch_train
        g_for_pr0 = sess.run(g,
            feed_dict={
                x_pix: batch_train,
                z: batch_z,
                weighted: weighted_})
        # Get probs using encodings.
        x_pred_pr0 = sess.run(label_pred_pr0,
            feed_dict={
                c_images_pix: x_for_pr0,
                dropout_pr: 1.0})
        g_pred_pr0 = sess.run(label_pred_pr0,
            feed_dict={
                c_images_pix: g_for_pr0,
                dropout_pr: 1.0})

        # Run full training step on pre-fetched data and simulations.
        # If ae_fake is much worse than ae_real, drop adversarial component of d_loss.
        #if step > 0 and ae_loss_fake_unweighted_ > 2 * ae_loss_real_unweighted_:
        #    lambda_mmd_d_setting_ = step 
        #    lambda_mmd_g_setting_ = lambda_mmd_g_setting
        #else:
        #    lambda_mmd_d_setting_ = lambda_mmd_d_setting
        #    lambda_mmd_g_setting_ = lambda_mmd_g_setting
        lambda_mmd_d_setting_ = step 
        lambda_mmd_g_setting_ = lambda_mmd_g_setting
        result = sess.run(fetch_dict,
            feed_dict={
                x_pix: batch_train,
                z: batch_z,
                x_pr0: np.reshape(x_pred_pr0, [-1, 1]),
                g_pr0: np.reshape(g_pred_pr0, [-1, 1]),
                x_prop0: np.mean(np.round(x_pred_pr0)),
                g_prop0: np.mean(np.round(g_pred_pr0)),
                classifier_accuracy: user_acc,
                lambda_ae: lambda_ae_setting,
                lambda_ae_fake: lambda_ae_fake_setting,
                lambda_mmd_d: lambda_mmd_d_setting_, 
                lambda_mmd_g: lambda_mmd_g_setting_, 
                lambda_fm: lambda_fm_setting,
                dropout_pr: 1.0,
                weighted: weighted_})

        ae_loss_real_unweighted_ = result['ae_loss_real_unweighted']
        ae_loss_fake_unweighted_ = result['ae_loss_fake_unweighted']

        # Log and save as needed.
        if step % log_step == 0:
            summary_writer.add_summary(result['summary'], step)
            summary_writer.flush()
            ae_loss_real_ = result['ae_loss_real']
            mmd_ = result['mmd']
            controlled_mmd_d_ = result['controlled_mmd_d']
            first_moment_loss_ = result['first_moment_loss']
            print(('\n\n[{}/{}] RAW losses: ae_real, real_u, fake_u: {:.3f}, {:.3f}, {:.3f} | '
                'mmd: {:.3f} | fm: {:.3f}').format(
                    step, max_step, ae_loss_real_, ae_loss_real_unweighted_,
                    ae_loss_fake_unweighted_, mmd_, first_moment_loss_))
            print(('D_losses: ae_real: {:.3f} | mmd_d: {:.3f}').format(
                    ae_loss_real_, controlled_mmd_d_))
            print(('G_losses: mmd_g: {:.3f}').format(mmd_ * lambda_mmd_g_setting_))
        if step % (save_step) == 0:
            print(str(config))
            z_rand= np.random.normal(0, 1, size=(batch_size, z_dim))
            gen_rand = generate(
                z_rand, log_dir, idx='rand'+str(step))
            gen_fixed = generate(
                z_fixed, log_dir, idx='fix'+str(step))
        if step % lr_update_step == lr_update_step - 1:
            sess.run([g_lr_update, d_lr_update])


def get_images_from_loader():
    x = x_loader.eval(session=sess)
    if data_format == 'NCHW':
        x = x.transpose([0, 2, 3, 1])
    return x
