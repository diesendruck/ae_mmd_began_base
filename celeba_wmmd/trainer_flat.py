from __future__ import print_function

import os
import pdb
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque
from PIL import Image
from scipy.misc import imsave

from models import *
from utils import save_image
# For MNIST classifier
from tensorflow.examples.tutorials.mnist import input_data
import tempfile


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



split = 'train'
data_path = '../../began/BEGAN-tensorflow/data/celeba'
model_dir = 'logs/test' 
dataset = 'celeba'
train_ratio = [20, 80]

optimizer = 'rmsprop'
batch_size = 64
use_mmd = True
lambda_mmd_setting = 100.
weighted = True

step = tf.Variable(0, name='step', trainable=False)
d_lr = tf.Variable(0.00008, name='d_lr', trainable=False)
g_lr = tf.Variable(0.00008, name='g_lr', trainable=False)
c_lr = tf.Variable(0.00008, name='c_lr', trainable=False)
d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, 0.00002), name='d_lr_update')
g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, 0.00002), name='g_lr_update')
c_lr_update = tf.assign(c_lr, tf.maximum(c_lr * 0.5, 0.00002), name='c_lr_update')

z_dim = 64
num_conv_filters = 128
scale_size = 64

use_gpu = True
data_format = 'NCHW'

repeat_num = int(np.log2(scale_size)) - 2
channel = 3

start_step = 0
log_step = 100
max_step = 200000
save_step = 1000
lr_update_step = 100000


# Set up all queues.
def img_and_lbl_queue_setup(filenames, labels):
    data_dir = os.path.dirname(filenames[0])
    labels_tensor = tf.constant(labels, dtype=tf.float32)
    
    filenames_tensor = tf.constant(filenames)
    fnq = tf.RandomShuffleQueue(capacity=200, min_after_dequeue=100, dtypes=tf.string)
    fnq_enq_op = fnq.enqueue_many(filenames_tensor)
    filename = fnq.dequeue()
    
    reader = tf.WholeFileReader()
    flnm, data = reader.read(fnq)
    image = tf.image.decode_jpeg(data, channels=3)
    image.set_shape([178, 218, 3])
    image = tf.image.crop_to_bounding_box(image, 50, 25, 128, 128)
    image = tf.to_float(image)
    
    image_index = [tf.cast(tf.string_to_number(tf.substr(flnm, len(data_dir), 6)) - 1, tf.int32)]
    n_labels = 2
    image_labels = tf.reshape(tf.gather(labels_tensor, indices=image_index, axis=0), [n_labels])
    imq = tf.RandomShuffleQueue(capacity=60, min_after_dequeue=30, dtypes=[tf.float32, tf.float32],
                                shapes=[[128, 128, 3], [n_labels]])
    imq_enq_op = imq.enqueue([image, image_labels])
    batch_size = 64 
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
    
    file_to_idx = {lines[i][:10]: i for i in xrange(2, len(lines))}
    labels = np.array([[int(it) for it in line.strip().split(
        ' ')[1:] if it != ''] for line in lines[2:]])[:, label_choices]
    
    return labels, label_names, file_to_idx

label_file = os.path.join(data_path, 'list_attr_celeba.txt')
labels, label_names, file_to_idx = load_labels(label_file)
feature_id = label_names.index('Eyeglasses')
labels = np.array(  # Binary one-hot labels.
    [[1.0, 0.0] if l[feature_id] == 1 else [0.0, 1.0] for l in labels]) 

tf.Graph().as_default()

# Create queues for each dataset.
user_dir = os.path.join(data_path, 'splits', 'user')
train_dir = os.path.join(data_path, 'splits', 'train')
test_dir = os.path.join(data_path, 'splits', 'test')

user_paths = glob('{}/*.jpg'.format(user_dir))
train_paths = glob('{}/*.jpg'.format(train_dir))
test_paths = glob('{}/*.jpg'.format(test_dir))

user_loader, user_label_loader, user_qr_f, user_qr_i = img_and_lbl_queue_setup(list(user_paths), labels)
x_loader, x_label_loader, train_qr_f, train_qr_i = img_and_lbl_queue_setup(list(train_paths), labels)
test_loader, test_label_loader, test_qr_f, test_qr_i = img_and_lbl_queue_setup(list(test_paths), labels)

###############################################################################
# Begin: build_model() 
###############################################################################

x_pix = x_loader
x = norm_img(x_pix)  # Converts pixels to [-1, 1].
x_label = x_label_loader
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
xe = x_enc 
ge = g_enc 
sigma_list = [1., 2., 4., 8., 16.]
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
lambda_mmd = tf.Variable(0., trainable=False, name='lambda_mmd')
lambda_ae = tf.Variable(0., trainable=False, name='lambda_ae')
lambda_fm = tf.Variable(0., trainable=False, name='lambda_fm')
ae_loss_real = tf.cond(
    weighted,
    lambda: (
        tf.reduce_sum(p1_weights_ae_normed * tf.reshape(
            tf.reduce_sum(tf.square(AE_x - x), [1, 2, 3]), [-1, 1]))),
    lambda: tf.reduce_mean(tf.square(AE_x - x)))
ae_loss_fake = tf.reduce_mean(tf.square(AE_g - g))
ae_loss = ae_loss_real
fm1 = tf.reduce_mean(ge, axis=0) - tf.reduce_mean(xe, axis=0)
fm2 = tf.nn.relu(fm1)
fm3 = tf.reduce_mean(fm2)
first_moment_loss = -1. * fm3
d_loss = (
    lambda_ae * ae_loss -
    lambda_mmd * mmd -
    lambda_fm * first_moment_loss)
g_loss = (
    lambda_mmd * mmd +
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
        [tf.clip_by_value(g, -0.01, 0.01) for g in enc_grads])
    d_grads = enc_grads_clipped + dec_grads
    d_vars = enc_vars + dec_vars
    d_optim = d_opt.apply_gradients(zip(d_grads, d_vars))
else:
    d_optim = d_opt.minimize(d_loss, var_list=ae_vars)
g_optim = g_opt.minimize(
    g_loss, var_list=g_var, global_step=step)

summary_op = tf.summary.merge([
    tf.summary.image("a_g", g_pix, max_outputs=10),
    tf.summary.image("b_AE_g", AE_g_pix, max_outputs=10),
    tf.summary.image("c_x", to_nhwc(x_pix, data_format), max_outputs=10),
    tf.summary.image("d_AE_x", AE_x_pix, max_outputs=10),
    tf.summary.scalar("loss/d_loss", d_loss),
    tf.summary.scalar("loss/ae_loss_real", ae_loss_real),
    tf.summary.scalar("loss/ae_loss_fake", ae_loss_fake),
    tf.summary.scalar("loss/mmd_u", mmd_unweighted),
    tf.summary.scalar("loss/mmd_w", mmd_weighted),
    tf.summary.scalar("misc/d_lr", d_lr),
    tf.summary.scalar("misc/g_lr", g_lr),
    tf.summary.scalar("prop/x_prop0", x_prop0),
    tf.summary.scalar("prop/g_prop0", g_prop0),
    tf.summary.scalar("prop/classifier_accuracy_user", classifier_accuracy),
])

###############################################################################
# End: BUILD MODEL
###############################################################################


init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(model_dir)
sv = tf.train.Supervisor(logdir=model_dir,
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


    # Begin: train() ##########################################################
    z_fixed = np.random.normal(0, 1, size=(batch_size, z_dim))

    # Save a sample.
    #ss = sess.run(x_label)
    pdb.set_trace()

    # Train generator.
    for step in trange(start_step, max_step):
        # TRAIN CLASSIFIER.
        sess.run(c_optim, {dropout_pr: 0.5})

        # Log classifier results.
        if step % log_step == 0:
            user_acc = sess.run(classifier_accuracy, {dropout_pr: 1.0})
            print('\nstep {},\nclassifer_acc on user {:.4f}'.format(step, user_acc))

        # TRAIN GAN.
        # Always run optim nodes, and sometimes, some logs.
        fetch_dict = {
            'd_optim': d_optim,
            'g_optim': g_optim,
        }
        if step % log_step == 0:
            fetch_dict.update({
                'summary': summary_op,
                'ae_loss_real': ae_loss_real,
                'ae_loss_fake': ae_loss_fake,
                'mmd': mmd,
                'keeping_probs': keeping_probs,
            })

        weighted = True 
        batch_train = sess.run(x_pix)
        batch_z = np.random.normal(0, 1, size=(batch_size, z_dim))
        # Pre-fetch data and simulations, and normalize for classification.
        x_for_pr0 = batch_train
        g_for_pr0 = sess.run(g,
            feed_dict={
                x_pix: batch_train,
                z: batch_z})
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
        result = sess.run(fetch_dict,
            feed_dict={
                x_pix: batch_train,
                z: batch_z,
                x_pr0: np.reshape(x_pred_pr0, [-1, 1]),
                g_pr0: np.reshape(g_pred_pr0, [-1, 1]),
                x_prop0: np.mean(np.round(x_pred_pr0)),
                g_prop0: np.mean(np.round(g_pred_pr0)),
                classifier_accuracy: user_acc,
                lambda_mmd: lambda_mmd_setting, 
                lambda_ae: 1.0,
                lambda_fm: 0.0,
                dropout_pr: 1.0,
                weighted: weighted})

        # Log and save as needed.
        if step % log_step == 0:
            summary_writer.add_summary(result['summary'], step)
            summary_writer.flush()
            ae_loss_real = result['ae_loss_real']
            ae_loss_fake = result['ae_loss_fake']
            mmd = result['mmd']
            print(('[{}/{}] LOSSES: ae_real/fake: {:.6f}, {:.6f} '
                'mmd: {:.6f}').format(
                    step, max_step, ae_loss_real, ae_loss_fake, mmd))
        if step % (save_step) == 0:
            z = np.random.normal(0, 1, size=(batch_size, z_dim))
            gen_rand = generate(
                z, model_dir, idx='rand'+str(step))
            gen_fixed = generate(
                z_fixed, model_dir, idx='fix'+str(step))
        if step % lr_update_step == lr_update_step - 1:
            sess.run([g_lr_update, d_lr_update])


def generate(inputs, root_path=None, path=None, idx=None, save=True):
    x = sess.run(g, {z: inputs})
    if path is None and save:
        path = os.path.join(root_path, 'G_{}.png'.format(idx))
        save_image(x, path)
        print("[*] Samples saved: {}".format(path))
    return x


def get_images_from_loader():
    x = x_loader.eval(session=sess)
    if data_format == 'NCHW':
        x = x.transpose([0, 2, 3, 1])
    return x
