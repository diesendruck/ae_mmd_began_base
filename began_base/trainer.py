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

from models import *
from utils import save_image
from data_loader import get_loader
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


def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


class Trainer(object):
    def __init__(self, config, data_loader, data_loader_target):
        self.config = config
        self.split = config.split
        self.data_path = config.data_path
        self.data_loader = data_loader
        self.data_loader_target = data_loader_target
        self.dataset = config.dataset

        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.use_mmd= config.use_mmd
        self.lambda_mmd_setting = config.lambda_mmd_setting
        self.weighted = config.weighted

        self.step = tf.Variable(0, name='step', trainable=False)

        self.d_lr = tf.Variable(config.d_lr, name='d_lr', trainable=False)
        self.g_lr = tf.Variable(config.g_lr, name='g_lr', trainable=False)
        self.c_lr = tf.Variable(config.c_lr, name='c_lr', trainable=False)
        self.d_lr_update = tf.assign(
            self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary),
            name='d_lr_update')
        self.g_lr_update = tf.assign(
            self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary),
            name='g_lr_update')
        self.c_lr_update = tf.assign(
            self.c_lr, tf.maximum(self.c_lr * 0.5, config.lr_lower_boundary),
            name='c_lr_update')

        self.z_dim = config.z_dim
        self.num_conv_filters = config.num_conv_filters

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, self.height, self.width, self.channel = \
                get_conv_shape(self.data_loader, self.data_format)
        self.scale_size = self.height 
        self.repeat_num = int(np.log2(self.height)) - 1  # 2 --> 1 for 28x28 mnist.

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False



    def build_model(self):
        self.x = self.data_loader
        x = norm_img(self.x)  # Converts 256 to [-1, 1].
        self.z = tf.random_normal([tf.shape(x)[0], self.z_dim])
        self.weighted = tf.placeholder(tf.bool, name='weighted')

        # Set up generator and autoencoder functions.
        g, self.g_var = GeneratorCNN(
            self.z, self.num_conv_filters, self.channel,
            self.repeat_num, self.data_format, reuse=False)
        d_out, d_enc, self.d_var_enc, self.d_var_dec = AutoencoderCNN(
            tf.concat([x, g], 0), self.channel, self.z_dim, self.repeat_num,
            self.num_conv_filters, self.data_format, reuse=False)
        AE_x, AE_g = tf.split(d_out, 2)
        self.x_enc, self.g_enc = tf.split(d_enc, 2)
        self.g = denorm_img(g, self.data_format)
        self.AE_g = denorm_img(AE_g, self.data_format)
        self.AE_x = denorm_img(AE_x, self.data_format)

        # Set up autoencoder with test input.
        self.test_input = tf.placeholder(tf.float32, name='test_input',
            shape=[None, self.channel, self.scale_size, self.scale_size])
        _, self.test_enc, _, _ = AutoencoderCNN(
            self.test_input, self.channel, self.z_dim, self.repeat_num,
            self.num_conv_filters, self.data_format, reuse=True)

        ## BEGIN: MMD DEFINITON
        on_encodings = 1
        if on_encodings:
            # Kernel on encodings.
            self.xe = self.x_enc 
            self.ge = self.g_enc 
            sigma_list = [1., 2., 4., 8., 16.]
        else:
            # Kernel on full imgs.
            self.xe = tf.reshape(x, [tf.shape(x)[0], -1]) 
            self.ge = tf.reshape(g, [tf.shape(g)[0], -1]) 
            sigma_list = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
        data_num = tf.shape(self.xe)[0]
        gen_num = tf.shape(self.ge)[0]
        v = tf.concat([self.xe, self.ge], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
        sqs_tiled_horiz = tf.tile(sqs, [1, tf.shape(sqs)[0]])
        exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma**2)
            K += tf.exp(-gamma * exp_object)
        self.K = K
        K_xx = K[:data_num, data_num:]
        K_yy = K[data_num:, data_num:]
        K_xy = K[:data_num, data_num:]
        K_xx_upper = (tf.matrix_band_part(K_xx, 0, -1) -
                      tf.matrix_band_part(K_xx, 0, 0))
        K_yy_upper = (tf.matrix_band_part(K_yy, 0, -1) -
                      tf.matrix_band_part(K_yy, 0, 0))
        num_combos_xx = tf.to_float(data_num * (data_num - 1) / 2)
        num_combos_yy = tf.to_float(gen_num * (gen_num - 1) / 2)

        # Build mnist classification, and get probabilities of keeping.
        self.build_mnist_classifier()
        self.x_pr0 = tf.Variable(tf.ones([self.batch_size, 1]),
            trainable=False, name='x_pr0')
        self.g_pr0 = tf.Variable(tf.ones([self.batch_size, 1]),
            trainable=False, name='g_pr0')
        self.x_prop0 = tf.Variable(0., trainable=False, name='x_prop0')
        self.g_prop0 = tf.Variable(0., trainable=False, name='g_prop0')
        self.x_prop0_pix = tf.Variable(0., trainable=False, name='x_prop0_pix')
        self.g_prop0_pix = tf.Variable(0., trainable=False, name='g_prop0_pix')
        self.classifier_accuracy_test = tf.Variable(0., trainable=False,
            name='classifier_accuracy_test')
        self.classifier_accuracy_test_pix = tf.Variable(0., trainable=False,
            name='classifier_accuracy_test_pix')
        self.x_normed_for_mnist_prediction = (tf.reshape(x,
            [self.batch_size, -1]) + 1.)/ 2.  # Maps [-1, 1] to [0, 1].
        self.g_normed_for_mnist_prediction = (tf.reshape(g,
            [self.batch_size, -1]) + 1.)/ 2.  # Maps [-1, 1] to [0, 1].

        thin_factor = 1. - float(min(self.config.pct)) / max(self.config.pct)
        self.keeping_probs = 1. - thin_factor * tf.reshape(
            self.x_pr0, [-1, 1])

        keeping_probs_tiled = tf.tile(self.keeping_probs, [1, gen_num])
        # Autoencoder weights.
        self.p1_weights_ae = 1. / self.keeping_probs
        self.p1_weights_ae_normed = (
            self.p1_weights_ae / tf.reduce_sum(self.p1_weights_ae))
        self.g_weights_ae = 1. / (
            1. - thin_factor * tf.reshape(self.g_pr0, [-1, 1]))
        self.g_weights_ae_normed = (
            self.g_weights_ae / tf.reduce_sum(self.g_weights_ae))
        # MMD weights.
        self.p1_weights_xy = 1. / keeping_probs_tiled
        self.p1_weights_xy_normed = (
            self.p1_weights_xy / tf.reduce_sum(self.p1_weights_xy))
        self.p1p2_weights_xx = (
            self.p1_weights_xy * tf.transpose(self.p1_weights_xy))
        self.p1p2_weights_xx_upper = (
            tf.matrix_band_part(self.p1p2_weights_xx, 0, -1) -
            tf.matrix_band_part(self.p1p2_weights_xx, 0, 0))
        self.p1p2_weights_xx_upper_normed = (
            self.p1p2_weights_xx_upper /
            tf.reduce_sum(self.p1p2_weights_xx_upper))
        Kw_xx_upper = K_xx_upper * self.p1p2_weights_xx_upper_normed
        Kw_xy = K_xy * self.p1_weights_xy_normed
        num_combos_xy = tf.to_float(data_num * gen_num)

        # Compute and choose between MMD values.
        self.mmd = tf.cond(
            self.weighted,
            lambda: (
                tf.reduce_sum(Kw_xx_upper) +
                tf.reduce_sum(K_yy_upper) / num_combos_yy -
                2 * tf.reduce_sum(Kw_xy)),
            lambda: (
                tf.reduce_sum(K_xx_upper) / num_combos_xx +
                tf.reduce_sum(K_yy_upper) / num_combos_yy -
                2 * tf.reduce_sum(K_xy) / num_combos_xy))
        self.mmd_weighted = (
            tf.reduce_sum(Kw_xx_upper) +
            tf.reduce_sum(K_yy_upper) / num_combos_yy -
            2 * tf.reduce_sum(Kw_xy))
        self.mmd_unweighted = (
            tf.reduce_sum(K_xx_upper) / num_combos_xx +
            tf.reduce_sum(K_yy_upper) / num_combos_yy -
            2 * tf.reduce_sum(K_xy) / num_combos_xy)
        ## END: MMD DEFINITON

        # Define losses.
        self.lambda_mmd = tf.Variable(0., trainable=False, name='lambda_mmd')
        self.lambda_ae = tf.Variable(0., trainable=False, name='lambda_ae')
        self.lambda_fm = tf.Variable(0., trainable=False, name='lambda_fm')
        self.ae_loss_real = tf.cond(
            self.weighted,
            lambda: (
                tf.reduce_sum(self.p1_weights_ae_normed * tf.reshape(
                    tf.reduce_sum(tf.square(AE_x - x), [1, 2, 3]), [-1, 1]))),
            lambda: tf.reduce_mean(tf.square(AE_x - x)))
        #self.ae_loss_real = tf.reduce_mean(tf.square(AE_x - x))
        self.ae_loss_fake = tf.reduce_mean(tf.square(AE_g - g))
        self.ae_loss = self.ae_loss_real
        self.fm1 = tf.reduce_mean(self.ge, axis=0) - tf.reduce_mean(self.xe, axis=0)
        self.fm2 = tf.nn.relu(self.fm1)
        self.fm3 = tf.reduce_mean(self.fm2)
        self.first_moment_loss = -1. * self.fm3
        self.d_loss = (
            self.lambda_ae * self.ae_loss -
            self.lambda_mmd * self.mmd -
            self.lambda_fm * self.first_moment_loss)
        self.g_loss = (
            self.lambda_mmd * self.mmd +
            self.lambda_fm * self.first_moment_loss)

        # Optimizer nodes.
        if self.optimizer == 'adam':
            ae_opt = tf.train.AdamOptimizer(self.d_lr)
            d_opt = tf.train.AdamOptimizer(self.d_lr)
            g_opt = tf.train.AdamOptimizer(self.g_lr)
        elif self.optimizer == 'rmsprop':
            ae_opt = tf.train.RMSPropOptimizer(self.d_lr)
            d_opt = tf.train.RMSPropOptimizer(self.d_lr)
            g_opt = tf.train.RMSPropOptimizer(self.g_lr)
        elif self.optimizer == 'sgd':
            ae_opt = tf.train.GradientDescentOptimizer(self.d_lr)
            d_opt = tf.train.GradientDescentOptimizer(self.d_lr)
            g_opt = tf.train.GradientDescentOptimizer(self.g_lr)


        # Set up optim nodes. Clip encoder only! 
        if 1:
            enc_grads_, enc_vars_ = zip(*ae_opt.compute_gradients(
                self.ae_loss_real, var_list=self.d_var_enc))
            dec_grads_, dec_vars_ = zip(*ae_opt.compute_gradients(
                self.ae_loss_real, var_list=self.d_var_dec))
            enc_grads_clipped_ = tuple(
                [tf.clip_by_value(g, -0.01, 0.01) for g in enc_grads_])
            ae_grads_ = enc_grads_clipped_ + dec_grads_
            ae_vars_ = enc_vars_ + dec_vars_
            self.ae_optim = ae_opt.apply_gradients(zip(ae_grads_, ae_vars_))

            enc_grads, enc_vars = zip(*d_opt.compute_gradients(
                self.d_loss, var_list=self.d_var_enc))
            dec_grads, dec_vars = zip(*d_opt.compute_gradients(
                self.d_loss, var_list=self.d_var_dec))
            enc_grads_clipped = tuple(
                [tf.clip_by_value(g, -0.01, 0.01) for g in enc_grads])
            d_grads = enc_grads_clipped + dec_grads
            d_vars = enc_vars + dec_vars
            self.d_optim = d_opt.apply_gradients(zip(d_grads, d_vars))
        else:
            ae_vars = self.d_var_enc + self.d_var_dec
            self.ae_optim = ae_opt.minimize(self.ae_loss_real, var_list=ae_vars)
            self.d_optim = d_opt.minimize(self.d_loss, var_list=ae_vars)
        self.g_optim = g_opt.minimize(
            self.g_loss, var_list=self.g_var, global_step=self.step)

        self.summary_op = tf.summary.merge([
            tf.summary.image("a_g", self.g, max_outputs=10),
            tf.summary.image("b_AE_g", self.AE_g, max_outputs=10),
            tf.summary.image("c_x", to_nhwc(self.x, self.data_format),
                max_outputs=10),
            tf.summary.image("d_AE_x", self.AE_x, max_outputs=10),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/ae_loss_real", self.ae_loss_real),
            tf.summary.scalar("loss/ae_loss_fake", self.ae_loss_fake),
            tf.summary.scalar("loss/mmd_u", self.mmd_unweighted),
            tf.summary.scalar("loss/mmd_w", self.mmd_weighted),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("prop/x_prop0", self.x_prop0),
            tf.summary.scalar("prop/g_prop0", self.g_prop0),
            tf.summary.scalar("prop/x_prop0_pix_reference", self.x_prop0_pix),
            tf.summary.scalar("prop/g_prop0_pix_reference", self.g_prop0_pix),
            tf.summary.scalar("prop/classifier_accuracy_test",
                self.classifier_accuracy_test),
            tf.summary.scalar("prop/classifier_accuracy_test_pix_reference",
                self.classifier_accuracy_test_pix),
        ])

        
    def build_mnist_classifier(self):
        # Classification data is [0, 1], from TF package.
        self.c_images_01 = tf.placeholder(tf.float32, [None, 784])
        self.c_labels = tf.placeholder(tf.float32, [None, 2])

        # Compute the data probabilities used in WMMD.
        # NOTE: Whatever is chose here is used downstream in the model.
        #       Other pred/prob/entropy/optims are just for reporting.
        self.dropout_pr = tf.placeholder(tf.float32)
        classify_on_pixels = 0
        if classify_on_pixels:
            self.label_pred, self.label_pred_pr, self.c_vars = mnistCNN(
                self.c_images_01, self.dropout_pr, reuse=False)
            self.label_pred_pr0 = self.label_pred_pr[:, 0]
        else:
            # Convert to [-1, 1] for autoencoder.
            c = tf.reshape(self.c_images_01 * 2. - 1.,
                [-1, self.channel, self.scale_size, self.scale_size])
            _, c_enc, _, _ = AutoencoderCNN(
                c, self.channel, self.z_dim, self.repeat_num,
                self.num_conv_filters, self.data_format, reuse=True)
            self.label_pred, self.label_pred_pr, self.c_vars = mnist_enc_NN(
                c_enc, self.dropout_pr, reuse=False)
            self.label_pred_pr0 = self.label_pred_pr[:, 0]

        # Get probs with pixels, for reporting only.
        self.label_pred_pix, self.label_pred_pr_pix, self.c_vars_pix = mnistCNN(
            self.c_images_01, self.dropout_pr, reuse=False)
        self.label_pred_pr0_pix = self.label_pred_pr_pix[:, 0]

        # Define classifier losses.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.c_labels, logits=self.label_pred)
        cross_entropy = tf.reduce_mean(cross_entropy)

        cross_entropy_pix = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.c_labels, logits=self.label_pred_pix)
        cross_entropy_pix = tf.reduce_mean(cross_entropy_pix)

        # Define optimization procedure.
        if 1:
            self.c_optim = tf.train.RMSPropOptimizer(self.c_lr).minimize(
                cross_entropy, var_list=self.c_vars)

            self.c_optim_pix = tf.train.RMSPropOptimizer(self.c_lr).minimize(
                cross_entropy_pix, var_list=self.c_vars_pix)
        else:
            # Same as  above, but with gradient clipping.
            c_opt = tf.train.AdamOptimizer(self.c_lr)
            c_gvs = c_opt.compute_gradients(
                cross_entropy, var_list=self.c_vars)
            c_capped_gvs = (
                [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in c_gvs])
            self.c_optim = c_opt.apply_gradients(c_capped_gvs) 

        # Compute accuracy.
        correct_prediction = tf.equal(
            tf.argmax(self.label_pred, 1), tf.argmax(self.c_labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.classifier_accuracy = tf.reduce_mean(correct_prediction)

        correct_prediction_pix = tf.equal(
            tf.argmax(self.label_pred_pix, 1), tf.argmax(self.c_labels, 1))
        correct_prediction_pix = tf.cast(correct_prediction_pix, tf.float32)
        self.classifier_accuracy_pix = tf.reduce_mean(correct_prediction_pix)

    def train(self):
        print('\n{}\n'.format(self.config))
        z_fixed = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        x_fixed = self.get_images_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        # Use tensorflow tutorial set for conveniently labeled mnist.
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.c_images_reference, self.c_labels_reference = self.prep_01_data(
            split='train', mix='5050', n=8000)
        self.c_images_user, self.c_labels_user = self.prep_01_data(
            split='classifier', mix='5050', n=108)
        self.c_images_test, self.c_labels_test = self.prep_01_data(
            split='test', mix='5050', n=1800)

        # Train generator.
        for step in trange(self.start_step, self.max_step):
            # First do classifier updates.
            batch_user = self.get_n_images_and_labels(
                self.batch_size, self.c_images_user, self.c_labels_user)
            batch_reference = self.get_n_images_and_labels(
                self.batch_size, self.c_images_reference, self.c_labels_reference)

            self.sess.run(self.c_optim,
                feed_dict={
                    self.c_images_01: batch_user[0],
                    self.c_labels: batch_user[1],
                    self.dropout_pr: 0.5})
            self.sess.run(self.c_optim_pix,
                feed_dict={
                    self.c_images_01: batch_reference[0],
                    self.c_labels: batch_reference[1],
                    self.dropout_pr: 0.5})

            if step % self.log_step == 0:
                user_acc, user_acc_pix = self.sess.run([
                        self.classifier_accuracy,
                        self.classifier_accuracy_pix],
                    feed_dict={
                        self.c_images_01: batch_user[0],
                        self.c_labels: batch_user[1],
                        self.dropout_pr: 1.0})
                test_acc, test_acc_pix = self.sess.run([
                        self.classifier_accuracy,
                        self.classifier_accuracy_pix],
                    feed_dict={
                        self.c_images_01: self.c_images_test,
                        self.c_labels: self.c_labels_test,
                        self.dropout_pr: 1.0})
                print('\nstep {},\nuser/test acc {:.4f}/{:.4f}'
                      '\nuser/test acc_pix {:.4f}/{:.4f}'.format(
                    step, user_acc, test_acc, user_acc_pix, test_acc_pix))
            # Set up basket of items to be run. Occasionally fetch items
            # useful for logging and saving.
            fetch_dict = {
                'd_optim': self.d_optim,
                'g_optim': self.g_optim,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    'summary': self.summary_op,
                    'ae_loss_real': self.ae_loss_real,
                    'ae_loss_fake': self.ae_loss_fake,
                    'mmd': self.mmd,
                    'keeping_probs': self.keeping_probs,
                })

            # Train a bit with mmd5050 (target), then switch to wmmd8020.
            if step < 10000:
                weighted = False 
                #x_ = self.get_images_from_loader_target().transpose(
                #    [0, 3, 1, 2])
                x_ = self.get_images_from_loader().transpose([0, 3, 1, 2])
                z_ = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
            else:
                weighted = True 
                x_ = self.get_images_from_loader().transpose([0, 3, 1, 2])
                z_ = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
            # Pre-fetch data and simulations, and normalize for classification.
            x_mnistcnn, g_mnistcnn = self.sess.run([
                    self.x_normed_for_mnist_prediction,
                    self.g_normed_for_mnist_prediction],
                feed_dict={
                    self.x: x_,
                    self.z: z_})
            # Get probs using encodings.
            x_pred_pr0 = self.sess.run(self.label_pred_pr0,
                feed_dict={
                    self.c_images_01: x_mnistcnn,
                    self.dropout_pr: 1.0})
            g_pred_pr0 = self.sess.run(self.label_pred_pr0,
                feed_dict={
                    self.c_images_01: g_mnistcnn,
                    self.dropout_pr: 1.0})
            # Get probs using pixels.
            x_pred_pr0_pix = self.sess.run(self.label_pred_pr0_pix,
                feed_dict={
                    self.c_images_01: x_mnistcnn,
                    self.dropout_pr: 1.0})
            g_pred_pr0_pix = self.sess.run(self.label_pred_pr0_pix,
                feed_dict={
                    self.c_images_01: g_mnistcnn,
                    self.dropout_pr: 1.0})

            xp0 = np.mean(np.round(x_pred_pr0_pix))
            gp0 = np.mean(np.round(g_pred_pr0_pix))

            # Run full training step on pre-fetched data and simulations.
            result = self.sess.run(fetch_dict,
                feed_dict={
                    self.x: x_,
                    self.z: z_,
                    self.x_pr0: np.reshape(x_pred_pr0, [-1, 1]),
                    self.g_pr0: np.reshape(g_pred_pr0, [-1, 1]),
                    self.x_prop0: np.mean(np.round(x_pred_pr0)),
                    self.g_prop0: np.mean(np.round(g_pred_pr0)),
                    self.x_prop0_pix: xp0,
                    self.g_prop0_pix: gp0,
                    self.classifier_accuracy_test: test_acc,
                    self.classifier_accuracy_test_pix: test_acc_pix,
                    self.lambda_mmd: self.lambda_mmd_setting, 
                    self.lambda_ae: 1.0,
                    self.lambda_fm: 0.0,
                    self.dropout_pr: 1.0,
                    self.weighted: weighted})

            # Log and save as needed.
            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()
                ae_loss_real = result['ae_loss_real']
                ae_loss_fake = result['ae_loss_fake']
                mmd = result['mmd']
                print(('[{}/{}] LOSSES: ae_real/fake: {:.6f}, {:.6f} '
                    'mmd: {:.6f}').format(
                        step, self.max_step, ae_loss_real, ae_loss_fake, mmd))
                with open(os.path.join(self.model_dir, 'xp0s.txt'), 'a') as xp0_file:
                    xp0_file.write(str(xp0)+'\n')
                with open(os.path.join(self.model_dir, 'gp0s.txt'), 'a') as gp0_file:
                    gp0_file.write(str(gp0)+'\n')

            if step % (self.save_step) == 0:
                z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                gen_rand = self.generate(
                    z, self.model_dir, idx='rand'+str(step))
                gen_fixed = self.generate(
                    z_fixed, self.model_dir, idx='fix'+str(step))
                if step == 0:
                    x_samp = self.get_images_from_loader()
                    save_image(x_samp, '{}/x_samp.png'.format(self.model_dir))
            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])


    def generate(self, inputs, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.g, {self.z: inputs})
        if path is None and save:
            path = os.path.join(root_path, 'G_{}.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x


    def get_images_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x


    def get_images_from_loader_target(self):
        x = self.data_loader_target.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x


    def prep_01_data(self, split='classifier', mix='5050', n=100):
        # Ensure that training mix has zero as target.
        prop0 = int(mix[:2])
        prop1 = int(mix[2:])
        #self.config.pct = [prop0, prop1]
        n0 = int(n * prop0 / 100.)
        n1 = int(n * prop1 / 100.)

        def pct(mix):
            amts = [int(mix[:2]), int(mix[2:])]
            pct = float(min(amts)) / max(amts)
            return pct

        def fetch_01_and_prep(zipped_images_and_labels, percent):
            # Fetch 01s and apportion according to mix percent.
            d = zipped_images_and_labels
            zero_ind = [i for i,v in enumerate(d) if v[1][0] == 1]
            one_ind = [i for i,v in enumerate(d) if v[1][0] != 1]
            assert n0 <= len(zero_ind) and n1 <= len(one_ind), (
                'Asking for {} zeros and {} ones, but have {} zeros and {} ones'
                ).format(n0, n1, len(zero_ind), len(one_ind))
            zero_ind = zero_ind[:n0]
            one_ind = one_ind[:n1]
            eligible_indices = np.concatenate((zero_ind, one_ind))
            d_01 = [v for i,v in enumerate(d) if i in eligible_indices]
            d_01 = np.random.permutation(d_01)  # Shuffle for random sampling.
            images = [v[0] for v in d_01]
            labels = [v[1] for v in d_01]

            # Reshape, rescale, recode.
            #images = np.reshape(images,
            #    [len(images), self.channel, self.scale_size, self.scale_size])
            images = np.reshape(images, [len(images), -1])  # [n, 784]
            #images = convert_01_to_n11(images)
            labels = [[1.0, 0.0] if i.tolist().index(1.0) == 0 else [0.0, 1.0]
                for i in labels]
            return np.array(images), np.array(labels)

        m = self.mnist
        if split == 'classifier':
            imgs_and_labs = zip(m.validation.images, m.validation.labels)
            images, labels = fetch_01_and_prep(imgs_and_labs, pct(mix))
        elif split == 'test':
            imgs_and_labs = zip(m.test.images, m.test.labels)
            images, labels = fetch_01_and_prep(imgs_and_labs, pct(mix))
        elif split == 'train':
            imgs_and_labs = zip(m.train.images, m.train.labels)
            images, labels = fetch_01_and_prep(imgs_and_labs, pct(mix))

        return images, labels


    def get_n_images_and_labels(self, n, images, labels):
        assert n <= len(images), 'n must be less than length of image set'
        n_random_indices = np.random.choice(range(len(images)), n, replace=False)
        n_images = [v for i,v in enumerate(images) if i in n_random_indices]
        n_labels = [v for i,v in enumerate(labels) if i in n_random_indices]
        return np.array(n_images), np.array(n_labels) 
