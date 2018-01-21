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
#from data_loader import get_loader
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


def convert_n11_01(image):
    return (image + 1.) / 2.


def convert_01_255(image):
    return np.round(image * 255)


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


class Trainer(object):
    def __init__(
            self, config):
        self.config = config
        self.split = config.split
        self.data_path = config.data_path
        self.dataset = config.dataset
        self.just01 = config.just01
        self.target_num = config.target_num
        self.train_mix = config.train_mix
        self.config.pct = [int(self.train_mix[:2]), int(self.train_mix[2:])]

        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.use_mmd= config.use_mmd
        self.lambda_mmd_setting = config.lambda_mmd_setting
        self.weighted = config.weighted

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')
        self.g_lr_update = tf.assign(
            self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary),
            name='g_lr_update')
        self.d_lr_update = tf.assign(
            self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary),
            name='d_lr_update')

        self.z_dim = config.z_dim
        self.num_conv_filters = config.num_conv_filters
        self.scale_size = config.scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        #_, height, width, self.channel = \
        #        get_conv_shape(self.data_loader, self.data_format)
        #self.repeat_num = int(np.log2(height)) - 1  # 2 --> 1 for 28x28 mnist.
        self.channel = 1
        self.repeat_num = 3  # 2 --> 1 for 28x28 mnist.

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
        #self.x = self.data_loader
        #x = norm_img(self.x)  # Converts to [-1, 1].
        self.x = tf.placeholder(tf.float32, name='x',
            shape=[None, self.channel, self.scale_size, self.scale_size])
        self.x_label = tf.placeholder(tf.float32, name='x_label',
            shape=[None, 2])
        self.z = tf.random_normal([tf.shape(self.x)[0], self.z_dim])
        self.weighted = tf.placeholder(tf.bool, name='weighted')

        # Set up generator and autoencoder functions.
        self.g, self.g_var = GeneratorCNN(
            self.z, self.num_conv_filters, self.channel,
            self.repeat_num, self.data_format, reuse=False)
        d_out, d_enc, self.d_var_enc, self.d_var_dec = AutoencoderCNN(
            tf.concat([self.x, self.g], 0), self.channel, self.z_dim,
            self.repeat_num, self.num_conv_filters, self.data_format,
            reuse=False)
        self.AE_x, self.AE_g = tf.split(d_out, 2)
        self.x_enc, self.g_enc = tf.split(d_enc, 2)
        self.x_pix = denorm_img(self.x, self.data_format)
        self.g_pix = denorm_img(self.g, self.data_format)
        self.AE_x_pix = denorm_img(self.AE_x, self.data_format)
        self.AE_g_pix = denorm_img(self.AE_g, self.data_format)

        # Encode a test input.
        self.test_input = tf.placeholder(tf.float32, name='test_input',
            shape=[None, self.channel, self.scale_size, self.scale_size])
        _, self.test_enc, _, _ = AutoencoderCNN(
            self.test_input, self.channel, self.z_dim, self.repeat_num,
            self.num_conv_filters, self.data_format, reuse=True)

        # Build mnist classification, and get probabilities of keeping.
        self.build_mnist_classifier()

        # BEGIN: MMD DEFINITON ##############################################
        on_encodings = 1
        if on_encodings:
            # Kernel on encodings.
            self.xe = self.x_enc 
            self.ge = self.g_enc 
            sigma_list = [1., 2., 4., 8., 16.]
        else:
            # Kernel on full images.
            self.xe = tf.reshape(self.x, [tf.shape(self.x)[0], -1]) 
            self.ge = tf.reshape(self.g, [tf.shape(self.g)[0], -1]) 
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

        # Define keeping probabilities, using classifier output.
        self.thin_factor = (
            1. - float(min(self.config.pct)) / max(self.config.pct))
        self.keeping_probs = 1. - self.thin_factor * tf.reshape(
            self.x_label_pred_pr0, [-1, 1])
        keeping_probs_tiled = tf.tile(self.keeping_probs, [1, gen_num])

        # Autoencoder weights.
        self.p1_weights_ae = 1. / self.keeping_probs
        self.p1_weights_ae_normed = (
            self.p1_weights_ae / tf.reduce_sum(self.p1_weights_ae))
        
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
        # END: MMD DEFINITON ##############################################

        # Define losses.
        self.lambda_mmd = tf.Variable(1., trainable=False, name='lambda_mmd')
        self.lambda_ae = tf.Variable(1., trainable=False, name='lambda_ae')
        self.lambda_fm = tf.Variable(0., trainable=False, name='lambda_fm')
        self.ae_loss_real = tf.cond(
            self.weighted,
            lambda: (
                tf.reduce_sum(self.p1_weights_ae_normed * tf.reshape(
                    tf.reduce_sum(tf.square(self.AE_x - self.x), [1, 2, 3]),
                    [-1, 1]))),
            lambda: tf.reduce_mean(tf.square(self.AE_x - self.x)))
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
            # Set up optim for autoencoder on real data.
            enc_grads_, enc_vars_ = zip(*ae_opt.compute_gradients(
                self.ae_loss_real, var_list=self.d_var_enc))
            dec_grads_, dec_vars_ = zip(*ae_opt.compute_gradients(
                self.ae_loss_real, var_list=self.d_var_dec))
            enc_grads_clipped_ = tuple(
                [tf.clip_by_value(g, -0.01, 0.01) for g in enc_grads_])
            ae_grads_ = enc_grads_clipped_ + dec_grads_
            ae_vars_ = enc_vars_ + dec_vars_
            self.ae_optim = ae_opt.apply_gradients(zip(ae_grads_, ae_vars_))

            # Set up optim for d_loss.
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
            tf.summary.image("a_g", self.g_pix, max_outputs=10),
            tf.summary.image("b_AE_g", self.AE_g_pix, max_outputs=10),
            tf.summary.image("c_x", self.x_pix, max_outputs=10),
            tf.summary.image("d_AE_x", self.AE_x_pix, max_outputs=10),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/ae_loss_real", self.ae_loss_real),
            tf.summary.scalar("loss/mmd", self.mmd),
            tf.summary.scalar("loss/mmd_u", self.mmd_unweighted),
            tf.summary.scalar("loss/mmd_w", self.mmd_weighted),
            tf.summary.scalar("loss/first_moment", self.first_moment_loss),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("prop/x_prop0", self.x_prop0),
            tf.summary.scalar("prop/g_prop0", self.g_prop0),
            tf.summary.scalar("prop/c_prop0", self.c_prop0),
            tf.summary.scalar("classifier/mnist_classifier_accuracy",
                self.mnist_classifier_accuracy),
        ])

        
    def build_mnist_classifier(self):
        self.dropout_pr = tf.placeholder(tf.float32)
        self.c = tf.placeholder(tf.float32, name='c',
            shape=[None, self.channel, self.scale_size, self.scale_size])
        self.c_label = tf.placeholder(tf.float32, name='c_label',
            shape=[None, 2])

        # Create encodings of zeros and nonzeros.
        self.c_ae, self.c_enc, self.c_var_enc, self.c_var_dec = AutoencoderCNN(
            self.c, self.channel, self.z_dim, self.repeat_num,
            self.num_conv_filters, self.data_format, reuse=True)

        # Get classifier probabilities for zeros/nonzeros.
        self.c_label_pred, self.c_label_pred_pr, self.classifier_vars = (
            mnist_encoding_classifier(self.c_enc, self.dropout_pr))
        self.c_label_pred_pr0 = self.c_label_pred_pr[:, self.target_num]
        self.c_prop0 = tf.reduce_mean(tf.round(self.c_label_pred_pr0))
        # Get classifier probabilities for training data.
        self.x_label_pred, self.x_label_pred_pr, _ = (
            mnist_encoding_classifier(self.x_enc, 1.0))  # NOTE: No dropout.
        self.x_label_pred_pr0 = self.x_label_pred_pr[:, self.target_num]
        self.x_prop0 = tf.reduce_mean(tf.round(self.x_label_pred_pr0))
        # Get classifier probabilities for simulations.
        self.g_label_pred, self.g_label_pred_pr, _ = (
            mnist_encoding_classifier(self.g_enc, 1.0))  # NOTE: No dropout.
        self.g_label_pred_pr0 = self.g_label_pred_pr[:, self.target_num]
        self.g_prop0 = tf.reduce_mean(tf.round(self.g_label_pred_pr0))

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.c_label, logits=self.c_label_pred)
            cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            if 1:
                self.c_optim = tf.train.AdamOptimizer(1e-4).minimize(
                    cross_entropy, var_list=self.classifier_vars)
            else:
                # Same as  above, but with gradient clipping.
                c_opt = tf.train.AdamOptimizer(1e-4)
                c_gvs = c_opt.compute_gradients(
                    cross_entropy, var_list=self.classifier_vars)
                c_capped_gvs = (
                    [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in c_gvs])
                self.c_optim = c_opt.apply_gradients(c_capped_gvs)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.c_label_pred, 1), tf.argmax(self.c_label, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.mnist_classifier_accuracy = tf.reduce_mean(correct_prediction)


    def train(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        if self.just01:
            (self.full_x_images,
             self.full_x_labels,
             self.full_t_images,
             self.full_t_labels,
             self.full_c_images,
             self.full_c_labels) = self.prep_01_data(train_mix=self.train_mix)
        print('\n{}\n'.format(self.config))
        z_fixed = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        x_fixed, x_fixed_label = self.get_mnist_images_and_labels(1,
            split='train')
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        # Train generator.
        for step in trange(self.start_step, self.max_step):
            # Set up basket of nodes to run. Occasionally fetch log items.
            fetch_dict = {
                'd_optim': self.d_optim,
                'g_optim': self.g_optim,
                'c_optim': self.c_optim,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    'summary': self.summary_op,
                    'ae_loss_real': self.ae_loss_real,
                    'first_moment_loss': self.first_moment_loss,
                    'mmd': self.mmd,
                })

            # TRAIN.
            # Optionally start on 5050, before going to unbalanced.
            if step < 0:
                weighted = False 
                x = norm_img(
                    self.get_image_from_loader_target().transpose([0, 3, 1, 2]))
                z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
            else:
                weighted = True 
                # NOTE: All inputs must be on [-1, 1].
                x, x_label = self.get_mnist_images_and_labels(
                    self.batch_size, split='train')
                c, c_label = self.get_mnist_images_and_labels(
                    self.batch_size, split='classifier')
                z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))

            # Run full training step on pre-fetched data and simulations.
            result = self.sess.run(fetch_dict,
                feed_dict={
                    self.x: norm_img(x),
                    self.x_label: x_label,
                    self.c: norm_img(c),
                    self.c_label: c_label,
                    self.lambda_mmd: self.lambda_mmd_setting, 
                    self.dropout_pr: 0.5,
                    self.weighted: weighted})

            # Log and save as needed.
            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()
                ae_loss_real = result['ae_loss_real']
                first_moment_loss = result['first_moment_loss']
                mmd = result['mmd']
                print(('[{}/{}] LOSSES: ae_real: {:.6f} '
                    'fm: {:.6f} mmd: {:.6f}').format(
                        step, self.max_step, ae_loss_real, first_moment_loss,
                        mmd))
            if step % (self.save_step) == 0:
                z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                gen_rand = self.generate(
                    z, self.model_dir, idx='rand'+str(step))
                gen_fixed = self.generate(
                    z_fixed, self.model_dir, idx='fix'+str(step))
                if step == 0:
                    x_samp, x_label = self.get_mnist_images_and_labels(
                        1, split='train')
                    save_image(x_samp, '{}/x_samp.png'.format(self.model_dir))
                #self.autoencode(x_samp, self.model_dir, idx=step)
            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])


    def generate(self, inputs, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.g, {self.z: inputs})
        if path is None and save:
            path = os.path.join(root_path, 'G_{}.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x


    def autoencode(self, inputs, path, idx=None, x_fake=None):
        items = {
            'real': inputs,
            'fake': x_fake,
        }
        for key, img in items.items():
            if img is None:
                continue
            if img.shape[3] in [1, 3]:
                img = img.transpose([0, 3, 1, 2])

            x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            x = self.sess.run(self.AE_x, {self.x: img})
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))


    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.test_enc, {self.test_input: inputs})


    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x


    def get_image_from_loader_target(self):
        x = self.data_loader_target.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x


    def prep_01_data(self,
            train_mix='2080', test_mix='5050', classifier_mix='5050'):
        # Ensure that training mix has zero as target.
        assert(int(train_mix[:2]) < int(train_mix[2:]),
            'zero class must be underrep')
        self.config.pct = [int(train_mix[:2]), int(train_mix[2:])]

        def pct(mix):
            amts = [int(mix[:2]), int(mix[2:])]
            pct = float(min(amts)) / max(amts)
            return pct

        def fetch_01_and_prep(zipped_images_and_labels, percent):
            d = zipped_images_and_labels
            zero_ind = [i for i,v in enumerate(d) if v[1][0] == 1]
            one_ind = [i for i,v in enumerate(d) if v[1][1] == 1]
            trim_count = min(len(zero_ind), len(one_ind))
            zero_ind = zero_ind[:int(trim_count * percent)]
            one_ind = one_ind[:trim_count]
            eligible_indices = np.concatenate((zero_ind, one_ind))
            d_01 = [v for i,v in enumerate(d) if i in eligible_indices]
            d_01 = np.random.permutation(d_01)
            images = [v[0] for v in d_01]
            labels = [v[1] for v in d_01]

            # Reshape, rescale, recode.
            images = np.reshape(images,
                [len(images), self.channel, self.scale_size, self.scale_size])
            images = convert_01_255(images)
            labels = [[1.0, 0.0] if i.tolist().index(1) == 0 else [0.0, 1.0]
                for i in labels]
            return np.array(images), np.array(labels)

        m = self.mnist
        train = zip(m.train.images, m.train.labels)
        test = zip(m.test.images, m.test.labels)
        classifier = zip(m.validation.images, m.validation.labels)
        x_images, x_labels = fetch_01_and_prep(train, pct(train_mix))
        t_images, t_labels = fetch_01_and_prep(test, pct(test_mix))
        c_images, c_labels = fetch_01_and_prep(classifier, pct(classifier_mix))
        print('DATA total counts: train {}, test {}, class {}'.format(
            len(x_images), len(t_images), len(c_images)))

        return x_images, x_labels, t_images, t_labels, c_images, c_labels


    def get_n_images_and_labels(self, n, images, labels):
        assert(n <= len(images), 'n must be less than length of image set')
        n_random_indices = np.random.randint(0, len(images), n)
        images = [v for i,v in enumerate(images) if i in n_random_indices]
        labels = [v for i,v in enumerate(labels) if i in n_random_indices]
        return np.array(images), np.array(labels)


    def get_mnist_images_and_labels(self, n, split='train'):
        '''Wraps other fetching functions.
        '''
        assert(split in ['train', 'test', 'classifier'])
        # NOTE: Trying 01 set.
        if self.just01:
            if split == 'train':
                i = self.full_x_images
                l = self.full_x_labels
                return self.get_n_images_and_labels(n, i, l)
            elif split == 'test':
                i = self.full_t_images
                l = self.full_t_labels
                return self.get_n_images_and_labels(n, i, l)
            elif split == 'classifier':
                i = self.full_c_images
                l = self.full_c_labels
                return self.get_n_images_and_labels(n, i, l)
        else:
            # TODO: Facilitate switching between 01 and full set.
            if split == 'train':
                batch = self.mnist.train.next_batch(n)
            elif split == 'test':
                batch = self.mnist.test.next_batch(n)
            elif split == 'classifier':
                batch = self.mnist.validation.next_batch(n)
            images = batch[0]
            images = np.reshape(images,
                [n, self.channel, self.scale_size, self.scale_size])
            images = convert_01_255(images)
            labels = batch[1]
            # Code for 0 as target group.
            labels = [[1.0, 0.0] if i.tolist().index(1) == 0 else [0.0, 1.0]
                for i in labels]
            return images, labels
