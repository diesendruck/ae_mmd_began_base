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
        self.do_k_update = config.do_k_update

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

        _, height, width, self.channel = \
                get_conv_shape(self.data_loader, self.data_format)
        self.repeat_num = int(np.log2(height)) - 1  # 2 --> 1 for 28x28 mnist.

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

            self.build_test_model()


    def build_model(self):
        self.x = self.data_loader
        x = norm_img(self.x)  # Converts to [-1, 1].
        self.z = tf.random_normal([tf.shape(x)[0], self.z_dim])
        self.k_t = tf.Variable(1., trainable=False, name='k_t')

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
        self.x_pr1 = tf.Variable(tf.ones([self.batch_size, 1]),
            trainable=False, name='x_pr1')
        self.x_prop1 = tf.Variable(0., trainable=False, name='x_prop1')
        self.g_prop1 = tf.Variable(0., trainable=False, name='g_prop1')
        self.x_normed_for_mnist_prediction = (tf.reshape(x,
            [self.batch_size, -1]) + 1.)/ 2.  # Maps [-1, 1] to [0, 1].
        self.g_normed_for_mnist_prediction = (tf.reshape(g,
            [self.batch_size, -1]) + 1.)/ 2.  # Maps [-1, 1] to [0, 1].
        thin_factor = 0.75

        self.keeping_probs = 1. - thin_factor * tf.reshape(
            self.x_pr1, [-1, 1])
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
        self.mmd_weighted = (
            tf.reduce_sum(Kw_xx_upper) +
            tf.reduce_sum(K_yy_upper) / num_combos_yy -
            2 * tf.reduce_sum(Kw_xy)) #/ tf.to_float(gen_num))
        self.mmd_unweighted = (
            tf.reduce_sum(K_xx_upper) / num_combos_xx +
            tf.reduce_sum(K_yy_upper) / num_combos_yy -
            2 * tf.reduce_sum(K_xy) / num_combos_xy)
        if self.weighted:
            print('WEIGHTED MMD')
            self.mmd = self.mmd_weighted
        else:
            self.mmd = self.mmd_unweighted 
        ## END: MMD DEFINITON

        # Define losses.
        self.lambda_mmd = tf.Variable(0., trainable=False, name='lambda_mmd')
        self.lambda_ae = tf.Variable(0., trainable=False, name='lambda_ae')
        self.lambda_fm = tf.Variable(0., trainable=False, name='lambda_fm')
        if self.weighted:
            print('WEIGHTED AE')
            self.ae_loss_real = tf.reduce_sum(
                self.p1_weights_ae_normed *
                tf.reshape(tf.reduce_sum(tf.square(AE_x - x), [1, 2, 3]),
                    [-1, 1]))
        else:
            self.ae_loss_real = tf.reduce_mean(tf.square(AE_x - x))
        self.ae_loss_fake = tf.reduce_mean(tf.square(AE_g - g))
        self.ae_loss = self.ae_loss_real + self.k_t * self.ae_loss_fake
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

        # BEGAN-style control.
        self.balance = self.ae_loss_real - self.ae_loss_fake
        with tf.control_dependencies([self.d_optim, self.g_optim]):
            #self.k_update = tf.assign(
            #    self.k_t,
            #    tf.clip_by_value(self.k_t - 0.1 * self.balance, 0, 500))
            self.k_update = tf.assign(self.k_t, 0)

        self.summary_op = tf.summary.merge([
            tf.summary.image("a_g", self.g, max_outputs=9),
            tf.summary.image("b_AE_g", self.AE_g, max_outputs=9),
            tf.summary.image("c_x", to_nhwc(self.x, self.data_format),
                max_outputs=9),
            tf.summary.image("d_AE_x", self.AE_x, max_outputs=9),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/ae_loss_real", self.ae_loss_real),
            tf.summary.scalar("loss/ae_loss_fake", self.ae_loss_fake),
            tf.summary.scalar("loss/mmd", self.mmd),
            tf.summary.scalar("loss/mmd_u", self.mmd_unweighted),
            tf.summary.scalar("loss/mmd_w", self.mmd_weighted),
            tf.summary.scalar("loss/first_moment", self.first_moment_loss),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
            tf.summary.scalar("prop/x_prop1", self.x_prop1),
            tf.summary.scalar("prop/g_prop1", self.g_prop1),
        ])

        
    def build_mnist_classifier(self):
        # Create the model
        self.x_mnist = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        self.y_mnist = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        self.dropout_pr = tf.placeholder(tf.float32)
        self.label_pred, self.label_pred_pr = mnistCNN(self.x_mnist, self.dropout_pr)
        self.label_pred_pr1 = self.label_pred_pr[:, 1] 

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_mnist, logits=self.label_pred)
            cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.label_pred, 1), tf.argmax(self.y_mnist, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.mnist_classifier_accuracy = tf.reduce_mean(correct_prediction)


    def train(self):
        #######################################################################
        # BEGIN mnist classifier. First train the thinning fn (aka Classifier)
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        for i in range(5000):
            batch = mnist.train.next_batch(64)
            if i % 100 == 0:
                train_accuracy = self.sess.run(self.mnist_classifier_accuracy,
                    feed_dict={
                        self.x_mnist: batch[0],
                        self.y_mnist: batch[1],
                        self.dropout_pr: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            self.sess.run(self.train_step,
                feed_dict={
                    self.x_mnist: batch[0],
                    self.y_mnist: batch[1],
                    self.dropout_pr: 0.5})
        print('test accuracy %g' % self.sess.run(
            self.mnist_classifier_accuracy,
                feed_dict={
                    self.x_mnist: mnist.test.images,
                    self.y_mnist: mnist.test.labels,
                    self.dropout_pr: 1.0}))
        # END mnist classifier.
        #######################################################################

        print('\n{}\n'.format(self.config))
        z_fixed = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        x_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        # Train generator.
        for step in trange(self.start_step, self.max_step):
            # Set up basket of items to be run. Occasionally fetch items
            # useful for logging and saving.
            if self.do_k_update:
                fetch_dict = {
                    'k_update': self.k_update,
                }
            else:
                fetch_dict = {
                    'd_optim': self.d_optim,
                    'g_optim': self.g_optim,
                }
            if step % self.log_step == 0:
                fetch_dict.update({
                    'summary': self.summary_op,
                    'ae_loss_real': self.ae_loss_real,
                    'ae_loss_fake': self.ae_loss_fake,
                    'fm1': self.fm1,
                    'fm2': self.fm2,
                    'fm3': self.fm3,
                    'first_moment_loss': self.first_moment_loss,
                    'mmd': self.mmd,
                    'k_t': self.k_t,
                    'keeping_probs': self.keeping_probs,
                })

            # Optionally pre-train autoencoder.
            pretrain1 = 0
            pretrain2 = 0
            pretrain_steps = pretrain1 + pretrain2
            if step < pretrain1:
                # Pretrain1: Autoencoder on real.
                aer = self.sess.run([self.ae_loss_real, self.ae_optim],
                    {self.dropout_pr: 1.0})
                if step % 500 == 0:
                    print('ae_loss_real: {}'.format(aer[0]))
            elif step >= pretrain1 and step < pretrain_steps:
                self.d_lr = self.config.d_lr
                # Pretrain2: Autoencoder on real and fake.
                aer = self.sess.run(
                    [self.ae_loss_real, self.ae_loss_fake, self.d_optim],
                    {self.dropout_pr: 1.0})
                if step % 500 == 0:
                    print('ae_loss_real: {}, ae_loss_fake: {}'.format(
                        aer[0], aer[1]))
            else:
                # Main train stage.
                # Train a bit with mmd5050 (target), then switch to wmmd8020.
                #if step < pretrain_steps + 5000:
                if step < 0:
                    self.weighted = False 
                    x_ = self.get_image_from_loader_target().transpose(
                        [0, 3, 1, 2])
                    z_ = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                else:
                    self.weighted = True 
                    x_ = self.get_image_from_loader().transpose([0, 3, 1, 2])
                    z_ = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                # Pre-fetch data and simulations, and compute classifier probs.
                #x_, z_ = self.sess.run([self.x, self.z])
                x_mnistcnn, g_mnistcnn = self.sess.run([
                        self.x_normed_for_mnist_prediction,
                        self.g_normed_for_mnist_prediction],
                    feed_dict={
                        self.x: x_,
                        self.z: z_})
                x_pred_pr1 = self.sess.run(
                    self.label_pred_pr1,
                    feed_dict={
                        self.x_mnist: x_mnistcnn,
                        self.dropout_pr: 1.0})
                g_pred_pr1 = self.sess.run(
                    self.label_pred_pr1,
                    feed_dict={
                        self.x_mnist: g_mnistcnn,
                        self.dropout_pr: 1.0})

                # Run full training step on pre-fetched data and simulations.
                if step % 500 == 0:
                    for _ in range(100):
                        self.sess.run(self.d_optim, {self.dropout_pr: 1.0})
                else:
                    if step > 10000:
                        for _ in range(19):
                            self.sess.run(self.g_optim, {self.dropout_pr: 1.0})
                result = self.sess.run(fetch_dict,
                    feed_dict={
                        self.x: x_,
                        self.z: z_,
                        self.x_pr1: np.reshape(x_pred_pr1, [-1, 1]),
                        self.x_prop1: np.mean(x_pred_pr1),
                        self.g_prop1: np.mean(g_pred_pr1),
                        self.lambda_mmd: self.lambda_mmd_setting, 
                        self.lambda_ae: 1.0,
                        self.lambda_fm: 0.0,
                        self.dropout_pr: 1.0})

            # Log and save as needed.
            if step > pretrain_steps and step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()
                ae_loss_real = result['ae_loss_real']
                ae_loss_fake = result['ae_loss_fake']
                first_moment_loss = result['first_moment_loss']
                mmd = result['mmd']
                k_t = result['k_t']
                print(('[{}/{}] LOSSES: ae_real/fake: {:.6f}, {:.6f} '
                    'fm: {:.6f} mmd: {:.6f}, k_t: {:.4f}').format(
                        step, self.max_step, ae_loss_real, ae_loss_fake,
                        first_moment_loss, mmd, k_t))
            if step % (self.save_step) == 0:
                z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                gen_rand = self.generate(z, self.model_dir, idx='r'+str(step))
                gen_fixed = self.generate(z_fixed, self.model_dir, idx=step)
                if step == 0:
                    x_samp = self.get_image_from_loader()
                    save_image(x_samp, '{}/x_samp.png'.format(self.model_dir))
                #self.autoencode(x_samp, self.model_dir, idx=step)
            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])


    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_dim],
                tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z)

        g_z_r, _ = GeneratorCNN(
            self.z_r, self.num_conv_filters, self.channel, self.repeat_num,
            self.data_format, reuse=True)

        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x - g_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss,
                var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))


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


    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size/2)

        self.sess.run(self.z_r_update)
        tf_real_batch = to_nchw_numpy(real_batch)
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run(
                [self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = (real_batch[:half_batch_size],
            real_batch[half_batch_size:])

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img,
                os.path.join(root_path, 'test{}_interp_g_{}.png'.format(
                    step, idx)),
                nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated,
            [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated,
            os.path.join(root_path, 'test{}_interp_G.png'.format(step)),
            nrow=10)


    def test(self):
        root_path = "./"#self.model_dir

        all_g_z = None
        for step in range(3):
            real1_batch = self.get_image_from_loader()
            real2_batch = self.get_image_from_loader()

            save_image(
                real1_batch,
                os.path.join(root_path, 'test{}_real1.png'.format(step)))
            save_image(
                real2_batch,
                os.path.join(root_path, 'test{}_real2.png'.format(step)))

            self.autoencode(
                real1_batch, self.model_dir,
                idx=os.path.join(root_path, "test{}_real1".format(step)))
            self.autoencode(
                real2_batch, self.model_dir,
                idx=os.path.join(root_path, "test{}_real2".format(step)))

            self.interpolate_G(real1_batch, step, root_path)

            z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            g_z = self.generate(z_fixed,
                path=os.path.join(root_path, "test{}_g_z.png".format(step)))

            if all_g_z is None:
                all_g_z = g_z
            else:
                all_g_z = np.concatenate([all_g_z, g_z])
            save_image(all_g_z, '{}/g_z{}.png'.format(root_path, step))

        save_image(all_g_z, '{}/all_g_z.png'.format(root_path), nrow=16)


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
