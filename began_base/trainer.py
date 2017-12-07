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
        self.conv_hidden_num = config.conv_hidden_num
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
        x = norm_img(self.x)
        self.z = tf.random_normal([tf.shape(x)[0], self.z_dim])
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        # Set up generator and autoencoder functions.
        g, self.g_var = GeneratorCNN(
            self.z, self.conv_hidden_num, self.channel,
            self.repeat_num, self.data_format, reuse=False)
        d_out, d_enc, self.d_var_enc, self.d_var_dec = AutoencoderCNN(
            tf.concat([x, g], 0), self.channel, self.z_dim, self.repeat_num,
            self.conv_hidden_num, self.data_format, reuse=False)
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
            self.conv_hidden_num, self.data_format, reuse=True)

        # DEFINE MMD.
        xe = self.x_enc 
        ge = self.g_enc 
        data_num = tf.shape(xe)[0]
        gen_num = tf.shape(ge)[0]
        v = tf.concat([xe, ge], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
        sqs_tiled_horiz = tf.tile(sqs, [1, tf.shape(sqs)[0]])
        exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
        sigma_list = [1., 2., 4., 8., 16.]
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma**2)
            K += tf.exp(-gamma * exp_object)
        K_xx = K[:data_num, data_num:]
        K_yy = K[data_num:, data_num:]
        K_xy = K[:data_num, data_num:]
        K_yy_upper = (tf.matrix_band_part(K_yy, 0, -1) -
                      tf.matrix_band_part(K_yy, 0, 0))
        num_combos_yy = tf.to_float(gen_num * (gen_num - 1) / 2)

        # Build mnist classification, and get probabilities of keeping.
        self.build_mnist_classifier()
        self.x_normed_for_mnist_prediction = (tf.reshape(x,
            [self.batch_size, -1]) + 1.)/ 2.
        self.batch_logit, self.batch_pr = mnistCNN(
            self.x_normed_for_mnist_prediction, self.keep_prob)
        self.batch_pr1 = self.batch_pr[:, 1]
        thin_factor = 0.75

        self.keeping_probs = 1. - thin_factor * tf.reshape(
            self.batch_pr1, [-1, 1])
        keeping_probs_tiled = tf.tile(self.keeping_probs, [1, gen_num])
        self.p1_weights_xy = 1. / keeping_probs_tiled
        self.p1_weights_xy_normed = (
            self.p1_weights_xy / tf.reduce_sum(self.p1_weights_xy))
        Kw_xy = K_xy * self.p1_weights_xy_normed
        if self.weighted:
            self.mmd = (tf.reduce_sum(K_yy_upper) / num_combos_yy -
                        2 * tf.reduce_sum(Kw_xy) / tf.to_float(gen_num))
        else:
            num_combos_xy = tf.to_float(data_num * gen_num)
            self.mmd = (tf.reduce_sum(K_yy_upper) / num_combos_yy -
                        2 * tf.reduce_sum(K_xy) / num_combos_xy)

        # Define losses.
        self.ae_loss_lambda = tf.placeholder(tf.float32, name='ae_loss_lambda')
        self.ae_loss_real = tf.reduce_sum(tf.square(AE_x - x))
        self.ae_loss_fake = tf.reduce_sum(tf.square(AE_g - g))
        self.ae_loss = self.ae_loss_real + self.ae_loss_fake - self.mmd
        self.g_loss = self.mmd

        # Optimizer nodes.
        if self.optimizer == 'adam':
            ae_opt = tf.train.AdamOptimizer(self.d_lr)
            g_opt = tf.train.AdamOptimizer(self.g_lr)
        elif self.optimizer == 'RMSProp':
            ae_opt = tf.train.RMSPropOptimizer(self.d_lr)
            g_opt = tf.train.RMSPropOptimizer(self.g_lr)
        elif self.optimizer == 'sgd':
            ae_opt = tf.train.GradientDescentOptimizer(self.d_lr)
            g_opt = tf.train.GradientDescentOptimizer(self.g_lr)


        # Clip encoder only! 
        if 1:
            enc_grads, enc_vars = zip(*ae_opt.compute_gradients(
                self.ae_loss, var_list=self.d_var_enc))
            dec_grads, dec_vars = zip(*ae_opt.compute_gradients(
                self.ae_loss, var_list=self.d_var_dec))
            enc_grads_clipped = tuple(
                [tf.clip_by_value(g, -0.01, 0.01) for g in enc_grads])
            ae_grads = enc_grads_clipped + dec_grads
            ae_vars = enc_vars + dec_vars
            self.ae_optim = ae_opt.apply_gradients(zip(ae_grads, ae_vars))
        else:
            all_ae_vars = self.d_var_enc + self.d_var_dec
            self.ae_optim = ae_opt.minimize(self.ae_loss, var_list=all_ae_vars)

        self.g_optim = g_opt.minimize(
            self.g_loss, var_list=self.g_var, global_step=self.step)

        # BEGAN-style control. NOT CURRENTLY IN USE.
        self.balance = self.ae_loss_real - self.ae_loss_fake
        with tf.control_dependencies([self.ae_optim, self.g_optim]):
            self.k_update = tf.assign(
                self.k_t,
                tf.clip_by_value(self.k_t + 0.001 * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.image("g", self.g),
            tf.summary.image("AE_g", self.AE_g),
            tf.summary.image("AE_x", self.AE_x),
            tf.summary.scalar("loss/ae_loss", self.ae_loss),
            tf.summary.scalar("loss/ae_loss_real", self.ae_loss_real),
            tf.summary.scalar("loss/ae_loss_fake", self.ae_loss_fake),
            tf.summary.scalar("loss/mmd", self.mmd),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ])

        
    def build_mnist_classifier(self):
        # Create the model
        self.x_mnist = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        self.y_mnist = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        self.keep_prob = tf.placeholder(tf.float32)
        self.y_pred, self.y_pred_pr = mnistCNN(self.x_mnist, self.keep_prob)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_mnist, logits=self.y_pred)
            cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.y_pred, 1), tf.argmax(self.y_mnist, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.mnist_classifier_accuracy = tf.reduce_mean(correct_prediction)


    def train(self):
        # BEGIN mnist classifier. First train the thinning fn (aka Classifier)
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        for i in range(1000):
            batch = mnist.train.next_batch(64)
            if i % 100 == 0:
                train_accuracy = self.sess.run(self.mnist_classifier_accuracy,
                    feed_dict={
                        self.x_mnist: batch[0],
                        self.y_mnist: batch[1],
                        self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            self.sess.run(self.train_step,
                feed_dict={
                    self.x_mnist: batch[0],
                    self.y_mnist: batch[1],
                    self.keep_prob: 0.5})
        print('test accuracy %g' % self.sess.run(
            self.mnist_classifier_accuracy,
                feed_dict={
                    self.x_mnist: mnist.test.images,
                    self.y_mnist: mnist.test.labels,
                    self.keep_prob: 1.0}))
        # END mnist classifier.

        print('\n{}\n'.format(self.config))
        z_fixed = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        x_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))


        # Train generator using pre-trained autoencoder.
        for step in trange(self.start_step, self.max_step):
            # Determine what will be fetched, and occasionally fetch items
            # useful for logging and saving.
            if self.use_mmd:
                fetch_dict = {
                    #'k_update': self.k_update,
                    'ae_optim': self.ae_optim,
                    'g_optim': self.g_optim,
                }
            else:
                fetch_dict = {
                    'k_update': self.k_update,
                }
            if step % self.log_step == 0:
                fetch_dict.update({
                    'summary': self.summary_op,
                    'ae_loss_real': self.ae_loss_real,
                    'ae_loss_fake': self.ae_loss_fake,
                    'mmd': self.mmd,
                    'k_t': self.k_t,
                    'keeping_probs': self.keeping_probs,
                })

            # Run the previously defined fetch items.
            if step < 25 or step % 500 == 0:
                for _ in range(10):
                    self.sess.run(self.ae_optim, {self.keep_prob: 1.0})
            else:
                for _ in range(5):
                    self.sess.run(self.ae_optim, {self.keep_prob: 1.0})

            result = self.sess.run(fetch_dict,
                feed_dict={
                    self.ae_loss_lambda: 1e-6,
                    self.keep_prob: 1.0})

            # Log and save as needed.
            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()
                ae_loss_real = result['ae_loss_real']
                ae_loss_fake = result['ae_loss_fake']
                mmd = result['mmd']
                k_t = result['k_t']
                print(('[{}/{}] AE_loss_real: {:.6f} AE_loss_fake: {:.6f} '
                       'MMD: {:.6f}, k_t: {:.4f}').format(
                           step, self.max_step, ae_loss_real, ae_loss_fake, mmd,
                           k_t))
            if step % (self.save_step) == 0:
                z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                x_fake = self.generate(z, self.model_dir, idx=step)
                x_samp = self.get_image_from_loader()
                save_image(x_samp, '{}/x_samp.png'.format(self.model_dir))
                self.autoencode(x_samp, self.model_dir, idx=step)
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
            self.z_r, self.conv_hidden_num, self.channel, self.repeat_num,
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
            path = os.path.join(root_path, '{}_G.png'.format(idx))
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
