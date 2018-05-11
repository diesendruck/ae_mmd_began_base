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

np.random.seed(123)


def next(loader):
    return loader.next()[0].data.numpy()


def vert(arr):
    return np.reshape(arr, [-1, 1])


def upper(mat):
    return tf.matrix_band_part(mat, 0, -1) - tf.matrix_band_part(mat, 0, 0)


def sum_normed(mat):
    return mat / tf.reduce_sum(mat)


def to_nhwc(image, data_format, is_tf=False):
    if is_tf:
        if data_format == 'NCHW':
            new_image = nchw_to_nhwc(image)
        else:
            new_image = image
        return new_image
    else:
        if data_format == 'NCHW':
            new_image = image.transpose([0, 2, 3, 1])
        else:
            new_image = image
        return new_image


def nhwc_to_nchw(image, is_tf=False):
    if is_tf:
        if image.get_shape().as_list()[3] in [1, 3]:
            new_image = tf.transpose(image, [0, 3, 1, 2])
        else:
            new_image = image
        return new_image
    else:
        if image.shape[3] in [1, 3]:
            new_image = image.transpose([0, 3, 1, 2])
        else:
            new_image = image
        return new_image


def convert_255_to_n11(image, data_format=None):
    ''' Converts pixel values to range [-1, 1].'''
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image


def convert_n11_to_255(image, data_format, is_tf=False):
    if is_tf:
        return tf.clip_by_value(to_nhwc((image + 1)*127.5, data_format, is_tf=True), 0, 255)
    else:
        return np.clip(to_nhwc((image + 1)*127.5, data_format), 0, 255)


def convert_01_to_n11(image):
    return image * 2. - 1.


def convert_01_to_255(image, data_format):
    return np.clip(to_nhwc(image * 255, data_format), 0, 255)


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
        self.w_lr = tf.Variable(config.w_lr, name='w_lr', trainable=False)
        self.d_lr_update = tf.assign(
            self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary),
            name='d_lr_update')
        self.g_lr_update = tf.assign(
            self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary),
            name='g_lr_update')
        self.w_lr_update = tf.assign(
            self.w_lr, tf.maximum(self.w_lr * 0.5, config.lr_lower_boundary),
            name='w_lr_update')

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
        self.z = tf.random_normal(shape=[self.batch_size, self.z_dim])
        # Images are NHWC on [0, 1].
        self.x = tf.placeholder(tf.float32,
            [self.batch_size, self.scale_size, self.scale_size, self.channel], name='x')
        self.x_predicted_weights = tf.placeholder(tf.float32, [self.batch_size, 1],
            name='x_predicted_weights')
        x = nhwc_to_nchw(convert_01_to_n11(self.x), is_tf=True)
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
        self.g = convert_n11_to_255(g, self.data_format, is_tf=True)
        self.AE_g = convert_n11_to_255(AE_g, self.data_format, is_tf=True)
        self.AE_x = convert_n11_to_255(AE_x, self.data_format, is_tf=True)

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
        K_xx_upper = upper(K_xx)
        K_yy_upper = upper(K_yy)
        num_combos_xx = tf.to_float(data_num * (data_num - 1) / 2)
        num_combos_yy = tf.to_float(gen_num * (gen_num - 1) / 2)

        # Build weights prediction.
        self.build_weights_prediction()

        self.x_predicted_weights_tiled = tf.tile(self.x_predicted_weights,
            [1, self.batch_size])
        # Autoencoder weights.
        self.p1_weights_ae = self.x_predicted_weights_tiled
        self.p1_weights_ae_normed = sum_normed(self.p1_weights_ae)
        # MMD weights.
        self.p1_weights = self.x_predicted_weights_tiled
        self.p1_weights_normed = sum_normed(self.p1_weights)
        self.p1p2_weights = self.p1_weights * tf.transpose(self.p1_weights)
        self.p1p2_weights_upper = upper(self.p1p2_weights)
        self.p1p2_weights_upper_normed = sum_normed(self.p1p2_weights_upper)
        Kw_xx_upper = K_xx_upper * self.p1p2_weights_upper_normed
        Kw_xy = K_xy * self.p1_weights_normed
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
        self.ae_loss_real = tf.cond(
            self.weighted,
            lambda: (
                tf.reduce_sum(self.p1_weights_ae_normed * tf.reshape(
                    tf.reduce_sum(tf.square(AE_x - x), [1, 2, 3]), [-1, 1]))),
            lambda: tf.reduce_mean(tf.square(AE_x - x)))
        #self.ae_loss_real = tf.reduce_mean(tf.square(AE_x - x))
        self.ae_loss_fake = tf.reduce_mean(tf.square(AE_g - g))
        self.ae_loss = self.ae_loss_real
        self.d_loss = self.lambda_ae * self.ae_loss - self.lambda_mmd * self.mmd
        self.g_loss = self.lambda_mmd * self.mmd

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
            tf.summary.image("c_x", self.x, max_outputs=10),
            tf.summary.image("d_AE_x", self.AE_x, max_outputs=10),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/ae_loss_real", self.ae_loss_real),
            tf.summary.scalar("loss/ae_loss_fake", self.ae_loss_fake),
            tf.summary.scalar("loss/mmd_u", self.mmd_unweighted),
            tf.summary.scalar("loss/mmd_w", self.mmd_weighted),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

        
    def build_weights_prediction(self):
        # Images are NHWC on [0, 1].
        self.w_images = tf.placeholder(tf.float32,
            [None, self.scale_size, self.scale_size, self.channel], name='w_images')
        self.w_weights = tf.placeholder(tf.float32, [None, 1], name='w_weights')

        # Convert NHWC [0, 1] to NCHW [-1, 1] for autoencoder.
        w_images_for_ae = convert_01_to_n11(nhwc_to_nchw(self.w_images, is_tf=True))
        _, w_enc, _, _ = AutoencoderCNN(
            w_images_for_ae, self.channel, self.z_dim, self.repeat_num,
            self.num_conv_filters, self.data_format, reuse=True)

        self.dropout_pr = tf.placeholder(tf.float32, name='dropout_pr')
        self.w_pred, self.w_vars = mnist_enc_NN_predict_weights(
            w_enc, self.dropout_pr, reuse=False)
        self.w_loss = tf.reduce_mean(tf.squared_difference(self.w_weights, self.w_pred))

        # Define optimization procedure.
        if 1:
            self.w_optim = tf.train.RMSPropOptimizer(self.w_lr).minimize(
                self.w_loss, var_list=self.w_vars)
        else:
            # Same as  above, but with gradient clipping.
            w_opt = tf.train.AdamOptimizer(self.w_lr)
            w_gvs = w_opt.compute_gradients(
                self.w_loss, var_list=self.w_vars)
            w_capped_gvs = (
                [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in w_gvs])
            self.w_optim = w_opt.apply_gradients(w_capped_gvs) 


    def train(self):
        print('\n{}\n'.format(self.config))

        # Save some fixed images once.
        z_fixed = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        x_fixed = self.get_images_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        # Use tensorflow tutorial set for conveniently labeled mnist.
        # NOTE: Data is on [0,1].
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.images_user = self.prep_data(split='user', n=100)  # NHWC
        self.images_user_weights = np.loadtxt('target_num_user_weights.txt', delimiter=',')
        self.images_train = self.prep_data(split='train', n=8000)

        # One time, save images in order to apply ratings.
        user_imgs_dir = os.path.join(self.model_dir, 'user_imgs')
        if not os.path.exists(user_imgs_dir):
            os.makedirs(user_imgs_dir)
        imgs = convert_01_to_255(self.images_user, 'NHWC')
        save_image(imgs, '{}/user_imgs/user.png'.format(self.model_dir))
        for i in range(len(self.images_user)):
            imgs = convert_01_to_255(self.images_user, 'NHWC')
            im = Image.fromarray(imgs[i][:, :, 0])
            im = im.convert('RGB')
            im.save('{}/user_imgs/user_{}.png'.format(self.model_dir, i))

        # Train generator.
        for step in trange(self.start_step, self.max_step):
            # First do weighting fn updates.
            batch_user, batch_user_weights = self.get_n_images_and_weights(
                self.batch_size, self.images_user, self.images_user_weights)
            #batch_reference = self.get_n_images_and_weights(
            #    self.batch_size, self.c_images_reference, self.w_weights_reference)

            self.sess.run(self.w_optim,
                feed_dict={
                    self.w_images: batch_user,
                    self.w_weights: batch_user_weights,
                    self.dropout_pr: 0.5})

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
                })

            weighted = True 
            batch_train = self.get_n_images(self.batch_size, self.images_train)
            # Now get weights for those in the training batch.
            batch_train_weights = self.sess.run(self.w_pred,
                feed_dict={
                    self.w_images: batch_train,
                    self.dropout_pr: 1.0})

            # Run full training step on pre-fetched data and simulations.
            result = self.sess.run(fetch_dict,
                feed_dict={
                    self.x: batch_train,
                    self.x_predicted_weights: batch_train_weights,
                    self.lambda_mmd: self.lambda_mmd_setting, 
                    self.lambda_ae: 1.0,
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


    def prep_data(self, split='user', n=100):
        target_num = 7
        print('\n\nFetching only number {}.\n\n'.format(target_num))

        def fetch_and_prep(zipped_images_and_labels):
            d = zipped_images_and_labels
            # Each element of d is a list of [image, one-hot label].
            ind = [i for i,v in enumerate(d) if v[1][target_num] == 1]
            d_target_num = [v for i,v in enumerate(d) if i in ind]
            # TODO: switch to no permutation, or !=. Will require relabeling target_num_user_weights.txt.
            if split != 'user':
                d_target_num = np.random.permutation(d_target_num)  # Shuffle for random sampling.
            images = [v[0] for v in d_target_num[:n]]
            labels = [v[1] for v in d_target_num[:n]]

            # Reshape, rescale, recode.
            images = np.reshape(images,
                [len(images), self.scale_size, self.scale_size, self.channel])
            return images

        m = self.mnist
        if split == 'user':
            imgs_and_labs = zip(m.validation.images, m.validation.labels)
            images = fetch_and_prep(imgs_and_labs)
        elif split == 'train':
            imgs_and_labs = zip(m.train.images, m.train.labels)
            images = fetch_and_prep(imgs_and_labs)
        elif split == 'test':
            imgs_and_labs = zip(m.test.images, m.test.labels)
            images = fetch_and_prep(imgs_and_labs)

        return images


    def get_n_images_and_weights(self, n, images, weights):
        assert n <= len(images), 'n must be less than length of image set'
        n_random_indices = np.random.choice(range(len(images)), n, replace=False)
        n_images = [v for i,v in enumerate(images) if i in n_random_indices]
        n_weights = [v for i,v in enumerate(weights) if i in n_random_indices]
        return np.array(n_images), vert(n_weights)
    

    def get_n_images(self, n, images):
        assert n <= len(images), 'n must be less than length of image set'
        n_random_indices = np.random.choice(range(len(images)), n, replace=False)
        n_images = [v for i,v in enumerate(images) if i in n_random_indices]
        return np.array(n_images)
