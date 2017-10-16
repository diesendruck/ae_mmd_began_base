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

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(
            self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary),
            name='g_lr_update')
        self.d_lr_update = tf.assign(
            self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary),
            name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_dim = config.z_dim
        self.conv_hidden_num = config.conv_hidden_num
        self.scale_size = config.scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, height, width, self.channel = \
                get_conv_shape(self.data_loader, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

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
        self.t_mean = tf.Variable(tf.zeros([self.z_dim, 1]), trainable=False,
                                  name='target_mean')
        self.t_cov_inv = tf.Variable(tf.zeros([self.z_dim, self.z_dim]),
                                     trainable=False, name='target_cov_inv')

        # Set up generator and autoencoder functions.
        g, self.g_var = GeneratorCNN(
            self.z, self.conv_hidden_num, self.channel,
            self.repeat_num, self.data_format, reuse=False)
        d_out, d_enc, self.d_var = AutoencoderCNN(
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
        _, self.test_enc, _ = AutoencoderCNN(
            self.test_input, self.channel, self.z_dim, self.repeat_num,
            self.conv_hidden_num, self.data_format, reuse=True)
        
        # Define autoencoder loss and MMD between encodings of data and gen.
        self.ae_loss = tf.reduce_mean(tf.abs(AE_x - x))

        # DEFINE MMD.
        #self.mmd = MMD(self.x_enc, self.g_enc, self.t_mean, self.t_cov_inv)
        if 1:
            xe = self.x_enc 
            ge = self.g_enc 
            t_mean = self.t_mean
            t_cov_inv = self.t_cov_inv
            data_num = tf.shape(xe)[0]
            gen_num = tf.shape(ge)[0]
            v = tf.concat([xe, ge], 0)
            VVT = tf.matmul(v, tf.transpose(v))
            sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
            sqs_tiled_horiz = tf.tile(sqs, [1, tf.shape(sqs)[0]])
            exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
            sigma = 1.
            K = tf.exp(-0.5 * (1 / sigma) * exp_object)
            K_yy = K[data_num:, data_num:]
            K_xy = K[:data_num, data_num:]
            K_yy_upper = (tf.matrix_band_part(K_yy, 0, -1) -
                          tf.matrix_band_part(K_yy, 0, 0))
            num_combos_yy = tf.to_float(gen_num * (gen_num - 1) / 2)
            def prob_of_keeping(x):
                xt_ = x - tf.transpose(t_mean)
                x_ = tf.transpose(xt_)
                pr = 1. - 0.5 * tf.exp(-10. * tf.matmul(tf.matmul(xt_, t_cov_inv), x_))
                return pr
            keeping_probs = tf.reshape(tf.map_fn(prob_of_keeping, xe), [-1, 1])
            keeping_probs_tiled = tf.tile(keeping_probs, [1, gen_num])
            p1_weights_xy = 1. / keeping_probs_tiled
            p1_weights_xy_normed = p1_weights_xy / tf.reduce_sum(p1_weights_xy)
            Kw_xy = K[:data_num, data_num:] * p1_weights_xy_normed
            self.mmd = (tf.reduce_sum(K_yy_upper) / num_combos_yy -
                        2 * tf.reduce_sum(Kw_xy))

        # Set up optimizer nodes.
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise ValueError('[!] Caution! Paper only used Adam')
        self.ae_optim = optimizer(self.d_lr).minimize(self.ae_loss,
                                                      var_list=self.d_var) 
        self.g_optim = optimizer(self.g_lr).minimize(self.mmd,
                                                     global_step=self.step,
                                                     var_list=self.g_var)

        self.summary_op = tf.summary.merge([
            tf.summary.image("g", self.g),
            tf.summary.image("AE_g", self.AE_g),
            tf.summary.image("AE_x", self.AE_x),
            tf.summary.scalar("loss/ae_loss", self.ae_loss),
            tf.summary.scalar("loss/mmd", self.mmd),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])


    def train(self):
        print('\n{}\n'.format(self.config))

        z_fixed = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        x_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        # Get mean and inverse covariance of target sample encodings.
        test_target = self.get_image_from_loader_target()
        try:
            t_mean = np.load('target_encoding_mean.npy') 
            t_cov_inv = np.load('target_encoding_cov_inv.npy') 
        except:
            print('Fetching target mean and covariance.')
            for i in trange(0, 1000):
                self.sess.run(self.ae_optim)
            target_enc = self.encode(test_target)
            t_mean = np.reshape(np.mean(target_enc, axis=0), [-1, 1])
            t_cov = np.cov(target_enc, rowvar=False)
            t_cov_inv = np.linalg.inv(t_cov)
            np.save('target_encoding_mean.npy', t_mean)
            np.save('target_encoding_cov_inv.npy', t_cov_inv)
            scipy.misc.imsave('target_encoding_cov.jpg', t_cov) 
            x_fake = self.generate(z_fixed, self.model_dir, idx=0)
            self.autoencode(x_fixed, self.model_dir, idx=0, x_fake=x_fake)

        # Train generator using pre-trained autoencoder.
        for step in trange(self.start_step, self.max_step):
            '''
            self.sess.run(self.g_optim,
                feed_dict={
                    self.t_mean: t_mean,
                    self.t_cov_inv: t_cov_inv})
            pdb.set_trace()
            '''

            fetch_dict = {
                'g_optim': self.g_optim,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    'summary': self.summary_op,
                    'ae_loss': self.ae_loss,
                    'mmd': self.mmd,
                })
            result = self.sess.run(fetch_dict,
                feed_dict={
                    self.t_mean: t_mean,
                    self.t_cov_inv: t_cov_inv})

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                ae_loss = result['ae_loss']
                mmd = result['mmd']

                print('[{}/{}] AE_loss: {:.6f} MMD: {:.6f}'.format(
                    step, self.max_step, ae_loss, mmd))

            if step % (self.save_step) == 0:
                x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])


    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_dim], tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z)

        g_z_r, _ = GeneratorCNN(
                self.z_r, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=True)

        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x - g_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

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
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_g_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)


    def test(self):
        root_path = "./"#self.model_dir

        all_g_z = None
        for step in range(3):
            real1_batch = self.get_image_from_loader()
            real2_batch = self.get_image_from_loader()

            save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))
            save_image(real2_batch, os.path.join(root_path, 'test{}_real2.png'.format(step)))

            self.autoencode(
                    real1_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real1".format(step)))
            self.autoencode(
                    real2_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real2".format(step)))

            self.interpolate_G(real1_batch, step, root_path)

            z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            g_z = self.generate(z_fixed, path=os.path.join(root_path, "test{}_g_z.png".format(step)))

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
