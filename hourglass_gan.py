# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Jul 10 19:13:56 2017

@author: Walid Benbihi
@mail : w.benbihi(at)gmail.com
@github : https://github.com/wbenbihi/hourglasstensorlfow/

Abstract:
    This python code creates a Stacked Hourglass Model
    (Credits : A.Newell et al.)
    (Paper : https://arxiv.org/abs/1603.06937)

    Code translated from 'anewell' github
    Torch7(LUA) --> TensorFlow(PYTHON)
    (Code : https://github.com/anewell/pose-hg-train)

    Modification are made and explained in the report
    Goal : Achieve Real Time detection (Webcam)
    ----- Modifications made to obtain faster results (trade off speed/accuracy)

    This work is free of use, please cite the author if you use it!
"""
import h5py
import time
import tensorflow as tf
import numpy as np
import sys
import os
import tensorflow.contrib.layers as tcl

from hourglass_tiny import HourglassModel


class HourglassModel_gan(HourglassModel):
    """ HourglassModel class: (to be renamed)
    Generate TensorFlow model to train and predict Human Pose from images (soon videos)
    Please check README.txt for further information on model management.
    """

    def __init__(self, **kwargs):
        self.lambdas = [0.01, 0.001] if kwargs['lambdas'] is None else kwargs['lambdas']
        self.k = 5  # Number of times to train discriminator before a single step of update
        del kwargs['lambdas']
        super().__init__(**kwargs)

    def generate_model(self):
        """ Create the complete graph
        """
        self.logger.info('Creating model')
        with tf.device(self.gpu):
            with tf.variable_scope('inputs'):
                # Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB)
                self.img_source = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img_source')
                self.img_target = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img_target')
                if self.w_loss:
                    self.weights = tf.placeholder(dtype=tf.float32, shape=(None, self.outDim))
                # Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
                self.gtMaps_source = tf.placeholder(dtype=tf.float32, shape=(None, self.nStack, 64, 64, self.outDim))
                self.gtMaps_target = tf.placeholder(dtype=tf.float32, shape=(None, self.nStack, 64, 64, self.outDim))

            # TODO : Implement weighted loss function
            # NOT USABLE AT THE MOMENT
            # weights = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 1, 1, self.outDim))

            with tf.variable_scope(self.model_name):
                self.output_source, enc_repre_source = self._graph_hourglass(self.img_source)
            with tf.variable_scope(self.model_name,reuse=True):
                self.output_target, enc_repre_target = self._graph_hourglass(self.img_target)

            # Stack the encoding features from different hourglass
            # enc_repre_source_flattened = enc_repre_source[-1]
            # enc_repre_target_flattened = enc_repre_target[-1]
            enc_repre_source_flattened = tf.concat(enc_repre_source, axis=3)
            enc_repre_target_flattened = tf.concat(enc_repre_target, axis=3)
            gt_source_flattened = self.gtMaps_source[:, -1, :, :, :]
            output_target_flattened = self.output_target[:, -1, :, :, :]

            with tf.variable_scope(self.dis_name):
                d_enc_source = self.discriminator(enc_repre_source_flattened,
                                                  trainable=True,
                                                  is_training=self.is_training)
                if self.lambdas[1] > 0:
                    d_pose_source, _ = self.discriminator_pose(gt_source_flattened, is_training=self.is_training)
            with tf.variable_scope(self.dis_name,reuse=True):
                d_enc_target = self.discriminator(enc_repre_target_flattened,
                                                  trainable=True,
                                                  is_training=self.is_training)
                if self.lambdas[1] > 0:
                    d_pose_target, _ = self.discriminator_pose(output_target_flattened, is_training=self.is_training)

            # Discriminator loss
            d_groundtruth = tf.concat([tf.zeros_like(d_enc_source), tf.ones_like(d_enc_target)], axis=0)
            # 1. Encoding
            d_enc_logits = tf.concat([d_enc_source, d_enc_target], axis=0)
            d_enc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=d_groundtruth, logits=d_enc_logits))
            # # 2. Pose
            if self.lambdas[1] > 0:
                d_pose_logits = tf.concat([d_pose_source, d_pose_target], axis=0)
                d_pose_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=d_groundtruth, logits=d_pose_logits))
                self.d_loss = self.lambdas[0] * d_enc_loss + self.lambdas[1] * d_pose_loss
            else:
                self.d_loss = self.lambdas[0] * d_enc_loss

            # Confusion loss
            c_desired = tf.fill(tf.shape(d_groundtruth), 0.5)  # Uniform distribution
            c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=c_desired, logits=d_enc_logits))
            if self.lambdas[1] > 0:
                c_pose_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=c_desired, logits=d_pose_logits))
            else:
                c_pose_loss = 0

            with tf.variable_scope('loss'):
                if self.w_loss:
                    self.p_loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss') + \
                                  self.lambdas[0]*c_loss + self.lambdas[1] * c_pose_loss
                else:
                    self.p_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_source, labels=self.gtMaps_source),
                        name='cross_entropy_loss')
                self.loss = self.p_loss + self.lambdas[0] * c_loss + self.lambdas[1] * c_pose_loss

            self.logger.info('Confusion weight: %f (enc), %f (pose)', self.lambdas[0], self.lambdas[1])

        with tf.device(self.cpu):
            with tf.variable_scope('accuracy'):
                self._accuracy_computation()
            with tf.variable_scope('steps'):
                self.train_step = tf.Variable(0, name='global_step', trainable=False)
            with tf.variable_scope('lr'):
                self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay,
                                                     staircase=True, name='learning_rate')
        with tf.device(self.gpu):
            with tf.variable_scope('rmsprop'):
                self.rmsprop_enc = tf.train.RMSPropOptimizer(learning_rate=self.lr)
                self.rmsprop_d = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            with tf.variable_scope('minimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):

                    # Get discriminator parameters, and create discriminator train_ip
                    var_list_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                    self.train_rmsprop_d = self.rmsprop_d.minimize(self.d_loss, var_list=var_list_d)  # step not updated

                    # Get main network parameters, and create main train_op
                    var_list_main = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v not in var_list_d]
                    self.train_rmsprop_enc = self.rmsprop_enc.minimize(self.loss, self.train_step, var_list=var_list_main)

        self.init = tf.global_variables_initializer()

        with tf.device(self.cpu):
            with tf.variable_scope('training'):
                tf.summary.scalar('loss', self.loss, collections=['train_enc'])
                tf.summary.scalar('Confusion loss', c_loss, collections=['train_enc'])
                tf.summary.scalar('Pose loss', self.p_loss, collections=['train_enc'])

                tf.summary.scalar('Discriminator', self.d_loss, collections=['train_d'])

                tf.summary.scalar('learning_rate', self.lr, collections=['train_enc'])

            with tf.variable_scope('summary'):
                for i in range(len(self.joints)):
                    tf.summary.scalar(self.joints[i], self.joint_accur[i], collections=['train_enc', 'test'])
        self.train_op_enc = tf.summary.merge_all('train_enc')
        self.train_op_d = tf.summary.merge_all('train_d')
        self.test_op = tf.summary.merge_all('test')
        self.weight_op = tf.summary.merge_all('weight')

    def _run_training(self, img_train, gt_train, img_train_target, gt_train_target, weight_train):

        if self.w_loss:
            _, loss_enc, summary_enc, step = \
                self.Session.run([self.train_rmsprop_enc, self.loss, self.train_op_enc, self.train_step],
                                  feed_dict={self.img_source: img_train,
                                             self.gtMaps_source: gt_train,
                                             self.img_target: img_train_target,
                                             self.gtMaps_target: gt_train_target,
                                             self.weights: weight_train,
                                             self.is_training: True})

            for i in range(self.k):
                _, loss_d, summary_d = self.Session.run([self.train_rmsprop_d, self.d_loss, self.train_op_d],
                                                        feed_dict={self.img_source: img_train,
                                                                   self.gtMaps_source: gt_train,
                                                                   self.img_target: img_train_target,
                                                                   self.gtMaps_target: gt_train_target,
                                                                   self.weights: weight_train,
                                                                   self.is_training: True})

        else:
            _, loss_enc, summary_enc, step = \
                self.Session.run([self.train_rmsprop_enc, self.loss, self.train_op_enc, self.train_step],
                                  feed_dict={self.img_source: img_train,
                                             self.gtMaps_source: gt_train,
                                             self.img_target: img_train_target,
                                             self.gtMaps_target: gt_train_target,
                                             self.is_training: True})

            for i in range(self.k):
                _, loss_d, summary_d, = \
                    self.Session.run([self.train_rmsprop_d, self.d_loss, self.train_op_d],
                                      feed_dict={self.img_source: img_train,
                                                 self.gtMaps_source: gt_train,
                                                 self.img_target: img_train_target,
                                                 self.gtMaps_target: gt_train_target,
                                                 self.is_training: True})

        losses = [loss_enc, loss_d]
        summaries = [summary_enc, summary_d]

        return losses, summaries, step

    def discriminator_pose(self, heatmap, is_training=True):
        with tf.variable_scope('discriminator_pose'):
            heatmap_small = tf.contrib.layers.max_pool2d(heatmap, [2, 2], [2, 2], padding='VALID')
            joints_pred = tcl.spatial_softmax(heatmap_small, temperature=0.3, trainable=False)
            net_j = tcl.fully_connected(joints_pred, 128)
            net_j = tf.layers.dropout(net_j, rate=self.dropout_rate, training=is_training)
            net_j = tcl.fully_connected(net_j, 128)
            net_j = tf.layers.dropout(net_j, rate=self.dropout_rate, training=is_training)
            pred = tcl.fully_connected(net_j, 1, activation_fn=None)

        return pred, joints_pred

    def discriminator(self, input, trainable=True, is_training=True):

        with tf.variable_scope('discriminator_enc'):
            h1=tf.nn.relu(tcl.batch_norm(tcl.conv2d(input,
                                                    num_outputs=1024,
                                                    kernel_size=[2,2],
                                                    stride=2,
                                                    padding='SAME',
                                                    scope='conv1'),
                                                    trainable=trainable,
                                                    scope="bn1",
                                                    is_training=is_training))
            h1 = tf.layers.dropout(h1, rate=self.dropout_rate, training=is_training, name='dropout_dis')
            h1_flatten=tcl.flatten(h1)

            # # only use features
            h1_cat = h1_flatten

            h2=tf.contrib.layers.fully_connected(h1_cat,512,scope="fc1")
            h3= tcl.fully_connected(h2, 1, activation_fn=None)  # 2 classes: Source or target

        return h3

if __name__ == '__main__':
    model = HourglassModel()
    model.generate_model()


