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
        super().__init__(**kwargs)

    def generate_model(self):
        """ Create the complete graph
        """
        startTime = time.time()
        print('CREATE MODEL:')
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
            inputTime = time.time()
            print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')

            with tf.variable_scope(self.model_name):
                self.output_source, self.enc_repre_source = self._graph_hourglass(self.img_source)
            with tf.variable_scope(self.model_name,reuse=True):
                self.output_target, self.enc_repre_target = self._graph_hourglass(self.img_target)

            with tf.variable_scope(self.dis_name):
                d_logits = self.discriminator(self.enc_repre_source[0],
                                              trainable=True,
                                              is_training=self.is_training)
            with tf.variable_scope(self.dis_name,reuse=True):
                d_logits_ = self.discriminator(self.enc_repre_target[0],
                                               trainable=True,
                                               is_training=self.is_training)

            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=tf.ones_like(d_logits)))
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=tf.zeros_like(d_logits_)))
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=tf.ones_like(d_logits_)))

            self.loss_d = d_loss_real + d_loss_fake

            # with tf.name_scope('linear'):
            #     w=tf.get_variable("linear_weigth",[1],dtype=tf.float32,initializer=tf.random_uniform_initializer(maxval=0))
            #     b=tf.get_variable("linear_bias",[1],dtype=tf.float32,initializer=tf.random_uniform_initializer())
            #     loss_joint_linear=tf.add(tf.multiply(w,self.loss_d),b)
                # loss_joint= self._compute_err(self.output_target,self.gtMaps_source)
                # self.loss_diff=tf.reduce_sum(loss_joint-loss_joint_linear)

            all_reg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)


            # self.d_reg = [var for var in all_reg if self.dis_name in var.name]
            # self.enc_reg = [var for var in all_reg if self.model_name in var.name]

            self.trainable_para=[var for var in all_reg if self.model_name in var.name ]+ [var for var in all_reg if 'linear' in var.name ]

            with tf.variable_scope('loss'):
                if self.w_loss:
                    self.loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss')+0.01*self.g_loss
                else:
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_source, labels=self.gtMaps_source),
                        name='cross_entropy_loss')+0.1*self.g_loss

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
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(self.update_ops):
                    self.train_rmsprop_enc = self.rmsprop_enc.minimize(self.loss, self.train_step)
                    self.train_rmsprop_d = self.rmsprop_d.minimize(self.loss_d, self.train_step)

        self.init = tf.global_variables_initializer()

        with tf.device(self.cpu):
            with tf.variable_scope('training'):
                tf.summary.scalar('loss', self.loss, collections=['train_enc'])
                # tf.summary.scalar('loss_diff', self.loss_diff, collections=['train_enc'])
                # tf.summary.scalar('enc_loss', self.g_loss, collections=['train_enc'])
                tf.summary.scalar('d_loss', self.loss_d, collections=['train_d'])
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
            _, loss_enc, summary_enc = self.Session.run([self.train_rmsprop_enc, self.loss, self.train_op_enc],
                                                        feed_dict={self.img_source: img_train,
                                                                   self.gtMaps_source: gt_train,
                                                                   self.img_target: img_train_target,
                                                                   self.gtMaps_target: gt_train_target,
                                                                   self.weights: weight_train,
                                                                   self.is_training: True})
            _, loss_d, summary_d = self.Session.run([self.train_rmsprop_d, self.loss_d, self.train_op_d],
                                                    feed_dict={self.img_source: img_train,
                                                               self.gtMaps_source: gt_train,
                                                               self.img_target: img_train_target,
                                                               self.gtMaps_target: gt_train_target,
                                                               self.weights: weight_train,
                                                               self.is_training: True})
        else:
            _, loss_enc, summary_enc = self.Session.run([self.train_rmsprop_enc, self.loss, self.train_op_enc],
                                                        feed_dict={self.img_source: img_train,
                                                                   self.gtMaps_source: gt_train,
                                                                   self.img_target: img_train_target,
                                                                   self.gtMaps_target: gt_train_target,
                                                                   self.is_training: True})
            _, loss_d, summary_d = self.Session.run([self.train_rmsprop_d, self.loss_d, self.train_op_d],
                                                    feed_dict={self.img_source: img_train,
                                                               self.gtMaps_source: gt_train,
                                                               self.img_target: img_train_target,
                                                               self.gtMaps_target: gt_train_target,
                                                               self.is_training: True})

        losses = [loss_enc, loss_d]
        summaries = [summary_enc, summary_d]

        return losses, summaries

    def discriminator(self, input, trainable=True, is_training=True):

        h1=tf.nn.relu(tcl.batch_norm(tcl.conv2d(input,
                                                num_outputs=1024,
                                                kernel_size=[3,3],
                                                stride=2,
                                                padding='SAME',
                                                scope='conv1'),
                                                trainable=trainable,
                                                scope="bn1",
                                                is_training=is_training))
        h1 = tf.layers.dropout(h1, rate=self.dropout_rate, training=is_training, name='dropout_dis')
        h1_flatten=tcl.flatten(h1)
        h2=tf.contrib.layers.fully_connected(h1_flatten,512,scope="fc1")
        h3= tcl.fully_connected(h2, 256, activation_fn=None)
        return h3

if __name__ == '__main__':
    model = HourglassModel()
    model.generate_model()


