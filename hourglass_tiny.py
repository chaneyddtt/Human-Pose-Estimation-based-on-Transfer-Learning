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
import logging
import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os

from tqdm import tqdm


class HourglassModel():
    """ HourglassModel class: (to be renamed)
    Generate TensorFlow model to train and predict Human Pose from images (soon videos)
    Please check README.txt for further information on model management.
    """

    def __init__(self, nFeat=512, nStack=4, nModules=1, nLow=3, outputDim=16, batch_size=16, drop_rate=0.2,
                 lear_rate=2.5e-4, decay=0.96, decay_step=2000, dataset_source=None, dataset_target=None,
                 logdir=None, w_loss=False, modif=False, name='hourglass',
                 joints=['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck',
                         'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist'], gpu=0):
        """ Initializer
        Args:
            nStack				: number of stacks (stage/Hourglass modules)
            nFeat				: number of feature channels on conv layers
            nLow				: number of downsampling (pooling) per module
            outputDim			: number of output Dimension (16 for MPII)
            batch_size			: size of training/testing Batch
            dro_rate			: Rate of neurons disabling for Dropout Layers
            lear_rate			: Learning Rate starting value
            decay				: Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
            decay_step			: Step to apply decay
            dataset			: Dataset (class DataGenerator)
            training			: (bool) True for training / False for prediction
            modif				: (bool) Boolean to test some network modification # DO NOT USE IT ! USED TO TEST THE NETWORK
            name				: name of the model
        """
        self.nStack = nStack  # Excludes hourglass_0, i.e. even if nStack is 1, there will be 2 hourglasses
        self.nFeat = nFeat
        self.nModules = nModules
        self.outDim = outputDim
        self.batchSize = batch_size
        self.dropout_rate = drop_rate
        self.learning_rate = lear_rate
        self.decay = decay
        self.modif = modif
        self.name = name
        self.decay_step = decay_step
        self.nLow = nLow
        self.dataset_source = dataset_source
        self.dataset_target = dataset_target
        self.cpu = '/cpu:0'
        self.gpu = '/gpu:%i' % gpu
        self.logdir = logdir
        self.logdir_with_time = None
        self.joints = joints
        self.w_loss = w_loss
        self.dis_name='discriminator'
        self.model_name='hourglass_first'
        self.is_training = tf.placeholder(tf.bool)

        self.logger = logging.getLogger(self.__class__.__name__)  # Logger
        self.logger.info('Running on GPU: %s' % self.gpu)

    # ACCESSOR

    def get_input(self):
        """ Returns Input (Placeholder) Tensor
        Image Input :
            Shape: (None,256,256,3)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.img_source

    def get_output(self):
        """ Returns Output Tensor
        Output Tensor :
            Shape: (None, nbStacks, 64, 64, outputDim)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.output_source

    def get_label(self):
        """ Returns Label (Placeholder) Tensor
        Image Input :
            Shape: (None, nbStacks, 64, 64, outputDim)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.gtMaps_source

    def get_loss(self):
        """ Returns Loss Tensor
        Image Input :
            Shape: (1,)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.loss

    def get_saver(self):
        """ Returns Saver
        /!\ USE ONLY IF YOU KNOW WHAT YOU ARE DOING
        Warning:
            Be sure to build the model first
        """
        return self.saver

    def generate_model(self):
        """ Create the complete graph
        """
        startTime = time.time()
        print('CREATE MODEL:')
        with tf.device(self.gpu):
            with tf.variable_scope('inputs'):
                # Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB)
                self.img_source = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img')

                if self.w_loss:
                    self.weights = tf.placeholder(dtype=tf.float32, shape=(None, self.outDim))
                # Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
                self.gtMaps_source = tf.placeholder(dtype=tf.float32, shape=(None, self.nStack, 64, 64, self.outDim))

            # TODO : Implement weighted loss function
            # NOT USABLE AT THE MOMENT
            # weights = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 1, 1, self.outDim))
            inputTime = time.time()
            print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')

            with tf.variable_scope('transfer_model'):
                self.output_source, self.enc_repre_source = self._graph_hourglass(self.img_source)

            with tf.variable_scope('loss'):
                if self.w_loss:
                    self.loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss')
                else:
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_source, labels=self.gtMaps_source),
                        name='cross_entropy_loss')

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
                self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            with tf.variable_scope('minimizer'):
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(self.update_ops):
                    self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
        self.init = tf.global_variables_initializer()

        with tf.device(self.cpu):
            with tf.variable_scope('training'):
                tf.summary.scalar('loss', self.loss, collections=['train'])
                tf.summary.scalar('learning_rate', self.lr, collections=['train'])
            with tf.variable_scope('summary'):
                for i in range(len(self.joints)):
                    tf.summary.scalar(self.joints[i], self.joint_accur[i], collections=['train', 'test'])
        self.train_op = tf.summary.merge_all('train')
        self.test_op = tf.summary.merge_all('test')
        self.weight_op = tf.summary.merge_all('weight')


    def restore(self, load=None):
        """ Restore a pretrained model
        Args:
            load	: Model to load (None if training from scratch) (see README for further information)
        """
        with tf.variable_scope('Session'):
            with tf.device(self.gpu):
                self._init_session()
                self._define_saver_summary(summary=False)
                if load is not None:
                    print('Loading Trained Model')
                    t = time.time()
                    self.saver.restore(self.Session, load)
                    print('Model Loaded (', time.time() - t, ' sec.)')
                else:
                    print('Please give a Model in args (see README for further information)')

    def _train(self, nEpochs=10, epochSize=1000, saveStep=500, validIter=10):
        """
        """
        with tf.variable_scope('Train'):

            self.dataset_source.generateSet()
            self.generator_source = self.dataset_source._aux_generator(self.batchSize, self.nStack, normalize=True,
                                                                       sample_set='train')
            self.validgen_source = self.dataset_source._aux_generator(self.batchSize, self.nStack, normalize=True,
                                                                   sample_set='valid')

            self.dataset_target.generateSet()
            self.generator_target = self.dataset_target._aux_generator(self.batchSize, self.nStack, normalize=True,
                                                                       sample_set='train')
            self.validgen_target = self.dataset_target._aux_generator(self.batchSize, self.nStack, normalize=True,
                                                                   sample_set='valid')
            self.resume = {}
            self.resume['accur_source'] = []
            self.resume['accur_target'] = []
            self.resume['accur_target2'] = []
            # self.resume['err'] = []

            for epoch in range(nEpochs):

                print()
                self.logger.info('Epoch: ' + str(epoch) + '/' + str(nEpochs))
            # Validation Set
                accuracy_array_source = np.array([0.0] * len(self.joint_accur))
                accuracy_array_target = np.array([0.0] * len(self.joint_accur))
                accuracy_array_target2 = np.array([0.0] * len(self.joint_accur))
                for i in range(validIter):
                    img_valid_source, gt_valid_source, weight_valid_source = next(self.validgen_source)
                    img_valid_target, gt_valid_target, weight_valid_target = next(self.validgen_target)
                    img_valid_target2, gt_valid_target2, weight_valid_target2 = next(self.generator_target)
                    accuracy_pred_source = self.Session.run(self.joint_accur,
                                                            feed_dict={self.img_source: img_valid_source,
                                                                       self.gtMaps_source: gt_valid_source,
                                                                       self.is_training: False})

                    accuracy_pred_target = self.Session.run(self.joint_accur,
                                                            feed_dict={self.img_source: img_valid_target,
                                                                       self.gtMaps_source: gt_valid_target,
                                                                       self.is_training: False})

                    accuracy_pred_target2 = self.Session.run(self.joint_accur,
                                                             feed_dict={self.img_source: img_valid_target2,
                                                                        self.gtMaps_source: gt_valid_target2,
                                                                        self.is_training: False})

                    valid_summary = self.Session.run(self.test_op, feed_dict={self.img_source: img_valid_target2,
                                                                              self.gtMaps_source: gt_valid_target2,
                                                                              self.is_training: False})
                    self.test_summary.add_summary(valid_summary, epoch * epochSize)
                    self.test_summary.flush()

                    accuracy_array_source += np.array(accuracy_pred_source, dtype=np.float32)
                    accuracy_array_target += np.array(accuracy_pred_target, dtype=np.float32)
                    accuracy_array_target2 += np.array(accuracy_pred_target2, dtype=np.float32)

                accuracy_array_source = accuracy_array_source / validIter
                accuracy_array_target = accuracy_array_target / validIter
                accuracy_array_target2 = accuracy_array_target2 / validIter
                accuracy_array_source_scalar = np.sum(accuracy_array_source) / len(accuracy_array_source)  # Convert to single scalar
                accuracy_array_target_scalar = np.sum(accuracy_array_target) / len(accuracy_array_target)
                accuracy_array_target2_scalar = np.sum(accuracy_array_target2) / len(accuracy_array_target2)

                self.logger.info('Avg. Accuracy (36M, test) = %.3f%%', accuracy_array_source_scalar * 100)
                self.logger.info('Avg. Accuracy (MP2, train) = %.3f%%', accuracy_array_target_scalar * 100)
                self.logger.info('Avg. Accuracy (MP2, test) = %.3f%%', accuracy_array_target2_scalar * 100)
                # Write to writer
                test_summary_to_write = tf.Summary(value=[
                    tf.Summary.Value(tag="acc_source", simple_value=accuracy_array_source_scalar),
                    tf.Summary.Value(tag="acc_target", simple_value=accuracy_array_target_scalar),
                    tf.Summary.Value(tag="acc_target2", simple_value=accuracy_array_target2_scalar),
                ])
                self.test_summary.add_summary(test_summary_to_write, epoch * epochSize)

                self.resume['accur_source'].append(accuracy_array_source)
                self.resume['accur_target'].append(accuracy_array_target)
                self.resume['accur_target2'].append(accuracy_array_target2)

                # Training Set
                for i in tqdm(range(epochSize)):
                    sys.stdout.flush()
                    img_train, gt_train, weight_train = next(self.generator_source)
                    img_train_target, gt_train_target, weight_target = next(self.generator_target)

                    losses, summaries = self._run_training(img_train, gt_train, img_train_target, gt_train_target,
                                                           weight_train)

                    # Save summary (Loss + Accuracy)
                    if i % 20 == 0:
                        for summary in summaries:
                            self.train_summary.add_summary(summary, epoch * epochSize + i)
                        self.train_summary.flush()

                    if epoch * epochSize + i % 250 == 0:
                        with tf.variable_scope('save'):
                            self.saver.save(self.Session, os.path.join(self.logdir_with_time, 'ckpt', 'model.ckpt'), epoch * epochSize + i)

            print('Training Done')

    def _run_training(self, img_train, gt_train, img_train_target, gt_train_target, weight_train):

        if self.w_loss:
            _, loss, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op],
                                             feed_dict={self.img_source: img_train, self.gtMaps_source: gt_train,
                                                        self.weights: weight_train,
                                                        self.is_training: True})
        else:
            _, loss, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op],
                                             feed_dict={self.img_source: img_train, self.gtMaps_source: gt_train,
                                                        self.is_training: True})

        losses = [loss]
        summaries = [summary]

        return losses, summaries

    def training_init(self, nEpochs=10, epochSize=1000, saveStep=500, load=None):
        """ Initialize the training
        Args:
            nEpochs		: Number of Epochs to train
            epochSize		: Size of one Epoch
            saveStep		: Step to save 'train' summary (has to be lower than epochSize)
            dataset		: Data Generator (see generator.py)
            load			: Model to load (None if training from scratch) (see README for further information)
        """
        with tf.variable_scope('Session'):
            with tf.device(self.gpu):
                self._init_weight()
                self._define_saver_summary()
                if load is not None:
                    self.saver.restore(self.Session, load)
                    self.test()
                else:
                    self._train(nEpochs, epochSize, saveStep, validIter=10)

    def weighted_bce_loss(self):
        """ Create Weighted Loss Function
        WORK IN PROGRESS
        """
        self.bceloss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_source, labels=self.gtMaps_source),
            name='cross_entropy_loss')
        e1 = tf.expand_dims(self.weights, axis=1, name='expdim01')
        e2 = tf.expand_dims(e1, axis=1, name='expdim02')
        e3 = tf.expand_dims(e2, axis=1, name='expdim03')
        return tf.multiply(e3, self.bceloss, name='lossW')


    def _accuracy_computation(self):
        """ Computes accuracy tensor
        """
        self.joint_accur = []
        for i in range(len(self.joints)):
            self.joint_accur.append(
                self._accur(self.output_source[:, self.nStack - 1, :, :, i], self.gtMaps_source[:, self.nStack - 1, :, :, i],
                            self.batchSize))

    def _define_saver_summary(self, summary=True):
        """ Create Summary and Saver
        Args:
            logdir_train		: Path to train summary directory
            logdir_test		: Path to test summary directory
        """
        if self.logdir is None:
            raise ValueError('Train/Test directory not assigned')
        else:
            with tf.device(self.cpu):
                self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=0.5)
            if summary:

                dn_prefix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                self.logdir_with_time = os.path.join(self.logdir, dn_prefix)
                self.logger.info('Summaries will be saved to %s' % self.logdir_with_time)

                with tf.device(self.gpu):
                    # self.train_summary = tf.summary.FileWriter(os.path.join(logdir, 'train'), tf.get_default_graph())
                    self.train_summary = tf.summary.FileWriter(os.path.join(self.logdir_with_time, 'train'))  # Do not save graph for speed
                    self.test_summary = tf.summary.FileWriter(os.path.join(self.logdir_with_time, 'test'))
                    # self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())


    def _init_weight(self):
        """ Initialize weights
        """
        self.logger.info('Session initialization')
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.Session = tf.Session(config=config)
        t_start = time.time()
        self.Session.run(self.init)
        self.logger.info('Sess initialized in %.2f sec' % (time.time() - t_start))

    def _init_session(self):
        """ Initialize Session
        """
        self.logger.info('Session initialization')
        t_start = time.time()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.Session = tf.Session(config=config)
        self.logger.info('Sess initialized in %.2f sec' % (time.time() - t_start))

    def _graph_hourglass(self, inputs):
        """Create the Network
        Args:
            inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
        """
        with tf.variable_scope('hourglass'):
        # with tf.name_scope(name=self.model_name,reuse=reuse):
            # if (reuse):
                # vs.reuse_variables()
            with tf.variable_scope('preprocessing'):
                # Input Dim : nbImages x 256 x 256 x 3
                pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
                # Dim pad1 : nbImages x 260 x 260 x 3
                conv1 = self._conv_bn_relu(pad1, filters=64, kernel_size=6, strides=2, name='conv_256_to_128')
                # Dim conv1 : nbImages x 128 x 128 x 64
                r1 = self._residual(conv1, numOut=128, name='r1')
                # Dim pad1 : nbImages x 128 x 128 x 128
                pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
                # Dim pool1 : nbImages x 64 x 64 x 128

                r2 = self._residual(pool1, numOut=int(self.nFeat / 2), name='r2')
                r3 = self._residual(r2, numOut=self.nFeat, name='r3')
            # Storage Table
            enc_repre = [None] * self.nStack
            hg = [None] * self.nStack
            ll = [None] * self.nStack
            ll_ = [None] * self.nStack
            drop = [None] * self.nStack
            out = [None] * self.nStack
            out_ = [None] * self.nStack
            sum_ = [None] * self.nStack

            with tf.variable_scope('stacks'):
                with tf.variable_scope('stage_0'):
                    hg[0], enc_repre[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
                    drop[0] = tf.layers.dropout(hg[0], rate=self.dropout_rate, training=self.is_training, name='dropout')
                    ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1, 1, 'VALID', name='conv')
                    ll_[0] = self._conv(ll[0], self.nFeat, 1, 1, 'VALID', 'll')
                    if self.modif:
                        # TEST OF BATCH RELU
                        out[0] = self._conv_bn_relu(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                    else:
                        out[0] = self._conv(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                    out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
                    sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
                for i in range(1, self.nStack):
                    with tf.variable_scope('stage_' + str(i)):
                        hg[i], enc_repre[i] = self._hourglass(sum_[i - 1], self.nLow, self.nFeat, 'hourglass')
                        drop[i] = tf.layers.dropout(hg[i], rate=self.dropout_rate, training=self.is_training,
                                                    name='dropout')
                        ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, 'VALID', name='conv')
                        ll_[i] = self._conv(ll[i], self.nFeat, 1, 1, 'VALID', 'll')
                        if self.modif:
                            out[i] = self._conv_bn_relu(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                        else:
                            out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                        out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                        sum_[i] = tf.add_n([out_[i], sum_[i - 1], ll_[0]], name='merge')
                with tf.variable_scope('stage_' + str(self.nStack)):
                    hg[self.nStack - 1], enc_repre[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow,
                                                                                      self.nFeat, 'hourglass')
                    drop[self.nStack - 1] = tf.layers.dropout(hg[self.nStack - 1], rate=self.dropout_rate,
                                                              training=self.is_training, name='dropout')
                    ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack - 1], self.nFeat, 1, 1, 'VALID', 'conv')
                    if self.modif:
                        out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID',
                                                                  'out')
                    else:
                        out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID', 'out')
            if self.modif:
                return tf.nn.sigmoid(tf.stack(out, axis=1, name='stack_output'), name='final_output')
            else:
                return tf.stack(out, axis=1, name='final_output'), enc_repre

    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv'):
        """ Spatial Convolution (CONV2D)
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		: Number of filters (channels)
            kernel_size	: Size of kernel
            strides		: Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
        Returns:
            conv			: Output Tensor (Convolved Input)
        """
        with tf.variable_scope(name):
            # Kernel for convolution, Xavier Initialisation
            kernel = tf.get_variable('weights', [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            return conv

    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu'):
        """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		: Number of filters (channels)
            kernel_size	: Size of kernel
            strides		: Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
        Returns:
            norm			: Output Tensor
        """
        with tf.variable_scope(name):
            kernel = tf.get_variable('weights', [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='VALID', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                is_training=self.is_training)
            return norm

    def _conv_block(self, inputs, numOut, name='conv_block'):
        """ Convolutional Block
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the block
        Returns:
            conv_3	: Output Tensor
        """

        with tf.variable_scope(name):
            with tf.variable_scope('norm_1'):
                norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=self.is_training)
                conv_1 = self._conv(norm_1, int(numOut / 2), kernel_size=1, strides=1, pad='VALID', name='conv')
            with tf.variable_scope('norm_2'):
                norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=self.is_training)
                pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                conv_2 = self._conv(pad, int(numOut / 2), kernel_size=3, strides=1, pad='VALID', name='conv')
            with tf.variable_scope('norm_3'):
                norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=self.is_training)
                conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad='VALID', name='conv')
            return conv_3

    def _skip_layer(self, inputs, numOut, name='skip_layer'):
        """ Skip Layer
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the bloc
        Returns:
            Tensor of shape (None, inputs.height, inputs.width, numOut)
        """
        with tf.variable_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self._conv(inputs, numOut, kernel_size=1, strides=1, name='conv')
                return conv

    def _residual(self, inputs, numOut, name='residual_block'):
        """ Residual Unit
        Args:
            inputs	: Input Tensor
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.variable_scope(name):
            convb = self._conv_block(inputs, numOut)
            skipl = self._skip_layer(inputs, numOut)
            if self.modif:
                return tf.nn.relu(tf.add_n([convb, skipl], name='res_block'))
            else:
                return tf.add_n([convb, skipl], name='res_block')

    def _hourglass(self, inputs, n, numOut, name='hourglass'):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.variable_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2, low_repre = self._hourglass(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')
                low_repre = low_2
            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            output = tf.add_n([up_2, up_1], name='out_hg')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                return output, low_repre

    def _argmax(self, tensor):
        """ ArgMax
        Args:
            tensor	: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            arg		: Tuple of max position
        """
        resh = tf.reshape(tensor, [-1])
        argmax = tf.argmax(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

    def _compute_err(self, u, v):
        """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
        Args:
            u		: 2D - Tensor (Height x Width : 64x64 )
            v		: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            (float) : Distance (in [0,1])
        """
        u_x, u_y = self._argmax(u)
        v_x, v_y = self._argmax(v)
        return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))),
                         tf.to_float(91))

    def _accur(self, pred, gtMap, num_image):
        """ Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
        returns one minus the mean distance.
        Args:
            pred		: Prediction Batch (shape = num_image x 64 x 64)
            gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
            num_image 	: (int) Number of images in batch
        Returns:
            (float)
        """
        err = tf.to_float(0)
        for i in range(num_image):
            err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
        return tf.subtract(tf.to_float(1), err / num_image)

    def test(self, validIter=10):
        """
        """
        with tf.variable_scope('Test'):
            self.dataset_source.generateSet()
            self.generator_source = self.dataset_source._aux_generator(self.batchSize, self.nStack, normalize=True,
                                                                       sample_set='train')
            self.validgen_source = self.dataset_source._aux_generator(self.batchSize, self.nStack, normalize=True,
                                                                      sample_set='valid')

            self.dataset_target.generateSet()
            self.generator_target = self.dataset_target._aux_generator(self.batchSize, self.nStack, normalize=True,
                                                                       sample_set='train')
            self.validgen_target = self.dataset_target._aux_generator(self.batchSize, self.nStack, normalize=True,
                                                                      sample_set='valid')

            accuracy_array_source = np.array([0.0] * len(self.joint_accur))
            accuracy_array_target = np.array([0.0] * len(self.joint_accur))
            accuracy_array_target2 = np.array([0.0] * len(self.joint_accur))
            for i in range(validIter):
                img_valid_source, gt_valid_source, weight_valid_source = next(self.validgen_source)
                img_valid_target, gt_valid_target, weight_valid_target = next(self.validgen_target)
                img_valid_target2, gt_valid_target2, weight_valid_target2 = next(self.generator_target)
                accuracy_pred_source = self.Session.run(self.joint_accur,
                                                        feed_dict={self.img_source: img_valid_source,
                                                                   self.gtMaps_source: gt_valid_source,
                                                                   self.is_training:False})

                accuracy_pred_target = self.Session.run(self.joint_accur,
                                                        feed_dict={self.img_source: img_valid_target,
                                                                   self.gtMaps_source: gt_valid_target,
                                                                   self.is_training: False})

                accuracy_pred_target2 = self.Session.run(self.joint_accur,
                                                         feed_dict={self.img_source: img_valid_target2,
                                                                    self.gtMaps_source: gt_valid_target2,
                                                                    self.is_training:False})

                accuracy_array_source += np.array(accuracy_pred_source, dtype=np.float32)
                accuracy_array_target += np.array(accuracy_pred_target, dtype=np.float32)
                accuracy_array_target2 += np.array(accuracy_pred_target2, dtype=np.float32)
            accuracy_array_source = accuracy_array_source / validIter
            accuracy_array_target = accuracy_array_target / validIter
            accuracy_array_target2 = accuracy_array_target2 / validIter
            print('--Avg. Accuracy =', str((np.sum(accuracy_array_source) / len(accuracy_array_source)) * 100)[:6], '%')
            print('--Avg. Accuracy =', str((np.sum(accuracy_array_target) / len(accuracy_array_target)) * 100)[:6], '%')
            print('--Avg. Accuracy =', str((np.sum(accuracy_array_target2) / len(accuracy_array_target2)) * 100)[:6], '%')


if __name__ == '__main__':
    model = HourglassModel()
    model.generate_model()
