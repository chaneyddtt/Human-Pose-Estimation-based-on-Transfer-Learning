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
import cv2

from tqdm import tqdm
from utils import draw_result, color_heatmap, get_max_positions, get_tensors_in_checkpoint_file


class HourglassModel():
    """ HourglassModel class: (to be renamed)
    Generate TensorFlow model to train and predict Human Pose from images (soon videos)
    Please check README.txt for further information on model management.
    """

    def __init__(self, nFeat=512, nStack=4, nModules=1, nLow=3, outputDim=16,
                 batch_size=4, val_batch_size=16, drop_rate=0.2,
                 lear_rate=2.5e-4, decay=0.96, decay_step=2000, dataset_source=None, dataset_target=None,
                 logdir=None, w_loss=False, modif=False, name='hourglass',
                 joints=None, save_graph=True, gpu=0):
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

        # Default arguments
        if joints is None:
            joints = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck',
                      'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']

        self.nStack = nStack  # Excludes hourglass_0, i.e. even if nStack is 1, there will be 2 hourglasses
        self.nFeat = nFeat
        self.nModules = nModules
        self.outDim = outputDim
        self.batchSize = batch_size
        self.valBatchSize = val_batch_size
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
        self.joints = joints
        self.w_loss = w_loss
        self.dis_name='discriminator'
        self.model_name='hourglass'
        self.is_training = tf.placeholder(tf.bool)
        self.save_graph = save_graph
        self.train_step = None

        self.init = None  # Initialization function

        self.logger = logging.getLogger(self.__class__.__name__)  # Logger
        self.logger.info('Running on GPU: %s' % self.gpu)

        self.logger.info('Dropout rate: %.2f', drop_rate)


        # For saving image summary tensors
        self.overlay_pl = tf.placeholder(tf.uint8, (None, None, None, 3))
        self.heatmap_pl = tf.placeholder(tf.uint8, (None, None, None, 3))

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
        self.logger.info('Creating model')
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

            with tf.variable_scope('hourglass'):
                self.output_source, self.enc_repre_source = self._graph_hourglass(self.img_source)

            self.logger.info('Hourglass output size %s' % self.output_source.shape)

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

    def _generate_im_summaries2(self, dataset_str, max_outputs=12):

        summary_overlay = tf.summary.image(dataset_str + "overlay", self.overlay_pl, max_outputs=max_outputs)
        summary_heatmap = tf.summary.image(dataset_str + "_heatmap", self.heatmap_pl, max_outputs=max_outputs)
        return summary_overlay, summary_heatmap

    def _generate_im_summaries(self, summary_ops, img, gt, pred):

        width_height = img.shape[1:3]
        overlays, heatmaps = [], []

        for j in range(img.shape[0]):
            overlays.append(draw_result(img[j, :, :, :], pred[j, 0, :, :, :], gt[j, 0, :, :, :]))
            heatmaps.append(color_heatmap(pred[j, 0, :, :, :], width_height, apply_sigmoid=True))

        overlays = np.stack(overlays, axis=0)
        heatmaps = np.stack(heatmaps, axis=0)

        im_summaries = self.Session.run(summary_ops, feed_dict={self.overlay_pl: overlays, self.heatmap_pl: heatmaps})
        return im_summaries

    def _train(self, nEpochs=10, epochSize=1000, saveStep=500, validIter=10):
        """
        """
        step = self.Session.run(self.train_step)
        self.logger.info('Starting at step %i', step)

        with tf.variable_scope('Train'):

            self.dataset_source.generateSet()
            self.dataset_target.generateSet()

            self.generator_source = self.dataset_source._aux_generator(self.batchSize, self.nStack,
                                                                       randomize=True, sample_set='train')
            self.generator_target = self.dataset_target._aux_generator(self.batchSize, self.nStack,
                                                                       randomize=True, sample_set='train')

            validation_names = ['source_val', 'target_val', 'target_test']
            val_image_summary_ops = []
            for vname in validation_names:
                val_image_summary_ops.append(self._generate_im_summaries2(vname))

            for epoch in range(nEpochs):

                '''
                Validation
                '''
                print()
                self.logger.info('Epoch: ' + str(epoch) + '/' + str(nEpochs))

                # Reset all validation generators
                validation_sources = [self.dataset_source._aux_generator(self.valBatchSize, self.nStack,
                                                                          randomize=False, sample_set='valid'),
                                      self.dataset_target._aux_generator(self.valBatchSize, self.nStack,
                                                                         randomize=False, sample_set='valid'),
                                      self.dataset_target._aux_generator(self.valBatchSize, self.nStack,
                                                                         randomize=False, sample_set='test')]
                validation_sizes = [self.dataset_source.data_sizes['valid'],
                                    self.dataset_target.data_sizes['valid'],
                                    self.dataset_target.data_sizes['test']]

                validation_accuracies = [0] * len(validation_sources)
                validation_pcks = [0] * len(validation_sources)
                im_summaries = [None] * len(validation_sources)

                for iVal in range(len(validation_sources)):

                    accuracy_array = np.array([0.0] * len(self.joint_accur))  # Accuracy of each joint
                    valGen = validation_sources[iVal]
                    iValIter = 0
                    num_iter = validation_sizes[iVal] // self.valBatchSize
                    num_below_pck05 = 0
                    num_joints_total = 0

                    for (imgs, gts, weights, mask, joints_gt, head_sz) in tqdm(valGen, total=num_iter, leave=False):

                        accuracy, pred = self.Session.run([self.joint_accur, self.output_source],
                                                          feed_dict={self.img_source: imgs, self.gtMaps_source: gts,
                                                                     self.is_training: False})

                        # valid_summary = self.Session.run(self.test_op, feed_dict={self.img_source: img_train_target,
                        #                                                           self.gtMaps_source: gt_train_target,
                        #                                                           self.is_training: False})
                        # self.test_summary.add_summary(valid_summary, epoch * epochSize)

                        # Compute PCK
                        for i in range(pred.shape[0]):
                            joints_pred = get_max_positions(pred[i,-1,:,:,:], imgs.shape[1]/pred.shape[2])
                            error = np.sqrt(np.sum(np.square(joints_pred - joints_gt[i]), axis=1))/head_sz[i]
                            num_below_pck05 += np.count_nonzero(error < 0.5)
                            num_joints_total += len(error)


                        if iValIter == 0:

                            if mask is not None:
                                # Black out the detections outside the bounding box mask
                                pred = pred[:, [-1], :, :, :]

                                mask_valid = np.expand_dims(np.expand_dims(mask, axis=1), axis=4) > 0
                                mask_valid = np.repeat(mask_valid, axis=4, repeats=pred.shape[4])
                                pred[np.logical_not(mask_valid)] = np.min(pred)

                            # Save image summaries
                            im_summaries[iVal] = self._generate_im_summaries(val_image_summary_ops[iVal], imgs, gts, pred)
                            pass

                        accuracy_array += accuracy
                        iValIter += 1

                    pass

                    # Compute accuracy and PCK
                    accuracy_array /= iValIter
                    validation_accuracies[iVal] = np.sum(accuracy_array) / len(accuracy_array)  # Convert to single scalar
                    validation_pcks[iVal] = num_below_pck05/num_joints_total
                    self.logger.info('Avg. Accuracy (%s) = %.3f%%, PCK@0.5 = %.3f%%', validation_names[iVal],
                                     validation_accuracies[iVal] * 100, validation_pcks[iVal] * 100)

                # Write to writer
                summaries = [tf.Summary.Value(tag="acc_{}".format(validation_names[i]), simple_value=validation_accuracies[i])
                             for i in range(len(validation_sources))] + \
                            [tf.Summary.Value(tag="pck_{}".format(validation_names[i]), simple_value=validation_pcks[i])
                             for i in range(len(validation_sources))]
                test_summary_to_write = tf.Summary(value=summaries)
                self.test_summary.add_summary(test_summary_to_write, step)
                for im_summary in im_summaries:
                    for topic in im_summary:
                        self.test_summary.add_summary(topic, step)
                self.test_summary.flush()

                '''
                Training
                '''

                # Training Set
                for i in tqdm(range(epochSize), leave=False):
                    sys.stdout.flush()

                    # Get training data
                    try:
                        img_train, gt_train, weight_train, _, _, _ = next(self.generator_source)
                    except StopIteration:
                        self.generator_source = self.dataset_source._aux_generator(self.batchSize, self.nStack,
                                                                                   randomize=True, sample_set='train')
                        img_train, gt_train, weight_train, _, _, _ = next(self.generator_source)

                    try:
                        img_train_target, gt_train_target, weight_target, mask_target, _, _ = next(self.generator_target)
                    except StopIteration:
                        self.generator_target = self.dataset_target._aux_generator(self.batchSize, self.nStack,
                                                                                   randomize=True, sample_set='train')
                        img_train_target, gt_train_target, weight_target, mask_target, _, _ = next(
                            self.generator_target)

                    # Run training
                    losses, summaries, step = self._run_training(img_train, gt_train, img_train_target, gt_train_target,
                                                           weight_train)

                    # Save summary (Loss + Accuracy)
                    if i % 20 == 0:
                        for summary in summaries:
                            self.train_summary.add_summary(summary, step)
                        self.train_summary.flush()

                    if step % 500 == 0 and step > 0:
                        self.saver.save(self.Session, os.path.join(self.logdir, 'ckpt', 'model.ckpt'), step)

            print('Training Done')

    def _run_training(self, img_train, gt_train, img_train_target, gt_train_target, weight_train):

        if self.w_loss:
            _, loss, summary, step = self.Session.run([self.train_rmsprop, self.loss, self.train_op, self.train_step],
                                             feed_dict={self.img_source: img_train, self.gtMaps_source: gt_train,
                                                        self.weights: weight_train,
                                                        self.is_training: True})
        else:
            _, loss, summary, step = self.Session.run([self.train_rmsprop, self.loss, self.train_op, self.train_step],
                                             feed_dict={self.img_source: img_train, self.gtMaps_source: gt_train,
                                                        self.is_training: True})

        losses = [loss]
        summaries = [summary]

        return losses, summaries, step

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
                    self.logger.info('Restoring from checkpoint: %s', load)
                    checkpoint_var_names = get_tensors_in_checkpoint_file(load)
                    model_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    missing = [m.op.name for m in model_var_list if m.op.name not in checkpoint_var_names]
                    for m in missing:
                        self.logger.warning('Variable missing from checkpoint: %s', m)
                    var_list = [m for m in model_var_list if m.op.name in checkpoint_var_names]

                    saver = tf.train.Saver(var_list)
                    saver.restore(self.Session, load)

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
                self.logger.info('Summaries will be saved to %s' % self.logdir)

                with tf.device(self.gpu):
                    if self.save_graph:
                        self.train_summary = tf.summary.FileWriter(os.path.join(self.logdir, 'train'), tf.get_default_graph())
                    else:
                        self.train_summary = tf.summary.FileWriter(os.path.join(self.logdir, 'train'))  # Do not save graph for speed
                    self.test_summary = tf.summary.FileWriter(os.path.join(self.logdir, 'test'))


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


        Zi Jian's modifications.
        1. Preprocessing convolution kernel size changed to 7
        2. Streamlined the creation of hourglass modules
        3. Remove self.modif conditions
        4. Corrected residual link ll_ to point correctly
        5. Remove variable scoping
        """

        with tf.variable_scope('preprocessing'):
            # Dim pad1 : nbImages x 256 x 256 x 3
            conv1 = self._conv_bn_relu(inputs, filters=64, kernel_size=7, strides=2, name='conv_256_to_128', pad='SAME')
            # Dim conv1 : nbImages x 128 x 128 x 64
            r1 = self._residual(conv1, numOut=128, name='r1')
            # Dim pad1 : nbImages x 128 x 128 x 128
            pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
            # Dim pool1 : nbImages x 64 x 64 x 128
            r2 = self._residual(pool1, numOut=int(self.nFeat / 2), name='r2')
            r3 = self._residual(r2, numOut=self.nFeat, name='r3')  # Input to hourglass units

        # Storage Table
        enc_repre = [None] * self.nStack
        hg = [None] * self.nStack
        ll = [None] * self.nStack
        ll_ = [None] * self.nStack
        drop = [None] * self.nStack
        out = [None] * self.nStack
        out_ = [None] * self.nStack
        sum_ = [None] * self.nStack

        hourglass_out = r3

        with tf.variable_scope('stacks'):

            for i in range(0, self.nStack):
                with tf.variable_scope('stage_' + str(i)):
                    hg[i], enc_repre[i] = self._hourglass(hourglass_out, self.nLow, self.nFeat, 'hourglass')
                    drop[i] = tf.layers.dropout(hg[i], rate=self.dropout_rate, training=self.is_training,
                                                name='dropout')
                    ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, 'VALID', name='conv')

                    out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')

                    if i < self.nStack-1:
                        ll_[i] = self._conv(ll[i], self.nFeat, 1, 1, 'VALID', 'll')
                        out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                        sum_[i] = tf.add_n([out_[i], hourglass_out, ll_[i]], name='merge')

                        hourglass_out = sum_[i]

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
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
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
            self.generator_source = self.dataset_source._aux_generator(self.batchSize, self.nStack,
                                                                       sample_set='train')
            self.validgen_source = self.dataset_source._aux_generator(self.batchSize, self.nStack,
                                                                      sample_set='valid')

            self.dataset_target.generateSet()
            self.generator_target = self.dataset_target._aux_generator(self.batchSize, self.nStack,
                                                                       sample_set='train')
            self.validgen_target = self.dataset_target._aux_generator(self.batchSize, self.nStack,
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
