# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation
 
Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Wed Jul 12 15:53:44 2017
 
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
from collections import deque
import logging
import logging.config
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm

import utils

class DataGenerator:
    """ DataGenerator Class : To generate Train, Validatidation and Test sets
    for the Deep Human Pose Estimation Model
    Formalized DATA:
        Inputs:
            Inputs have a shape of (Number of Image) X (Height: 256) X (Width: 256) X (Channels: 3)
        Outputs:
            Outputs have a shape of (Number of Image) X (Number of Stacks) X (Heigth: 64) X (Width: 64) X (OutputDimendion: 16)
    Joints:
        We use the MPII convention on joints numbering
        List of joints:
            00 - Right Ankle
            01 - Right Knee
            02 - Right Hip
            03 - Left Hip
            04 - Left Knee
            05 - Left Ankle
            06 - Pelvis (Not present in other dataset ex : LSP)
            07 - Thorax (Not present in other dataset ex : LSP)
            08 - Neck
            09 - Top Head
            10 - Right Wrist
            11 - Right Elbow
            12 - Right Shoulder
            13 - Left Shoulder
            14 - Left Elbow
            15 - Left Wrist
    # TODO : Modify selection of joints for Training

    How to generate Dataset:
        Create a TEXT file with the following structure:
            image_name.jpg[LETTER] box_xmin box_ymin box_xmax b_ymax joints
            [LETTER]:
                One image can contain multiple person. To use the same image
                finish the image with a CAPITAL letter [A,B,C...] for
                first/second/third... person in the image
            joints :
                Sequence of x_p y_p (p being the p-joint)
                /!\ In case of missing values use -1

    The Generator will read the TEXT file to create a dictionnary
    Then 2 options are available for training:
        Store image/heatmap arrays (numpy file stored in a folder: need disk space but faster reading)
        Generate image/heatmap arrays when needed (Generate arrays while training, increase training time - Need to compute arrays at every iteration)
    """
    def __init__(self, joints_name = None, img_dir=None, train_data_file = None, remove_joints = None):
        """ Initializer
        Args:
            joints_name			: List of joints condsidered
            img_dir				: Directory containing every images
            train_data_file		: Text file with training set data
            remove_joints		: Joints List to keep (See documentation)
        """
        if joints_name == None:
            self.joints_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
        else:
            self.joints_list = joints_name
        self.toReduce = False
        if remove_joints is not None:
            self.toReduce = True
            self.weightJ = remove_joints

        self.letter = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
        self.img_dir = img_dir
        self.train_data_file = train_data_file
        self.images = os.listdir(img_dir)
        self.data_sizes = {}

        self.logger = logging.getLogger(self.__class__.__name__)  # Logger

    # --------------------Generator Initialization Methods ---------------------


    def _reduce_joints(self, joints):
        """ Select Joints of interest from self.weightJ
        """
        j = []
        for i in range(len(self.weightJ)):
            if self.weightJ[i] == 1:
                j.append(joints[2*i])
                j.append(joints[2*i + 1])
        return j

    def _create_train_table(self):
        """ Create Table of samples from TEXT file
        """
        self.train_table = []
        self.no_intel = []
        self.data_dict = {}
        input_file = open(self.train_data_file, 'r')
        self.logger.info('Reading train data')
        for line in input_file:
            line = line.strip()
            line = line.split(' ')
            name = line[0]
            center = [int(l) for l in line[1:3]]
            scale = float(line[3])
            head_sz = float(line[4])
            joints = list(map(float,line[5:]))
            if self.toReduce:
                joints = self._reduce_joints(joints)
            if np.all(joints == -1):
                self.no_intel.append(name)
            else:
                joints = np.reshape(joints, (-1,2))
                w = (joints[:, 0] != -1).astype(np.int)
                self.data_dict[name] = {'center' : center, 'scale': scale, 'head_sz': head_sz, 'joints' : joints, 'weights' : w}
                self.train_table.append(name)
        input_file.close()

    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.train_table)

    def _complete_sample(self, name):
        """ Check if a sample has no missing value
        Args:
            name 	: Name of the sample
        """
        for i in range(self.data_dict[name]['joints'].shape[0]):
            if np.array_equal(self.data_dict[name]['joints'][i],[-1,-1]):
                return False
        return True

    def _give_batch_name(self, batch_size = 16, set = 'train'):
        """ Returns a List of Samples
        Args:
            batch_size	: Number of sample wanted
            set				: Set to use (valid/train)
        """
        list_file = []
        for i in range(batch_size):
            if set == 'train':
                list_file.append(random.choice(self.train_set))
            elif set == 'valid':
                list_file.append(random.choice(self.valid_set))
            else:
                self.logger.error('Set must be : train/valid')
                raise ValueError()
                break
        return list_file


    def _create_sets(self, validation_rate = 0.05, test_rate = 0.1):
        """ Select Elements to feed training and validation set
        Args:
            validation_rate		: Percentage of validation data (in ]0,1[, don't waste time use 0.1)
        """
        # TODO Filter to consider only complete samples
        num_before = len(self.train_table)
        self.train_table = [elem for elem in self.train_table if self._complete_sample(elem)]
        self.logger.info('After filtering complete data, %i / %i instances left', len(self.train_table), num_before)

        nsamples = len(self.train_table)
        valid_nsamples = int(nsamples * validation_rate)
        test_nsamples = int(nsamples * test_rate)
        train_nsamples = nsamples - valid_nsamples - test_nsamples
        self.data_sizes = {'train': train_nsamples, 'valid': valid_nsamples, 'test': test_nsamples}

        self.train_set = self.train_table[0 : train_nsamples]
        self.valid_set = self.train_table[train_nsamples : train_nsamples + valid_nsamples]
        self.test_set = self.train_table[train_nsamples + valid_nsamples : ]

        # np.save('Dataset-Validation-Set', self.valid_set)
        # np.save('Dataset-Training-Set', self.train_set)
        # np.save('Dataset-Test-Set', self.test_set)
        self.logger.info('--Training set: %i samples', len(self.train_set))
        self.logger.info('--Validation set: %i samples', len(self.valid_set))
        self.logger.info('--Test set: %i samples', len(self.test_set))

    def generateSet(self, rand = False):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._create_train_table()
        if rand:
            self._randomize()
        self._create_sets()

    # ---------------------------- Generating Methods --------------------------

    def _makeGaussian(self, height, width, sigma = 1, center=None):
        """ Make a square gaussian kernel with sigma 1
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        assert sigma == 1
        rc_cen = np.array([center[1], center[0]], dtype=np.int64)

        # kernel = np.array(
        #     [[1.96519161240319e-05, 0.000239409349497270, 0.00107295826497866, 0.00176900911404382, 0.00107295826497866,
        #       0.000239409349497270, 1.96519161240319e-05],
        #      [0.000239409349497270, 0.00291660295438644, 0.0130713075831894, 0.0215509428482683, 0.0130713075831894,
        #       0.00291660295438644, 0.000239409349497270],
        #      [0.00107295826497866, 0.0130713075831894, 0.0585815363306070, 0.0965846250185641, 0.0585815363306070,
        #       0.0130713075831894, 0.00107295826497866],
        #      [0.00176900911404382, 0.0215509428482683, 0.0965846250185641, 0.159241125690702, 0.0965846250185641,
        #       0.0215509428482683, 0.00176900911404382],
        #      [0.00107295826497866, 0.0130713075831894, 0.0585815363306070, 0.0965846250185641, 0.0585815363306070,
        #       0.0130713075831894, 0.00107295826497866],
        #      [0.000239409349497270, 0.00291660295438644, 0.0130713075831894, 0.0215509428482683, 0.0130713075831894,
        #       0.00291660295438644, 0.000239409349497270],
        #      [1.96519161240319e-05, 0.000239409349497270, 0.00107295826497866, 0.00176900911404382, 0.00107295826497866,
        #       0.000239409349497270, 1.96519161240319e-05]],
        #     dtype=np.float32)
        kernel = np.array(
            [[0.000123409804086680, 0.00150343919297757, 0.00673794699908547, 0.0111089965382423, 0.00673794699908547,
              0.00150343919297757, 0.000123409804086680],
             [0.00150343919297757, 0.0183156388887342, 0.0820849986238988, 0.135335283236613, 0.0820849986238988,
              0.0183156388887342, 0.00150343919297757],
             [0.00673794699908547, 0.0820849986238988, 0.367879441171442, 0.606530659712633, 0.367879441171442,
              0.0820849986238988, 0.00673794699908547],
             [0.0111089965382423, 0.135335283236613, 0.606530659712633, 1, 0.606530659712633, 0.135335283236613,
              0.0111089965382423],
             [0.00673794699908547, 0.0820849986238988, 0.367879441171442, 0.606530659712633, 0.367879441171442,
              0.0820849986238988, 0.00673794699908547],
             [0.00150343919297757, 0.0183156388887342, 0.0820849986238988, 0.135335283236613, 0.0820849986238988,
              0.0183156388887342, 0.00150343919297757],
             [0.000123409804086680, 0.00150343919297757, 0.00673794699908547, 0.0111089965382423, 0.00673794699908547,
              0.00150343919297757, 0.000123409804086680]],
            dtype=np.float32
        )

        kernel_sz = np.array(kernel.shape)
        kernel_sz_half = (0.5 * (kernel_sz - 1)).astype(np.int64)

        dst = np.zeros((height, width), np.float32)

        dst_tl = rc_cen - kernel_sz_half
        dst_br = rc_cen + kernel_sz_half + 1
        clip_tl = np.maximum(-dst_tl, 0)
        clip_br = np.maximum(dst_br - np.array([height, width]), 0)

        dst_tl2 = dst_tl + clip_tl
        dst_br2 = np.maximum(dst_br - clip_br, 0)
        kernel_tl = clip_tl
        kernel_br = np.maximum(kernel_sz - clip_br, 0)

        dst[dst_tl2[0]:dst_br2[0], dst_tl2[1]:dst_br2[1]] = \
            kernel[kernel_tl[0]:kernel_br[0], kernel_tl[1]:kernel_br[1]]

        return dst

    def _generate_hm(self, height, width ,joints, maxlenght, weight):
        """ Generate a full Heap Map for every joints in an array
        Args:
            height			: Wanted Height for the Heat Map
            width			: Wanted Width for the Heat Map
            joints			: Array of Joints
            maxlenght		: Lenght of the Bounding Box
        """
        num_joints = joints.shape[0]
        hm = np.zeros((height, width, num_joints), dtype = np.float32)
        for i in range(num_joints):
            if not(np.array_equal(joints[i], [-1,-1])) and weight[i] == 1:
                # s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
                s = 1
                hm[:,:,i] = self._makeGaussian(height, width, sigma=s, center=(joints[i,0], joints[i,1]))
            else:
                hm[:,:,i] = np.zeros((height,width))
        return hm

    def _crop_data(self, height, width, box, joints, boxp = 0.05):
        """ Automatically returns a padding vector and a bounding box given
        the size of the image and a list of joints.
        Args:
            height		: Original Height
            width		: Original Width
            box			: Bounding Box
            joints		: Array of joints
            boxp		: Box percentage (Use 20% to get a good bounding box)
        """
        padding = [[0,0],[0,0],[0,0]]
        j = np.copy(joints)
        if box[0:2] == [-1,-1]:
            j[joints == -1] = 1e5
            box[0], box[1] = min(j[:,0]), min(j[:,1])
        crop_box = [box[0] - int(boxp * (box[2]-box[0])), box[1] - int(boxp * (box[3]-box[1])), box[2] + int(boxp * (box[2]-box[0])), box[3] + int(boxp * (box[3]-box[1]))]
        if crop_box[0] < 0: crop_box[0] = 0
        if crop_box[1] < 0: crop_box[1] = 0
        if crop_box[2] > width -1: crop_box[2] = width -1
        if crop_box[3] > height -1: crop_box[3] = height -1
        new_h = int(crop_box[3] - crop_box[1])
        new_w = int(crop_box[2] - crop_box[0])
        crop_box = [crop_box[0] + new_w //2, crop_box[1] + new_h //2, new_w, new_h]
        if new_h > new_w:
            bounds = (crop_box[0] - new_h //2, crop_box[0] + new_h //2)
            if bounds[0] < 0:
                padding[1][0] = abs(bounds[0])
            if bounds[1] > width - 1:
                padding[1][1] = abs(width - bounds[1])
        elif new_h < new_w:
            bounds = (crop_box[1] - new_w //2, crop_box[1] + new_w //2)
            if bounds[0] < 0:
                padding[0][0] = abs(bounds[0])
            if bounds[1] > width - 1:
                padding[0][1] = abs(height - bounds[1])
        crop_box[0] += padding[1][0]
        crop_box[1] += padding[0][0]
        return padding, crop_box

    def _crop_img(self, img, padding, crop_box):
        """ Given a bounding box and padding values return cropped image
        Args:
            img			: Source Image
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode = 'constant')
        max_lenght = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght //2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght //2]
        return img

    def _crop(self, img, hm, padding, crop_box):
        """ Given a bounding box and padding values return cropped image and heatmap
        Args:
            img			: Source Image
            hm			: Source Heat Map
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode = 'constant')
        hm = np.pad(hm, padding, mode = 'constant')
        max_lenght = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght //2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght //2]
        hm = hm[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght//2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        return img, hm

    def _relative_joints(self, box, padding, joints, to_size = 64):
        """ Convert Absolute joint coordinates to crop box relative joint coordinates
        (Used to compute Heat Maps)
        Args:
            box			: Bounding Box
            padding	: Padding Added to the original Image
            to_size	: Heat Map wanted Size
        """
        new_j = np.copy(joints)
        max_l = max(box[2], box[3])
        new_j = new_j + [padding[1][0], padding[0][0]]
        new_j = new_j - [box[0] - max_l //2,box[1] - max_l //2]
        new_j = new_j * to_size / (max_l + 0.0000001)
        return new_j.astype(np.int32)

    def _augment(self, img, hm, mask=None, max_rotation=30):
        """ # TODO : IMPLEMENT DATA AUGMENTATION
        """
        if random.choice([0, 1]):  # 50% chance of augmenting

            # Lighting
            mu = np.mean(img)
            img = random.normalvariate(1.0, 0.1) * (img - mu) + mu # contrast
            img += random.uniform(-32/255, 32/255)  # Brightness
            img = np.clip(img, 0.0, 1.0)  # Clip back to 0-1

            # Rotate + scale
            r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            scale = np.clip(random.normalvariate(1.0, 0.1), 0.75, 1.25)
            img = utils.rotate_about_center(img, r_angle, scale)
            hm = utils.rotate_about_center(hm, r_angle, scale)
            if mask is not None:
                mask = utils.rotate_about_center(mask, r_angle, scale, cv2.INTER_NEAREST)

            # Clip again
            img = np.clip(img, 0.0, 1.0)  # Clip back to 0-1

        if mask is None:
            return img, hm
        else:
            return img, hm, mask

    # ----------------------- Batch Generator ----------------------------------

    def _generate_mask(self, joints_small):
        box_2d_cropped = np.concatenate((np.min(joints_small, axis=0, keepdims=True),
                                        np.max(joints_small, axis=0, keepdims=True)), axis=0).astype(np.int)
        mask = np.zeros((64, 64), dtype=np.uint8)
        return cv2.rectangle(mask,
                      (box_2d_cropped[0,0], box_2d_cropped[0,1]), (box_2d_cropped[1,0], box_2d_cropped[1,1]),
                      (255,255,255), -1 )

    def _aux_generator(self, batch_size = 16, stacks = 4, normalize=False, sample_set = 'train', randomize=True):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """

        assert sample_set in ['train', 'valid', 'test']

        # Initialize
        if sample_set == 'train':
            dataset = self.train_set
        elif sample_set == 'valid':
            dataset = self.valid_set
        else:
            dataset = self.test_set
        indices = deque(np.random.permutation(len(dataset))) if randomize else deque(list(range(len((dataset)))))

        while True:
            train_img = np.zeros((batch_size, 256,256,3), dtype = np.float32)
            train_gtmap = np.zeros((batch_size, stacks, 64, 64, len(self.joints_list)), np.float32)
            train_mask = np.zeros((batch_size, 64, 64), np.uint8)
            train_weights = np.zeros((batch_size, len(self.joints_list)), np.float32)
            joint_pos = []  # For storing full res joint positions
            head_sz_all = []

            for i in range(batch_size):

                try:
                    idx = indices.pop()
                    name = dataset[idx]
                except IndexError:
                    return

                joints = self.data_dict[name]['joints']
                weight = self.data_dict[name]['weights']
                center = self.data_dict[name]['center']
                scale = self.data_dict[name]['scale']
                head_sz = self.data_dict[name]['head_sz']

                train_weights[i] = weight
                img = self.open_img(name)

                # Generate crop
                s = (256/200) / scale * 0.65
                M = np.array([[s, 0, 128-(center[0]*s)],
                              [0, s, 128-(center[1]*s)]], dtype=np.float64)  # Affine matrix
                img = cv2.warpAffine(img[:, :, :], M, (256, 256))

                relative_joints_full = np.dot(M, np.pad(joints.T, ((0, 1), (0, 0)), mode='constant', constant_values=1)).T

                relative_joints_small = relative_joints_full / 4
                hm = self._generate_hm(64, 64, relative_joints_small, 64, weight)
                mask = self._generate_mask(relative_joints_small)

                if sample_set == 'train':
                    img, hm, mask = self._augment(img, hm, mask)

                hm = np.expand_dims(hm, axis = 0)
                hm = np.repeat(hm, stacks, axis = 0)
                if normalize:
                    train_img[i] = img.astype(np.float32) / 255
                else :
                    train_img[i] = img.astype(np.float32)
                train_gtmap[i] = hm
                train_mask[i,:,:] = mask
                joint_pos.append(relative_joints_full)
                head_sz_all.append(head_sz * s)

            yield train_img, train_gtmap, train_weights, train_mask, joint_pos, head_sz_all

    def generator(self, batchSize = 16, stacks = 4, norm = True, sample = 'train'):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch
            stacks 	 	: Stacks in HG model
            norm 	 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'valid' Default: 'train'
        """
        return self._aux_generator(batch_size=batchSize, stacks=stacks, normalize=norm, sample_set=sample)

    # ---------------------------- Image Reader --------------------------------
    def open_img(self, name, color = 'RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        if name[-1] in self.letter:
            name = name[:-1]
        img = cv2.imread(os.path.join(self.img_dir, name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color == 'BGR':
            pass
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            self.logger.error('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')
            raise NotImplementedError()

        # Convert to float in range [0, 1]
        img = img.astype(np.float32) / 255.0
        return img

    def plot_img(self, name, plot = 'plt'):
        """ Plot an image
        Args:
            name	: Name of the Sample
            plot	: Library to use (cv2: OpenCV, plt: matplotlib)
        """
        if plot == 'cv2':
            img = self.open_img(name, color = 'BGR')
            cv2.imshow('Image', img)
        elif plot == 'plt':
            img = self.open_img(name, color = 'RGB')
            plt.imshow(img)
            plt.show()

    def test(self, toWait = 0.2):
        """ TESTING METHOD
        You can run it to see if the preprocessing is well done.
        Wait few seconds for loading, then diaporama appears with image and highlighted joints
        /!\ Use Esc to quit
        Args:
            toWait : In sec, time between pictures
        """
        self._create_train_table()
        self._create_sets()
        for i in range(len(self.train_set)):
            img = self.open_img(self.train_set[i])
            w = self.data_dict[self.train_set[i]]['weights']
            padd, box = self._crop_data(img.shape[0], img.shape[1], self.data_dict[self.train_set[i]]['box'], self.data_dict[self.train_set[i]]['joints'], boxp= 0.0)
            new_j = self._relative_joints(box,padd, self.data_dict[self.train_set[i]]['joints'], to_size=256)
            rhm = self._generate_hm(256, 256, new_j,256, w)
            rimg = self._crop_img(img, padd, box)
            # See Error in self._generator
            #rimg = cv2.resize(rimg, (256,256))
            rimg = scm.imresize(rimg, (256,256))
            #rhm = np.zeros((256,256,16))
            #for i in range(16):
            #	rhm[:,:,i] = cv2.resize(rHM[:,:,i], (256,256))

            # grimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
            # cv2.imshow('image', grimg / 255 + np.sum(rhm,axis = 2))
            plt.imshow(rimg)
            plt.show()

        # Wait
        # time.sleep(toWait)
        # if cv2.waitKey(1) == 27:
        # 	print('Ended')
        # 	cv2.destroyAllWindows()
        # 	break



    # ------------------------------- PCK METHODS-------------------------------
    def pck_ready(self, idlh = 3, idrs = 12, testSet = None):
        """ Creates a list with all PCK ready samples
        (PCK: Percentage of Correct Keypoints)
        """
        id_lhip = idlh
        id_rsho = idrs
        self.total_joints = 0
        self.pck_samples = []
        for s in self.data_dict.keys():
            if testSet == None:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1:
                    self.pck_samples.append(s)
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts = True)
                    self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
            else:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1 and s in testSet:
                    self.pck_samples.append(s)
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts = True)
                    self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
        print('PCK PREPROCESS DONE: \n --Samples:', len(self.pck_samples), '\n --Num.Joints', self.total_joints)

    def getSample(self, sample = None):
        """ Returns information of a sample
        Args:
            sample : (str) Name of the sample
        Returns:
            img: RGB Image
            new_j: Resized Joints
            w: Weights of Joints
            joint_full: Raw Joints
            max_l: Maximum Size of Input Image
        """
        if sample != None:
            try:
                joints = self.data_dict[sample]['joints']
                box = self.data_dict[sample]['box']
                w = self.data_dict[sample]['weights']
                img = self.open_img(sample)
                padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp = 0.2)
                new_j = self._relative_joints(cbox,padd, joints, to_size=256)
                joint_full = np.copy(joints)
                max_l = max(cbox[2], cbox[3])
                joint_full = joint_full + [padd[1][0], padd[0][0]]
                joint_full = joint_full - [cbox[0] - max_l //2,cbox[1] - max_l //2]
                img = self._crop_img(img, padd, cbox)
                img = img.astype(np.uint8)
                img = scm.imresize(img, (256,256))
                return img, new_j, w, joint_full, max_l
            except:
                return False
        else:
            print('Specify a sample name')

    def plot_hotmap(self, hp, gt):
        hp_one = np.sum(hp, axis=4)
        gt_one = np.sum(gt, axis=4)
        for i in range(hp_one.shape[0]):
            plt.figure()
            plt.imshow(hp_one[i, 0, :, :], cmap='hot')
            plt.figure()
            plt.imshow(gt_one[i, 0, :, :], cmap='hot')
            plt.show()


if __name__ == '__main__':
    logging.config.fileConfig('logging.conf')

    data_gen = DataGenerator(None, '/home/lichen/Downloads/HumanPose/MP2',
                             'dataset.txt', remove_joints=None)
    data_gen.generateSet()
    generator_source = data_gen._aux_generator(32, 4, normalize=True, sample_set='train')

    plt.figure()
    while True:
        imgs, b, c, _, _, sz = next(generator_source)
        plt.figure()
        plt.imshow(imgs[0,:,:,:]*255)
        print('img range: ', np.min(imgs[0,:,:,:]), '-', np.max(imgs[0,:,:,:]))
        print(sz[0])
        plt.show()

    exit()