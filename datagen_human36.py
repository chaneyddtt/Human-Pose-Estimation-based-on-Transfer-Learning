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
import logging, logging.config
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
from scipy.special import expit
import h5py


class DataGenerator_human36():
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


    def __init__(self, joints_name=None, img_dir=None, remove_joints = None):
        """ Initializer
        Args:
            joints_name			: List of joints condsidered
            img_dir				: Directory containing every images
            train_data_file		: Text file with training set data
            remove_joints		: Joints List to keep (See documentation)
        """
        if joints_name == None:
            self.joints_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax',
                                'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
        else:
            self.joints_list = joints_name
        if img_dir == None:
            self.img_dir='/home/lichen/Downloads/HumanPose/Human36/images'
        else:
            self.img_dir=img_dir

        self.toReduce = False
        if remove_joints is not None:
            self.toReduce = True
            self.weightJ = remove_joints
        self.images= os.listdir(self.img_dir)
        self.Jnum = 16
        self.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
                 [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
                 [6, 8], [8, 9]]
        self.data_dict = {}

        self.logger = logging.getLogger(self.__class__.__name__)  # Logger

    # --------------------Generator Initialization Methods ---------------------


    def _reduce_joints(self, joints):
        """ Select Joints of interest from self.weightJ
        """
        j = []
        for i in range(len(self.weightJ)):
            if self.weightJ[i] == 1:
                j.append(joints[2 * i])
                j.append(joints[2 * i + 1])
        return j

    def getData(self, tmpFile):
        data = h5py.File(tmpFile, 'r')
        d = {}
        for k, v in data.items():
            d[k] = np.asarray(data[k])
        data.close()
        return d

    def _create_train_table(self):
        """ Create Table of samples from TEXT file
        """
        self.train_table = []
        self.no_intel = []
        input_file = '/home/lichen/Downloads/HumanPose/Human36/annot_train.h5'
        annot_train = self.getData(input_file)
        self.logger.info('Reading train data')
        subj=annot_train['subject']
        act=annot_train['action']
        subact=annot_train['subaction']
        cam=annot_train['camera']
        id= annot_train['id']

        names = ['s_%02d_act_%02d_subact_%02d_ca_%02d_%06d.jpg' % (subj[i], act[i], subact[i], cam[i], id[i]) for i in range(len(id))]
        self.data_dict.update(dict.fromkeys(names))

        for index in range(len(id)):
            box=  annot_train['bbox'][index]
            joints= annot_train['joint_2d'][index]
            if self.toReduce:
                joints = self._reduce_joints(joints)
            if np.all(joints == -1):
                self.no_intel.append(names[index])
                del self.data_dict[names[index]]
            else:
                w = (joints[:, 0] != -1).astype(np.int)
                self.data_dict[names[index]] = {'box': box, 'joints': joints, 'weights': w}
                self.train_table.append(names[index])

    def _create_valid_table(self):
        """ Create Table of samples from TEXT file
        """
        self.valid_table = []
        input_file = '/home/lichen/Downloads/HumanPose/Human36/annot_val.h5'
        annot_train = self.getData(input_file)
        self.logger.info('Reading validation data')
        subj = annot_train['subject']
        act = annot_train['action']
        subact = annot_train['subaction']
        cam = annot_train['camera']
        id = annot_train['id']

        names = ['s_%02d_act_%02d_subact_%02d_ca_%02d_%06d.jpg' % (subj[i], act[i], subact[i], cam[i], id[i]) for i in range(len(id))]
        self.data_dict.update(dict.fromkeys(names))

        for index in range(len(id)):
            box = annot_train['bbox'][index]
            joints = annot_train['joint_2d'][index]
            if self.toReduce:
                joints = self._reduce_joints(joints)
            if np.all(joints == -1):  # if joints == [-1] * len(joints):
                self.no_intel.append(names[index])
                del self.data_dict[names[index]]
            else:
                w = (joints[:, 0] != -1).astype(np.int)
                self.data_dict[names[index]] = {'box': box, 'joints': joints, 'weights': w}
                self.valid_table.append(names[index])

    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.train_table)

    def _complete_sample(self, name):
        """ Check if a sample has no missing value
        Args:
            name 	: Name of the sample
        """

        # for i in range(self.data_dict[name]['joints'].shape[0]):
        #     if np.array_equal(self.data_dict[name]['joints'][i], [-1, -1]):
        #         return False
        # return True

        return (not np.any(self.data_dict[name]['joints'] < 0))


    def _give_batch_name(self, batch_size=16, set='train'):
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
                break
        return list_file

    def _create_sets(self):
        """ Select Elements to feed training and validation set
        Args:
            validation_rate		: Percentage of validation data (in ]0,1[, don't waste time use 0.1)
        """
        self.train_set=[]
        self.valid_set=[]
        self.logger.info('Start set creation')

        self.train_set = [t for t in self.train_table if self._complete_sample(t)]
        self.valid_set = [v for v in self.valid_table if self._complete_sample(v)]

        self.logger.info('Set created')

        self.logger.info('--Training set: %i samples', len(self.train_set))
        self.logger.info('--Validation set: %i samples', len(self.valid_set))

    def generateSet(self, rand=False):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._create_train_table()
        self._create_valid_table()
        if rand:
            self._randomize()
        self._create_sets()

    # ---------------------------- Generating Methods --------------------------


    def _makeGaussian(self, height, width, sigma=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def _relative_joints(self, max_l, joints, to_size=64):
        """ Convert Absolute joint coordinates to crop box relative joint coordinates
        (Used to compute Heat Maps)
        Args:
            box			: Bounding Box
            padding	: Padding Added to the original Image
            to_size	: Heat Map wanted Size
        """
        new_j = np.copy(joints)
        new_j = new_j * to_size / (max_l + 0.0000001)
        return new_j.astype(np.int32)

    def _generate_hm(self, height, width, joints, maxlenght, weight):
        """ Generate a full Heap Map for every joints in an array
        Args:
            height			: Wanted Height for the Heat Map
            width			: Wanted Width for the Heat Map
            joints			: Array of Joints
            maxlenght		: Lenght of the Bounding Box
        """
        num_joints = joints.shape[0]
        hm = np.zeros((height, width, num_joints), dtype=np.float32)
        for i in range(num_joints):
            if not (np.array_equal(joints[i], [-1, -1])) and weight[i] == 1:
                s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
                hm[:, :, i] = self._makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm

    def _augment(self, img, hm, max_rotation=30):
        """ # TODO : IMPLEMENT DATA AUGMENTATION
        """
        if random.choice([0, 1]):
            r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            # img = 	transform.rotate(img, r_angle, preserve_range = True)
            img = transform.rotate(img, r_angle)
            hm = transform.rotate(hm, r_angle)
        return img, hm

    # ----------------------- Batch Generator ----------------------------------

    def _generator(self, batch_size=16, stacks=4, set='train', stored=False, normalize=True, debug=False):
        """ Create Generator for Training
        Args:
            batch_size	: Number of images per batch
            stacks			: Number of stacks/module in the network
            set				: Training/Testing/Validation set # TODO: Not implemented yet
            stored			: Use stored Value # TODO: Not implemented yet
            normalize		: True to return Image Value between 0 and 1
            _debug			: Boolean to test the computation time (/!\ Keep False)
        # Done : Optimize Computation time
            16 Images --> 1.3 sec (on i7 6700hq)
        """
        while True:
            if debug:
                t = time.time()
            train_img = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, stacks, 64, 64, len(self.joints_list)), np.float32)
            files = self._give_batch_name(batch_size=batch_size, set=set)
            for i, name in enumerate(files):
                if name[:-1] in self.images:
                    try:
                        img = self.open_img(name)
                        joints = self.data_dict[name]['joints']

                        hm = self._generate_hm(64, 64, joints, 64, weight)
                        img = img.astype(np.uint8)
                        img = scm.imresize(img, (256, 256))
                        img, hm = self._augment(img, hm)
                        hm = np.expand_dims(hm, axis=0)
                        hm = np.repeat(hm, stacks, axis=0)
                        if normalize:
                            train_img[i] = img.astype(np.float32) / 255
                        else:
                            train_img[i] = img.astype(np.float32)
                        train_gtmap[i] = hm
                    except:
                        i = i - 1
                else:
                    i = i - 1
            if debug:
                print('Batch : ', time.time() - t, ' sec.')
            yield train_img, train_gtmap

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train'):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        while True:
            train_img = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, stacks, 64, 64, len(self.joints_list)), np.float32)
            train_weights = np.zeros((batch_size, len(self.joints_list)), np.float32)
            i = 0
            while i < batch_size:
                # try:
                if sample_set == 'train':
                    name = random.choice(self.train_set)
                elif sample_set == 'valid':
                    name = random.choice(self.valid_set)
                joints = self.data_dict[name]['joints']
                weight = self.data_dict[name]['weights']
                train_weights[i] = weight
                img = self.open_img(name)
                joints = self._relative_joints(224, joints, to_size=64)
                hm = self._generate_hm(64, 64, joints, 64, weight)
                img = img.astype(np.uint8)
                img = scm.imresize(img, (256, 256))
                if sample_set == 'train':
                    img, hm = self._augment(img, hm)
                hm = np.expand_dims(hm, axis=0)
                hm = np.repeat(hm, stacks, axis=0)
                if normalize:
                    train_img[i] = img.astype(np.float32) / 255
                else:
                    train_img[i] = img.astype(np.float32)
                train_gtmap[i] = hm
                i = i + 1
            # except :
            # 	print('error file: ', name)
            yield train_img, train_gtmap, train_weights

    def generator(self, batchSize=16, stacks=4, norm=True, sample='train'):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch
            stacks 	 	: Stacks in HG model
            norm 	 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'valid' Default: 'train'
        """
        return self._aux_generator(batch_size=batchSize, stacks=stacks, normalize=norm, sample_set=sample)

    # ---------------------------- Image Reader --------------------------------
    def open_img(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        img = cv2.imread(os.path.join(self.img_dir, name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

    def plot_img(self, name, c=(0,0,255)):
        """ Plot an image
        Args:
            name	: Name of the Sample
            plot	: Library to use (cv2: OpenCV, plt: matplotlib)
        """
        joints=self.data_dict[name]['joints']
        # img = self.open_img(name, color='BGR')
        img = self.open_img(name, color='RGB')
        img = scm.imresize(img, (256, 256))
        # for j in range(self.Jnum):
        plt.imshow(img)
        plt.show()

            # cv2.circle(img, (int(joints[j, 0]), int(joints[j, 1])), 3,c,-1)
        # for e in self.edges:
        #     cv2.line(img, (int(joints[e[0], 0]), int(joints[e[0], 1])),
        #              (int(joints[e[1], 0]), int(joints[e[1], 1])), c,2)
        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # ------------------------------- PCK METHODS-------------------------------
    def pck_ready(self, idlh=3, idrs=12, testSet=None):
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
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts=True)
                    self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
            else:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][
                    id_rsho] == 1 and s in testSet:
                    self.pck_samples.append(s)
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts=True)
                    self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
        print('PCK PREPROCESS DONE: \n --Samples:', len(self.pck_samples), '\n --Num.Joints', self.total_joints)

    def plot_hotmap(self,hp,gt):
        hp_one = np.sum(hp, axis=4)
        gt_one = np.sum(gt, axis=4)
        for i in range(hp_one.shape[0]):
            plt.figure()
            plt.imshow(hp_one[i, 0, :, :], cmap='hot')
            plt.figure()
            plt.imshow(gt_one[i, 0, :, :], cmap='hot')
            plt.show()


def get_max_positions(hm, scale):
    hm_flattened = np.reshape(hm, (hm.shape[0] * hm.shape[1], hm.shape[2]))
    max_idx = np.argmax(hm_flattened, axis=0)
    max_r, max_c = np.unravel_index(max_idx, hm.shape[0:2])
    joints = np.array((max_c, max_r)).transpose().astype(np.float)
    joints *= scale

    return joints


def draw_result(img, pred, gt=None):

    dst = (img / np.max(np.abs(img)) * 255).astype(np.uint8)
    nPts = pred.shape[2]

    joints = get_max_positions(pred, scale = img.shape[0] / pred.shape[0])

    edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
             [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
             [6, 8], [8, 9]]
    for e in edges:
        cv2.line(dst, (int(joints[e[0], 0]), int(joints[e[0], 1])),
                 (int(joints[e[1], 0]), int(joints[e[1], 1])), (0,255,0), 2)
    for j in range(nPts):
        cv2.circle(dst, (int(joints[j, 0]), int(joints[j, 1])), 3, (0,0,255), -1)

    if gt is not None:
        nPts = gt.shape[2]
        joints = get_max_positions(gt, scale=img.shape[0] / gt.shape[0])
        for j in range(nPts):
            cv2.circle(dst, (int(joints[j, 0]), int(joints[j, 1])), 3, (255, 0, 0), -1)

    return dst


def color_heatmap(hm, resize_to=None, apply_sigmoid=False):
    ''' Apply hot colormap for visualization
    :param hm:
    :param resize_to:
    :return:
    '''

    if apply_sigmoid:
        hm = expit(hm)

    if hm.ndim == 3 and hm.shape[2] > 1:
        hm = np.sum(hm, axis=2)
        hm /= np.max(hm)

    hm_uint8 = (np.clip(hm * 255, 0, 255)).astype(np.uint8)
    dst = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_HOT)

    if resize_to is not None:
        dst = cv2.resize(dst, resize_to)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    return dst


if __name__ == '__main__':

    logging.config.fileConfig('logging.conf')

    data_gen=DataGenerator_human36()
    data_gen.generateSet()
    for name in data_gen.data_dict:
        data_gen.plot_img(name)
        # img = self.open_img(name)
        # joints = self.data_dict[name]['joints']
        # img = img.astype(np.uint8)
        # img = scm.imresize(img, (256, 256))
        joints = data_gen.data_dict[name]['joints']
        joints = data_gen._relative_joints(224, joints, to_size=64)
        weight = data_gen.data_dict[name]['weights']
        hm= data_gen._generate_hm(64, 64, joints, 64, weight)
        hm=np.sum(hm,axis=2)
        plt.imshow(hm, cmap='hot')
        plt.show()
    generator=data_gen._aux_generator(4,4,True,'train')
    img_train, gt_train, weight_train = next(generator)
    gt_train=np.sum(gt_train,axis=4)
    for i in range(img_train.shape[0]):
        plt.figure()
        plt.imshow(img_train[i,:,:,:]*255)
        plt.figure()
        plt.imshow(gt_train[0,i,:,:]*255,cmap='hot')
        plt.show()

    path=os.path.join(os.getcwd(),'/models/' ,'haha_0')
    print (path)

    ##
    # img = data_gen.open_img(name)
    #
    # hm_flattened = np.reshape(hm, (hm.shape[0]*hm.shape[1], hm.shape[2]))
    # max_idx = np.argmax(hm_flattened, axis=0)
    # max_r, max_c = np.unravel_index(max_idx, hm.shape[0:2])
    # joints = np.array((max_c, max_r)).transpose().astype(np.float)
    # scale = img.shape[0] / hm.shape[0]
    # joints *= scale
    #
    # edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
    #          [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
    #          [6, 8], [8, 9]]
    #
    # for e in edges:
    #     cv2.line(img, (int(joints[e[0], 0]), int(joints[e[0], 1])),
    #              (int(joints[e[1], 0]), int(joints[e[1], 1])), (0,255,0), 2)
    # for j in range(len(max_r)):
    #     cv2.circle(img, (int(joints[j, 0]), int(joints[j, 1])), 3, (0,0,255), -1)
    #
    # cv2.imshow('', img)
    # cv2.waitKey()
