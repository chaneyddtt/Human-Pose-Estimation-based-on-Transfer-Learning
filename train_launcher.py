"""
TRAIN LAUNCHER 

"""
from __future__ import print_function

import configparser
import datetime
from hourglass_gan import HourglassModel_gan
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
from datagen_human36 import DataGenerator_human36
import argparse
import logging, logging.config
import os
import shutil

def process_config(conf_file):
    """
    """
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'DataSetHG':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Train':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Validation':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Saver':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params

def make_dir_if_not_exist(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

# argparse
parse = argparse.ArgumentParser()
parse.add_argument("--network", help="choose a network", default='hourglass_gan', type=str)
parse.add_argument("--gpu", help="Select GPU (default: 0)", default=0, type=int)
parse.add_argument("--name", help="Name of run (used to name tf.Summary)", default='', type=str)
args = parse.parse_args()

# Parse config
params = process_config('config.cfg')

# Set up logging, logs will also be written to file in logdir
dn_prefix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
args.name = args.name.strip('\'')
if len(args.name) > 0:
    dn_prefix += ' (' + args.name + ')'
logdir = os.path.join(params['log_dir'], dn_prefix)
make_dir_if_not_exist(logdir)

logging.config.fileConfig('logging.conf')
logger = logging.getLogger()
fileHandler = logging.FileHandler("{0}/log.txt".format(logdir))
logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
logger.info("Logs will be written to %s" % logdir)
shutil.copy('config.cfg', logdir)


if __name__ == '__main__':

    #
    # print('--Creating Dataset')
    # dataset = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'], remove_joints=params['remove_joints'])
    # dataset._create_train_table()
    # dataset._randomize()
    # dataset._create_sets()
    # generator=dataset._aux_generator(16,4,True,'train')
    #
    # model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],  w_loss=params['weighted_loss'] , joints= params['joint_list'])
    # model.generate_model()
    # model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset = None)


    dataset_source = DataGenerator_human36()
    dataset_target = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'], remove_joints=params['remove_joints'])
    # model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
    # 					   nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],
    # 					   training=True, drop_rate=params['dropout_rate'], lear_rate=params['learning_rate'],
    # 					   decay=params['learning_rate_decay'], decay_step=params['decay_step'],
    # 					   dataset_source=dataset_source,dataset_target=dataset_target,
    # 					   name=params['name'], logdir_train=params['log_dir_train'],
    # 					   logdir_test=params['log_dir_test'], w_loss=params['weighted_loss'],
    # 					   joints=params['joint_list'])
    # model.generate_model()
    # model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],load=None)

    if args.network=='hourglass_tiny':
        model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
                               nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],
                               drop_rate=params['dropout_rate'], lear_rate=params['learning_rate'],
                               decay=params['learning_rate_decay'], decay_step=params['decay_step'],
                               dataset_source=dataset_source,dataset_target=dataset_target,
                               name=params['name'], logdir=logdir,
                               w_loss=params['weighted_loss'],
                               joints=params['joint_list'], gpu=args.gpu)
    elif args.network == 'hourglass_gan':
        model = HourglassModel_gan(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
                               nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],
                               drop_rate=params['dropout_rate'], lear_rate=params['learning_rate'],
                               decay=params['learning_rate_decay'], decay_step=params['decay_step'],
                               dataset_source=dataset_source, dataset_target=dataset_target,
                               name=params['name'], logdir=logdir,
                               w_loss=params['weighted_loss'],
                               joints=params['joint_list'], gpu=args.gpu)
    else:
        raise NotImplementedError()

    model.generate_model()
    # modelPath='/home/lichen/pose_estimation/hourglasstensorlfow/hourglassModel_tiny_1stack/hg_refined_200_200'
    model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],load=None)
