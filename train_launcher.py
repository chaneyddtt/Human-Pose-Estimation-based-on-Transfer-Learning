"""
TRAIN LAUNCHER 

"""

import configparser
from hourglass_gan import HourglassModel_gan
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
from datagen_human36 import DataGenerator_human36
import argparse
import logging, logging.config

parse = argparse.ArgumentParser()

# argparse
parse.add_argument("--network", help="choose a network", default='hourglass_gan', type=str)
parse.add_argument("--gpu", help="Select GPU (default: 0)", default=0, type=int)
args = parse.parse_args()

# logging
logging.config.fileConfig('logging.conf')

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



if __name__ == '__main__':
	print('--Parsing Config File')
	params = process_config('config.cfg')
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
							   name=params['name'], logdir=params['log_dir'],
                               w_loss=params['weighted_loss'],
							   joints=params['joint_list'], gpu=args.gpu)
	else:
		model = HourglassModel_gan(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
							   nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],
							   drop_rate=params['dropout_rate'], lear_rate=params['learning_rate'],
							   decay=params['learning_rate_decay'], decay_step=params['decay_step'],
							   dataset_source=dataset_source, dataset_target=dataset_target,
							   name=params['name'], logdir=params['log_dir'],
							   w_loss=params['weighted_loss'],
							   joints=params['joint_list'], gpu=args.gpu)
	model.generate_model()
	# modelPath='/home/lichen/pose_estimation/hourglasstensorlfow/hourglassModel_tiny_1stack/hg_refined_200_200'
	model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],load=None)
