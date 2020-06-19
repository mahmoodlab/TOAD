from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_MTL_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dicts = [{}, {}, {}],
		ignore=[],
		patient_strat=False,
		label_cols = ['label', 'site', 'sex'],
		patient_voting = 'max',
		multi_site = False,
		filter_dict = {},
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
			patient_voting (string): Rule for deciding the patient-level label
		"""
		self.custom_test_ids = None
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		self.label_cols = label_cols

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)

		if multi_site:
			label_dicts[0] = self.init_multi_site_label_dict(slide_data, label_dicts[0])

		self.label_dicts = label_dicts
		self.num_classes=[len(set(label_dict.values())) for label_dict in self.label_dicts]

		slide_data = self.df_prep(slide_data, self.label_dicts, ignore, self.label_cols, multi_site)
		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes[0])]		
		for i in range(self.num_classes[0]):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes[0])]
		for i in range(self.num_classes[0]):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def init_multi_site_label_dict(slide_data, label_dict):
		print('initiating multi-source label dictionary')
		sites = np.unique(slide_data['site'].values)
		multi_site_dict = {}
		num_classes = len(label_dict)
		for key, val in label_dict.items():
			for idx, site in enumerate(sites):
				site_key = (key, site)
				site_val = val+idx*num_classes
				multi_site_dict.update({site_key:site_val})
				print('{} : {}'.format(site_key, site_val))
		return multi_site_dict

	@staticmethod
	def filter_df(df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	@staticmethod
	def df_prep(data, label_dicts, ignore, label_cols, multi_site=False):
		if label_cols[0] != 'label':
			data['label'] = data[label_cols[0]].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			if multi_site:
				site = data.loc[i, 'site']
				key = (key, site)
			data.at[i, 'label'] = label_dicts[0][key]

		for idx, (label_dict, label_col) in enumerate(zip(label_dicts[1:], label_cols[1:])):
			print(label_dict, label_col)
			data[label_col] = data[label_col].map(label_dict)

		return data

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):

		for task in range(len(self.label_dicts)):
			print('task: ', task)
			print("label column: {}".format(self.label_cols[task]))
			print("label dictionary: {}".format(self.label_dicts[task]))
			print("number of classes: {}".format(self.num_classes[task]))
			print("slide-level counts: ", '\n', self.slide_data[self.label_cols[task]].value_counts(sort = False))
		
		for i in range(self.num_classes[0]):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def sample_held_out(self, test_num = (40, 40)):

		test_ids = []
		np.random.seed(self.seed) #fix seed
		
		if self.patient_strat:
			cls_ids = self.patient_cls_ids
		else:
			cls_ids = self.slide_cls_ids

		for c in range(len(test_num)):
			test_ids.extend(np.random.choice(cls_ids[c], test_num[c], replace = False)) # validation ids

		if self.patient_strat:
			slide_ids = [] 
			for idx in test_ids:
				case_id = self.patient_data['case_id'][idx]
				slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
				slide_ids.extend(slide_indices)

			return slide_ids
		else:
			return test_ids

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): 
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits=None, split_key='train', split=None):
		if split is None:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].dropna().reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].dropna().reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):


		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)
			
			else:
				test_split = None
			
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path)
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids, task):
		if task > 0:
			return self.slide_data[self.label_cols[task]][ids]
		else:
			return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):
		if return_descriptor:
			dfs = []
			for task in range(len(self.label_dicts)):
				index = [list(self.label_dicts[task].keys())[list(self.label_dicts[task].values()).index(i)] for i in range(self.num_classes[task])]
				columns = ['train', 'val', 'test']
				df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)
				dfs.append(df)
		



		for task in range(len(self.label_dicts)):
			count = len(self.train_ids)
			print('\nnumber of training samples: {}'.format(count))
			index = [list(self.label_dicts[task].keys())[list(self.label_dicts[task].values()).index(i)] for i in range(self.num_classes[task])]
			labels = self.getlabel(self.train_ids, task)
			unique, counts = np.unique(labels, return_counts=True)
			missing_classes = np.setdiff1d(np.arange(self.num_classes[task]), unique)
			unique = np.append(unique, missing_classes)
			counts = np.append(counts, np.full(len(missing_classes), 0))
			inds = unique.argsort()
			counts = counts[inds]
			for u in range(len(unique)):
				print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
				if return_descriptor:
					dfs[task].loc[index[u], 'train'] = counts[u]
		
			count = len(self.val_ids)
			print('\nnumber of val samples: {}'.format(count))
			labels = self.getlabel(self.val_ids, task)
			unique, counts = np.unique(labels, return_counts=True)
			missing_classes = np.setdiff1d(np.arange(self.num_classes[task]), unique)
			unique = np.append(unique, missing_classes)
			counts = np.append(counts, np.full(len(missing_classes), 0))
			inds = unique.argsort()
			counts = counts[inds]
			for u in range(len(unique)):
				print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
				if return_descriptor:
					dfs[task].loc[index[u], 'val'] = counts[u]

			count = len(self.test_ids)
			print('\nnumber of test samples: {}'.format(count))
			labels = self.getlabel(self.test_ids, task)
			unique, counts = np.unique(labels, return_counts=True)
			missing_classes = np.setdiff1d(np.arange(self.num_classes[task]), unique)
			unique = np.append(unique, missing_classes)
			counts = np.append(counts, np.full(len(missing_classes), 0))
			inds = unique.argsort()
			counts = counts[inds]
			for u in range(len(unique)):
				print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
				if return_descriptor:
					dfs[task].loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			df = pd.concat(dfs, axis=0) 
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


# class Generic_MIL_MTL_Dataset(Generic_WSI_MTL_Dataset):
# 	def __init__(self,
# 		data_dir, 
# 		**kwargs):
# 		super(Generic_MIL_MTL_Dataset, self).__init__(**kwargs)
# 		self.data_dir = data_dir
# 		self.use_h5 = False

# 	def load_from_h5(self, toggle):
# 		self.use_h5 = toggle

# 	def __getitem__(self, idx):
# 		slide_id = self.slide_data['slide_id'][idx]
# 		label = self.slide_data['label'][idx]
# 		site = self.slide_data[self.label_cols[1]][idx]
# 		if type(self.data_dir) == dict:
# 			source = self.slide_data['source'][idx]
# 			data_dir = self.data_dir[source]
# 		else:
# 			data_dir = self.data_dir

# 		if not self.use_h5:
# 			if self.data_dir:
# 				full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
# 				features = torch.load(full_path)
# 				return features, label, site
			
# 			else:
# 				return slide_id, label, site

# 		else:
# 			full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
# 			with h5py.File(full_path,'r') as hdf5_file:
# 				features = hdf5_file['features'][:]
# 				coords = hdf5_file['coords'][:]

# 			features = torch.from_numpy(features)
# 			return features, label, site, coords

class Generic_MIL_Fusion_Dataset(Generic_WSI_MTL_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
		super(Generic_MIL_Fusion_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		sex = self.slide_data[self.label_cols[2]][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
				features = torch.load(full_path)
				return features, label, sex
			
			else:
				return slide_id, label, sex

		else:
			full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, sex, coords



class Generic_Split(Generic_MIL_Fusion_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2, label_cols=None):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes[0])]
		self.label_cols = label_cols
		for i in range(self.num_classes[0]):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
		

class Generic_WSI_Inference_Dataset(Dataset):
	def __init__(self,
		data_dir,
		csv_path = None,
		print_info = True,
		label_dict = {'F': 0, 'M': 1}
		):
		self.data_dir = data_dir
		self.print_info = print_info

		if csv_path is not None:
			data = pd.read_csv(csv_path)
			self.slide_data = data
			self.slide_data['sex'] = self.slide_data['sex'].map(label_dict)

		if print_info:
			print('total number of slides to infer: ', len(self.slide_data))

	def __len__(self):
		return len(self.slide_data)

	def __getitem__(self, idx):
		slide_file = self.slide_data['slide_id'][idx]+'.pt'
		sex = self.slide_data['sex']
		full_path = os.path.join(self.data_dir, 'pt_files',slide_file)
		features = torch.load(full_path)
		return features, sex
