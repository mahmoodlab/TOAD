import pdb
import os
import pandas as pd
from datasets.dataset_mtl_concat import Generic_WSI_MTL_Dataset, Generic_MIL_MTL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= -1,
										help='fraction of labels (default: [1.0])')
parser.add_argument('--seed', type=int, default=1,
										help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
										help='number of splits (default: 10)')
parser.add_argument('--hold_out_test', action='store_true', default=False,
										help='fraction to hold out (default: 0)')
parser.add_argument('--split_code', type=str, default=None)
parser.add_argument('--task', type=str, choices=['dummy_mtl_concat'])

args = parser.parse_args()

if args.task == 'dummy_mtl_concat':
    args.n_classes=18
    dataset = Generic_WSI_MTL_Dataset(csv_path = 'dataset_csv/dummy_dataset.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts = [{'Lung':0, 'Breast':1, 'Colorectal':2, 'Ovarian':3, 
                                                                'Pancreatic':4, 'Adrenal':5, 
                                                                'Skin':6, 'Prostate':7, 'Renal':8, 'Bladder':9, 
                                                                'Esophagagostric':10,  'Thyroid':11,
                                                                'Head Neck':12,  'Glioma':13, 
                                                                'Germ Cell':14, 'Endometrial': 15, 'Cervix': 16, 'Liver': 17},
                                            {'Primary':0,  'Metastatic':1},
                                            {'F':0, 'M':1}],
                            label_cols = ['label', 'site', 'sex'],
                            patient_strat= False)

         
else:
	raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.floor(num_slides_cls * 0.1).astype(int)
test_num = np.floor(num_slides_cls * 0.2).astype(int)

print(val_num)
print(test_num)

if __name__ == '__main__':
		if args.label_frac > 0:
			label_fracs = [args.label_frac]
		else:
			label_fracs = [1.0]

		if args.hold_out_test:
			custom_test_ids = dataset.sample_held_out(test_num=test_num)
		else:
			custom_test_ids = None
		for lf in label_fracs:
			if args.split_code is not None:
				split_dir = 'splits/'+ str(args.split_code) + '_{}'.format(int(lf * 100))
			else:
				split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
			
			dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf, custom_test_ids=custom_test_ids)

			os.makedirs(split_dir, exist_ok=True)
			for i in range(args.k):
				if dataset.split_gen is None:
					ids = []
					for split in ['train', 'val', 'test']:
						ids.append(dataset.get_split_from_df(pd.read_csv(os.path.join(split_dir, 'splits_{}.csv'.format(i))), split_key=split, return_ids_only=True))
					
					dataset.train_ids = ids[0]
					dataset.val_ids = ids[1]
					dataset.test_ids = ids[2]
				else:
					dataset.set_splits()

				descriptor_df = dataset.test_split_gen(return_descriptor=True)
				descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))
				
				splits = dataset.return_splits(from_id=True)
				save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
				save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
				



