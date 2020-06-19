from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_mtl_concat import Generic_MIL_MTL_Dataset, save_splits
import h5py
from utils.eval_utils_mtl_concat import *

# Training settings
parser = argparse.ArgumentParser(description='TOAD Evaluation Script')
parser.add_argument('--data_root_dir', type=str, help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=1, help='number of folds (default: 1)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['dummy_mtl_concat'])

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoding_size = 1024

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'drop_out': args.drop_out,
            'micro_avg': args.micro_average}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)


if args.task == 'dummy_mtl_concat':
    args.n_classes=18
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/dummy_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir,'DATASET_DIR'),
                            shuffle = False, 
                            print_info = True,
                            label_dicts = [{'Lung':0, 'Breast':1, 'Colorectal':2, 'Ovarian':3, 
                                                                'Pancreatic':4, 'Adrenal':5, 
                                                                'Melanoma':6, 'Prostate':7, 'Renal':8, 'Bladder':9, 
                                                                'Esophagastric':10,  'Thyroid':11,
                                                                'Head Neck':12,  'Glioma':13, 
                                                                'Germ Cell Tumor':14, 'Endometrial': 15, 'Cervix': 16, 'Liver': 17},
                                            {'Primary':0,  'Metastatic':1},
                                            {'F':0, 'M':1}],
                            label_cols = ['label', 'site', 'sex'],
                            patient_strat= False)

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":

    all_cls_auc = []
    all_cls_acc = []
    all_site_auc = []
    all_site_acc = []
    all_cls_top3_acc = []
    all_cls_top5_acc = []
    
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
            csv_path = None
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]

        model, results_dict = eval(split_dataset, args, ckpt_paths[ckpt_idx])

        for cls_idx in range(len(results_dict['cls_aucs'])):
            print('class {} auc: {}'.format(cls_idx, results_dict['cls_aucs'][cls_idx]))

        all_cls_auc.append(results_dict['cls_auc'])
        all_cls_acc.append(1-results_dict['cls_test_error'])
        all_site_auc.append(results_dict['site_auc'])
        all_site_acc.append(1-results_dict['site_test_error'])
        all_cls_top3_acc.append(results_dict['top3_acc'])
        all_cls_top5_acc.append(results_dict['top5_acc'])
        df = results_dict['df']
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)


    df_dict = {'folds': folds, 'cls_test_auc': all_cls_auc, 'cls_test_acc': all_cls_acc, 'cls_top3_acc': all_cls_top3_acc, 'cls_top5_acc': all_cls_top5_acc,
                'site_test_auc': all_site_auc, 'site_test_acc': all_site_acc}

    final_df = pd.DataFrame(df_dict)
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
