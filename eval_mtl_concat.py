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
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default='/media/fedshyvana/ssd1',
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam', 'mil', 'attention_mil', 'clam_simple'], default='clam', 
                    help='type of model (default: clam)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--calc_features', action='store_true', default=False, 
                    help='calculate features for pca/tsne')
parser.add_argument('--summarize', action='store_true', default=False, 
                    help='summarize')
parser.add_argument('--return_topk', type=int, default=-1, 
                    help='calculate features for pca/tsne')
parser.add_argument('--k', type=int, default=1, help='number of folds (default: 1)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--infer_only', action='store_true', default=False) 
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--merge_splits', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['study_v2_mtl_sex', 'study_v2_mtl_sex_met',  'study_v2_mtl_sex_primary', 'study_v2_mtl_sex_osh',
 'study_adeno_mtl_sex',  'study_squamous_mtl_sex', 'study_v2_mtl_sex_liver_site', 'study_v2_mtl_sex_lymph_site'])

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoding_size = 1024

if not args.summarize:
    args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
    args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'attention_scores'), exist_ok=True)

    if args.splits_dir is None:
        args.splits_dir = args.models_dir

    assert os.path.isdir(args.models_dir)
    assert os.path.isdir(args.splits_dir)

    settings = {'task': args.task,
                'split': args.split,
                'save_dir': args.save_dir, 
                'models_dir': args.models_dir,
                'model_type': args.model_type,
                'drop_out': args.drop_out,
                'model_size': args.model_size,
                'micro': args.micro_average}

    with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    print(settings)


if args.task == 'study_v2_mtl_sex_osh':
    args.n_classes = 18
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/study_v2_osh_clean.csv',
                        data_dir= {'Oncopanel Primary':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                   'Oncopanel Metastatic':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                   'TCGA-KIRC':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-KICH':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-KIRP':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-LGG':os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                    'TCGA-GBM':os.path.join(args.data_root_dir,'tcga_gbm_20x_features'),
                                    'TCGA-PRAD':os.path.join(args.data_root_dir,'tcga_prostate_20x_features'),
                                    'TCGA-PAAD':os.path.join(args.data_root_dir,'tcga_pancreas_20x_features'),
                                    'TCGA-HNSC':os.path.join(args.data_root_dir,'tcga_head_and_neck_20x_features'),
                                    'TCGA-LUAD':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                    'TCGA-LUSC':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                    'TCGA-BRCA':os.path.join(args.data_root_dir,'tcga_breast_20x_features'),
                                    'TCGA-ACC':os.path.join(args.data_root_dir,'tcga_adrenal_20x_features'),
                                    'TCGA-COAD':os.path.join(args.data_root_dir,'tcga_colorectal_20x_features'),
                                    'TCGA-CESC':os.path.join(args.data_root_dir,'tcga_cervical_20x_features'),
                                    'TCGA-LIHC':os.path.join(args.data_root_dir,'tcga_liver_20x_features'),
                                    'TCGA-OV':os.path.join(args.data_root_dir,'tcga_ovary_20x_features'),
                                    'TCGA-SKCM':os.path.join(args.data_root_dir,'tcga_skin_20x_features'),
                                    'TCGA-ESCA':os.path.join(args.data_root_dir,'tcga_esophagus_20x_features'),
                                    'TCGA-STAD':os.path.join(args.data_root_dir,'tcga_stomach_20x_features'),
                                    'TCGA-TGCT':os.path.join(args.data_root_dir,'tcga_germ_cell_20x_features'),
                                    'TCGA-UCEC':os.path.join(args.data_root_dir,'tcga_endometrial_20x_features'),
                                    'TCGA-BLCA':os.path.join(args.data_root_dir,'tcga_bladder_20x_features'),
                                    'TCGA-THCA':os.path.join(args.data_root_dir,'tcga_thyroid_20x_features'),
                                    'TCGA-READ':os.path.join(args.data_root_dir,'tcga_rectum_20x_features')},
                        shuffle = False, 
                        print_info = True,
                        label_dicts = [{'Lung':0, 'Breast':1, 'Colorectal':2, 'Ovarian':3, 
                                                            'Pancreatic':4, 'Adrenal':5, 
                                                            'Melanoma':6, 'Prostate':7, 'Renal':8, 'Bladder':9, 
                                                            'Esophagastric':10,  'Thyroid':11,
                                                            'Head Neck':12,  'Glioma':13, 
                                                            'Germ Cell Tumor':14, 'Endometrial': 15, 'Cervix': 16, 'Liver': 17},
                                        {'Primary':0, 'Metastatic Recurrence':1, 'TCGA Primary Tumor':0, 'TCGA Metastatic':1},
                                        {'F': 0, 'M': 1}],
                        label_cols = ['label', 'site', 'sex'],
                        filter_dict = {},
                        patient_strat= False,
                        ignore=[])

elif args.task == 'study_v2_mtl_sex_met':
    args.n_classes = 18
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/study_v2_no_osh_clean.csv',
                        data_dir= {'Oncopanel Primary':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                   'Oncopanel Metastatic':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                   'TCGA-KIRC':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-KICH':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-KIRP':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-LGG':os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                    'TCGA-GBM':os.path.join(args.data_root_dir,'tcga_gbm_20x_features'),
                                    'TCGA-PRAD':os.path.join(args.data_root_dir,'tcga_prostate_20x_features'),
                                    'TCGA-PAAD':os.path.join(args.data_root_dir,'tcga_pancreas_20x_features'),
                                    'TCGA-HNSC':os.path.join(args.data_root_dir,'tcga_head_and_neck_20x_features'),
                                    'TCGA-LUAD':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                    'TCGA-LUSC':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                    'TCGA-BRCA':os.path.join(args.data_root_dir,'tcga_breast_20x_features'),
                                    'TCGA-ACC':os.path.join(args.data_root_dir,'tcga_adrenal_20x_features'),
                                    'TCGA-COAD':os.path.join(args.data_root_dir,'tcga_colorectal_20x_features'),
                                    'TCGA-CESC':os.path.join(args.data_root_dir,'tcga_cervical_20x_features'),
                                    'TCGA-LIHC':os.path.join(args.data_root_dir,'tcga_liver_20x_features'),
                                    'TCGA-OV':os.path.join(args.data_root_dir,'tcga_ovary_20x_features'),
                                    'TCGA-SKCM':os.path.join(args.data_root_dir,'tcga_skin_20x_features'),
                                    'TCGA-ESCA':os.path.join(args.data_root_dir,'tcga_esophagus_20x_features'),
                                    'TCGA-STAD':os.path.join(args.data_root_dir,'tcga_stomach_20x_features'),
                                    'TCGA-TGCT':os.path.join(args.data_root_dir,'tcga_germ_cell_20x_features'),
                                    'TCGA-UCEC':os.path.join(args.data_root_dir,'tcga_endometrial_20x_features'),
                                    'TCGA-BLCA':os.path.join(args.data_root_dir,'tcga_bladder_20x_features'),
                                    'TCGA-THCA':os.path.join(args.data_root_dir,'tcga_thyroid_20x_features'),
                                    'TCGA-READ':os.path.join(args.data_root_dir,'tcga_rectum_20x_features')},
                        shuffle = False, 
                        print_info = True,
                        label_dicts = [{'Lung':0, 'Breast':1, 'Colorectal':2, 'Ovarian':3, 
                                                            'Pancreatic':4, 'Adrenal':5, 
                                                            'Melanoma':6, 'Prostate':7, 'Renal':8, 'Bladder':9, 
                                                            'Esophagastric':10,  'Thyroid':11,
                                                            'Head Neck':12,  'Glioma':13, 
                                                            'Germ Cell Tumor':14, 'Endometrial': 15, 'Cervix': 16, 'Liver': 17},
                                        {'Primary':0, 'Metastatic Recurrence':1, 'TCGA Primary Tumor':0, 'TCGA Metastatic':1},
                                        {'F': 0, 'M': 1}],
                        label_cols = ['label', 'site', 'sex'],
                        filter_dict = {'site': ['Metastatic Recurrence', 'TCGA Metastatic']},
                        patient_strat= False,
                        ignore=[])

elif args.task == 'study_v2_mtl_sex_primary':
    args.n_classes = 18
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/study_v2_no_osh_clean.csv',
                        data_dir= {'Oncopanel Primary':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                   'Oncopanel Metastatic':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                   'TCGA-KIRC':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-KICH':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-KIRP':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-LGG':os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                    'TCGA-GBM':os.path.join(args.data_root_dir,'tcga_gbm_20x_features'),
                                    'TCGA-PRAD':os.path.join(args.data_root_dir,'tcga_prostate_20x_features'),
                                    'TCGA-PAAD':os.path.join(args.data_root_dir,'tcga_pancreas_20x_features'),
                                    'TCGA-HNSC':os.path.join(args.data_root_dir,'tcga_head_and_neck_20x_features'),
                                    'TCGA-LUAD':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                    'TCGA-LUSC':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                    'TCGA-BRCA':os.path.join(args.data_root_dir,'tcga_breast_20x_features'),
                                    'TCGA-ACC':os.path.join(args.data_root_dir,'tcga_adrenal_20x_features'),
                                    'TCGA-COAD':os.path.join(args.data_root_dir,'tcga_colorectal_20x_features'),
                                    'TCGA-CESC':os.path.join(args.data_root_dir,'tcga_cervical_20x_features'),
                                    'TCGA-LIHC':os.path.join(args.data_root_dir,'tcga_liver_20x_features'),
                                    'TCGA-OV':os.path.join(args.data_root_dir,'tcga_ovary_20x_features'),
                                    'TCGA-SKCM':os.path.join(args.data_root_dir,'tcga_skin_20x_features'),
                                    'TCGA-ESCA':os.path.join(args.data_root_dir,'tcga_esophagus_20x_features'),
                                    'TCGA-STAD':os.path.join(args.data_root_dir,'tcga_stomach_20x_features'),
                                    'TCGA-TGCT':os.path.join(args.data_root_dir,'tcga_germ_cell_20x_features'),
                                    'TCGA-UCEC':os.path.join(args.data_root_dir,'tcga_endometrial_20x_features'),
                                    'TCGA-BLCA':os.path.join(args.data_root_dir,'tcga_bladder_20x_features'),
                                    'TCGA-THCA':os.path.join(args.data_root_dir,'tcga_thyroid_20x_features'),
                                    'TCGA-READ':os.path.join(args.data_root_dir,'tcga_rectum_20x_features')},
                        shuffle = False, 
                        print_info = True,
                        label_dicts = [{'Lung':0, 'Breast':1, 'Colorectal':2, 'Ovarian':3, 
                                                            'Pancreatic':4, 'Adrenal':5, 
                                                            'Melanoma':6, 'Prostate':7, 'Renal':8, 'Bladder':9, 
                                                            'Esophagastric':10,  'Thyroid':11,
                                                            'Head Neck':12,  'Glioma':13, 
                                                            'Germ Cell Tumor':14, 'Endometrial': 15, 'Cervix': 16, 'Liver': 17},
                                        {'Primary':0, 'Metastatic Recurrence':1, 'TCGA Primary Tumor':0, 'TCGA Metastatic':1},
                                        {'F': 0, 'M': 1}],
                        label_cols = ['label', 'site', 'sex'],
                        filter_dict = {'site': ['Primary', 'TCGA Primary Tumor']},
                        patient_strat= False,
                        ignore=[])

elif args.task == 'study_v2_mtl_sex':
    args.n_classes = 18
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/study_v2_no_osh_clean.csv',
                        data_dir= {'Oncopanel Primary':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                   'Oncopanel Metastatic':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                   'TCGA-KIRC':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-KICH':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-KIRP':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                    'TCGA-LGG':os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                    'TCGA-GBM':os.path.join(args.data_root_dir,'tcga_gbm_20x_features'),
                                    'TCGA-PRAD':os.path.join(args.data_root_dir,'tcga_prostate_20x_features'),
                                    'TCGA-PAAD':os.path.join(args.data_root_dir,'tcga_pancreas_20x_features'),
                                    'TCGA-HNSC':os.path.join(args.data_root_dir,'tcga_head_and_neck_20x_features'),
                                    'TCGA-LUAD':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                    'TCGA-LUSC':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                    'TCGA-BRCA':os.path.join(args.data_root_dir,'tcga_breast_20x_features'),
                                    'TCGA-ACC':os.path.join(args.data_root_dir,'tcga_adrenal_20x_features'),
                                    'TCGA-COAD':os.path.join(args.data_root_dir,'tcga_colorectal_20x_features'),
                                    'TCGA-CESC':os.path.join(args.data_root_dir,'tcga_cervical_20x_features'),
                                    'TCGA-LIHC':os.path.join(args.data_root_dir,'tcga_liver_20x_features'),
                                    'TCGA-OV':os.path.join(args.data_root_dir,'tcga_ovary_20x_features'),
                                    'TCGA-SKCM':os.path.join(args.data_root_dir,'tcga_skin_20x_features'),
                                    'TCGA-ESCA':os.path.join(args.data_root_dir,'tcga_esophagus_20x_features'),
                                    'TCGA-STAD':os.path.join(args.data_root_dir,'tcga_stomach_20x_features'),
                                    'TCGA-TGCT':os.path.join(args.data_root_dir,'tcga_germ_cell_20x_features'),
                                    'TCGA-UCEC':os.path.join(args.data_root_dir,'tcga_endometrial_20x_features'),
                                    'TCGA-BLCA':os.path.join(args.data_root_dir,'tcga_bladder_20x_features'),
                                    'TCGA-THCA':os.path.join(args.data_root_dir,'tcga_thyroid_20x_features'),
                                    'TCGA-READ':os.path.join(args.data_root_dir,'tcga_rectum_20x_features')},
                        shuffle = False, 
                        print_info = True,
                        label_dicts = [{'Lung':0, 'Breast':1, 'Colorectal':2, 'Ovarian':3, 
                                                            'Pancreatic':4, 'Adrenal':5, 
                                                            'Melanoma':6, 'Prostate':7, 'Renal':8, 'Bladder':9, 
                                                            'Esophagastric':10,  'Thyroid':11,
                                                            'Head Neck':12,  'Glioma':13, 
                                                            'Germ Cell Tumor':14, 'Endometrial': 15, 'Cervix': 16, 'Liver': 17},
                                        {'Primary':0, 'Metastatic Recurrence':1, 'TCGA Primary Tumor':0, 'TCGA Metastatic':1},
                                        {'F': 0, 'M': 1}],
                        label_cols = ['label', 'site', 'sex'],
                        filter_dict = {},
                        patient_strat= False,
                        ignore=[])


elif args.task == 'study_v2_mtl_sex_liver_site':
    args.n_classes=18
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/study_v2_no_osh_clean.csv',
                                        data_dir= os.path.join(args.data_root_dir, 'oncopanel_met_primary'),
                                        shuffle = False, 
                                        print_info = True,
                                        label_dicts = [{'Lung':0, 'Breast':1, 'Colorectal':2, 'Ovarian':3, 
                                                            'Pancreatic':4, 'Adrenal':5, 
                                                            'Melanoma':6, 'Prostate':7, 'Renal':8, 'Bladder':9, 
                                                            'Esophagastric':10,  'Thyroid':11,
                                                            'Head Neck':12,  'Glioma':13, 
                                                            'Germ Cell Tumor':14, 'Endometrial': 15, 'Cervix': 16, 'Liver': 17},
                                                      {'Primary':0, 'Metastatic Recurrence':1, 'TCGA Primary Tumor':0, 'TCGA Metastatic':1},
                                                      {'F':0, 'M':1}],
                                        label_cols = ['label', 'site', 'sex'],
                                        patient_strat= False,
                                        filter_dict={'label': ['Colorectal', 'Pancreatic', 'Breast', 'Lung']},
                                        ignore=[])

elif args.task == 'study_v2_mtl_sex_lymph_site':
    args.n_classes=18
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/study_v2_no_osh_clean.csv',
                                        data_dir= os.path.join(args.data_root_dir, 'oncopanel_met_primary'),
                                        shuffle = False, 
                                        print_info = True,
                                        label_dicts = [{'Lung':0, 'Breast':1, 'Colorectal':2, 'Ovarian':3, 
                                                            'Pancreatic':4, 'Adrenal':5, 
                                                            'Melanoma':6, 'Prostate':7, 'Renal':8, 'Bladder':9, 
                                                            'Esophagastric':10,  'Thyroid':11,
                                                            'Head Neck':12,  'Glioma':13, 
                                                            'Germ Cell Tumor':14, 'Endometrial': 15, 'Cervix': 16, 'Liver': 17},
                                                      {'Primary':0, 'Metastatic Recurrence':1, 'TCGA Primary Tumor':0, 'TCGA Metastatic':1},
                                                      {'F':0, 'M':1}],
                                        label_cols = ['label', 'site', 'sex'],
                                        patient_strat= False,
                                        filter_dict={'label': ['Lung', 'Breast', 'Melanoma', 'Thyroid']},
                                        ignore=[])


elif args.task == 'study_adeno_mtl_sex':
    args.n_classes = 5
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/study_adeno_clean.csv',
                                  data_dir= {'Oncopanel Primary':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                       'Oncopanel Metastatic':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                       'TCGA-KIRC':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                        'TCGA-KICH':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                        'TCGA-KIRP':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                        'TCGA-LGG':os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                        'TCGA-GBM':os.path.join(args.data_root_dir,'tcga_gbm_20x_features'),
                                        'TCGA-PRAD':os.path.join(args.data_root_dir,'tcga_prostate_20x_features'),
                                        'TCGA-PAAD':os.path.join(args.data_root_dir,'tcga_pancreas_20x_features'),
                                        'TCGA-HNSC':os.path.join(args.data_root_dir,'tcga_head_and_neck_20x_features'),
                                        'TCGA-LUAD':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                        'TCGA-LUSC':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                        'TCGA-BRCA':os.path.join(args.data_root_dir,'tcga_breast_20x_features'),
                                        'TCGA-ACC':os.path.join(args.data_root_dir,'tcga_adrenal_20x_features'),
                                        'TCGA-COAD':os.path.join(args.data_root_dir,'tcga_colorectal_20x_features'),
                                        'TCGA-CESC':os.path.join(args.data_root_dir,'tcga_cervical_20x_features'),
                                        'TCGA-LIHC':os.path.join(args.data_root_dir,'tcga_liver_20x_features'),
                                        'TCGA-OV':os.path.join(args.data_root_dir,'tcga_ovary_20x_features'),
                                        'TCGA-SKCM':os.path.join(args.data_root_dir,'tcga_skin_20x_features'),
                                        'TCGA-ESCA':os.path.join(args.data_root_dir,'tcga_esophagus_20x_features'),
                                        'TCGA-STAD':os.path.join(args.data_root_dir,'tcga_stomach_20x_features'),
                                        'TCGA-TGCT':os.path.join(args.data_root_dir,'tcga_germ_cell_20x_features'),
                                        'TCGA-UCEC':os.path.join(args.data_root_dir,'tcga_endometrial_20x_features'),
                                        'TCGA-BLCA':os.path.join(args.data_root_dir,'tcga_bladder_20x_features'),
                                        'TCGA-THCA':os.path.join(args.data_root_dir,'tcga_thyroid_20x_features'),
                                        'TCGA-READ':os.path.join(args.data_root_dir,'tcga_rectum_20x_features')},
                                    shuffle = False, 
                                    print_info = True, 
                                    label_dicts = [{'Lung':0, 'Colorectal':1, 'Esophagastric':2, 'Pancreatic':3, 'Prostate':4},
                                                   {'Primary':0, 'Metastatic Recurrence':1, 'TCGA Primary Tumor':0, 'TCGA Metastatic':1},
                                                   {'F':0, 'M':1}],
                                    patient_strat= False,
                                    filter_dict={'label': ['Lung', 'Colorectal', 'Esophagastric', 'Pancreatic', 'Prostate']},
                                    label_cols = ['label', 'site', 'sex'],
                                    ignore=[])

elif args.task == 'study_squamous_mtl_sex':
    args.n_classes = 4
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/study_squamous_clean.csv',
                                  data_dir= {'Oncopanel Primary':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                       'Oncopanel Metastatic':os.path.join(args.data_root_dir,'oncopanel_met_primary'),
                                       'TCGA-KIRC':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                        'TCGA-KICH':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                        'TCGA-KIRP':os.path.join(args.data_root_dir,'tcga_kidney_20x_features'),
                                        'TCGA-LGG':os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                        'TCGA-GBM':os.path.join(args.data_root_dir,'tcga_gbm_20x_features'),
                                        'TCGA-PRAD':os.path.join(args.data_root_dir,'tcga_prostate_20x_features'),
                                        'TCGA-PAAD':os.path.join(args.data_root_dir,'tcga_pancreas_20x_features'),
                                        'TCGA-HNSC':os.path.join(args.data_root_dir,'tcga_head_and_neck_20x_features'),
                                        'TCGA-LUAD':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                        'TCGA-LUSC':os.path.join(args.data_root_dir,'tcga_lung_20x_features'),
                                        'TCGA-BRCA':os.path.join(args.data_root_dir,'tcga_breast_20x_features'),
                                        'TCGA-ACC':os.path.join(args.data_root_dir,'tcga_adrenal_20x_features'),
                                        'TCGA-COAD':os.path.join(args.data_root_dir,'tcga_colorectal_20x_features'),
                                        'TCGA-CESC':os.path.join(args.data_root_dir,'tcga_cervical_20x_features'),
                                        'TCGA-LIHC':os.path.join(args.data_root_dir,'tcga_liver_20x_features'),
                                        'TCGA-OV':os.path.join(args.data_root_dir,'tcga_ovary_20x_features'),
                                        'TCGA-SKCM':os.path.join(args.data_root_dir,'tcga_skin_20x_features'),
                                        'TCGA-ESCA':os.path.join(args.data_root_dir,'tcga_esophagus_20x_features'),
                                        'TCGA-STAD':os.path.join(args.data_root_dir,'tcga_stomach_20x_features'),
                                        'TCGA-TGCT':os.path.join(args.data_root_dir,'tcga_germ_cell_20x_features'),
                                        'TCGA-UCEC':os.path.join(args.data_root_dir,'tcga_endometrial_20x_features'),
                                        'TCGA-BLCA':os.path.join(args.data_root_dir,'tcga_bladder_20x_features'),
                                        'TCGA-THCA':os.path.join(args.data_root_dir,'tcga_thyroid_20x_features'),
                                        'TCGA-READ':os.path.join(args.data_root_dir,'tcga_rectum_20x_features')},
                                    shuffle = False, 
                                    print_info = True,
                                    label_dicts = [{'Lung':0, 'Head Neck':1, 'Esophagastric':2, 'Cervix':3},
                                                  {'Primary':0, 'Metastatic Recurrence':1, 'TCGA Primary Tumor':0, 'TCGA Metastatic':1},
                                                  {'F':0, 'M':1}],
                                    label_cols = ['label', 'site', 'sex'],
                                    patient_strat= False,
                                    filter_dict={'label': ['Lung', 'Head Neck', 'Esophagastric', 'Cervix']},

                                    ignore=[])


else:
    raise NotImplementedError

if args.summarize:
    pdb.set_trace()

    csv_path = 'splits/study_v2_mtl_sex_100/splits_0.csv'
    ids = []
    for split in ['train', 'val', 'test']:
        ids.append(dataset.get_split_from_df(pd.read_csv(csv_path), split_key=split, return_ids_only=True))
    pdb.set_trace()
    ids = np.concatenate(ids)
    dataset.slide_data = dataset.slide_data.loc[ids].reset_index(drop=True)
    df_path = os.path.join('summary_files', args.task+'.csv')
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path)
        bag_sizes = df['bag_size'].values
        print('Max: {}, Min: {}, Mean: {}, Median:{}'.format(bag_sizes.max(), bag_sizes.min(), bag_sizes.mean(), np.median(bag_sizes)))
    else:
        bag_sizes = np.zeros(len(dataset), dtype=np.int32)
        for idx in range(len(dataset)):
            features, _, _, _ = dataset[idx]
            bag_sizes[idx] = len(features)

        print('Max: {}, Min: {}, Mean: {}, Median:{}'.format(bag_sizes.max(), bag_sizes.min(), bag_sizes.mean(), np.median(bag_sizes)))
        print('Done')
        df = pd.DataFrame({'slide_id':dataset.slide_data['slide_id'].values, 'bag_size': bag_sizes})
        df.to_csv(df_path)
    exit()



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
    all_aucs = []
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
        
        if args.infer_only:
            split_dataset.infer_only(True)
            class_labels = ['Lung', 'Breast', 'Colorectal', 'Ovarian', 
                                          'Pancreatic', 'Adrenal', 
                                          'Melanoma', 'Prostate', 'Renal', 'Bladder', 
                                          'Esophagastric',  'Thyroid',
                                          'Head Neck',  'Glioma', 
                                          'Germ Cell Tumor', 'Endometrial', 'Cervix', 'Liver']
            site_labels = ['Primary', 'Met']
            _, df  = infer(split_dataset, args, ckpt_paths[ckpt_idx], class_labels, site_labels)
            df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

        else:
            model, results_dict = eval(split_dataset, args, ckpt_paths[ckpt_idx])

            for cls_idx in range(len(results_dict['cls_aucs'])):
                print('class {} auc: {}'.format(cls_idx, results_dict['cls_aucs'][cls_idx]))

            all_cls_auc.append(results_dict['cls_auc'])
            all_cls_acc.append(1-results_dict['cls_test_error'])
            all_site_auc.append(results_dict['site_auc'])
            all_site_acc.append(1-results_dict['site_test_error'])
            all_cls_top3_acc.append(results_dict['top3_acc'])
            all_cls_top5_acc.append(results_dict['top5_acc'])
            if len(results_dict['cls_aucs']) > 0:
                all_aucs.append(results_dict['cls_aucs'])
            df = results_dict['df']
            df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

            if args.calc_features:
                if csv_path and args.merge_splits:
                    pdb.set_trace()
                    splits = pd.read_csv(csv_path)
                    split = pd.concat([splits['val'], splits['test']]).reset_index(drop=True).dropna()
                    split_dataset = dataset.get_split_from_df(split=split)
                compute_features(split_dataset, args, ckpt_paths[ckpt_idx], args.save_dir, model=model)

    if args.infer_only:
        exit()

    df_dict = {'folds': folds, 'cls_test_auc': all_cls_auc, 'cls_test_acc': all_cls_acc, 'cls_top3_acc': all_cls_top3_acc, 'cls_top5_acc': all_cls_top5_acc,
                'site_test_auc': all_site_auc, 'site_test_acc': all_site_acc}

    if args.n_classes > 2:
        all_aucs = np.vstack(all_aucs)
        for i in range(args.n_classes):
            df_dict.update({'class_{}_ovr_auc'.format(i):all_aucs[:,i]})

    final_df = pd.DataFrame(df_dict)
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
