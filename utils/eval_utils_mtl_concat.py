import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_toad import TOAD_fc_mtl_concat
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils_mtl_concat import EarlyStopping,  Accuracy_Logger
from utils.file_utils import save_pkl, load_pkl
from sklearn.metrics import roc_auc_score, roc_curve, auc
import h5py
from models.resnet_custom import resnet50_baseline
import math
from sklearn.preprocessing import label_binarize

def initiate_model(args, ckpt_path=None):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    model = TOAD_fc_mtl_concat(**model_dict)    

    model.relocate()
    print_network(model)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)

    print('Init Loaders')
    loader = get_simple_loader(dataset)
    results_dict = summary(model, loader, args)

    print('cls_test_error: ', results_dict['cls_test_error'])
    print('cls_auc: ', results_dict['cls_auc'])
    print('site_test_error: ', results_dict['site_test_error'])
    print('site_auc: ', results_dict['site_auc'])

    return model, results_dict

# Code taken from pytorch/examples for evaluating topk classification on on ImageNet
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def summary(model, loader, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_logger = Accuracy_Logger(n_classes=args.n_classes)
    site_logger = Accuracy_Logger(n_classes=2)
    model.eval()
    cls_test_error = 0.
    cls_test_loss = 0.
    site_test_error = 0.
    site_test_loss = 0.

    all_cls_probs = np.zeros((len(loader), args.n_classes))
    all_cls_labels = np.zeros(len(loader))
    all_site_probs = np.zeros((len(loader), 2))
    all_site_labels = np.zeros(len(loader))
    all_sexes = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, site, sex) in enumerate(loader):
        data =  data.to(device)
        label = label.to(device)
        site = site.to(device)
        sex = sex.float().to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            model_results_dict = model(data, sex)

        logits, Y_prob, Y_hat  = model_results_dict['logits'], model_results_dict['Y_prob'], model_results_dict['Y_hat']
        site_logits, site_prob, site_hat = model_results_dict['site_logits'], model_results_dict['site_prob'], model_results_dict['site_hat']
        del model_results_dict

        cls_logger.log(Y_hat, label)
        site_logger.log(site_hat, site)
        cls_probs = Y_prob.cpu().numpy()
        all_cls_probs[batch_idx] = cls_probs
        all_cls_labels[batch_idx] = label.item()

        all_sexes[batch_idx] = sex.item()

        site_probs = site_prob.cpu().numpy()
        all_site_probs[batch_idx] = site_probs
        all_site_labels[batch_idx] = site.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'cls_prob': cls_probs, 'cls_label': label.item(), 
                                'site_prob': site_probs, 'site_label': site.item()}})
        cls_error = calculate_error(Y_hat, label)
        cls_test_error += cls_error
        site_error = calculate_error(site_hat, site)
        site_test_error += site_error

    cls_test_error /= len(loader)
    site_test_error /= len(loader)

    all_cls_preds = np.argmax(all_cls_probs, axis=1)
    all_site_preds = np.argmax(all_site_probs, axis=1)

    if args.n_classes > 2:
        if args.n_classes > 5:
            topk = (1,3,5)
        else:
            topk = (1,3)
        topk_accs = accuracy(torch.from_numpy(all_cls_probs), torch.from_numpy(all_cls_labels), topk=topk)
        for k in range(len(topk)):
            print('top{} acc: {:.3f}'.format(topk[k], topk_accs[k].item()))

    if len(np.unique(all_cls_labels)) == 1:
        cls_auc = -1
        cls_aucs = []
    else:
        if args.n_classes == 2:
            cls_auc = roc_auc_score(all_cls_labels, all_cls_probs[:, 1])
            cls_aucs = []
        else:
            cls_aucs = []
            binary_labels = label_binarize(all_cls_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_cls_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_cls_probs[:, class_idx])
                    cls_aucs.append(auc(fpr, tpr))
                else:
                    cls_aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_cls_labels, classes=[i for i in range(args.n_classes)])
                valid_classes = np.where(np.any(binary_labels, axis=0))[0]
                binary_labels = binary_labels[:, valid_classes]
                valid_cls_probs = all_cls_probs[:, valid_classes]
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), valid_cls_probs.ravel())
                cls_auc = auc(fpr, tpr)
            else:
                cls_auc = np.nanmean(np.array(cls_aucs))
    
    if len(np.unique(all_site_labels)) == 1:
        site_auc = -1
    else:
        site_auc = roc_auc_score(all_site_labels, all_site_probs[:, 1])

    results_dict = {'slide_id': slide_ids, 'sex': all_sexes, 'Y': all_cls_labels, 'Y_hat': all_cls_preds, 
                    'site': all_site_labels, 'site_hat': all_site_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_cls_probs[:,c]})

    results_dict.update({'site_p': all_site_probs[:,1]})

    df = pd.DataFrame(results_dict)
    inference_results = {'patient_results': patient_results, 'cls_test_error': cls_test_error,
                     'cls_auc': cls_auc, 'cls_aucs': cls_aucs,
               'site_test_error': site_test_error, 'site_auc': site_auc, 'loggers': (cls_logger, site_logger), 'df':df}

    for k in range(len(topk)):
        inference_results.update({'top{}_acc'.format(topk[k]): topk_accs[k].item()})

    return inference_results
