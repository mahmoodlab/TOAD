import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM, CLAM_Simple
from models.model_attention_mil import MIL_Attention_fc
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import EarlyStopping,  Accuracy_Logger
from utils.file_utils import save_pkl, load_pkl
from sklearn.metrics import roc_auc_score, roc_curve, auc
import h5py
from models.resnet_custom import resnet50_baseline
import math
from sklearn.preprocessing import label_binarize

def initiate_model(args, ckpt_path=None):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam', 'attention_mil', 'clam_simple']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam':
        model = CLAM(**model_dict)
    elif args.model_type =='clam_simple':
        model = CLAM_Simple(**model_dict)
    elif args.model_type == 'attention_mil':
        model = MIL_Attention_fc(**model_dict)    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

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
    patient_results, test_error, auc, aucs, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    for cls_idx in range(len(aucs)):
        print('class {} auc: {}'.format(cls_idx, aucs[cls_idx]))
    return model, patient_results, test_error, auc, aucs, df

def infer(dataset, args, ckpt_path, class_labels):
    model = initiate_model(args, ckpt_path)
    df = infer_dataset(model, dataset, args, class_labels)
    return model, df

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
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)
    if args.n_classes > 2:
        pdb.set_trace()
        acc1, acc3 = accuracy(torch.from_numpy(all_probs), torch.from_numpy(all_labels), topk=(1, 3))
        print('top1 acc: {:.3f}, top3 acc: {:.3f}'.format(acc1.item(), acc3.item()))
        
    if len(np.unique(all_labels)) == 1:
        auc_score = -1
    else:
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, aucs, df, acc_logger

def infer_dataset(model, dataset, args, class_labels, k=3):
    model.eval()
    all_probs = np.zeros((len(dataset), k))
    all_preds = np.zeros((len(dataset), k))
    all_preds_str = np.full((len(dataset), k), ' ', dtype=object)
    slide_ids = dataset.slide_data
    for batch_idx, data in enumerate(dataset):
        data = data.to(device)
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        probs, ids = torch.topk(Y_prob, k)
        probs = probs.cpu().numpy()
        ids = ids.cpu().numpy()
        all_probs[batch_idx] = probs
        all_preds[batch_idx] = ids
        all_preds_str[batch_idx] = np.array(class_labels)[ids]
    del data
    results_dict = {'slide_id': slide_ids}
    for c in range(k):
        results_dict.update({'Pred_{}'.format(c): all_preds_str[:, c]})
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)
    return df

# def infer_dataset(model, dataset, args, class_labels, k=3):
#     model.eval()

#     all_probs = np.zeros((len(dataset), args.n_classes))
#     all_preds = np.zeros(len(dataset))
#     all_str_preds = np.full(len(dataset), ' ', dtype=object)

#     slide_ids = dataset.slide_data
#     for batch_idx, data in enumerate(dataset):
#         data = data.to(device)
#         with torch.no_grad():
#             logits, Y_prob, Y_hat, _, results_dict = model(data)
        
#         probs = Y_prob.cpu().numpy()
#         all_probs[batch_idx] = probs
#         all_preds[batch_idx] = Y_hat.item()
#         all_str_preds[batch_idx] = class_labels[Y_hat.item()]
#     del data

#     results_dict = {'slide_id': slide_ids, 'Prediction': all_str_preds, 'Y_hat': all_preds}
#     for c in range(args.n_classes):
#         results_dict.update({'p_{}_{}'.format(c, class_labels[c]): all_probs[:,c]})
#     df = pd.DataFrame(results_dict)
#     return df

def compute_features(dataset, args, ckpt_path, save_dir, model=None, feature_dim=512):
    if model is None:
        model = initiate_model(args, ckpt_path)

    names = dataset.get_list(np.arange(len(dataset))).values
    file_path = os.path.join(save_dir, 'features.h5')

    initialize_features_hdf5_file(file_path, len(dataset), feature_dim=feature_dim, names=names)
    for i in range(len(dataset)):
        print("Progress: {}/{}".format(i, len(dataset)))
        save_features(dataset, i, model, args, file_path)

def save_features(dataset, idx, model, args, save_file_path):
    name = dataset.get_list(idx)
    print(name)
    features, label = dataset[idx]
    features = features.to(device)
    with torch.no_grad():
        if type(model) == CLAM:
            _, Y_prob, Y_hat, _, results_dict = model(features, instance_eval=False, return_features=True)
            bag_feat = results_dict['features'][Y_hat.item()]
        else:
            _, Y_prob, Y_hat, _, results_dict = model(features, return_features=True)
            bag_feat = results_dict['features']
    del features
    Y_hat = Y_hat.item()
    Y_prob = Y_prob.view(-1).cpu().numpy()
    bag_feat = bag_feat.view(1, -1).cpu().numpy()

    with h5py.File(save_file_path, 'r+') as file:
        print('label', label)
        file['features'][idx, :] = bag_feat
        file['label'][idx] = label
        file['Y_hat'][idx] = Y_hat
        file['Y_prob'][idx] = Y_prob[Y_hat]

def initialize_features_hdf5_file(file_path, length, feature_dim=512, names = None):
    
    file = h5py.File(file_path, "w")

    dset = file.create_dataset('features', 
                                shape=(length, feature_dim), chunks=(1, feature_dim), dtype=np.float32)

    # if names is not None:
    #     names = np.array(names, dtype='S')
    #     dset.attrs['names'] = names
    if names is not None:
        dt = h5py.string_dtype()
        label_dset = file.create_dataset('names', 
                                        shape=(length, ), chunks=(1, ), dtype=dt)
        file['names'][:] = names
    
    label_dset = file.create_dataset('label', 
                                        shape=(length, ), chunks=(1, ), dtype=np.int32)

    pred_dset = file.create_dataset('Y_hat', 
                                        shape=(length, ), chunks=(1, ), dtype=np.int32)

    prob_dset = file.create_dataset('Y_prob', 
                                        shape=(length, ), chunks=(1, ), dtype=np.float32)

    file.close()
    return file_path

