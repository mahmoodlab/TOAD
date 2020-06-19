import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_attention_mil import MIL_Attention_fc_mtl
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
        raise NotImplementedError
    elif args.model_type =='clam_simple':
        raise NotImplementedError
    elif args.model_type == 'attention_mil':
        model = MIL_Attention_fc_mtl(**model_dict)    
    else: # args.model_type == 'mil'
        raise NotImplementedError

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
    loader = get_simple_loader(dataset, collate_fn='MIL_mtl')
    results_dict = summary(model, loader, args)

    print('cls_test_error: ', results_dict['cls_test_error'])
    print('cls_auc: ', results_dict['cls_auc'])
    print('site_test_error: ', results_dict['site_test_error'])
    print('site_auc: ', results_dict['site_auc'])

    for cls_idx in range(len(results_dict['cls_aucs'])):
        print('class {} auc: {}'.format(cls_idx, results_dict['cls_aucs'][cls_idx]))

    return model, results_dict
    # patient_results, test_error, auc, aucs, df

def infer(dataset, args, ckpt_path, class_labels, site_labels):
    model = initiate_model(args, ckpt_path)
    df = infer_dataset(model, dataset, args, class_labels, site_labels)
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

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, site) in enumerate(loader):
        data =  data.to(device)
        label = label.to(device)
        site = site.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            model_results_dict = model(data)

        logits, Y_prob, Y_hat  = model_results_dict['logits'], model_results_dict['Y_prob'], model_results_dict['Y_hat']
        site_logits, site_prob, site_hat = model_results_dict['site_logits'], model_results_dict['site_prob'], model_results_dict['site_hat']
        del model_results_dict

        cls_logger.log(Y_hat, label)
        site_logger.log(site_hat, site)
        cls_probs = Y_prob.cpu().numpy()
        all_cls_probs[batch_idx] = cls_probs
        all_cls_labels[batch_idx] = label.item()

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
        acc1, acc3 = accuracy(torch.from_numpy(all_cls_probs), torch.from_numpy(all_cls_labels), topk=(1, 3))
        print('top1 acc: {:.3f}, top3 acc: {:.3f}'.format(acc1.item(), acc3.item()))

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
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_cls_probs.ravel())
                cls_auc = auc(fpr, tpr)
            else:
                cls_auc = np.nanmean(np.array(cls_aucs))
    
    if len(np.unique(all_site_labels)) == 1:
        site_auc = -1
    else:
        site_auc = roc_auc_score(all_site_labels, all_site_probs[:, 1])

    results_dict = {'slide_id': slide_ids, 'Y': all_cls_labels, 'Y_hat': all_cls_preds, 
                    'site': all_site_labels, 'site_hat': all_site_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_cls_probs[:,c]})

    results_dict.update({'site_p': all_site_probs[:,1]})

    df = pd.DataFrame(results_dict)
    inference_results = {'patient_results': patient_results, 'cls_test_error': cls_test_error,  'top1_acc': acc1.item(), 'top3_acc':acc3.item(), 
                        'cls_auc': cls_auc, 'cls_aucs': cls_aucs,
                        'site_test_error': site_test_error, 'site_auc': site_auc, 'loggers': (cls_logger, site_logger), 'df':df}

    return inference_results


def infer_dataset(model, dataset, args, class_labels, site_labels, k=3):
    model.eval()
    all_probs_cls = np.zeros((len(dataset), k))
    all_probs_site = np.zeros((len(dataset),2))

    all_preds_cls = np.zeros((len(dataset), k))
    all_preds_cls_str = np.full((len(dataset), k), ' ', dtype=object)
    all_preds_site = np.full((len(dataset)), ' ', dtype=object)

    slide_ids = dataset.slide_data
    for batch_idx, data in enumerate(dataset):
        data = data.to(device)
        with torch.no_grad():
            results_dict = model(data)
        
        Y_prob, Y_hat = results_dict['Y_prob'], results_dict['Y_hat']
        site_prob, site_hat = results_dict['site_prob'], results_dict['site_hat']
        del results_dict
        probs, ids = torch.topk(Y_prob, k)
        probs = probs.cpu().numpy()
        site_prob = site_prob.cpu().numpy()
        ids = ids.cpu().numpy()
        all_probs_cls[batch_idx] = probs
        all_preds_cls[batch_idx] = ids
        all_preds_cls_str[batch_idx] = np.array(class_labels)[ids]

        all_probs_site[batch_idx] = site_prob
        all_preds_site[batch_idx] = np.array(site_labels)[site_hat.item()]
        
    del data
    results_dict = {'slide_id': slide_ids}
    for c in range(k):
        results_dict.update({'Pred_{}'.format(c): all_preds_cls_str[:, c]})
        results_dict.update({'p_{}'.format(c): all_probs_cls[:, c]})
    results_dict.update({'Site_Pred': all_preds_site, 'Site_p': all_probs_site[:, 1]})
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
    features, label, site = dataset[idx]
    features = features.to(device)
    with torch.no_grad():
        results_dict = model(features, return_features=True) 
        Y_prob, Y_hat = results_dict['Y_prob'], results_dict['Y_hat']
        site_prob, site_hat = results_dict['site_prob'], results_dict['site_hat']
        bag_feat = results_dict['features'][0]
        site_feat = results_dict['features'][1]

    del results_dict 
    del features
    Y_hat = Y_hat.item()
    Y_prob = Y_prob.view(-1).cpu().numpy()
    site_hat = site_hat.item()
    site_prob = site_prob.view(-1).cpu().numpy()
    bag_feat = bag_feat.view(1, -1).cpu().numpy()
    site_feat = site_feat.view(1, -1).cpu().numpy()
    with h5py.File(save_file_path, 'r+') as file:
        print('label', label)
        file['features'][idx, :] = bag_feat
        file['site_features'][idx, :] = site_feat
        file['label'][idx] = label
        file['Y_hat'][idx] = Y_hat
        file['Y_prob'][idx] = Y_prob[Y_hat]
        file['site'][idx] = site
        file['site_hat'][idx] = site_hat
        file['site_prob'][idx] = site_prob[1]

def initialize_features_hdf5_file(file_path, length, feature_dim=512, names = None):
    
    file = h5py.File(file_path, "w")

    dset = file.create_dataset('features', 
                                shape=(length, feature_dim), chunks=(1, feature_dim), dtype=np.float32)

    dset = file.create_dataset('site_features', 
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

    label_dset = file.create_dataset('site', 
                                        shape=(length, ), chunks=(1, ), dtype=np.int32)

    site_pred_dset = file.create_dataset('site_hat', 
                                        shape=(length, ), chunks=(1, ), dtype=np.int32)

    site_prob_dset = file.create_dataset('site_prob', 
                                        shape=(length, ), chunks=(1, ), dtype=np.float32)

    file.close()
    return file_path

