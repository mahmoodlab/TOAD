import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
from PIL import ImageFilter
import matplotlib.pyplot as plt
from datasets.wsi_dataset import Wsi_Region
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, annotation=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    if annotation is not None:
        wsi_object.initXML(annotation)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

def compute_fineheatmap(wsi_object, label=None, model=None, feature_extractor=None, batch_size=512, seg_params=None, filter_params=None, 
    vis_params=None, save_path=None, mask_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size']
    
    # wsi = wsi_object.getOpenSlide()
    # if vis_params['vis_level'] < 0:
    #     best_level = wsi.get_best_level_for_downsample(32)
    #     vis_params['vis_level'] = best_level
    # mask = wsi_object.visWSI(**vis_params)
    # mask.save(mask_path)

    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size)
    print('total number of patches to process: ', len(roi_loader) * batch_size)
    num_batches = len(roi_loader)
    print('number of batches: ', len(roi_loader))
    mode = "w"
    for idx, (roi, coords) in enumerate(roi_loader):
        roi = roi.to(device)
        with torch.no_grad():
            features = feature_extractor(roi)
            A = model(features, attention_only=True)
        if idx % math.ceil(num_batches * 0.05) == 0:
            print('procssed {} / {}'.format(idx, num_batches))
        
        A = A.view(-1, 1).cpu().numpy()
        coords = coords.numpy()
        if ref_scores is not None:
            for score_idx in range(len(A)):
                A[score_idx] = score2percentile(A[score_idx], ref_scores)

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        # save_path = save_attention_hdf5_file(A, coords, save_path, mode=mode)
        asset_dict = {'attention_scores': A, 'coords': coords}
        save_path = save_hdf5(save_path, asset_dict, mode=mode)
        mode = "a"
    return save_path, wsi_object