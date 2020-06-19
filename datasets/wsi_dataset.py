from torchvision import transforms
from wsi_core.WholeSlideImage import WholeSlideImage
import pandas as pd
import numpy as np
import time
import pdb
import PIL.Image as Image
import h5py
from torch.utils.data import Dataset
from datasets.dataset_generic import Generic_MIL_Dataset
import os
import torch
import cv2
from wsi_core.util_classes import Contour_Checking_fn, isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard

def default_transforms(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean = mean, std = std)])
    return t

def get_contour_check_fn(contour_fn='four_pt_hard', cont=None, ref_patch_size=None, center_shift=None):
    if contour_fn == 'four_pt_hard':
        cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size, center_shift=center_shift)
    elif contour_fn == 'four_pt_easy':
        cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size, center_shift=0.5)
    elif contour_fn == 'center':
        cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size)
    elif contour_fn == 'basic':
        cont_check_fn = isInContourV1(contour=cont)
    else:
        raise NotImplementedError
    return cont_check_fn

class Wsi_Region(Dataset):
    def __init__(self, wsi_object, top_left, bot_right, level=0, patch_size = (256, 256), step_size=(256, 256), 
                 contour_fn='four_pt_hard',
                 t=None, custom_downsample=1, use_center_shift=False):
        
        self.custom_downsample =custom_downsample
        self.ref_downsample = wsi_object.level_downsamples[level]
        self.ref_size = tuple((np.array(patch_size) * np.array(self.ref_downsample)).astype(int)) 
        if self.custom_downsample > 1:
            assert custom_downsample == 2
            assert level == 0
            # self.target_patch_size = tuple((np.array(patch_size) / 2).astyep(int))
            self.target_patch_size = patch_size
            patch_size = tuple(np.array(patch_size) * 2)
            step_size = tuple(np.array(step_size) * 2)
            self.ref_size = patch_size
        else:
            step_size = tuple((np.array(step_size) * np.array(self.ref_downsample)).astype(int))
            self.ref_size = tuple((np.array(patch_size) * np.array(self.ref_downsample)).astype(int)) 
        self.wsi = wsi_object.wsi
        self.level = level
        self.patch_size = patch_size
        if top_left is None:
            x1, y1 = (0, 0)
        else:
            x1, y1 = top_left
            x1, y1 = int(x1), int(y1)

        if bot_right is None:
            x2, y2 = self.wsi.level_dimensions[0]

        else:
            x2, y2 = bot_right
            x2, y2 = int(x2), int(y2)
        
        w = x2 - x1
        h = y2 - y1
        print('input ROI is {} x {}'.format(w, h))
        x_res = (w - patch_size[0]) % step_size[0]
        y_res = (h - patch_size[1]) % step_size[1]
        x2 -= x_res
        y2 -= y_res
        w = x2 - x1
        h = y2 - y1
        print('trimmed ROI to be {} x {}'.format(w, h))
        
        if not use_center_shift:
            center_shift = 0.
        else:
            overlap = 1 - float(step_size[0] / patch_size[0])
            if overlap < 0.25:
                center_shift = 0.375
            elif overlap >= 0.25 and overlap < 0.75:
                center_shift = 0.5
            elif overlap >=0.75 and overlap < 0.95:
                center_shift = 0.5
            else:
                center_shift = 0.625
            #center_shift = 0.375 # 25% overlap
            #center_shift = 0.625 #50%, 75% overlap
            #center_shift = 1.0 #95% overlap
        filtered_coords = []
        for cont_idx, contour in enumerate(wsi_object.contours_tissue): #iterate through tissue contours
            print('processing {}/{} contours'.format(cont_idx, len(wsi_object.contours_tissue)))
            cont_check_fn= get_contour_check_fn(contour_fn, contour, self.ref_size[0], center_shift)
            coord_results, _ = wsi_object.process_contour(contour, wsi_object.holes_tissue[cont_idx], level, '', 
                            patch_size = patch_size[0], step_size = step_size[0], contour_fn=cont_check_fn,
                            use_padding=True, top_left = top_left, bot_right = bot_right)
            if len(coord_results) > 0:
                filtered_coords.append(coord_results['coords'])
        
        coords=np.vstack(filtered_coords)

        self.coords = coords
        print('filtered a total of {} coordinates'.format(len(self.coords)))
        if t is None:
            self.transforms = default_transforms()
        else:
            self.transforms = t

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        coord = self.coords[idx]
        patch = self.wsi.read_region(tuple(coord), self.level, self.patch_size).convert('RGB')
        if self.custom_downsample > 1:
            patch = patch.resize(self.target_patch_size)
        patch = self.transforms(patch).unsqueeze(0)
        return patch, coord 

class Wsi_Clustering_Dataset(Generic_MIL_Dataset):
    def __init__(self,
        sample_ratio = 0.02,
        **kwargs):
    
        super(Wsi_Clustering_Dataset, self).__init__(**kwargs)
        self.sample_ratio = sample_ratio

    
    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]

        full_path = os.path.join(self.data_dir,'{}.h5'.format(slide_id))
        with h5py.File(full_path,'r') as hdf5_file:
            features = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]

        np.random.seed(self.seed)
        sample_num = int(self.sample_ratio * len(coords))
        if sample_num > 10000:
            sample_num = 10000
        if sample_num == 0:
            sample_num = 1
        sample_ids = np.random.choice(np.arange(len(coords)), sample_num,  replace=False)
        features = torch.from_numpy(features)
        coords = coords
        return features, label, coords, sample_ids

