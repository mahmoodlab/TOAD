<!-- CLAM <img src="logo.jpg" width="350px" align="right" /> -->
===========
Tumor Origin Assessement via Deep-learning on Whole Slide Images.

ArXiv | Interactive Demo 

*TL;DR: .*

## TOAD: Tumor Origin Assessement via Deep-learning on Whole Slide Images

## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 16)
* Python (3.7.5), h5py (2.10.0), matplotlib (3.1.1), numpy (1.17.3), opencv-python (4.1.1.26), openslide-python (1.1.1), openslides (3.4.1), pandas (0.25.3), pillow (6.2.1), PyTorch (1.3.1), scikit-learn (0.22.1), scipy (1.3.1), tensorflow (1.14.0), tensorboardx (1.9), torchvision (0.4.2).

### Installation Guide for Linux (using anaconda)
[Installation Guide](INSTALLATION.md)

<!-- ## Weakly-Supervised Learning using Slide-Level Labels with CLAM -->

<!-- <img src="CLAM2.jpg" width="1000px" align="center" /> -->

### Data Preparation
We chose to encode each tissue patch with a 1024-dim feature vector using a truncated, pretrained ResNet50. For each WSI, these features are expected to be saved as matrices of torch tensors of size N x 1024, where N is the number of patches from each WSI (varies from slide to slide):
```bash
FEATURES_DIRECTORY/
	├── slide_1.pt
	├── slide_2.pt
	└── ...
```
Please refer to CLAM for examples on how perform this feature extraction step.

### Datasets
Datasets are expected to be prepared in a csv format containing at least 3 columns: **case_id**, **slide_id**, **sex**, and labels columns for the slide-level labels: **label**, **site**. Each **case_id** is a unique identifier for a patient, while the **slide_id** is a unique identifier for a slide that correspond to the name of an extracted feature .pt file. This is necessary because often one patient has multiple slides, which might also have different labels. When train/val/test splits are created, we also make sure that slides from the same patient do not go to different splits. The slide ids should be consistent with what was used during the feature extraction step. We provide a dummy example of a dataset csv file in the **dataset_csv** folder, named **dummy_dataset.csv**. 

Dataset objects used for actual training/validation/testing can be constructed using the **Generic_MIL_MTL_Dataset** Class (defined in **datasets/dataset_mtl_concat.py**). Examples of such dataset objects passed to the models can be found in both **main_mtl_concat.py** and **eval_mtl_concat.py**. 

For training, look under main.py:
```python 
if args.task == 'dummy_mtl_concat':
    args.n_classes=18
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/dummy_dataset.csv',
                            data_dir= 'Oncopanel Primary':os.path.join(args.data_root_dir,'DUMMY_DATA_DIR')
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts = [{'Lung':0, 'Breast':1, 'Colorectal':2, 'Ovarian':3, 
                                            'Pancreatic':4, 'Adrenal':5, 
                                             'Melanoma':6, 'Prostate':7, 'Renal':8, 'Bladder':9, 
                                             'Esophagastric':10,  'Thyroid':11,
                                             'Head Neck':12,  'Glioma':13, 
                                             'Germ Cell Tumor':14, 'Endometrial': 15, 
                                             'Cervix': 16, 'Liver': 17},
                                            {'Primary':0,  'Metastatic':1},
                                            {'F':0, 'M':1}],
                            label_cols = ['label', 'site', 'sex'],
                            patient_strat= False)
```
In addition to the number of classes (args.n_classes), the following arguments need to be specified:
* csv_path (str): Path to the dataset csv file
* data_dir (str): Path to saved .pt features for the dataset
* label_dicts (list of dict): List of dictionaries with key, value pairs for converting str labels to int for each label column
* label_cols (list of str): List of column headings to use as labels and map with label_dicts

Finally, the user should add this specific 'task' specified by this dataset object to be one of the choices in the --task arguments as shown below:

```python
parser.add_argument('--task', type=str, choices=['dummy_mtl_concat'])
```

### Training Splits
For evaluating the algorithm's performance, multiple folds (e.g. 10-fold) of train/val/test splits can be used. Example 10-fold 80/10/10 splits for camelyon and tcga-kidney, using 50% of training data can be found under the **splits** folder. These splits can be automatically generated using the create_splits_seq.py script with minimal modification just like with **main.py**. For example, camelyon splits with 75% of training data can be created by calling:
 
``` shell
python create_splits_seq.py --task camelyon_40x_cv --seed 1 --label_frac 0.75 --k 10
```
The script uses the **Generic_WSI_Classification_Dataset** Class for which the constructor expects the same arguments as 
**Generic_MIL_Dataset** (without the data_dir argument). For details, please refer to the dataset definition in **datasets/dataset_generic.py**

### Training
``` shell
CUDA_VISIBLE_DEVICES=0,1 python main_mtl_concat.py --drop_out --early_stopping --lr 2e-4 --k 1 --exp_code study_v2_mtl_sex_100  --task study_v2_mtl_sex  --log_data 
```
By default results will be saved to **results/exp_code** corresponding to the exp_code input argument from the user. If tensorboard logging is enabled (with the arugment toggle --log_data), the user can go into the results folder for the particular experiment, run:
``` shell
tensorboard --logdir=.
```
This should open a browser window and show the logged training/validation statistics in real time. 
For information on each argument, see:
``` shell
python main.py -h
```

### Evaluation 
User also has the option of using the evluation script to test the performances of trained models. Examples corresponding to the models trained above are provided below:
``` shell
CUDA_VISIBLE_DEVICES=0,1 python eval.py --drop_out --k 1 --models_exp_code study_v2_mtl_sex_100_s1 --save_exp_code study_v2_mtl_sex_100_s1_all --task study_v2_mtl_sex  --results_dir results
```

Information on each commandline argument, see:
``` shell
python eval.py -h
```

To test trained models on your own custom datasets, first add them into **eval_mtl_concat.py**, the same way as you do for **main_mtl_concat.py**.

<!-- <img src="fig-gh3.jpg" width="1000px" align="center" />	 -->

## Issues
- Please report all issues on the public forum.

## License
© [Mahmood Lab](http://www.mahmoodlab.org) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes. 

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our paper:
```
@inproceedings{lu2020clam,
  title     = {Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images},
  author    = {Ming Y. Lu, Drew F. K. Williamson, Tiffany Y. Chen, Richard J. Chen, Matteo Barbieri, Faisal Mahmood},
  booktitle = {arXiv},
  year = {2020}
}
```
