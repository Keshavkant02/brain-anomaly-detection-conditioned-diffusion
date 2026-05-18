from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio
import pandas as pd
from pytorch_lightning import LightningDataModule
import os

# Fix for SimpleITK threading issues in Linux
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")

# --- 1. NAMED UTILITIES ---

def cast_to_float(tensor):
    return tensor.float()

def sitk_reader(path):
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if not 'mask' in str(path).lower() and not 'seg' in str(path).lower():
        image_nii = sitk.CurvatureFlow(image1=image_nii, timeStep=0.125, numberOfIterations=3)
    vol = sitk.GetArrayFromImage(image_nii).transpose(2, 1, 0)
    return vol, None

# --- 2. CORE DATASET FUNCTIONS ---

def Train(csv, cfg, preload=True):
    subjects = []
    for _, sub in csv.iterrows():
        subject_dict = {
            'vol' : tio.ScalarImage(sub.img_path, reader=sitk_reader, type=tio.INTENSITY), 
            'age' : sub.age,
            'ID' : sub.img_name,
            'label' : sub.label,
            'Dataset' : sub.setname,
            'stage' : sub.settype,
            'path' : sub.img_path
        }
        
        mask_path = getattr(sub, 'mask_path', None)
        if pd.notna(mask_path) and mask_path and str(mask_path).lower() != 'none': 
            subject_dict['mask'] = tio.LabelMap(mask_path, reader=sitk_reader, type=tio.LABEL)
        else: 
            img_data = tio.ScalarImage(sub.img_path, reader=sitk_reader).data
            subject_dict['mask'] = tio.LabelMap(tensor=img_data > 0, type=tio.LABEL)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    
    if preload: 
        ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
        ds = preload_wrapper(ds, augment=get_augment(cfg))
    else: 
        ds = tio.SubjectsDataset(subjects, transform=tio.Compose([get_transform(cfg), get_augment(cfg)]))
    return ds

def Eval(csv, cfg): 
    subjects = []
    for _, sub in csv.iterrows():
        subject_dict = {
            'vol' : tio.ScalarImage(sub.img_path, reader=sitk_reader, type=tio.INTENSITY),
            'vol_orig' : tio.ScalarImage(sub.img_path, reader=sitk_reader, type=tio.INTENSITY), 
            'age' : sub.age,
            'ID' : sub.img_name,
            'label' : sub.label,
            'Dataset' : sub.setname,
            'stage' : sub.settype,
            'seg_available': False,
            'path' : sub.img_path 
        }
        
        seg_path = getattr(sub, 'seg_path', None)
        if pd.notna(seg_path) and seg_path and str(seg_path).lower() != 'none': 
            subject_dict['seg'] = tio.LabelMap(seg_path, reader=sitk_reader, type=tio.LABEL)
            subject_dict['seg_orig'] = tio.LabelMap(seg_path, reader=sitk_reader, type=tio.LABEL)
            subject_dict['seg_available'] = True
        
        mask_path = getattr(sub, 'mask_path', None)
        if pd.notna(mask_path) and mask_path and str(mask_path).lower() != 'none': 
            subject_dict['mask'] = tio.LabelMap(mask_path, reader=sitk_reader, type=tio.LABEL)
            subject_dict['mask_orig'] = tio.LabelMap(mask_path, reader=sitk_reader, type=tio.LABEL)
        else: 
            img_data = tio.ScalarImage(sub.img_path, reader=sitk_reader).data
            subject_dict['mask'] = tio.LabelMap(tensor=img_data > 0, type=tio.LABEL)
            subject_dict['mask_orig'] = tio.LabelMap(tensor=img_data > 0, type=tio.LABEL)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    
    ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
    ds = preload_wrapper(ds)
    return ds

# --- 3. CACHING AND WRAPPERS ---

class preload_wrapper(Dataset):
    def __init__(self, ds, augment=None):
        self.ds = ds
        self.augment = augment
        self._cache = {}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        if index not in self._cache:
            self._cache[index] = self.ds.__getitem__(index)
        subject = self._cache[index]
        
        if self.augment:
            subject = self.augment(subject)
        
        output = {
            'vol': {tio.DATA: subject['vol'].data},
            'mask': {tio.DATA: subject['mask'].data},
            'age': subject['age'],
            'ID': subject['ID'],
            'label': subject['label'],
            'Dataset': subject['Dataset'],
            'stage': subject['stage']
        }
        
        if 'seg' in subject:
            output['seg'] = {tio.DATA: subject['seg'].data}
            
        return output

# --- 4. PREPROCESSING ---

def get_transform(cfg):
    h, w, d = tuple(cfg.get('imageDim', (160, 192, 160)))
    exclude_from_resampling = ['vol_orig', 'mask_orig', 'seg_orig'] if not cfg.resizedEvaluation else None
    
    return tio.Compose([
        tio.CropOrPad((h, w, d), padding_mode=0),
        tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 1), cfg.get('perc_high', 99)), masking_method='mask'),
        tio.Lambda(cast_to_float),
        tio.Resample(cfg.get('rescaleFactor', 1.0), image_interpolation='bspline', exclude=exclude_from_resampling),
    ])

def get_augment(cfg):
    augmentations = []
    if cfg.get('aug_intensity', False):
        augmentations.extend([tio.RandomGamma(p=0.5), tio.RandomBiasField(p=0.25)])
    return tio.Compose(augmentations)

# --- 5. LIGHTNING MODULE ---

class GoldDataModule(LightningDataModule):
    def __init__(self, csv, cfg, fold=0, csv_val=None):
        super().__init__()
        self.csv = csv
        self.csv_val = csv_val
        self.cfg = cfg
        self.fold = fold

    def setup(self, stage=None):
        train_df = pd.read_csv(self.csv)

        if self.csv_val is not None:
            val_df = pd.read_csv(self.csv_val)
        else:
            val_df = train_df

        self.train_ds = Train(train_df, self.cfg)
        self.val_ds = Eval(val_df, self.cfg)

        print(f"[GoldDataModule] Train rows: {len(train_df)}")
        print(f"[GoldDataModule] Val rows: {len(val_df)}")

        if self.csv_val is None:
            print("[GoldDataModule] WARNING: csv_val not provided; using train CSV for monitoring validation.")
        else:
            overlap = set(train_df["img_name"].astype(str)) & set(val_df["img_name"].astype(str))
            print(f"[GoldDataModule] Train/val overlap: {len(overlap)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=False)
