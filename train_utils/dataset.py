import numpy as np
import numpy as np
from collections.abc import Callable, Sequence
from typing import Dict, List, Any, Union, Tuple
from monai.data import Dataset as MonaiDataset,load_decathlon_datalist,DataLoader
from monai.transforms import Compose,apply_transform
import time
from monai.data.meta_tensor import MetaTensor
import os
import json
from tqdm import tqdm
import nibabel as nib
import torch
import math
import pydicom
import cv2
import SimpleITK as sitk
from PIL import Image, ImageEnhance
import random
from torchvision import transforms
import monai
# import albumentations as albu
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import nibabel as nib  

def get_slices_paths(image_dict, data_item):
    slice_set = 10
    cur_mod_dir = os.path.join(data_item['mod_parent'], "PSIR").replace('../data', 'data')
    if not os.path.exists(cur_mod_dir):
        print(data_item['mod_parent'])
        raise ValueError
    
    slice_files = sorted([os.path.join(cur_mod_dir, i) for i in os.listdir(cur_mod_dir) 
                          if (i not in [".DS_Store", 'mask']) and not (i.endswith('bmp')) and not (i.endswith('png'))])
    slices = []
    slices_paths = []
    norm_trans = monai.transforms.ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True, channel_wise=False)
    
    for slice_file in slice_files:
        if os.path.isdir(slice_file):
            inner_files = sorted([os.path.join(slice_file, i) for i in os.listdir(slice_file)])
            for inner_file in inner_files:
                ds = pydicom.dcmread(inner_file, force=True)
                window_center = ds[0x0028, 0x1050].value # DS 类型
                window_width = ds[0x0028, 0x1051].value # DS 类型
                min_value = window_center - window_width // 2
                max_value = window_center + window_width // 2
                if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
                    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                pixel_array = ds.pixel_array.astype(float)
                pixel_array = np.clip(pixel_array, min_value, max_value)
                
                slices.append(pixel_array)
                slices_paths.append(inner_file)
        else:
            ds = pydicom.dcmread(slice_file, force=True)
            window_center = ds[0x0028, 0x1050].value # DS 类型
            window_width = ds[0x0028, 0x1051].value # DS 类型
            min_value = window_center - window_width // 2
            max_value = window_center + window_width // 2
            if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            pixel_array = ds.pixel_array.astype(float)
            pixel_array = ds.pixel_array.astype(float)
            pixel_array = np.clip(pixel_array, min_value, max_value)
            slices.append(pixel_array)
            slices_paths.append(slice_file)
    
    if len(slices_paths) == 0:
        print(cur_mod_dir)
        raise ValueError
    slices, slices_paths = sample_slice_files(slices, slices_paths, slice_set)
    slices = [(norm_trans(i)* 255.0).astype(np.uint8) for i in slices]
    
    
    return slices, slices_paths


def sample_slice_files(slice_files, slices_paths,slice_set):
    n = len(slice_files)
    assert len(slices_paths)==n,f"file path not consistent"
    
    if n > slice_set:
        # 等间隔采样
        indices = np.linspace(0, n - 1, slice_set, dtype=int)
        sampled_files = [slice_files[i] for i in indices]
        sampled_files_paths = [slices_paths[i] for i in indices]
    else:
        # 重复采样
        sampled_files = []
        sampled_files_paths = []
        for i in range(slice_set):
            sampled_files.append(slice_files[i % n])
            sampled_files_paths.append(slices_paths[i % n])
            
    return sampled_files,sampled_files_paths


    
class SurvivalDataset(MonaiDataset):
    def __init__(self,args, data: Sequence,tokenizer,training:bool, transform: Union[Callable[..., Any], None] = None) -> None:
        # sample data 
        transform = None
        self.data = data
        self.img_size = 256
        self.tokenizer = tokenizer
        self.max_length = 100000
        self.text_inputs_keys = args.text_inputs_keys
        self.pure_text_inputs = args.pure_text_inputs
        self.img_rpt_path = args.img_rpt_path
        self.training = training
        # for data_item in self.data:
        #     data_item['disc_label'] = torch.tensor(data_item['disc_label']) if int(data_item['mace'])==1 else torch.tensor(4.0)
        new_data = []
        for item in self.data:
            feat_path = os.path.join(self.img_rpt_path,item['processed_name'],"img_rpt.npy")
            if not os.path.exists(feat_path):
                print("skip path: ",feat_path)
                continue
            else:
                new_data.append(item)
        self.data = new_data
        
        self.mace1_list = [int(i['disc_label']) for i in self.data if int(i['mace'])==1]
        self.mace0_list = [int(i['disc_label']) for i in self.data if int(i['mace'])==0]

        # norm_trans = monai.transforms.ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True, channel_wise=False)
        
        # if self.training:
        #     self.data_augmentation_trans = transforms.Compose([
        #     transforms.Resize((self.img_size,self.img_size)),  # 调整图像大小
        #     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        #     transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        #     transforms.RandomRotation(degrees=15),  # 随机旋转
        #     transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0)),  # 随机裁剪并调整大小
        #     # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机仿射变换
        #     transforms.ToTensor(),  # 转换为张量
        #     norm_trans,
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        # ])
        # else:
        #     self.data_augmentation_trans = transforms.Compose([
        #     transforms.Resize((self.img_size,self.img_size)),  # 调整图像大小
        #     transforms.ToTensor(),  # 转换为张量
        #     norm_trans,
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        # ])
        labels0,label_counts0 = np.unique(self.mace0_list,return_counts=True)
        labels1,label_counts1 = np.unique(self.mace1_list,return_counts=True)
        dataset_name = self.data[0]['processed_name'].split('/')[0]
        print("="*70)
        print(f"[!]Dataset {dataset_name} length: ",len(data))
        print('mace1',labels1,label_counts1)
        print('mace0',labels0,label_counts0)
        print("="*70)
        super().__init__(self.data, transform)
        
    @property
    def weights(self):
        cls_cnt = {}
        for data_i in self.data:
            label_i = data_i['disc_label']
            if label_i not in cls_cnt: cls_cnt[label_i] = 0
            cls_cnt[label_i] += 1
        cls_cnt = {k: 1 / cls_cnt[k] for k in cls_cnt}
        weights = np.zeros((len(self.data),), dtype=np.float32)
        for idx, data_i in enumerate(self.data):
            label_i = data_i['disc_label']
            weights[idx] = cls_cnt[label_i]
        return weights
    
    @property
    def weights_dst(self):
        cls_cnt = {}
        for data_i in self.data:
            label_i = data_i['processed_name'].split('/')[0]
            # print(label_i)
            if label_i not in cls_cnt: cls_cnt[label_i] = 0
            cls_cnt[label_i] += 1
        cls_cnt = {k: 1 / cls_cnt[k] for k in cls_cnt}
        weights = np.zeros((len(self.data),), dtype=np.float32)
        for idx, data_i in enumerate(self.data):
            label_i = data_i['processed_name'].split('/')[0]
            weights[idx] = cls_cnt[label_i]
        return weights

    
    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        
        data_i = self.load_preprocessed_data(self.data[index])
        return data_i

        
    def load_preprocessed_data(self,data_item:dict):
        data_dict = {}
        feat_path = os.path.join(self.img_rpt_path,data_item['processed_name'],"img_rpt.npy")
        
        data_dict['img_feats'] = torch.tensor(np.load(feat_path))
        data_dict['y_disc'] = torch.tensor(data_item['disc_label'])
        data_dict['event_time'] = torch.tensor(data_item['survival_months'])
        data_dict['censor'] = torch.tensor(data_item['censorship'])
        
        # data_dict['Imaging_Findings'] = data_item['Imaging_Findings'] + data_item['Imaging_Diagnosis']
        data_dict['Imaging_Findings'] = data_item['Imaging_Findings']
        data_dict['imaging_diagnosis'] = data_item['Imaging_Diagnosis']
        ques_replace = "\nBased on the information above, please predict the likelihood of the patient experiencing a major adverse cardiovascular event (MACE) within 80 months following the imaging examination.  \nPlease answer with either **high** or **low**."
        
        for key in self.text_inputs_keys:
            if key not in data_dict:
                data_dict[key] = data_item[key]
                # print("yes")
                if key =="question":
                    data_dict[key] = data_dict[key].replace(ques_replace,"")
                    # we use mean pooling for input text tokens and found that it does not need replacement and this occasion will help image feature fusion
                    data_dict[key] = data_dict[key]
                if key == "Refined_Rationale_WithLabel":
                    if self.training:
                        pass
                    else:
                        data_dict[key] =  data_item["Initial_Model_Response"]
                    # data_dict[key] = data_item['question'].replace(ques_replace,"") + data_dict[key]
                if key == "Refined_Rationale_Model":
                    if self.training:
                        pass
                    else:
                        data_dict[key] = data_item["Initial_Model_Response"]

        return data_dict
    
    def custom_collate_fn(self,batch):
        """
        自定义 collate 函数：
        - 对 "texts" 使用 tokenizer 处理
        - 对其他 Key 使用默认的 collate 方式
        """
        batch_dict = {}

        # 遍历 batch 中的每个样本

        for key in batch[0].keys():
            if key in self.text_inputs_keys:
                # 对 "texts" 使用 tokenizer
                texts = [sample[key] for sample in batch]
                tokenized = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt",max_length=100000)
                batch_dict[key] = tokenized
            else:
                # 对其他 Key 使用默认的 collate 方式
                values = [sample[key] for sample in batch]
                if isinstance(values[0], torch.Tensor):
                    # 如果是 Tensor，使用 pad_sequence 或者直接 stack
                    if values[0].dim() == 0:  # 标量 Tensor
                        batch_dict[key] = torch.stack(values)
                    else:  # 非标量 Tensor
                        batch_dict[key] = torch.stack(values)
                        # batch_dict[key] = pad_sequence(values, batch_first=True, padding_value=0)
                elif isinstance(values[0], dict):
                    # 如果是嵌套字典，递归处理
                    batch_dict[key] = self.custom_collate_fn(values)
                else:
                    # 如果是其他类型（如列表、数字等），直接转换为 Tensor
                    batch_dict[key] = values
                        

        return batch_dict
        
        
