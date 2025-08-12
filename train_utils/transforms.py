from __future__ import annotations
from monai.transforms import MapTransform,Randomizable, CropForeground, CropForegroundd,Cropd
from typing import Optional
from numpy import ndarray
import torch
import random

"""
A collection of "vanilla" transforms for crop and pad operations.
"""
from collections.abc import Callable, Sequence
import numpy as np
import torch
from monai.config import IndexSelection
from monai.transforms.croppad.functional import crop_func, pad_func
from monai.transforms.transform import LazyTransform, Randomizable
from monai.transforms.utils import (
    is_positive,
)
from monai.utils import ImageMetaKey as Key
from monai.utils import (
    PytorchPadMode,
    deprecated_arg_default,
    ensure_tuple_rep,
)
from collections.abc import Callable, Hashable, Mapping, Sequence
import numpy as np
import torch
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms.croppad.array import (
    CropForeground,
)
from monai import transforms
from monai.transforms.transform import LazyTransform, MapTransform, Randomizable
from monai.transforms.utils import is_positive
from monai.utils import MAX_SEED, Method, PytorchPadMode, deprecated_arg_default, ensure_tuple_rep

def create_collate_fn(pad_keys=[]):
    def collate_fn(batch):
        other_keys = {key: [] for key in batch[0].keys() if key not in pad_keys}
        for key in other_keys:
            other_keys[key] = torch.tensor([item[key] for item in batch])

        
        pad_keys_list = {key: [] for key in batch[0].keys() if key in pad_keys}
        
        for pad_key in pad_keys_list:
            max_z = max(item[pad_key].shape[3] for item in batch)
            padded_data = []
            for item in batch:
                data = item[pad_key]
                # print(data.shape,max_z)
                pad_width = max_z - data.shape[3]
                padded_data.append(torch.nn.functional.pad(data, (0, pad_width)))
            padded_data = torch.stack(padded_data, dim=0)
            pad_keys_list[pad_key]  = padded_data
        # 将 'psir_data' 和其他键的值合并成一个字典
        batch_dict = {**pad_keys_list, **other_keys}
        
        return batch_dict
    return collate_fn

class Randomzoomd(MapTransform,Randomizable):
    def __init__(
        self,
        keys,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        wrap_sequence: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
    
        super().__init__(keys, allow_missing_keys)
    

    def __call__(self, data):
        d = dict(data) 
        zoom_size = random.uniform(0.5, 1)
        zoom_trans = transforms.Zoom(zoom=zoom_size,align_corners=True,keep_size=True,mode="trilinear")
        d['image'] = zoom_trans(d['image']) 
        return d
    
    


import numpy as np
from monai.transforms import MapTransform

class CropToMaskd(MapTransform):
    def __init__(self, keys, mask_key):
        """
        Args:
            keys (list): Keys of the data to be cropped.
            mask_key (str): Key of the mask used for cropping.
        """
        self.keys = keys
        self.mask_key = mask_key

    def __call__(self, data):
        mask = data[self.mask_key]
        # print(mask.shape)
        mask_indices = np.where(mask != 0)
        # print(mask_indices,len(mask_indices))
        # if len(mask_indices[0]) == 0:
        #     raise ValueError("Mask does not contain any non-zero values.")

        # Determine height and width boundaries (H and W axes)
        # print(mask_indices[0].shape)
        min_y = np.min(mask_indices[1])
        max_y = np.max(mask_indices[1])
        min_x = np.min(mask_indices[2])
        max_x = np.max(mask_indices[2])

        # Original image dimensions
        original_height, original_width = mask.shape[1], mask.shape[2]
        original_aspect_ratio = original_height / original_width

        # Crop height and width
        crop_height = max_y - min_y + 1
        crop_width = max_x - min_x + 1
        crop_aspect_ratio = crop_height / crop_width

        # Adjust crop dimensions to match the original aspect ratio
        if crop_aspect_ratio > original_aspect_ratio:
            # Height is relatively larger, adjust width
            target_width = int(crop_height / original_aspect_ratio)
            center_x = (min_x + max_x) // 2
            min_x = max(0, center_x - target_width // 2)
            max_x = min(original_width - 1, center_x + target_width // 2)
        elif crop_aspect_ratio < original_aspect_ratio:
            # Width is relatively larger, adjust height
            target_height = int(crop_width * original_aspect_ratio)
            center_y = (min_y + max_y) // 2
            min_y = max(0, center_y - target_height // 2)
            max_y = min(original_height - 1, center_y + target_height // 2)

        # Ensure the adjusted crop is within the image boundaries
        min_y = max(0, min_y)
        max_y = min(original_height - 1, max_y)
        min_x = max(0, min_x)
        max_x = min(original_width - 1, max_x)

        # Crop the mask
        data[self.mask_key] = data[self.mask_key][:, min_y:max_y+1, min_x:max_x+1]

        # Crop the data for all specified keys
        for key in self.keys:
            data[key] = data[key][:, min_y:max_y+1, min_x:max_x+1]

        return data


from monai.transforms import MapTransform
import torch

class SymmetricSliced(MapTransform):
    """
    自定义 MONAI 变换，用于对称地在指定维度上选取固定数量的切片。
    适用于字典形式的数据。

    参数:
        keys (list[str]): 要应用变换的键列表。
        num_slices (int): 选取的切片数量。
        dim (int): 在哪个维度进行切片，默认为 -1 (最后一维)。
    """
    def __init__(self, keys, num_slices=8, dim=-1):
        super().__init__(keys)
        self.num_slices = num_slices
        self.dim = dim

    def __call__(self, data):
        for key in self.keys:
            tensor = data[key]
            size = tensor.shape[self.dim]
            if size == self.num_slices:
                continue
            elif size < self.num_slices:
                # raise ValueError
                continue

            # 计算中心点及切片范围
            center = size // 2
            start = max(center - self.num_slices // 2, 0)
            end = min(start + self.num_slices, size)

            # 调整切片范围确保不越界
            if end - start < self.num_slices:
                start = max(end - self.num_slices, 0)

            # 索引切片
            indices = torch.arange(start, end, device=tensor.device)
            data[key] = tensor.index_select(self.dim, indices)
        return data

class EquallySpacedSliced(MapTransform):
    """
    自定义 MONAI 变换，用于在指定维度上等距选取固定数量的切片。
    适用于字典形式的数据。

    参数:
        keys (list[str]): 要应用变换的键列表。
        num_slices (int): 选取的切片数量。
        dim (int): 在哪个维度进行切片，默认为 -1 (最后一维)。
    """
    def __init__(self, keys, num_slices=8, dim=-1):
        super().__init__(keys)
        self.num_slices = num_slices
        self.dim = dim

    def __call__(self, data):
        for key in self.keys:
            tensor = data[key]
            size = tensor.shape[self.dim]
            if size == self.num_slices:
                continue
            elif size < self.num_slices:
                # 如果切片数量大于维度大小，可以选择抛出异常或跳过
                # raise ValueError(f"Number of slices ({self.num_slices}) is greater than the dimension size ({size}).")
                continue

            # 计算等距采样的索引
            step = size / self.num_slices
            indices = torch.linspace(0, size - 1, self.num_slices, dtype=torch.long, device=tensor.device)

            # 索引切片
            data[key] = tensor.index_select(self.dim, indices)
        return data

# Example usage:
# transform = CropToMaskd(keys=["psir_data", "t2w_data", 'psir_mask', 't2w_mask'], mask_key='psir_mask')


class ComposeTranForVaryinputs(MapTransform):
   
    def __init__(self,keys,img_size=256):
        self.img_size = img_size
        super().__init__(keys)
        

    def __call__(self, data):
        # data : {'image_dict'} 
        process_keys = data['image_dict'].keys()
        print(data['image_dict']['PSIR'].shape)
        compose_trans = transforms.Compose(
            [
            transforms.EnsureChannelFirstd(keys=process_keys,channel_dim=0),
            # transforms.Resized(keys=process_keys, spatial_size=(self.img_size, self.img_size,-1), mode="trilinear"),
            # transforms.ScaleIntensityRangePercentilesd(keys=process_keys, lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True, channel_wise=True),
            # transforms.RandRotated(keys=process_keys,range_x=0, range_y=0, range_z=(-45, 45), prob=1.0, keep_size=True),
            # transforms.RandAdjustContrastd(keys=process_keys,prob=1.0, gamma=(0.8, 1.2)),
            # transforms.RandFlipd(keys=process_keys,spatial_axis=2, prob=0.5),
            ]
        )
        data['image_dict'] = compose_trans(data['image_dict'])
        print(data['image_dict']['PSIR'].shape)
        exit()
        return data
    
def create_collate_fn_text_image(tokenizer,model):
    def collate_fn(batch):
        other_keys = {key: [] for key in batch[0].keys() if key not in ["text_inputs"]}
        for key in other_keys:
            other_keys[key] = torch.stack([item[key] for item in batch])

        text_embedding = {"text_embedding":[]}
        device = model.device
        print(device)
        text_list = [i['text_inputs'] for i in batch]
        inputs = tokenizer(text_list, return_tensors="pt", truncation = True, max_length = 1024,padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        text_embedding[text_embedding] = model(**inputs).last_hidden_state.mean(0)
        batch_dict = {**text_embedding, **other_keys}
        
        return batch_dict
    return collate_fn
        
        