# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from monai.data import CacheDataset,PersistentDataset, ThreadDataLoader,DataLoader, Dataset, DistributedSampler, SmartCacheDataset
import re
import os
import os.path as osp
import json
from utils.custom_trans import Sample_fix_seqd,RecordAffined,CenterCropForegroundd
from .masked_input_dataset import MaskedInputDataset
from dataset.dataset import PretrainDataset

def preprocess_data(ds_json):
    with open(ds_json, 'r') as f:
        ds_data = json.load(f)
        ds_data = ds_data['validation']+ ds_data['test']
        print("[!] exclued downstream validation and test: ",len(ds_data))
    replace_str = ""
    if "center2" in ds_json:
        nii_dir = "../data_processed/center2"
    elif "center1" in ds_json:
        nii_dir = "../data_processed/center1"
    elif "center3" in ds_json:
        nii_dir = "../data_processed/center3"
    elif "center4" in ds_json:
        nii_dir = "../data_processed/center4"
    else:
        raise ValueError("No such dataset")
    ds_data = {item['processed_name']:item for item in ds_data} # center2/...
    
    # we need to exclude the validation and test data for downstream tasks
    mod_parents = []
    skip_num = 0
    for root, dirs, files in os.walk(nii_dir):
        if files != []:
            if ".nii.gz" in files[0]:
                relative_path = os.path.relpath(root, start="../data_processed")
                if relative_path in ds_data:
                    # print(f"Skip {relative_path}")
                    skip_num += 1
                    continue
                else:
                    mod_parents.append(relative_path)
                # root: cur mod_parent
    print("skip num: ",skip_num)
    mod_parents = [os.path.join("../data_processed", item) for item in mod_parents] 
    return mod_parents

def get_loader(args):
    num_workers = args.num_workers
    
    total_data = []
    for idx,dataset_name in enumerate(args.dataset):
        mod_parents = preprocess_data(dataset_name)
        total_data.extend(mod_parents)
        
    if args.debug:
        total_data = total_data[:500]
    
    pretrain_ds = PretrainDataset(total_data)
    print("[!] len of pretrain_ds: ",len(pretrain_ds))

    if args.distributed:
        train_sampler = DistributedSampler(dataset=pretrain_ds, num_replicas=args.world_size, rank=args.rank)
    else:
        from torch.utils.data import RandomSampler
        train_sampler = RandomSampler(pretrain_ds)

    train_loader = DataLoader(
        pretrain_ds,
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory = True,
        # collate_fn=pad_list_data_collate
    )
        
    return train_loader