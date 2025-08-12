from monai.data import (DataLoader,CacheDataset)
from monai import transforms
import random
import os
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler as Sampler
from train_utils.sample import DistributedWeightedRandomSampler
from train_utils.transforms import *
from train_utils.dataset import *
from collections import defaultdict
from transformers import AutoTokenizer
import pandas as pd 
import re
    
# def get_item_by_id(data_list,id):

def check_data_cross(training, validation, test):
    # 将每个集合的 ID 提取出来
    train_ids = {item["ID"] for item in training}
    val_ids = {item["ID"] for item in validation}
    test_ids = {item["ID"] for item in test}
    
    # 检查训练集和验证集是否有交集
    train_val_intersection = train_ids.intersection(val_ids)
    if train_val_intersection:
        print("训练集和验证集有重复数据！")
        print("重复的 ID:", train_val_intersection)
        print("重复的 item:")
        for item in training:
            if item["ID"] in train_val_intersection:
                print(item)
        for item in validation:
            if item["ID"] in train_val_intersection:
                print(item)
        raise ValueError
    
    # 检查训练集和测试集是否有交集
    train_test_intersection = train_ids.intersection(test_ids)
    if train_test_intersection:
        print("训练集和测试集有重复数据！")
        print("重复的 ID:", train_test_intersection)
        print("重复的 item:")
        for item in training:
            if item["ID"] in train_test_intersection:
                print(item)
        for item in test:
            if item["ID"] in train_test_intersection:
                print(item)
        raise ValueError
    
    # 检查验证集和测试集是否有交集
    val_test_intersection = val_ids.intersection(test_ids)
    if val_test_intersection:
        print("验证集和测试集有重复数据！")
        print("重复的 ID:", val_test_intersection)
        print("重复的 item:")
        for item in validation:
            if item["ID"] in val_test_intersection:
                print(item)
        for item in test:
            if item["ID"] in val_test_intersection:
                print(item)
        raise ValueError

    # 如果没有交集
    if not train_val_intersection and not train_test_intersection and not val_test_intersection:
        print("训练集、验证集和测试集之间没有数据交叉。")


def hirachy_sample(args,data_list):
    random.seed(args.random_seed)  # 你可以选择任何整数作为种子
    use_hirachy_sample = True
    # use_hirachy_sample = False
    if use_hirachy_sample:
        grouped_data = defaultdict(list)
        for item in data_list:
            # print(item)
            grouped_data[str(int(item["disc_label"]))].append(item)
            # grouped_data[str(int(item["survival_months"]))].append(item)
            
            # grouped_data[str(int(item["mace"]))].append(item)
            
        training = []
        validation = []
        test = []
        for mace_value, items in grouped_data.items():
            random.shuffle(items)
            total = len(items)
            train_size = int(0.6 * total)
            val_size = int(0.2 * total)
            
            training.extend(items[:train_size])
            validation.extend(items[train_size:train_size + val_size])
            test.extend(items[train_size + val_size:])
    else:
        random.shuffle(data_list)
        random.shuffle(data_list)
        random.shuffle(data_list)
        training = []
        validation = []
        test = []
        
        total = len(data_list)
        train_size = int(0.6 * total)
        val_size = int(0.2 * total)
        
        training.extend(data_list[:train_size])
        validation.extend(data_list[train_size:train_size + val_size])
        test.extend(data_list[train_size + val_size:])
        

    check_data_cross(training, validation, test)
    return training,validation,test


def get_loader(args,tokenizer=None):
    with open(args.dataset[0],'r') as f:
        total_data = json.load(f)
        if "training" in total_data:
            train_list = total_data['training']
            val_list = total_data['validation'] 
            test_list = total_data['test']
            # we replace item with merged files
            # save data to the final folder
            if "Refined_Rationale_WithLabel" not in  train_list[0]:
                print("[!] using merged dataset for replacing items")
                dataset_name = args.dataset_name
                with open(f"./jsons/CoT_final/{dataset_name}/merged_results.json",'r') as f:
                    merged_total_data = json.load(f)
                merged_train_list = []
                merged_val_list = []
                merged_test_list = []
                for item in merged_total_data:
                    if item['split'] == "training":
                        merged_train_list.append(item)
                    elif item['split'] == "validation":
                        merged_val_list.append(item)
                    elif item['split'] == "test":
                        merged_test_list.append(item)
                    else:
                        raise ValueError
                merged_train_list = {item['ID']:item for item in merged_train_list}
                merged_val_list = {item['ID']:item for item in merged_val_list}
                merged_test_list = {item['ID']:item for item in merged_test_list}
                # print(merged_train_list.keys())
                for i in range(len(train_list)):
                    ID = train_list[i]["ID"]
                    train_list[i] = merged_train_list[ID]

                for i in range(len(val_list)):
                    ID = val_list[i]["ID"]
                    val_list[i] = merged_val_list[ID]

                for i in range(len(test_list)):
                    ID = test_list[i]["ID"]
                    test_list[i] = merged_test_list[ID]

                del merged_total_data
                del merged_train_list
                del merged_val_list
                del merged_test_list
                print("[!] replace successfully.....")
        else:
            pattern = r"\(reference range[^)]*\)"
            for data in total_data:
                data["Imaging_Findings"] = re.sub(pattern, "", data["Imaging_Findings"])
            train_list,val_list,test_list = hirachy_sample(args,total_data)
            
            # save selected seed
            total_save = {"training":train_list,"validation":val_list,"test":test_list}
            dataset_name = args.dataset.split('/')[-1].replace('.json','')
            save_file = os.path.join("./jsons/select_seed",f'{dataset_name}_{args.random_seed}.json')
            with open(save_file,'w') as f:
                json.dump(total_save,f,indent=4,ensure_ascii=False)
            print("="*40)
            print("[!] save elected json to ",save_file)
            print("="*40)
    
    check_data_cross(train_list, val_list, test_list)
    # save dataset to the final full info json
    # with open("./jsons/jsons_multi_center_wCoT")
    if args.rank==0:print(f"[!]Random seed {args.random_seed} dataset length: {len(train_list+val_list+test_list)} len(train) {len(train_list)} len(val) {len(val_list)} len(test) {len(test_list)}")
    test_list = train_list+val_list+test_list
    
    
    args.vary_slices = True
    if args.vary_slices:
        train_transforms = None
        val_transforms = train_transforms

    if args.test_mode:
        test_ds = CustomizedMonaiDataset(data=test_list, transform=None)
        val_sampler =  None
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, sampler=val_sampler, pin_memory=True,)
        loader = test_loader
    else:
        train_ds = SurvivalDataset(args,data=train_list,tokenizer=tokenizer, transform=train_transforms,training=True)
        val_ds = SurvivalDataset(args,data=val_list, tokenizer=tokenizer,transform=val_transforms,training=False)
        test_ds = SurvivalDataset(args,data=test_list,tokenizer=tokenizer, transform=val_transforms,training=False)
        
        from torch.utils.data import RandomSampler
        # train_sampler = WeightedRandomSampler(weights=train_ds.weights,num_samples=len(train_ds))
        train_sampler = RandomSampler(train_ds)
        # train_sampler = None
        if train_sampler!= None:
            print('using balanced sampling')
        # train_sampler = Sampler(train_ds) if args.distributed else None
        collate_fn = train_ds.custom_collate_fn
        # eval_bs = 1
        eval_bs = args.batch_size
        
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            # shuffle=None if train_sampler!=None else None,
            num_workers=args.num_workers,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
            collate_fn = collate_fn
            )
        val_loader = DataLoader(
            val_ds, 
            batch_size=eval_bs,
            shuffle=False,
            num_workers=args.num_workers, 
            sampler=None, 
            pin_memory=True, 
            collate_fn = collate_fn
        )
        test_loader = DataLoader(
            test_ds, 
            batch_size=eval_bs,
            shuffle=False,
            num_workers=args.num_workers, 
            sampler=None, 
            pin_memory=True, 
            collate_fn = collate_fn
        )
        shap_loader = DataLoader(
            test_ds, 
            batch_size=eval_bs,
            shuffle=False,
            num_workers=args.num_workers, 
            sampler=None, 
            pin_memory=True, 
        )
    loader = [train_loader, val_loader,test_loader,shap_loader]

    return loader
    
