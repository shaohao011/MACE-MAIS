import json
import pandas as pd
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.data import CacheDataset
from train_utils.sample import DistributedWeightedRandomSampler
from train_utils.transforms import *
from train_utils.dataset import *
from train_utils.data_utils_survival import hirachy_sample, check_data_cross
from torch.utils.data import RandomSampler

def generate_bin_edges(all_data, label_col="survival_months", censorship_col="censorship", n_bins=4, eps=1e-4):
    df = pd.DataFrame(all_data)
    uncensored_df = df[df[censorship_col] < 1]
    disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False, duplicates='drop')
    q_bins[0] = df[label_col].min() - eps
    q_bins[-1] = df[label_col].max() + eps
    return q_bins.tolist()

def apply_bins_to_data(data_list, bin_edges, label_col="survival_months"):
    df = pd.DataFrame(data_list)
    disc_labels = pd.cut(df[label_col], bins=bin_edges, labels=False, right=False, include_lowest=True)
    df['disc_label'] = disc_labels.astype(int)
    for i, item in enumerate(data_list):
        item['disc_label'] = int(df.iloc[i]['disc_label'])
    return data_list

def replace_with_merged_info(dataset_name):
    dst_path = os.path.join("./jsons/jsons_final_full_info", f"{dataset_name}.json")
    with open(dst_path, "r") as f:
        ori_data = json.load(f)
    train_list = ori_data['training']
    val_list = ori_data['validation']
    test_list = ori_data['test']
    
    print(f"[!] Dataset {dataset_name}: using merged dataset for replacing items")
    merged_path = os.path.join("./jsons/CoT_final", dataset_name, "merged_results.json")
    with open(merged_path, "r") as f:
        merged_total_data = json.load(f)
    merged_train_list = {item['ID']: item for item in merged_total_data if item['split'] == "training"}
    merged_val_list = {item['ID']: item for item in merged_total_data if item['split'] == "validation"}
    merged_test_list = {item['ID']: item for item in merged_total_data if item['split'] == "test"}

    for i in range(len(train_list)):
        ID = train_list[i]["ID"]
        train_list[i] = merged_train_list[ID]
    for i in range(len(val_list)):
        ID = val_list[i]["ID"]
        val_list[i] = merged_val_list[ID]
    for i in range(len(test_list)):
        ID = test_list[i]["ID"]
        test_list[i] = merged_test_list[ID]

    print(f"[!] Dataset {dataset_name}: replace successfully.....")
    return train_list, val_list, test_list

def get_loader(args, tokenizer=None):
    internal_train_list = []
    internal_test_dict = {}

    for ds_name in args.dataset:
        train_list, val_list, test_list = replace_with_merged_info(ds_name)
        check_data_cross(train_list, val_list, test_list)
        internal_train_list.extend(train_list + val_list + test_list)
        internal_test_dict[ds_name] = test_list  

    # 获取外部数据集名（作为验证+测试来源）
    full_set = {"center1", "center2", "center3", "center4"}
    internal_set = set(args.dataset)
    external_set = full_set - internal_set
    if len(external_set) != 1:
        raise ValueError("外部数据集应唯一，请确认输入的 args.dataset")
    external_ds = external_set.pop()
    external_train_list, external_val_list, external_test_list = replace_with_merged_info(external_ds)

    # 验证集 = 外部中心的 val + test
    external_val_list = external_val_list + external_test_list

    # 构造统一分箱边界
    all_merged_data = []
    for center in ["center1", "center2", "center3", "center4"]:
        merged_path = os.path.join("./jsons/CoT_final", center, "merged_results.json")
        with open(merged_path, "r") as f:
            merged_data = json.load(f)
        all_merged_data.extend(merged_data)

    bin_edges = generate_bin_edges(all_merged_data, label_col="survival_months", censorship_col="censorship", n_bins=4)

    # 应用统一分箱
    print("=" * 70)
    print("[!] Use same bin split for each center......")
    internal_train_list = apply_bins_to_data(internal_train_list, bin_edges)[:50]
    external_val_list = apply_bins_to_data(external_val_list, bin_edges)
    external_train_list = apply_bins_to_data(external_train_list, bin_edges)  # 用作最终测试集
    
    external_val_list = external_val_list + external_train_list
    print("=" * 70)

    if args.rank == 0:
        print(f"[!] Unified bin edges: {bin_edges}")
        print(f"[!] Train set (from 3 centers) size: {len(internal_train_list)}")
        print(f"[!] Val set (external center val+test): {len(external_val_list)}")
        print(f"[!] Test set (external center train): {len(external_train_list)}")

    args.vary_slices = True
    train_transforms = None if args.vary_slices else None
    val_transforms = train_transforms
    # eval_bs = 1
    eval_bs = args.batch_size

    if args.test_mode:
        val_ds = CustomizedMonaiDataset(data=external_val_list, transform=None)
        test_ds = CustomizedMonaiDataset(data=external_train_list, transform=None)

        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, sampler=None, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, sampler=None, pin_memory=True)

        loader = {"val": val_loader, "external_test": test_loader}
    else:
        train_ds = SurvivalDataset(args, data=internal_train_list, tokenizer=tokenizer, transform=train_transforms, training=True)
        val_ds = SurvivalDataset(args, data=external_val_list, tokenizer=tokenizer, transform=val_transforms, training=False)
        test_ds = SurvivalDataset(args, data=external_train_list, tokenizer=tokenizer, transform=val_transforms, training=False)

        # train_sampler = WeightedRandomSampler(weights=train_ds.weights_dst, num_samples=len(train_ds))
        train_sampler = RandomSampler(train_ds)
        collate_fn = train_ds.custom_collate_fn

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=train_sampler, pin_memory=True, drop_last=False, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False,
                                num_workers=args.num_workers, sampler=None, pin_memory=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False,
                                 num_workers=args.num_workers, sampler=None, pin_memory=True, collate_fn=collate_fn)

        loader = {
            "train": train_loader,
            "val": val_loader,
            "external_test": test_loader
        }

    return loader
