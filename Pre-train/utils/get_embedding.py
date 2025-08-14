import sys
sys.path.append('../')
from models.Uniformer import RecModel
import torch
import os
class ARGS():
    def __init__(self) -> None:
        self.device = 'cuda:0'
        self.in_channels = 3
        self.roi_x = 96
        self.initial_checkpoint = ""

args = ARGS()
model = RecModel(args, dim=512) # Dimension of the inter feature
ckpt_dir = "../runs/uniformer_4_center"
ckpt = torch.load(os.path.join(ckpt_dir,"model_300.pt"),map_location='cpu')
# model.load_state_dict(ckpt['state_dict'], strict=True)
new_state_dict = {}
for k, v in ckpt['state_dict'].items():
    new_key = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict, strict=True)

# we need to create a 
from utils.data_utils import get_loader

dataset_name = "center4"
json_path = f"../../jsons/jsons_final_full_info/{dataset_name}.json" # you should preprocess tongji first

from dataset.dataset import GetImbeddingDataset
rpt_dataset = GetImbeddingDataset(json_path)

rpt_loader = torch.utils.data.DataLoader(rpt_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,drop_last=False)

import os
import numpy as np
from tqdm import tqdm
model.to(args.device)
model.eval()

save_folder = f"../../img_rpt/uniformer_4_center"
for case in tqdm(rpt_loader):
    for key in case:
        if key in  ["non_cine","cinesa"]:
            case[key] = case[key].to(args.device)

    # print(original_path)
    with torch.no_grad():
        losses, scores, img_rpt = model(case) 

    
    save_path = os.path.join(save_folder,case['processed_name'][0])
    os.makedirs(save_path,exist_ok=True)
    save_path = os.path.join(save_path,'img_rpt.npy')

    np.save(save_path, img_rpt.detach().cpu().numpy())