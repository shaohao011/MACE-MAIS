import json
import os
import numpy as np
import torch
import nibabel as nib
import monai
from torchvision import transforms
from torch.nn.functional import interpolate
import random

def get_data_replace(input_str):
    if "anzhen" in input_str:
        replace_str = "data/anzhen/anzhen_data-12-30-full/"
        nii_dir = "../data_processed/anzhen/"
    elif "renji" in input_str:
        replace_str = "./data/renji/renji-full/"
        nii_dir = "../data_processed/renji/"
    elif "tongji" in input_str:
        replace_str = "data/tongji/tongji_full/"
        nii_dir = "../data_processed/tongji/"
    elif "gulou" in input_str:
        replace_str = "data/gulou/images/"
        nii_dir = "../data_processed/gulou/"
    else:
        raise ValueError("No such dataset")
    return [replace_str, nii_dir]

def find_other_time_exam(item,original_path=None):

    print(f"[WARN] Folder not found: {original_path}")
    
    patient_dir = os.path.dirname(original_path)
    
    if os.path.isdir(patient_dir):
        candidates = []
        for subdir in os.listdir(patient_dir):
            sub_path = os.path.join(patient_dir, subdir)
            if os.path.isdir(sub_path):
                file_list = os.listdir(sub_path)
                if len(file_list) > 1:
                    candidates.append((subdir, len(file_list)))

        if candidates:
            best_candidate = sorted(candidates, key=lambda x: -x[1])[0][0]
            print(f"[INFO] Replacing with alternative folder: {best_candidate}")
            new_path = os.path.join(patient_dir, best_candidate)
            
            item['mod_parent'] = new_path
            mod_list = os.listdir(new_path)
        else:
            print(f"[ERROR] No valid alternative found in {patient_dir}")
            mod_list = []  # 或 raise Exception
    else:
        print(f"[ERROR] Parent folder does not exist: {patient_dir}")
        mod_list = []  # 或 raise Exception
    return mod_list,item

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, total_list):
        self.img_size = 224
        
        self.total_data = total_list
        self.norm_trans = monai.transforms.ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True, channel_wise=False)
        self.total_list = []
        img_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/shaohao/renji/R1_Cardio/data_processed"
        
        for mod_parent in self.total_data:
            # print(mod_parent['mod_parent'])
            if not os.path.exists(mod_parent):
                continue
            
            mod_list = os.listdir(mod_parent)
            if "anzhen" in mod_parent:
                try:
                    if "cinesa.nii.gz" not in mod_list or "psir.nii.gz" not in mod_list:
                        mod_list,mod_parent = find_other_time_exam(mod_parent)
                except:
                    mod_list,mod_parent = find_other_time_exam(mod_parent)
                    
            # print(os.listdir(mod_parent))
            if "cinesa.nii.gz" not in mod_list:
                pass
                print("[!]CINE-SA not found in ",mod_parent)
            
            if len(os.listdir(mod_parent))==1:
                continue
            
            self.total_list.append(mod_parent)
        del self.total_data   
        print("dataset length: ",len(self.total_list))
        super(PretrainDataset, self).__init__()
    
    def __len__(self):
        return len(self.total_list)
    
    
    def _resize_img(self, img):
        # Resize H and W dimensions to self.img_size
        # img shape could be [T, H, W, D] or [1, H, W, D]
            # Ensure tensor is contiguous
        img = img.contiguous()
        
        # Get original shape
        orig_shape = img.shape
        
        # For non-cine cases where we have [1, H, W, D], we need to handle differently
        if orig_shape[0] == 1:  # Non-cine case
            # Remove the channel dimension temporarily
            img = img.squeeze(0)  # [H, W, D]
            
            # Add batch dimension for interpolation
            img = img.unsqueeze(0)  # [1, H, W, D]
            try:
                img = img.permute(0, 3, 1, 2)  # [1, D, H, W]
            except:
                print(img.shape)
            
            # Resize
            # print("before resize",img.shape)
            img = interpolate(img, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            # print("after resize",img.shape)
            
            # Permute back and restore original shape
            img = img.permute(0, 2, 3, 1)  # [1, H, W, D]
            img = img.squeeze(0)  # [H, W, D]
            img = img.unsqueeze(0)  # [1, H, W, D]
        
        else:  # Cine case [T, H, W, D]
            # Correct processing for cine sequences:
            # 1. First permute to [T, D, H, W] for interpolation
            img = img.permute(0, 3, 1, 2)  # [T, D, H, W]
            
            # 2. Resize each frame separately
            # We need to reshape to [T*D, 1, H, W] for 2D interpolation
            t, d, h, w = img.shape
            img = img.reshape(t * d, 1, h, w)  # [T*D, 1, H, W]
            
            # Use bilinear interpolation since we're processing 2D slices
            img = interpolate(img, size=(self.img_size, self.img_size),
                            mode='bilinear', align_corners=False)
            
            # 3. Reshape back to original dimensions
            img = img.reshape(t, d, self.img_size, self.img_size)  # [T, D, H', W']
            
            # 4. Permute back to [T, H, W, D]
            img = img.permute(0, 2, 3, 1)
        return img

    def _process_cine(self, img):
        # img shape: [T, H, W, D]
        T = img.shape[0]
        
        # Select temporal frames (middle and adjacent frames)
        if T >= 3:
            mid = T // 2
            selected_frames = [mid-1, mid, mid+1] if T > 2 else [0, mid, -1]
            selected_frames = [min(max(f, 0), T-1) for f in selected_frames]
            img = img[selected_frames]
        else:
            # If fewer than 3 frames, randomly select with replacement
            selected_indices = np.random.choice(T, size=3, replace=True if T < 3 else False)
            img = img[selected_indices]
        
        # Process depth dimension (D) - ensure 25 slices
        D = img.shape[3]
        frame_num = 10
        if D < frame_num:
            # If fewer than frame_num slices, replicate existing slices to reach frame_num
            repeat_times = (frame_num + D - 1) // D  # Ceiling division
            img = img.repeat(1, 1, 1, repeat_times)[:, :, :, :frame_num]
        elif D > 10:
            # If more than frame_num slices, sample evenly
            sample_indices = torch.linspace(0, D-1, steps=frame_num).long()
            img = img[:, :, :, sample_indices]
        
        return img
    
    def _process_non_cine(self, img):
        # img shape: [1, H, W, D]
        D = img.shape[3]
        
        # Select middle 3 slices in depth dimension
        if D >= 3:
            mid = D // 2
            selected_slices = [mid-1, mid, mid+1] if D > 2 else [0, mid, -1]
            selected_slices = [min(max(s, 0), D-1) for s in selected_slices]
            img = img[:, :, :, selected_slices]
        else:
            # If fewer than 3 slices, randomly select with replacement
            selected_indices = np.random.choice(D, size=3, replace=True if D < 3 else False)
            img = img[:, :, :, selected_indices]
        
        return img
    
    
    def __getitem__(self, idx):
        mod_parent = self.total_list[idx]
        # print(mod_parent)
        mod_dict = {}
        mod_dict['non_cine'] = []
        mod_dict['mod_parent'] = mod_parent
        mod_name = []
        # we should do sample here during pre-training
        k = 2
        # we should do sample here
        filtered_list = [item for item in os.listdir(mod_parent) if "cinesa.nii.gz" not in item]
        if len(filtered_list) >= k:
            sampled_list = random.sample(filtered_list, k)
        else:
            sampled_list = random.choices(filtered_list, k=k)
           
            
        sampled_list.append("cinesa.nii.gz")
        count = 1
        # print(sampled_list)
        for mod in sampled_list:
            # try:
            # Load image
            file_path = os.path.join(mod_parent, mod) 
            # print(file_path) 
            if not os.path.exists(file_path):
                file_path = file_path.replace('cinesa','psir')
                img = nib.load(file_path)
            else:
                img = nib.load(file_path)
            img = img.get_fdata()
            img = torch.tensor(img).float()
            # Process based on modality type
            if "cine" in mod:
                try:
                    img = img.permute(3, 0, 1, 2)  # [T, H, W, D]
                except:
                    img = img.unsqueeze(0).repeat(3,1,1,1)
                img = self._process_cine(img)
            else:
                if len(img.shape) == 4:
                    # img = img.squeeze(0)
                    continue
                else:
                    img = img.unsqueeze(0)  # [1, H, W, D]
                    # print(img.shape)
                    img = self._process_non_cine(img)
            # Resize H and W dimensions
            img = self._resize_img(img)
             # Normalize
            img = self.norm_trans(img)
            cur_mod = mod.replace('.nii.gz', '')
            if cur_mod in mod_dict and "cine" not in cur_mod:
                cur_mod+=f"_{count}"
            
            if "cine" not in cur_mod:
                mod_dict['non_cine'].append(img) # maybe overlap
                mod_name.append(cur_mod)
            else:
                mod_dict[cur_mod] = img # maybe overlap
            # except Exception as e:
            #     print(f"Error processing {mod}: {str(e)}")
            #     continue
        mod_dict['mod_name'] = ",".join(mod_name)
        # print(len(mod_dict['non_cine']))
        try:
            mod_dict['non_cine'] = torch.stack(mod_dict['non_cine'])
        except:
            if mod_dict['non_cine'] == []:
                mod_dict['non_cine'] = mod_dict['cinesa'][1:2,:,:,:3].unsqueeze(dim=0).repeat(k,1,1,1,1)
        # we need to concat the non-cine modalities
        if mod_dict['non_cine'].shape[0] !=k:
            print("yes",mod_parent)
            mod_dict['non_cine'] = mod_dict['non_cine'].repeat(k,1,1,1,1)
            
        return mod_dict
    

    
class GetImbeddingDataset(PretrainDataset):
    def __init__(self, json_path):
        self.img_size = 224
        with open(json_path,'r') as f:
            total_data = json.load(f)
        try:
            total_list = total_data['training'] + total_data['validation'] + total_data['test']
        except:
            total_list = total_data
        total_list = total_list
        print("preload length: ",len(total_list))
        # we need to know its name
        self.total_data = total_list
        self.norm_trans = monai.transforms.ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True, channel_wise=False)
        # exclude case without cine
        self.total_list = []
        img_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/ruishaohao-240108100042/shaohao/renji/R1_Cardio/data_processed"
        
        for item in self.total_data:
            # print(mod_parent['mod_parent'])
            cur_mod_path = os.path.join(img_dir,item['processed_name'])
            if not os.path.exists(cur_mod_path):
                continue
            try:
                mod_list = os.listdir(cur_mod_path)
                if "cinesa.nii.gz" not in mod_list and "anzhen" in item['processed_name']:
                    mod_list = find_other_time_exam(item,item['mod_parent'])
            except:
                if "anzhen" in item['processed_name']:
                    mod_list = find_other_time_exam(item,item['mod_parent'])
                
            # print(os.listdir(mod_parent))
            if "cinesa.nii.gz" not in mod_list:
                print("[!]CINE-SA not found in ",cur_mod_path)
                # continue
                pass
            elif len(os.listdir(cur_mod_path))==1:
                # continue
                pass
            item['img_rpt_path'] = cur_mod_path
            self.total_list.append(item)
        del self.total_data   
        print("dataset length: ",len(self.total_list))
        super(PretrainDataset, self).__init__()
    
    def __len__(self):
        return len(self.total_list)
    
    def __getitem__(self, idx):
        case_item = self.total_list[idx]
        cur_mods_path = case_item['img_rpt_path']
        
        mod_dict = {}
        mod_dict['non_cine'] = []
        mod_dict['img_rpt_path'] = cur_mods_path
        
        mod_name = []
        # we should do sample here
        filtered_list = [item for item in os.listdir(cur_mods_path) if "cinesa.nii.gz" not in item]
       
            
        sampled_list = filtered_list
        sampled_list.append("cinesa.nii.gz")
        count = 1
        # print(sampled_list)
        for mod in sampled_list:
            # try:
            # Load image
            file_path = os.path.join(cur_mods_path, mod) 
            # print(file_path) 
            if not os.path.exists(file_path):
                file_path = file_path.replace('cinesa','psir')
                img = nib.load(file_path)
            else:
                img = nib.load(file_path)
            img = img.get_fdata()
            img = torch.tensor(img).float()
            # Process based on modality type
            if "cine" in mod:
                try:
                    img = img.permute(3, 0, 1, 2)  # [T, H, W, D]
                except:
                    img = img.unsqueeze(0).repeat(3,1,1,1)
                img = self._process_cine(img)
            else:
                if len(img.shape) == 4:
                    # img = img.squeeze(0)
                    continue
                else:
                    img = img.unsqueeze(0)  # [1, H, W, D]
                    # print(img.shape)
                    img = self._process_non_cine(img)
            # Resize H and W dimensions
            img = self._resize_img(img)
            # except:
            #     # print(img.shape)
            #     # print(mod)   
            #     # print(mod_parent)
            #     raise ValueError
                
            
             # Normalize
            img = self.norm_trans(img)
            cur_mod = mod.replace('.nii.gz', '')
            if cur_mod in mod_dict and "cine" not in cur_mod:
                cur_mod+=f"_{count}"
        
            
            if "cine" not in cur_mod:
                mod_dict['non_cine'].append(img) # maybe overlap
                mod_name.append(cur_mod)
            else:
                mod_dict[cur_mod] = img # maybe overlap
            # except Exception as e:
            #     print(f"Error processing {mod}: {str(e)}")
            #     continue
        mod_dict['mod_name'] = ",".join(mod_name)
        # print(len(mod_dict['non_cine']))
        try:
            mod_dict['non_cine'] = torch.stack(mod_dict['non_cine'])
        except:
            if mod_dict['non_cine'] == []:
                mod_dict['non_cine'] = mod_dict['cinesa'][1:2].unsqueeze(dim=0)
        # we need to concat the non-cine modalities
        mod_dict['processed_name'] = case_item['processed_name']
        return mod_dict
    
if __name__=="__main__":
    ds_json="../jsons/anzhen.json"
    nii_dir="../data_processed/anzhen"
    dst = PretrainDataset(ds_json,nii_dir)
    print(dst[0].keys())
    print(dst[0]['cinesa'].shape)
    print(dst[0]['cinesa'].max(),dst[0]['cinesa'].min())
    