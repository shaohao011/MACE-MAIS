import os
import pydicom
import torch
import SimpleITK as sitk
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

check = False
# renji 
# dicom_root = 'data/renji/renji-full/'
# dest_root = "./data_processed/renji"
# # anzhen
# dicom_root = 'data/anzhen/anzhen_data-12-30-full/'
# dest_root = "./data_processed/anzhen"
# # tongji
# dicom_root = 'data/tongji/tongji_full/'
# dest_root = "./data_processed/tongji"
# # gulou
dicom_root = 'data/gulou/images/'
dest_root = "./data_processed/gulou"

cases = os.listdir(dicom_root)
new_cases = []
for case in cases:
    for root, dir, files in os.walk(os.path.join(dicom_root,case)):
        dir = [i for i in dir if not '._' in i]
        # if "1810081045/1810081045" in root:print(dir)
        if "CINE-SA" in dir:
            new_cases.append(root.replace(dicom_root,''))
        elif "CINE-4CH" in dir:
            new_cases.append(root.replace(dicom_root,''))
        elif "CINE-3CH" in dir:
            new_cases.append(root.replace(dicom_root,''))
            
            # print(new_cases)
            # exit()
cases = new_cases
views = ['cinesa', 'psir', 'et1m', 'nt1m','t2m','t2star','t2w']
info_mark_list = ['.DS_Store', 'DICOMDIR', 'LOCKFILE']

multi_exam_case = []
error_case = []
count_dict = {i:[] for i in views}
lock = threading.Lock()  # For thread-safe operations on shared resources

def organize_dicom(dicom_folder):
    for root ,dirs,files in os.walk(dicom_folder):
        if ".dcm" in files[0]:
            dicom_folder = root
    dicom_files = sorted(os.listdir(dicom_folder), key = lambda x: x.encode("gb2312"))
    temporal_dict = defaultdict(list)  # 用于按 SliceLocation 存储数据
    flag = False
    for i in dicom_files:
        if i.endswith('.dcm'):
            flag = True
            break
    if flag:
        dicom_files = [i for i in dicom_files if i.endswith('.dcm')]

    for dcm in dicom_files:
        # if not dcm.endswith('.dcm'):
        #     continue

        pattern = dcm
        dcm_path = f"{dicom_folder}/{dcm}"
        dicom_data = pydicom.dcmread(dcm_path,force=True)
        
        instance_number = dicom_data.get((0x0020, 0x0013), None).value
        image_position = tuple(dicom_data.get((0x0020, 0x0032), None).value)
        
        if image_position is not None and instance_number is not None:
            temporal_dict[image_position].append((instance_number, pattern))

    temporal_list = []
    for slice_location in sorted(temporal_dict.keys()):
        instances_sorted = sorted(temporal_dict[slice_location], key=lambda x: x[0])
        temporal_list.append([item[1] for item in instances_sorted])

    return temporal_list[::-1],dicom_folder

def load_and_process_dicom(idx_list, dicom_folder, case, view):
    all_slices = []

    for sublist in idx_list:
        temporal_slices = []
        
        for base_name in sublist:
            dicom_path = os.path.join(dicom_folder, f"{base_name}")
            dicom_data = pydicom.dcmread(dicom_path,force=True)
            pixel_array = dicom_data.pixel_array
            image = sitk.GetImageFromArray(pixel_array)
            temporal_slices.append(image)

        temporal_stack = sitk.JoinSeries(temporal_slices)
        all_slices.append(temporal_stack)

    final_result = sitk.JoinSeries(all_slices)
    return final_result

def load_dicom_nocine(dicom_folder):
    mod_slices = []
    
    dicom_files = []
    for root, _, files in os.walk(dicom_folder):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
            else:
                if "renji" in dicom_root:
                    continue
                else:
                    if file and root.split('/')[-1].lower().replace(' ','').replace('-','').replace('_','') in  views:
                        dicom_files.append(os.path.join(root, file))
    
    dicom_files.sort(key=lambda x: os.path.basename(x).encode("gb2312"))
    
    for dcm_path in dicom_files:
        dicom_data = pydicom.dcmread(dcm_path,force=True)
        pixel_array = dicom_data.pixel_array
        image = sitk.GetImageFromArray(pixel_array)
        mod_slices.append(image)
    
    mod_res = sitk.JoinSeries(mod_slices)
    return mod_res


def save_sitk_as_nifti(sitk_image, output_path):
    sitk.WriteImage(sitk_image, output_path)
    print(f"NIfTI file saved at: {output_path}")

def process_case(case):
    if case in error_case or case in info_mark_list:
        return
    
    case_path = os.path.join(dicom_root,case)
    if "._" in os.path.basename(case_path):
        return
    # case = case.split('/')[-2] if "anzhen" in dicom_root else case.split('/')[-1]
    mod_mapping = {}
    
    # First count all views for this case
    with lock:
        for view in views:
            if view not in [i.lower().replace(' ','').replace('-','').replace('_','').replace('cine4ch','cinesa').replace('cine3ch','cinesa') 
                           for i in os.listdir(case_path)]:
                count_dict[view].append(0)
                
    # Create mapping for existing views
    for i in os.listdir(case_path):
        mod_name = i.lower().replace(' ','').replace('-','').replace('_','').replace('cine4ch','cinesa').replace('cine3ch','cinesa') 
        for view in views:
            if view in mod_name:
                mod_mapping[i] = view
                break

    # Process each view for this case
    
    if os.path.exists(f"{dest_root}/{case}/cinesa.nii.gz"):
        print("[!!!!]no error,skip..................")
        return
    for view in mod_mapping:
        view_std = mod_mapping[view]
        # if os.path.exists(f"{dest_root}/{case}/{view_std}.nii.gz"):
        #     continue
        modal_path = f"{case_path}/{view}"
        if not os.path.exists(modal_path):
            continue
            
        
        try:
            if view_std == 'cinesa':
                try:
                    idx_list,dicom_folder = organize_dicom(modal_path)
                except:
                    idx_list,dicom_folder = organize_dicom(modal_path.replace('CINE-SA','CINE-4CH'))
                    
                sitk_4D_data = load_and_process_dicom(idx_list, dicom_folder, case, view)
                
                if sitk_4D_data is not None:
                    # print(f"{case} - {view_std}: {sitk_4D_data.GetSize()}")
                    os.makedirs(f"{dest_root}/{case}", exist_ok=True)
                    output_path = f"{dest_root}/{case}/{view_std}.nii.gz"
                    if not os.path.exists(output_path):
                        save_sitk_as_nifti(sitk_4D_data, output_path)
                    with lock:
                        count_dict[view_std].append(sitk_4D_data.GetSize()[-2])
                else:
                    with lock:
                        count_dict[view_std].append(0)
                    
            else:
                sitk_3D_data = load_dicom_nocine(modal_path)
                if sitk_3D_data is not None:
                    # print(f"{case} - {view_std}: {sitk_3D_data.GetSize()}")
                    os.makedirs(f"{dest_root}/{case}", exist_ok=True)
                    output_path = f"{dest_root}/{case}/{view_std}.nii.gz"
                    if not os.path.exists(output_path):
                        save_sitk_as_nifti(sitk_3D_data, output_path)
                    with lock:
                        count_dict[view_std].append(sitk_3D_data.GetSize()[-1])
                else:
                    with lock:
                        count_dict[view_std].append(0)
        except Exception as e:
            print(f"Error processing case {case}, view {view}: {str(e)}")
            with lock:
                error_case.append(case)


print(f"Total cases to process: {len(cases)}")

# Process cases with multi-threading
max_workers = 64  # Adjust based on your CPU cores and memory
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_case, case) for case in cases]
    
    for future in as_completed(futures):
        try:
            future.result()  # Get the result to catch any exceptions
        except Exception as e:
            print(f"Error in processing: {str(e)}")

print("Processing completed. Saving count dictionary...")
np.save('count_dict.npy', count_dict, allow_pickle=True)
print(count_dict)