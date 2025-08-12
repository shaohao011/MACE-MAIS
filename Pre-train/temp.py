import os
import pydicom
import torch
import SimpleITK as sitk
from collections import defaultdict
import numpy as np

check = False
dicom_root = '../data/renji/renji-full'
dest_root = "../data_processed/renji"
cases = os.listdir(dicom_root)
# views = ['CINE-SA', 'CINE-2CH', 'CINE-3CH', 'CINE-4CH'][0:1]
views = ['cinesa', 'psir', 'et1m', 'nt1m','t2m','t2star','t2w']
info_mark_list = ['.DS_Store', 'DICOMDIR', 'LOCKFILE']

multi_exam_case = []
error_case = []
# 错误日志文件
# error_log_file = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/suhaoyang-240107100018/research/Monai/Data_process_new/data_process_renji/renji_error_samples_sitk.txt"
# 定义异常处理函数
# def log_error(case, view, pattern, error_message):
#     with open(error_log_file, "a") as f:
#         f.write(f"Error occurred for case: {case}, view: {view}, pattern: {pattern}\n")
#         f.write(f"Error message: {error_message}\n")
#         f.write("=" * 50 + "\n")

def organize_dicom(dicom_folder):
    dicom_files = sorted(os.listdir(dicom_folder), key = lambda x: x.encode("gb2312"))
    temporal_dict = defaultdict(list)  # 用于按 SliceLocation 存储数据

    # 遍历文件夹中的 DICOM 文件
    for dcm in dicom_files:  # 确保文件按名称排序
        if not dcm.endswith('.dcm'):
            continue

        pattern = dcm.split('.dcm')[0]
        dcm_path = f"{dicom_folder}/{dcm}"
        dicom_data = pydicom.dcmread(dcm_path)
        
        # 提取 SliceLocation 和 InstanceNumber
        instance_number = dicom_data.get((0x0020, 0x0013), None).value
        image_position = tuple(dicom_data.get((0x0020, 0x0032), None).value)
        # print(instance_number) # 1, 
        # print(image_position) # ('26.0997489909641', '-144.79894559888', '233.792237067362')
        # 如果 SliceLocation 和 InstanceNumber 存在，存储数据
        if image_position is not None and instance_number is not None:
            temporal_dict[image_position].append((instance_number, pattern))
    # print(temporal_dict)
    # exit()
    # 将数据从字典转为列表，并排序
    temporal_list = []
    for slice_location in sorted(temporal_dict.keys()):  # 按 SliceLocation 排序
        # 按 InstanceNumber 对同一 SliceLocation 的数据排序
        instances_sorted = sorted(temporal_dict[slice_location], key=lambda x: x[0])
        temporal_list.append([item[1] for item in instances_sorted])  # 只保留 pattern

    return temporal_list[::-1] # 逆序返回

def load_dicom_nocine(dicom_folder):
    mod_slices = []
    
    # 递归获取所有 DICOM 文件
    dicom_files = []
    for root, _, files in os.walk(dicom_folder):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    # 按照 gb2312 编码的文件名排序
    dicom_files.sort(key=lambda x: os.path.basename(x).encode("gb2312"))
    
    # 读取 DICOM 文件
    for dcm_path in dicom_files:
        dicom_data = pydicom.dcmread(dcm_path)
        pixel_array = dicom_data.pixel_array  # 形状为 (H, W)
        image = sitk.GetImageFromArray(pixel_array)
        mod_slices.append(image)
    
    # 组合成 3D 图像
    mod_res = sitk.JoinSeries(mod_slices)
    return mod_res

def load_and_process_dicom(idx_list, dicom_folder, case, view):
    """
    Args:
        idx_list (list of list): 二维列表，存储 DICOM 文件的基础名称（无后缀）
        dicom_folder (str): DICOM 文件的文件夹路径

    Returns:
        SimpleITK.Image: 4D 图像
    """
    all_slices = []  # 存储所有子列表处理后的结果

    for sublist in idx_list:
        temporal_slices = []  # 存储当前子列表的时间维度拼接结果 (H, W, T)
        
        for base_name in sublist:
            # 构造完整路径并读取 DICOM 文件
            dicom_path = os.path.join(dicom_folder, f"{base_name}.dcm")
            dicom_data = pydicom.dcmread(dicom_path)

            # 提取像素数据并转换为 SimpleITK 图像对象
            pixel_array = dicom_data.pixel_array  # 形状为 (H, W)
            image = sitk.GetImageFromArray(pixel_array)
            temporal_slices.append(image)

        # 将当前子列表的所有帧按照时间维度拼接
        temporal_stack = sitk.JoinSeries(temporal_slices)  # 最后一维为时间 T
        all_slices.append(temporal_stack)

    # 将所有子列表结果再沿新维度 D 拼接
    final_result = sitk.JoinSeries(all_slices)  # 最后一维为 D
    return final_result

def save_sitk_as_nifti(sitk_image, output_path):
    """
    将 SimpleITK 图像保存为 NIfTI 格式
    """
    sitk.WriteImage(sitk_image, output_path)
    print(f"NIfTI file saved at: {output_path}")

#'20201221 caomingsheng', '20220819 wangshiliang', '20210716 fenghuiliang'
index = cases.index('20210716 fenghuiliang')
cases = cases[index+1:]
print(len(cases))

count_dict = {i:[] for i in views}

for case in cases:
    if case in error_case:
        continue
    print(case)
    if case in info_mark_list:
        continue
    case_path = f"{dicom_root}/{case}"
    mod_mapping = {}
    # count std view number
    for view in views:
        if view not in [i.lower().replace(' ','').replace('-','').replace('_','') for i in os.listdir(case_path)]:
            count_dict[view].append(0)
            
    for i in os.listdir(case_path):
        mod_name = i.lower().replace(' ','').replace('-','').replace('_','')
        for view in views:
            if view in mod_name:
                mod_mapping[i] = view
                break

    for view in mod_mapping:
        view_std = mod_mapping[view]
        modal_path = f"{case_path}/{view}"
        if not os.path.exists(modal_path):
            continue
        dcm_files = os.listdir(modal_path)
        if len(dcm_files) == 0:
            continue
        if view_std == 'cinesa':
            idx_list = organize_dicom(modal_path)
            sitk_4D_data = load_and_process_dicom(idx_list, modal_path, case, view)
            # print(sitk_4D_data.GetSize()) # ((256, 256, 30, 14)) [H W T D] 需要将T采样13, D中间采样3
            # we save the original 4D data and do sample later  
            if sitk_4D_data is not None:
                print(sitk_4D_data.GetSize())  # 打印输出 4D 图像的大小
                os.makedirs(f"{dest_root}/{case}", exist_ok=True)
                output_path = f"{dest_root}/{case}/{view_std}.nii.gz"
                save_sitk_as_nifti(sitk_4D_data, output_path)
                count_dict[view_std].append(sitk_4D_data.GetSize()[-2])
            else:
                print("Error in processing DICOM files.")
                count_dict[view_std].append(0)
                
        else:
            # read the rest of the modalities
            sitk_3D_data = load_dicom_nocine(modal_path)
            if sitk_3D_data is not None:
                print(sitk_3D_data.GetSize())
                os.makedirs(f"{dest_root}/{case}", exist_ok=True)
                output_path = f"{dest_root}/{case}/{view_std}.nii.gz"
                save_sitk_as_nifti(sitk_3D_data, output_path)
                count_dict[view_std].append(sitk_3D_data.GetSize()[-1])
            else:
                print("Error in processing DICOM files.")
                count_dict[view_std].append(0)
                
print(count_dict)  
np.save('count_dict.npy', os.path.join(dest_root,count_dict),allow_pickle=True)  
 