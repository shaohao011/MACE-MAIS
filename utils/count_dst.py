import os
import glob
import matplotlib.pyplot as plt
import json
def count_dcm_files(base_directory, modalities):
    # 预处理模态名称（统一小写并去除空格和下划线）
    modalities = [modality.lower().replace(' ', '').replace('_', '').replace('-','') for modality in modalities]
    mod_len_list = {modality: [] for modality in modalities}
    
    # 遍历目录树
    for root, dirs, files in os.walk(base_directory):
        # 预处理当前路径用于匹配
        processed_root = root.lower().replace(' ', '').replace('_', '').replace('-','')
        # if root=="data/renji/renji-full/20220726 wujian/T2 star":
        #     print(modalities)
        #     print(processed_root)
        #     exit()
        if "gulou" in base_directory:
            with open('jsons/jsons_ori/gulou.json','r')as f:
                total_data = json.load(f)['total']
            mod_parents = {item['mod_parent']: item for item in total_data}
            if not any(mod_parent in root for mod_parent in mod_parents):
                continue
        # 检查每个模态是否出现在路径中
        show_flag = False
        for modality in modalities:
            if (modality=="nt1m" or modality=="et1m") and "tongji" in base_directory: 
                modality_temp = modality.replace('m','')
            else:
                modality_temp = modality
            if modality_temp in processed_root:
                # 统计当前目录的.dcm文件数
                dcm_files = glob.glob(os.path.join(root, '*.dcm'))
                file_count = len(dcm_files)
                mod_len_list[modality].append(file_count)
                show_flag = True
            #
        if not show_flag and not "CINE" in root:
            print(root)
            
    return mod_len_list

# 示例用法
# base_directory = 'data/renji/renji-full'
# base_directory = "data/anzhen/anzhen_data-12-30-full"
# base_directory = "data/tongji/tongji_full"
base_directory = "data/gulou/images"
modalities = ['PSIR', 'T2W','T2WI','T2M','T2-star','eT1M','nT1M','ECV','CINE-SA']
result = count_dcm_files(base_directory, modalities)

# # 计算每个模态的总文件数
# mod_totals = {mod: sum(counts) for mod, counts in result.items()}

# # 按总文件数排序模态
# modalities_sorted = sorted(mod_totals.keys(), key=lambda x: x, reverse=True)
# totals_sorted = [mod_totals[mod] for mod in modalities_sorted]

# # 创建柱状图
# plt.figure(figsize=(12, 6))
# bars = plt.bar(modalities_sorted, totals_sorted, color='skyblue')

# # 添加数据标签
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height,
#              f'{height}',
#              ha='center', va='bottom')

# # 图表装饰
# plt.xlabel('Modality', fontsize=12)
# plt.ylabel('Total DICOM Files', fontsize=12)
# plt.title(f'{base_directory.split("/")[-1]} Distribution by Modality', fontsize=14)
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# plt.savefig(f'data/data_stats/{base_directory.split("/")[-1]}_hist.png')

# def count_nii_per_dir(dir_name):
#     views = ['cinesa', 'psir', 'et1m', 'nt1m','t2m','t2star','t2w']
#     slice_mod = 
#     for root, dir, files in os.walk(dir_name):
#         if ".nii.gz" in files[0]:
#             # cur root is parent 
        # we decide to count the detailed information ourselves
            