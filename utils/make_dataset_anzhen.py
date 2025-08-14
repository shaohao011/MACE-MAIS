import matplotlib.pyplot as plt
import os
import statistics
import pandas as pd
import os
import json


def hist_count(json_file_path,out_hist_dir):
    with open(json_file_path,'r') as file:
        total_data = json.load(file)['total']
    modality_list = {}
    os.makedirs(out_hist_dir,exist_ok=True)
    for item in total_data:
        for modality, slice_counts in item['mod_slices'].items():
            if modality not in modality_list: modality_list[modality] = []
            modality_list[modality].append(slice_counts)
    for modality in modality_list:  
        slice_counts = modality_list[modality]    
        if slice_counts:  # 检查是否有数据
            try:
                mode = statistics.mode(slice_counts)  # 众数
            except statistics.StatisticsError:
                mode = "No unique mode"  # 如果没有唯一众数
            median = statistics.median(slice_counts)  # 中位数
            plt.figure()
            plt.hist(slice_counts, bins=20, color='blue', alpha=0.7)
            plt.title(f'Slice Count Distribution for {modality}\nMode: {mode}, Median: {median}')
            plt.xlabel('Number of Slices')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(f'./{out_hist_dir}/{modality}.png')
        else:
            print(f"No data available for {modality}")

def type_judge(d1,d2,d3,d4):
    flag = False
    try:
        d1 = float(d1)
        d2 = float(d2)
        d3 = float(d3)
        d4 = float(d4)
        flag = True
    except:
        pass
    return flag
    


def creat_anzhen_json(data_root,info_path,out_json_path,mace_only_csv):
    non_NaN = []
    count_paths = {'PSIR':[],'T2-star':[],'T2m':[],'T2W':[],'eT1m':[],'nT1m':[]}
    count_number = {'PSIR':[],'T2-star':[],'T2m':[],'T2W':[],'eT1m':[],'nT1m':[]}
    all_label = pd.read_csv(info_path)
    mace_only_label = pd.read_csv(mace_only_csv)
    # print(all_label['PatientID'][0].dtype)
    total_data = []
    full_label_data = []
    for patient_dir in sorted(os.listdir(data_root)):
        try:
            p_index,p_id,p_name = patient_dir.split('_',maxsplit=2)
        except:
            continue
        visit_folder = sorted(os.listdir(os.path.join(data_root,patient_dir)))
        if '.DS_Store' in visit_folder:visit_folder.remove('.DS_Store')
        if len(visit_folder)==0:continue
        # 多个随访记录
        try:
            p_id_stripped = p_id.lstrip('0')  # 结果为 "3398931"
            all_label['PatientID_stripped'] = all_label['PatientID'].astype(str).str.lstrip('0')
            # print(p_id_stripped)
            row = all_label[all_label['PatientID_stripped'] == p_id_stripped].iloc[0]
            # row = all_label[all_label['PatientID']==str(p_id)].iloc[0]
        except:
            continue
            # row = all_label[all_label['PatientID']==float(p_id)]
            # print(row)
            print(p_id)
            print(patient_dir)
            exit()
            
        if len(visit_folder)>1:
            time = [float(i.replace('-','')) for i in visit_folder]
            ref_time = float(str(row['examID'])[2:10])
            abs_time = [abs(t-ref_time) for t in time]
            closest_index = abs_time.index(min(abs_time))
            visit_folder = visit_folder[closest_index:closest_index+1]
            
        
        visit_folder = os.path.join(data_root,patient_dir,visit_folder[0])
        mods_folder = os.listdir(visit_folder)
        item_count_number = {'PSIR':0,'T2-star':0,'T2m':0,'T2W':0,'eT1m':0,'nT1m':0}

        for mod_folder in mods_folder:
            if mod_folder in count_paths.keys():
                slice_num = len(os.listdir(os.path.join(visit_folder,mod_folder)))
                if slice_num>0:
                    count_paths[mod_folder].append(visit_folder)
                    count_number[mod_folder].append(slice_num)
                    item_count_number[mod_folder] = slice_num
        mace = row['MACE']
        if pd.isna(mace): 
            try:
                p_id_stripped = p_id.lstrip('0')  # 结果为 "3398931"
                mace_only_label['PatientID_stripped'] = mace_only_label['PatientID'].astype(str).str.lstrip('0')
                row_new = mace_only_label[mace_only_label['PatientID_stripped'] == p_id_stripped].iloc[0]
                mace = row_new['MACE']
            except:
                pass
        else:
            pass
        if not pd.isna(mace): 
            non_NaN.append(patient_dir)
            
        mace_time = row['MACE时间'] if mace>0 else "2023/12/31"
        
        item = {
            'ID':str(p_id_stripped),
            
            'mod_parent':visit_folder.replace('../data','data'),
             "GENDER": row['GENDER'],
            "AGE": row['AGE'],
            'mod_slices':item_count_number,
            "examTime":row['examTime'],
            "maceTime":mace_time,
            
            # "NTproBNP":row['NTproBNP'],
            # "TNIpeak":row['TNIpeak'],
            
            # "LGE":row['LGE'],
            # "LV_EF": row['LV_EF'],
            # "RV_EF": row['RV_EF'],
           
            "Imaging_Findings":row['Imaging_Findings'],
            "Imaging_Diagnosis":row['Imaging_Diagnosis'],
            
            # "Microcirculation_Dysfunction_r":row["Microcirculation_Dysfunction_r"],
            # "Intramyocardial_Hemorrhage_r":row["Intramyocardial_Hemorrhage_r"],
            # "Ventricular_Thrombus_r":row["Ventricular_Thrombus_r"],
            # "Ventricular_Aneurysm_r":row["Ventricular_Aneurysm_r"],
            
            "Microcirculation_Dysfunction_r":row["微循环障碍right_"],
            "Intramyocardial_Hemorrhage_r":row["心肌内出血right_"],
            "Ventricular_Thrombus_r":row["心室血栓right_"],
            "Ventricular_Aneurysm_r":row["室壁瘤right_"],
            'mace':mace,

        }
        if (not pd.isna(row['GENDER'])) and (not pd.isna(mace_time)) and (not pd.isna(row['AGE'])) and (item_count_number['PSIR']!=0) and (not pd.isna(mace)) and type_judge(row["微循环障碍right_"],row["心肌内出血right_"],row["心室血栓right_"],row["室壁瘤right_"]) and (not pd.isna(row['Imaging_Findings']) and (not pd.isna(row['Imaging_Diagnosis']))): 
        # if (not pd.isna(row['AGE'])) and (not pd.isna(row['GENDER'])) and (not pd.isna(mace)) and type_judge(row["微循环障碍right_"],row["心肌内出血right_"],row["心室血栓right_"],row["室壁瘤right_"]) and (item['mod_slices']['PSIR']!=0): 
            full_label_data.append(item)
        total_data.append(item)
  
    print("total sample:",len(total_data))
    print("non_NaN sample:",len(non_NaN))
    print("non_NaN sample:",len(full_label_data))

    # with open(out_json_path,'w')as f:
    #     json.dump({'total':total_data},f,indent=4,ensure_ascii=False)
    with open(out_json_path.replace('dataset','full_label_dataset_1'),'w')as f:
        json.dump({'total':full_label_data},f,indent=4,ensure_ascii=False)
    
if __name__=="__main__":
    data_dir = "../data/anzhen/anzhen_data-12-30-full"
    info_path = "../data/anzhen/anzhen-full.csv"
    out_json_path = "../json_miccai/anzhen_dataset.json"
    mace_only_csv = "../data/anzhen/anzhen_mace_only_refined.csv"
    creat_anzhen_json(data_dir,info_path,out_json_path,mace_only_csv)
    
    
    # json_file_path = out_json_path
    # out_hist_dir = "../data/data_stats/anzhen"
    # hist_count(json_file_path,out_hist_dir)
