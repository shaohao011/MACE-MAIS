import matplotlib.pyplot as plt
import os
import statistics
import pandas as pd
import os
import json
import pydicom



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
    


def creat_renji_json(data_root,info_path,out_json_path):
    non_NaN = []
    count_paths = {'PSIR':[],'T2-star':[],'T2M':[],'T2W':[],'eT1m':[],'nT1m':[],'T2WI':[],'ECV':[]}
    count_number = {'PSIR':[],'T2-star':[],'T2M':[],'T2W':[],'eT1m':[],'nT1m':[],"T2WI":[],'ECV':[]}
    all_label = pd.read_excel(info_path,skiprows=1)
    # print(all_label['PatientID'][0].dtype)
    total_data = []
    full_label_data = []
    # all_label['病人编号1'] = all_label['病人编号'].astype(str)
    all_label['病人编号1'] = all_label['放射编号'].astype(str)
    for patient_dir in sorted(os.listdir(data_root)):

        # p_id_stripped = p_id.lstrip('0')  # 结果为 "3398931"
        # all_label['PatientID_stripped'] = all_label['PatientID'].astype(str).str.lstrip('0')
        # # print(p_id_stripped)
        # row = all_label[all_label['PatientID_stripped'] == p_id_stripped].iloc[0]
        
        visit_folder = os.path.join(data_root,patient_dir)
        try:
            psir_dir = os.path.join(data_root,patient_dir,'PSIR')
            psir_file = os.path.join(psir_dir,[i for i in os.listdir(psir_dir) if i.endswith('.dcm')][0])
            dicom_data = pydicom.dcmread(psir_file)
            # exam_time = dicom_data.get('AcquisitionTime',None)
            
        except:
            try:
                psir_dir = os.path.join(data_root,patient_dir,'T2W')
                psir_file = os.path.join(psir_dir,[i for i in os.listdir(psir_dir) if i.endswith('.dcm')][0])
                dicom_data = pydicom.dcmread(psir_file)
            except:
                psir_dir = os.path.join(data_root,patient_dir,'T2WI')
                psir_file = os.path.join(psir_dir,[i for i in os.listdir(psir_dir) if i.endswith('.dcm')][0])
                dicom_data = pydicom.dcmread(psir_file)
            
        # patient_id = dicom_data.get("PatientID", "未找到 Patient ID")  # 如果不存在 PatientID，返回默认值
        patient_id = dicom_data.get("AccessionNumber", "未找到 AN")  # 如果不存在 PatientID，返回默认值
        exam_date = dicom_data.get('AcquisitionDate',None)
        
        # print(acession_number)
        # exit()
        # print(type(all_label['病人编号1'][0]),type(str(patient_id)))
        # print(all_label['病人编号'][0])
        # print(str(patient_id))
        try:
            row = all_label[all_label['病人编号1'] == str(patient_id)].iloc[0]
        except:
            print(patient_dir,patient_id)
            print('error')
            continue
            exit()
        # print(row)
        # exit()

        mods_folder = os.listdir(visit_folder)
        item_count_number = {'PSIR':0,'T2-star':0,'T2m':0,'T2W':0,'eT1m':0,'nT1m':0,"T2WI":0,"ECV":0}

        for mod_folder in mods_folder:
            if mod_folder in count_paths.keys():
                slice_num = len(os.listdir(os.path.join(visit_folder,mod_folder)))
                if slice_num>0:
                    count_paths[mod_folder].append(visit_folder)
                    count_number[mod_folder].append(slice_num)
                    item_count_number[mod_folder] = slice_num
                    
                 
        mace = row['Mace']
        if not pd.isna(mace): 
            non_NaN.append(patient_dir)
        mace_time = row['Mace时间'] if mace>0 else "2023/12/31"
        item = {
            'ID':str(patient_id),
            
            'mod_parent':os.path.join(data_root,patient_dir),
            "GENDER": float(row['GENDER']),
            "AGE": float(row['AGE']),
            'mod_slices':item_count_number,
            "examTime":exam_date,
            "maceTime":mace_time,
            
            "NTproBNP":float(row['NTproBNP']),
            "TNIpeak":float(row['TNIpeak']),
            
            "LGE":float(row['LGE']),
            "LV_EF": float(row['LV_EF']),
            "RV_EF": float(row['RV_EF']),
            
            "Imaging_Findings":row['影像学表现'],
            "Imaging_Diagnosis":row['影像学诊断'],
            
            # "Microcirculation_Dysfunction_r":row["Microcirculation_Dysfunction_r"],
            # "Intramyocardial_Hemorrhage_r":row["Intramyocardial_Hemorrhage_r"],
            # "Ventricular_Thrombus_r":row["Ventricular_Thrombus_r"],
            # "Ventricular_Aneurysm_r":row["Ventricular_Aneurysm_r"],
            
            "Microcirculation_Dysfunction_r":float(row["微循环障碍"]),
            "Intramyocardial_Hemorrhage_r":float(row["心肌内出血"]),
            "Ventricular_Thrombus_r":float(row["心室血栓"]),
            "Ventricular_Aneurysm_r":float(row["室壁瘤"]),
            'mace':float(mace),

        }
        if (exam_date!=None) and (not pd.isna(mace_time)) and (not pd.isna(row['GENDER'])) and (not pd.isna(row['AGE'])) and (item_count_number['PSIR']!=0) and (not pd.isna(mace)) and type_judge(row["微循环障碍"],row["心肌内出血"],row["心室血栓"],row["室壁瘤"]) and (not pd.isna(row['影像学表现']) and (not pd.isna(row['影像学诊断']))): 
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
    data_dir = "./data/renji/renji-full"
    info_path = "./data/renji/renji-full.xlsx"
    out_json_path = "./jsons/jsons_ori/renji.json"
    creat_renji_json(data_dir,info_path,out_json_path)
