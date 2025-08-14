import matplotlib.pyplot as plt
import os
import statistics
import pandas as pd
import os
import json
import pydicom
import numpy as np



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
    

def creat_tongji_json(data_root,info_path,out_json_path,extra_info_path,mace_time):
    non_NaN = []
    count_paths = {'PSIR':[],'T2-star':[],'T2M':[],'T2W':[],'eT1m':[],'nT1m':[],'T2WI':[],'ECV':[]}
    count_number = {'PSIR':[],'T2-star':[],'T2M':[],'T2W':[],'eT1m':[],'nT1m':[],"T2WI":[],'ECV':[]}
    all_label = pd.read_csv(info_path)
    extra_label = pd.read_excel(extra_info_path,skiprows=1)
    mace_time_label = pd.read_excel(mace_time)
    # print(all_label['PatientID'][0].dtype)
    total_data = []
    full_label_data = []
    # all_label['病人编号1'] = all_label['病人编号'].astype(str)
    all_label['病人编号1'] = all_label['dicom_accession_id'].astype(str)
    extra_label['dicom_accession_id'] = extra_label['放射编号'].astype(str).str.strip()
    for patient_dir in sorted(os.listdir(data_root)):
        if patient_dir==".DS_Store":continue
        visit_folder = os.path.join(data_root,patient_dir)
        try:
            psir_dir = os.path.join(data_root,patient_dir,'PSIR','img')
            psir_file = os.path.join(psir_dir,[i for i in os.listdir(psir_dir) if i.endswith('.dcm')][0])
            dicom_data = pydicom.dcmread(psir_file)
        except:
            try:
                psir_dir = os.path.join(data_root,patient_dir,'T2W','img')
                psir_file = os.path.join(psir_dir,[i for i in os.listdir(psir_dir) if i.endswith('.dcm')][0])
                dicom_data = pydicom.dcmread(psir_file)
                print('1111')
            except:
                try:
                    psir_dir = os.path.join(data_root,patient_dir,'T2WI','img')
                    psir_file = os.path.join(psir_dir,[i for i in os.listdir(psir_dir) if i.endswith('.dcm')][0])
                    dicom_data = pydicom.dcmread(psir_file)
                    
                except:
                    psir_dir = os.path.join(data_root,patient_dir,'PSIR')
                    psir_file = os.path.join(psir_dir,[i for i in os.listdir(psir_dir)][0])
                    dicom_data = pydicom.dcmread(psir_file)
                    print('222')
            
        # patient_id = dicom_data.get("PatientID", "未找到 Patient ID")  # 如果不存在 PatientID，返回默认值
        patient_id = dicom_data.get("AccessionNumber", "未找到 AN")  # 如果不存在 PatientID，返回默认值
        ref_patient_id = dicom_data.get("PatientID", "未找到 AN")  # 如果不存在 PatientID，返回默认值
        exam_date = dicom_data.get('AcquisitionDate',None)
        # print(patient_id)
        # row_mace_time = mace_time_label[mace_time_label['放射编号']==int(patient_id)]
        # MACETIME = row_mace_time['Mace时间or随访截止时间'].values[0]
        # print(MACETIME)
        # exit()
        try:
            row = all_label[all_label['病人编号1'] == str(patient_id)].iloc[0]
            try:
                row_extra = extra_label[extra_label['dicom_accession_id'] == str(ref_patient_id)].iloc[0]
                row_mace_time = mace_time_label[mace_time_label['放射编号']==int(patient_dir.split('_')[1])]
                MACETIME = row_mace_time['Mace时间or随访截止时间'].values[0]
                Imaging_Findings = row_extra['影像描述']
                Imaging_Diagnosis = row_extra['影像诊断']
            except:
                Imaging_Findings = np.NaN
                Imaging_Diagnosis = np.NaN
                MACETIME = np.NaN
        except:
            print(patient_dir,patient_id)
            print('error')
            continue
        

        mods_folder = os.listdir(visit_folder)
        item_count_number = {'PSIR':0,'T2-star':0,'T2m':0,'T2W':0,'eT1m':0,'nT1m':0,"T2WI":0,"ECV":0}

        for mod_folder in mods_folder:
            if mod_folder in count_paths.keys():
                slice_num = len(os.listdir(os.path.join(visit_folder,mod_folder)))
                if slice_num>0:
                    count_paths[mod_folder].append(visit_folder)
                    count_number[mod_folder].append(slice_num)
                    item_count_number[mod_folder] = slice_num
                    
                 
        mace = row["Mace.1"]
        # for key, value in row.items():
        #     print(f"{key}: {value}")
        # print(row)
        # print(mace)
        # exit()
        if not pd.isna(mace): 
            non_NaN.append(patient_dir)
        item = {
            # 'accession_id': str(patient_id),
            # 'patient_id': str(ref_patient_id),
            'ID':str(patient_id),
            'mod_parent':os.path.join(data_root,patient_dir),
            "GENDER": float(row['GENDER']),
            "AGE": float(row['AGE']),
            'mod_slices':item_count_number,
            "examTime":exam_date,
            "maceTime":str(MACETIME),
            
            # "NTproBNP":float(row['NTproBNP']),
            # "NTproBNP":row['NTproBNP'],
            # # "TNIpeak":float(row['TNIpeak肌钙蛋白']),
            # "TNIpeak":row['TNIpeak肌钙蛋白'],
            
            # "LGE":float(row['LGE']),
            
            # "LV_EF": float(row['EF']),
            # # "RV_EF": float(row['RV_EF']),
            # "RV_EF": np.NaN,
            
            "Imaging_Findings":Imaging_Findings,
            "Imaging_Diagnosis":Imaging_Diagnosis,
            
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
        if (exam_date!=None) and (not pd.isna(str(MACETIME))) and (not pd.isna(row['AGE'])) and (not pd.isna(row['GENDER'])) and (item_count_number['PSIR']!=0) and (not pd.isna(mace)) and type_judge(row["微循环障碍"],row["心肌内出血"],row["心室血栓"],row["室壁瘤"]) and (not pd.isna(Imaging_Findings)) and (not pd.isna(Imaging_Diagnosis)): 
        
        # if not pd.isna(mace) and type_judge(row["微循环障碍"],row["心肌内出血"],row["心室血栓"],row["室壁瘤"]): 
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
    data_dir = "data/tongji/tongji_full"
    info_path = "data/tongji/tongji_info.csv"
    mace_time = "data/tongji/mace_time.xlsx"
    extra_info_path = "data/tongji/tongji_diagnosis.xlsx"
    out_json_path = "jsons/jsons_ori/tongji.json"
    creat_tongji_json(data_dir,info_path,out_json_path,extra_info_path,mace_time)
    

