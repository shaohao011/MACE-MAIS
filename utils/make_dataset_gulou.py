import matplotlib.pyplot as plt
import os
import statistics
import pandas as pd
import os
import json
from retrying import retry
from googletrans import Translator # 4.0.0-rc1 attention please
from tqdm import tqdm   


# class DeepSeek:
#     def __init__(self, model_name, api_url, api_key):
#         self.model_name = model_name
#         self.api_url = api_url
#         self.api_key = api_key
#         self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
#         # print(f"Using model: {self.model_name}")

#     def call(self, content, additional_args={}):
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant"},
#                 {"role": "user", "content": content},
#             ],
#             **additional_args
#         )
#         return response.choices[0].message.content

#     @retry(wait_fixed=3000, stop_max_attempt_number=3)
#     def retry_call(self, content, additional_args={"max_tokens": 8192}):
#         return self.call(content, additional_args)

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
    


def creat_gulou_json(data_root,info_path,out_json_path,extra_info,translated_json):
    trans = True
    if trans:
        use_google = True
        if use_google:
            translator = Translator()
        else:
            model_name = "deepseek-chat"
            api_url = "https://api.deepseek.com"
            api_key = "sk-c94cb439c1e04d3a874d95a2cc553b2c"
            deepseek_instance = DeepSeek(model_name, api_url, api_key)
            translate_prompt = "[{}]\n\n将上述内容翻译成英文，只能返回一段语句，译文不要出现中文。"
    
    # with open(translated_json,'r') as f:
    #     trans_json = json.load(f)['total']
    # trans_json = {item['ID']:item for item in trans_json}
    
    all_label = pd.read_excel(info_path)
    extra_label = pd.read_csv(extra_info)
    total_data = []
    full_label_data = []
    
    label_dict = {str(i['检查号']): i for i in all_label.to_dict(orient='records')}
    extra_dict = {str(i['检查流水号']): i for i in extra_label.to_dict(orient='records')}
    error_list = []
    # loop data dir
    for case_folder in tqdm(os.listdir(data_root)):
        if '.zip' in case_folder or not str(case_folder) in label_dict:
            continue
        print(len(label_dict))
        row = label_dict[str(case_folder)]
        mod_parent = str(row['检查号'])
        
        case_folder = os.path.join('data/gulou/images',mod_parent)
        try:
            if "PSIR" in os.listdir(case_folder):
                pass
            else:
                case_folder = os.path.join(case_folder,mod_parent)
                if "PSIR" in os.listdir(case_folder):
                    pass
                else:
                    raise ValueError(f"no image {mod_parent}")
        except:
            error_list.append(mod_parent)
        # if i==5: break
        patient_id = str(row['检查流水号'])   
        if "查询不到" ==patient_id:
            continue
        extra_finding = extra_dict[patient_id]
        # 1 male
        mace_time = str(row['MACE时间']).replace('年','/').replace('月','').replace('日','') 
        if mace_time == 'nan':
            mace_time = "2024/12/30"
        if pd.isna(extra_finding['影像学描述']) or pd.isna(extra_finding['影像学诊断']):
            continue
        image_findings_ori = extra_finding['影像学描述']
        image_diagnosis_ori = extra_finding['影像学诊断']
        # translate
        
        # if use_google:
        #     image_findings = translator.translate(image_findings_ori,  src='zh-cn', dest='en').text
        #     image_diagnosis = translator.translate(image_diagnosis_ori,  src='zh-cn', dest='en').text
        # else:
        #     image_findings = deepseek_instance.retry_call(translate_prompt.format(image_findings_ori))
        #     image_diagnosis = deepseek_instance.retry_call(translate_prompt.format(image_diagnosis_ori))
        # image_findings = ''
        # image_diagnosis = ''
           
        image_findings = translator.translate(image_findings_ori,  src='zh-cn', dest='en').text
        image_diagnosis = translator.translate(image_diagnosis_ori,  src='zh-cn', dest='en').text
        

        item = {
            
            'ID':str(patient_id),
            'mod_parent':case_folder,
            "GENDER": 1 if row['性别']=="男" else 0,
            "AGE": str(row['年龄']).replace('岁',''),
            'mod_slices':{},
            "examTime":row['检查流水号'][:9],
            "maceTime":mace_time,
            
           
            "Imaging_Findings":image_findings,
            "Imaging_Diagnosis":image_diagnosis, # need translate
            
            # "Microcirculation_Dysfunction_r":row["微循环障碍"],
            # "Intramyocardial_Hemorrhage_r":row["心肌内出血"],
            # "Ventricular_Thrombus_r":row["心室血栓"],
            # "Ventricular_Aneurysm_r":row["室壁瘤"],
            'mace':row['MACE'],
            "image_findings_ori":image_findings_ori,
            "image_diagnosis_ori":image_diagnosis_ori

        }
        full_label_data.append(item)
        total_data.append(item)
  
    print("total sample:",len(total_data))
    print("non_NaN sample:",len(full_label_data))
    print("ERROR PARENT")
    print(error_list)

    # with open(out_json_path,'w')as f:
    #     json.dump({'total':total_data},f,indent=4,ensure_ascii=False)
    with open(out_json_path,'w')as f:
        json.dump({'total':full_label_data},f,indent=4,ensure_ascii=False)
    
if __name__=="__main__":
    data_dir = "data/gulou/images"
    info_path = "data/gulou/gulou.xlsx"
    extra_info = "data/gulou/extra_findings.csv"
    out_json_path = "jsons/jsons_ori/gulou.json"
    translated_json = "jsons/jsons_ori/gulou.json"
    creat_gulou_json(data_dir,info_path,out_json_path,extra_info,translated_json)
    
