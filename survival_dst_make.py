import json
from datetime import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


def get_item_by_id(data_list,id):
    items_dict = {item["ID"]: item for item in data_list}
    return items_dict.get(id, None)  

def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d')  
    raise TypeError(f"Type {type(obj)} not serializable")

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, default=json_serializer, ensure_ascii=False, indent=4)

def convert2stdtime(item):
    try:
        item['examTime'] = datetime.strptime(item['examTime'], '%Y%m%d')
    except:
        try:
            item['examTime'] = datetime.strptime(item['examTime'], '%Y-%m-%d %H:%M:%S')  
        except:
            try:
                item['examTime'] = datetime.strptime(item['examTime'], '%Y/%m/%d')
            except:
                item['examTime'] = datetime.strptime(item['examTime'], '%Y%m%d')
                
    try:
        item['maceTime'] = datetime.strptime(item['maceTime'], '%Y-%m-%d %H:%M:%S')  
    except:
        try:
            item['maceTime'] = datetime.strptime(item['maceTime'], '%Y/%m/%d')
        except:
            try:
                item['maceTime'] = datetime.strptime(item['maceTime'], '%Y/%m/%d %H:%M')
            except:
                try:
                    item['maceTime'] = datetime.strptime(item['maceTime'], '%Y-%m-%d')
                except:
                    print(item['maceTime'])
                    item['maceTime'] = datetime.strptime(item['maceTime'], '%Y.%m.%d')
                    
                
            
    return item

def process_data(args,data,translated_data,output_file):
    new_data = []
    for item in data:
        item['mod_parent'] = str(item['mod_parent'])
        if "cen" in item['mod_parent']:
            item['examTime'] =item['examTime'][:-1]

        item['examTime_ori'] = item['examTime']
        item['maceTime_ori'] = item['maceTime']
        item = convert2stdtime(item)
        if translated_data!=None:
            item['Imaging_Findings_ori'] = item['Imaging_Findings']
            item['Imaging_Diagnosis_ori'] = item['Imaging_Diagnosis']
            item['Imaging_Findings'] = get_item_by_id(translated_data,item["ID"])['Imaging_Findings']
            item['Imaging_Diagnosis'] = get_item_by_id(translated_data,item["ID"])['Imaging_Diagnosis']
        else:
            item['Imaging_Findings_ori'] = item['image_findings_ori']
            item['Imaging_Diagnosis_ori'] = item['image_diagnosis_ori']
            item['Imaging_Findings'] =item['Imaging_Findings']
            item['Imaging_Diagnosis'] = item['Imaging_Diagnosis']
        
        item['time_diff'] = (item['maceTime'] - item['examTime']).days
        if item['time_diff']<0:
            print(item['examTime'],item['maceTime'])
            continue
        else:
            diff = relativedelta(item['maceTime'], item['examTime'])
            item['survival_months'] = diff.years * 12 + diff.months
            new_data.append(item)
            
    df = pd.DataFrame(new_data)
    df.rename(columns={
        'time_diff_months': 'survival_months',
    }, inplace=True)
    df['censorship']  =  1- df['mace']
    
    # refferd from PROMISE
    label_col = "survival_months"
    n_bins = args.n_bins
    eps = 1e-6
    slide_data = df
    
    patients_df = slide_data.drop_duplicates(['ID']).copy()
    uncensored_df = patients_df[patients_df['censorship'] < 1] 
    split_type = "interval"
    
    if split_type=="interval":
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        print(disc_labels,len(disc_labels))
        print(q_bins)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        print(q_bins)
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))
    elif split_type=="time":
        disc_labels, q_bins = pd.cut(uncensored_df[label_col], bins=n_bins, retbins=True, labels=False)
        print(disc_labels,len(disc_labels))
        print(q_bins)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        print(q_bins)
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))
    elif split_type=="whole_interval":
        disc_labels, q_bins = pd.qcut(patients_df[label_col], q=n_bins, retbins=True, labels=False)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))
    elif split_type=="whole_time":
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=n_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))
    else:
        raise ValueError

    slide_data = patients_df
    slide_data.reset_index(drop=True, inplace=True)
    slide_data = slide_data.assign(slide_id=slide_data['ID'])

    label_dict = {}
    key_count = 0
    for i in range(len(q_bins)-1):
        for c in [0, 1]:
            print('{} : {}'.format((i, c), key_count))
            label_dict.update({(i, c):key_count})
            key_count+=1

    label_dict = label_dict
    for i in slide_data.index:
        key = slide_data.loc[i, 'label']
        slide_data.at[i, 'disc_label'] = key
        censorship = slide_data.loc[i, 'censorship']
        key = (key, int(censorship))
        slide_data.at[i, 'label'] = label_dict[key] 

    bins = q_bins
    num_classes=len(label_dict) 
    patients_df = slide_data.drop_duplicates(['ID'])
    patient_data = {'ID':patients_df['ID'].values, 'label':patients_df['label'].values}

    #new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2]) ### ICCV
    new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])  ### PORPOISE
    slide_data = slide_data[new_cols]
    slide_data = slide_data
    metadata = ['disc_label', 'Unnamed: 0', 'ID', 'label', 'slide_id', 'age', 'site', 'survival_months', 'censorship', 'is_female', 'oncotree_code', 'train']
    metadata = slide_data.columns[:12]
    
    for col in slide_data.drop(metadata, axis=1).columns:
        if not pd.Series(col).str.contains('|_cnv|_rnaseq|_rna|_mut')[0]:
            print(col)
    #pdb.set_trace()

    assert metadata.equals(pd.Index(metadata))
    #pdb.set_trace()
    label,label_counts = np.unique(list(slide_data['disc_label']),return_counts=True)
    print(label,label_counts)
    # exit()
    slide_data.to_json(output_file.replace('.json',f"_{split_type}.json"), orient='records', indent=4,force_ascii=False)

    # slide_data.to_csv(output_file, index=False, encoding='utf-8')

def main(args):
    dataset_name = "center_1"
    input_file = f'jsons/jsons_ori/{dataset_name}.json'  
    if "gulou" in input_file:
        translated_data = None
    else:
        translated_file = f"jsons/translated/translated-deepseek_{dataset_name}.json"
        translated_data = read_json(translated_file)
    try:
        data = read_json(input_file)['total']
    except:
        data = read_json(input_file)
        
    output_file = f'jsons/jsons_bin_split/{dataset_name}.json'  
    _ = process_data(args,data,translated_data,output_file=output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--n_bins', default=4, type=int)
    parser.add_argument('--use_ml_model', action='store_true', default=False,)
    args = parser.parse_args()
    main(args)