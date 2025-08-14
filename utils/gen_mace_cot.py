import os
import json
import argparse
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from retrying import retry
import re
# --- GPT-4o Interface ---
class GPT4o:
    def __init__(self, model_name, api_url, api_key):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        print(f"Using model: {self.model_name}")

    def call(self, content, additional_args={}):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ],
            **additional_args
        )
        return response.choices[0].message.content

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, content, additional_args={"max_tokens": 4096,"temperature": 0.1}):
        return self.call(content, additional_args)

def get_answer_from_response(response):

    ration_match = re.search(r'<rationale>(.*?)</rationale>', response, re.DOTALL)
    pred_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    # print(response)
    # print(ration_match)
    # print(pred_match)
    rationale = None
    prediction = None

    if ration_match and pred_match:
        rationale = ration_match.group(1).strip()
        prediction = pred_match.group(1).strip()
        # print(rationale)
        # print(prediction)
        return rationale, prediction
    else:
        raise ValueError
        print("Failed to extract rationale or prediction from response.")
        return None, None

    
def format_initial_prompt(question):
    return f"""
You are a cardiovascular assistant tasked with analyzing a clinical case.

<question>{question}</question>
Please respond to the above question <question> through step-by-step reasoning.

**MACE Evaluation Guidelines:**  
- **Imaging Features:** Ischemia, edema, and especially **late gadolinium enhancement (LGE)** indicating myocardial scar or necrosis.  
- **Cardiac Function:** **Left ventricular ejection fraction (LVEF)**, **right ventricular ejection fraction (RVEF)**; reduced EF is associated with higher MACE risk.  
- **Biomarkers:** Elevated **TNIpeak**, **NTproBNP**, **hsCRP**, or **HbA1c** may suggest acute injury, inflammation, or metabolic risk.  
- **Clinical Risk Factors:** Smoking, hypertension, diabetes, and dyslipidemia are known contributors to MACE risk.  
- **Perfusion Status:** **TIMI grade pre- and post-PCI** reflects coronary perfusion success; suboptimal reperfusion increases risk.  
- **Extent of Myocardial Damage:** Estimate of infarct size or necrotic volume (e.g., LGE % of LV mass).

The output format must strictly follow the structure below:

<rationale>Provide your reasoning step by step, considering all provided information. Use clinical knowledge and MACE evaluation guidelines.</rationale>
<answer>high or low</answer>
"""

def format_prompt_with_tags(question, response, label=None):
    rationale,prediction = get_answer_from_response(response)
    
    if label== None:
        prompt = f"""You are acting as a clinical assistant to an experienced cardiologist.
        <question>{question}</question>

        **Model's Existing Rationale:**  
        <rationale>{rationale}</rationale>
        
        **Model's Existing Prediction:**  
        <answer>{prediction}</answer>
        
        **Task:**  
        Evaluate whether the rationale is medically sound and clinically relevant to the question. Use the following MACE-related evaluation criteria as guidance:
        
        **MACE Evaluation Guidelines:**  
        - **Imaging Features:** Ischemia, edema, and especially **late gadolinium enhancement (LGE)** indicating myocardial scar or necrosis.  
        - **Cardiac Function:** **Left ventricular ejection fraction (LVEF)**, **right ventricular ejection fraction (RVEF)**; reduced EF is associated with higher MACE risk.  
        - **Biomarkers:** Elevated **TNIpeak**, **NTproBNP**, **hsCRP**, or **HbA1c** may suggest acute injury, inflammation, or metabolic risk.  
        - **Clinical Risk Factors:** Smoking, hypertension, diabetes, and dyslipidemia are known contributors to MACE risk.  
        - **Perfusion Status:** **TIMI grade pre- and post-PCI** reflects coronary perfusion success; suboptimal reperfusion increases risk.  
        - **Extent of Myocardial Damage:** Estimate of infarct size or necrotic volume (e.g., LGE % of LV mass).

        **Your Tasks:**  
        1. **Assess Validity**: Does the rationale correctly connect the imaging, biomarkers, and clinical context to the likelihood of MACE?  
        2. **Refine or Rewrite**:  
        - If the rationale is appropriate, **summarize and improve it** for clarity and conciseness, focusing on high-impact features.  
        - If the rationale is incomplete or incorrect, **rewrite it** with a more clinically accurate explanation based on the data.  
        3. Focus on **clinical reasoning** over superficial description. Avoid repeating the full patient case unless necessary.

        Return the revised or validated rationale within <refined rationale> </refined rationale> tag and final answer within <answer>high or low</answer> tag.
    """
    else:
        prompt = f"""
        You are assisting in improving clinical reasoning for cardiovascular event risk assessment based on a real patient case.

        <question>{question}</question>

        **Model's Existing Rationale:**  
        <rationale>{rationale}</rationale>
        
        **Model's Existing Prediction:**  
        <answer>{prediction}</answer>

        **Expert Feedback (reference only):**  
        The final clinical outcome label for this case is: **{label}**

        **Task:**
        - Use this reference label as guidance to reflect on and improve the original rationale.  
        - Identify any flaws, gaps, or misinterpretations in the original reasoning.  
        - Rewrite the rationale to be more clinically accurate and logically consistent with established risk factors, without directly or indirectly referencing the known label.  
        - Your revised rationale should appear as a refined clinical judgment, not a correction based on hindsight.  
        - Do **not** state, suggest, or imply that the true outcome is known.

        Please return only the improved rationale within <refined rationale> </refined rationale> tag and final answer within <answer>high or low</answer> tag.
        """
    return prompt

def build_question(d):
    return f"""A {d['AGE']}-year-old {'man' if int(float(d['GENDER']))==1 else 'woman'} with a BMI of {d['BMI']} underwent a cardiac MRI. 

**Note:** The values below may include binary indicators where  
- **1 = Yes / Positive**,  
- **0 = No / Negative**,  
- **unknown = Information not available or not measured**.

(1) **Imaging Findings**  
{d['Imaging_Findings']}

(2) **Imaging Diagnosis**  
{d['Imaging_Diagnosis']}

(3) **Cardiac/Biochemical Markers**  
- TNIpeak: {d['TNIpeak']}
- NTproBNP: {d['NTproBNP']}
- hsCRP: {d['hsCRP']}
- HbA1c: {d['HbA1c']}

(4) **Medical History / Risk Factors**  
- Smoking={d['Smoke']}
- HBP={d['HBP']}
- Diabetes={d['Diabetes']}
- Dyslipidemia={d['Dyslipidemia']}

(5) **Coronary Flow Status (TIMI grade)**  
- TIMI Grade (Pre-PCI): {d['TIMI_grade_pre']}, 
- TIMI Grade (Post-PCI): {d['TIMI_grade_post']}

Based on the information above, please predict the likelihood of the patient experiencing a major adverse cardiovascular event (MACE) within {d['survival_months']} months following the imaging examination.  
Please answer with either **high** or **low**."""

# --- Processing Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument("--save_dir", type=str, required=False)
    parser.add_argument("--api_key", type=str, required=False)
    parser.add_argument("--api_url", type=str, required=False)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--num_process", type=int, default=16)
    parser.add_argument("--limit_num", type=int, default=0)
    args = parser.parse_args()

    dataset_name = "anzhen"
    args.data_path = f"./jsons/jsons_final_full_info/{dataset_name}.json"
    args.save_dir = f"./jsons/CoT_final/{dataset_name}"
    
    os.makedirs(args.save_dir, exist_ok=True)
    args.model_name = "gpt-4o"
    args.api_key = ""
    args.api_url = ""
    
    
    model = GPT4o(model_name=args.model_name, api_url=args.api_url, api_key=args.api_key)

    # Load data from each split
    with open(args.data_path) as f:
        raw_json = json.load(f)
        all_data = []
        for split_name in ['training', 'validation', 'test']:
            for item in raw_json.get(split_name, []):
                item['split'] = split_name
                all_data.append(item)

    tmp_id = 1
    for da in all_data:
        da['process_id'] = tmp_id
        tmp_id += 1

    if args.limit_num:
        all_data = all_data[:args.limit_num]

    def process_item(d):
        try:
            question = build_question(d)
            label = "high" if int(float(d['mace'])) == 1 else "low"
            d['question'] = question
            prompt_initial = format_initial_prompt(question)
            initial_response = model.retry_call(prompt_initial)
            d['Initial_Model_Response'] = initial_response

            prompt_label = format_prompt_with_tags(question, initial_response, label=label)
            d['prompt_label'] = prompt_label    
            d['Refined_Rationale_WithLabel'] = model.retry_call(prompt_label)

            # Save to file
            save_path = os.path.join(args.save_dir, f"{d['ID']}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"❌ Error processing ID {d.get('ID')}: {e}")
            traceback.print_exc()
            return False
    
    def deduplicate_data(data, processed_data):
        processed_ids = {item['process_id'] for item in processed_data}
        return [item for item in data if item['process_id'] not in processed_ids]

    def merge_saved_files(save_dir):
        _, _, filenames = [i for i in os.walk(save_dir)][0]
        json_files = [f for f in filenames if f.endswith('.json') and "merge" not in f]
        res = []
        for file_path in json_files:
            try:
                with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                    da = json.loads(f.read())
                    assert 'Refined_Rationale_WithLabel' in da and 'Refined_Rationale_Model' in da 
                    res.append(da)
            except Exception as e:
                continue
        return res 
    
    processed_data = merge_saved_files(args.save_dir) 
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(all_data, processed_data) 
    print(f"Items remaining for processing: {len(input_data)}")
    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(tqdm(executor.map(process_item, input_data), total=len(input_data), desc="Processing"))

    print("✅ All items processed. Merging into one JSON...")
    all_results = []
    for fname in os.listdir(args.save_dir):
        if fname.endswith(".json") and "merged" not in fname:
            with open(os.path.join(args.save_dir, fname), encoding="utf-8") as f:
                all_results.append(json.load(f))

    merged_output_path = os.path.join(args.save_dir, "merged_results.json")
    with open(merged_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"📦 Merged file saved to {merged_output_path}")

if __name__ == '__main__':
    main()
