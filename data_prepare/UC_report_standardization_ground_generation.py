import os
import pandas as pd

def generate_report_from_BUC_ground_truth(ground_truth_row: pd.Series) -> str:
    """
    Generate a standardized pathology report for Bladder Urothelial Carcinoma (BUC) based on the provided ground truth data.
    """
    pT_stage_to_invasion = {
        "T0": "no evidence of primary tumor",
        "Ta": "tumor confined to mucosa",
        "Tis": "carcinoma in situ",
        "T1": "tumor invades subepithelial connective tissue",
        "T2a": "tumor invades superficial muscle layer",
        "T2b": "tumor invades deep muscle layer",
        "T2": "tumor invades muscle layer",
        "T3a": "tumor penetrates muscle layer into perivesical tissue (microscopically invades perivesical tissue)",
        "T3b": "tumor penetrates muscle layer into perivesical tissue (macroscopically invades perivesical tissue)",
        "T3": "tumor penetrates muscle layer into perivesical tissue",
        "T4a": "tumor invades adjacent organs (directly invades organs like prostatic stroma, seminal vesicles, uterus)",
        "T4b": "tumor invades adjacent organs (directly invades organs like pelvic wall or abdominal wall)",
        "T4": "tumor invades adjacent organs",
        "TX": "not specified"
    }
    pN_stage_to_lymph_node = {
        "N0": "no regional lymph node metastasis",
        "N1": "single regional lymph node metastasis",
        "N2": "multiple regional lymph node metastases",
        "N3": "extensive regional lymph node metastases",
        "NX": "regional lymph nodes not assessed"
    }
    pM_stage_to_distant_metastasis = {
        "M0": "no distant metastasis",
        "M1": "distant metastasis present",
        "M1a": "metastasis in nonregional lymph node",
        "M1b": "metastasis in other distant site",
        "MX": "not specified"
    }
    # --- New logic for combined grade and variants ---
    grade_code = str(ground_truth_row.get("grade", "1")) 
    if grade_code == '0':
        grade_description = "Low-Grade Urothelial Carcinoma"
    else:
        grade_description = "High-Grade Urothelial Carcinoma"

    # Check for histologic variants and append if present
    if pd.notna(ground_truth_row.get("histologic_variants")) and ground_truth_row.get("histologic_variants") == 1:
        grade_description += " with histologic variants"
    # --- End of new logic ---


    age = ground_truth_row.get("age")
    gender = ground_truth_row.get("gender", "not specified")
    pT = ground_truth_row.get("T", "TX")
    pN = ground_truth_row.get("N", "NX")
    pM = ground_truth_row.get("M", "MX")

    age_str = f"{int(age)}-year-old" if pd.notna(age) else "age not specified"
    gender_str = gender if pd.notna(gender) else "gender not specified"
    basic_info = f"PATIENT INFO: {age_str}, {gender_str}. "

    pathology_template = (
        "PATHOLOGICAL DIAGNOSIS: {grade}. "
        "PRIMARY TUMOR: {invasion}, pathological T stage {pT}. "
        "LYMPH NODES: {lymph_nodes}, pathological N stage {pN}. "
        "DISTANT METASTASIS: {distant_metastasis}, pathological M stage {pM}. "
        "LYMPHOVASCULAR INVASION: {lvi}."
    )

    pathology_report = pathology_template.format(
        grade=grade_description,
        invasion=pT_stage_to_invasion.get(pT, "not specified"),
        pT=pT,
        lymph_nodes=pN_stage_to_lymph_node.get(pN, "regional lymph nodes not assessed"),
        pN=pN,
        distant_metastasis=pM_stage_to_distant_metastasis.get(pM, "not specified"),
        pM=pM,
        lvi=ground_truth_row.get("LVI", "not reported")
    )
    return basic_info + pathology_report


def generate_report_from_UTUC_ground_truth(ground_truth_row: pd.Series) -> str:
    """
    Generate a standardized pathology report for Upper Tract Urothelial Carcinoma (UTUC) based on the provided ground truth data.
    """
    pT_stage_to_invasion = {
        "T0": "no evidence of primary tumor",
        "Ta": "tumor confined to mucosa",
        "Tis": "carcinoma in situ",
        "T1": "tumor invades subepithelial connective tissue",
        "T2": "tumor invades muscle layer",
        "T3": "Tumor invades beyond muscularis into peripelvic/periureteric fat",
        "T4": "tumour invades adjacent organs",
        "TX": "not specified"
    }
    pN_stage_to_lymph_node = {
        "N0": "no regional lymph node metastasis",
        "N1": "metastasis in a single lymph node 2 cm or less in the greatest dimension",
        "N2": "metastasis in a single lymph node more than 2 cm, or multiple lymph nodes",
        "NX": "regional lymph nodes cannot be assessed"
    }
    pM_stage_to_distant_metastasis = {
        "M0": "no distant metastasis",
        "M1": "distant metastasis",
        "MX": "not specified"
    }

    # --- New logic for combined grade and variants ---
    grade_code = str(ground_truth_row.get("grade", "1")) # Default to High-Grade
    if grade_code == '0':
        grade_description = "Low-Grade Urothelial Carcinoma"
    else:
        grade_description = "High-Grade Urothelial Carcinoma"

    # Check for histologic variants and append if present
    if pd.notna(ground_truth_row.get("histologic_variants")) and ground_truth_row.get("histologic_variants") == 1:
        grade_description += " with histologic variants"
    # --- End of new logic ---

    
    pT = ground_truth_row.get("T", "TX")
    pN = ground_truth_row.get("N", "NX")
    pM = ground_truth_row.get("M", "MX")

  
    invasion_description = pT_stage_to_invasion.get(pT, "not specified")

    age = ground_truth_row.get("age")
    gender = ground_truth_row.get("gender", "not specified")
    age_str = f"{int(age)}-year-old" if pd.notna(age) else "age not specified"
    gender_str = gender if pd.notna(gender) else "gender not specified"
    basic_info = f"PATIENT INFO: {age_str}, {gender_str}. "

    pathology_template = (
        "PATHOLOGICAL DIAGNOSIS: {grade}. "
        "PRIMARY TUMOR: {invasion}, pathological T stage {pT}. "
        "LYMPH NODES: {lymph_nodes}, pathological N stage {pN}. "
        "DISTANT METASTASIS: {distant_metastasis}, pathological M stage {pM}. "
        "LYMPHOVASCULAR INVASION: {lvi}."
    )
    margins_status = ground_truth_row.get("margins", "not reported")
    pathology_report = pathology_template.format(
        grade=grade_description,
        invasion=invasion_description,  
        pT=pT,
        lymph_nodes=pN_stage_to_lymph_node.get(pN, "regional lymph nodes not assessed"),
        pN=pN,
        distant_metastasis=pM_stage_to_distant_metastasis.get(pM, "not specified"),
        pM=pM,
        margins=margins_status,
        lvi=ground_truth_row.get("LVI", "not reported")
    )
    return basic_info + pathology_report


input_csv_path = "path/to/buc.csv"
output_dir = "./BUC_report_standardization_ground"
os.makedirs(output_dir, exist_ok=True)


all_results = []

try:
    df_input = pd.read_csv(input_csv_path)

except FileNotFoundError:
    print(f"error at {input_csv_path}")
    exit()

for index, row in df_input.iterrows():
    original_text = row.get('Report', '')
    doc_id = row.get('id', f"row_{index}")

    gt_standardized_report_text = generate_report_from_BUC_ground_truth(row)

    
    result_row = {
       
        'ID': doc_id,
        'GT_Standardized_Report': gt_standardized_report_text
    }
    all_results.append(result_row)


df_output = pd.DataFrame(all_results)
output_csv_filename = f"{output_dir}/BUC_extraction_evaluation_results.csv"
df_output.to_csv(output_csv_filename, index=False, encoding='utf-8-sig')


