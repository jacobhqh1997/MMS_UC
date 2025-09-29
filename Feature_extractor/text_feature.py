from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
from tqdm import tqdm

class TextPreprocessor:
    def __init__(self, model_name='/path/to/ClinicalBERT', device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()  # Set to evaluation mode
        self.device = device

    @torch.no_grad() # Disable gradient calculation
    def process(self, report_text: str) -> torch.Tensor:
        """
        Processes a single pathology report text into a fixed-size tensor.
        """
        # Handle empty or NaN text
        if pd.isna(report_text) or not isinstance(report_text, str):
            report_text = ""
        
        # 1. Tokenize
        inputs = self.tokenizer(report_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        
        # 2. Get BERT embeddings
        outputs = self.model(**inputs)
        
        # 3. Get the [CLS] token's embedding (represents the whole sequence)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 4. Remove batch dimension and return [1, 768]
        return cls_embedding  # [768]

    def process_csv(self, csv_path: str, output_dir: str = None):
        """
        Process CSV file and save macro features as pt files.
        """
        # Read CSV file
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Check required columns
        if 'id' not in df.columns or 'GT_Standardized_Report' not in df.columns:
            raise ValueError("CSV file must contain 'id' and 'GT_Standardized_Report' columns")

        processed_data = {}
        failed_ids = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing reports"):
            patient_id = row['id']
            report_text = row['GT_Standardized_Report']

            try:
                # Process the report
                macro_features = self.process(report_text)
                # Save individual pt file
                output_path = os.path.join(output_dir, f"{patient_id}.pt")
                torch.save(macro_features, output_path)
            except Exception as e:
                print(f"Failed to process patient {patient_id}: {str(e)}")
                failed_ids.append(patient_id)
        
    
        if failed_ids:
            print(f"Failed patient IDs: {failed_ids}")

        return processed_data


if __name__ == "__main__":
    # Initialize preprocessor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    output_dir = '/path/to/text'
    preprocessor = TextPreprocessor(device=device)
    # Process CSV file
    csv_path = "/path/to/combine.csv"

    processed_data = preprocessor.process_csv(csv_path, output_dir=output_dir)
