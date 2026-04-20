import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class FinancialEncoder:
    def __init__(self, pooling_strategy="cls"):
        if pooling_strategy not in ["cls", "mean"]:
            raise ValueError("pooling_strategy must be either 'cls' or 'mean'")
            
        self.pooling_strategy = pooling_strategy
        self.model_name = "prajjwal1/bert-mini"
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=32):
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and move to device
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            
            if self.pooling_strategy == "cls":
                # CLS pooling: Take the representation of the [CLS] token (first token)
                embeddings = outputs.last_hidden_state[:, 0, :]
                
            elif self.pooling_strategy == "mean":
                # Mean pooling: Average of all non-padding token representations
                attention_mask = encoded_input['attention_mask']
                last_hidden = outputs.last_hidden_state
                
                # Expand attention mask to match hidden state dimensions
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                
                # Zero out padding tokens and sum
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                
                # Count non-padding tokens (clamp to avoid division by zero)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Average
                embeddings = sum_embeddings / sum_mask
                
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.concatenate(all_embeddings, axis=0)
