import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def load_data(file_path="Sentences_AllAgree.txt"):
    texts, labels = [], []
    # Labels are usually positive, negative, neutral
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # Split by the last '@' to separate text and label
            parts = line.rsplit('@', 1)
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(label_map[parts[1].lower()])
                
    return texts, labels

def evaluate_vader(texts, labels):
    analyzer = SentimentIntensityAnalyzer()
    preds = []
    
    for text in texts:
        scores = analyzer.polarity_scores(text)
        comp = scores['compound']
        
        # Map compound score to 0 (negative), 1 (neutral), 2 (positive)
        if comp >= 0.05:
            preds.append(2)
        elif comp <= -0.05:
            preds.append(0)
        else:
            preds.append(1)
            
    return f1_score(labels, preds, average='macro'), confusion_matrix(labels, preds)

def frozen_lr(embeddings_train, embeddings_test, y_train, y_test):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(embeddings_train, y_train)
    preds = clf.predict(embeddings_test)
    return f1_score(y_test, preds, average='macro'), confusion_matrix(y_test, preds)

def frozen_mlp(embeddings_train, embeddings_test, y_train, y_test):
    # MLP mapping 256 -> 128 -> 3
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42)
    clf.fit(embeddings_train, y_train)
    preds = clf.predict(embeddings_test)
    return f1_score(y_test, preds, average='macro'), confusion_matrix(y_test, preds)

# --- End-to-End Fine-Tuning Components ---

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

class BertMeanPoolClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super(BertMeanPoolClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("prajjwal1/bert-mini")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return loss, logits

def finetune_model(train_texts, train_labels, test_texts, test_labels):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
    
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    model = BertMeanPoolClassifier(num_labels=3)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Train for 5 epochs
    model.train()
    for epoch in range(5):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, _ = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            _, logits = model(input_ids, attention_mask)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            
    return f1_score(test_labels, preds, average='macro'), confusion_matrix(test_labels, preds), model
