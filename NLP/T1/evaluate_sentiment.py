import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer
from encoder import FinancialEncoder
from sentiment_pipeline import load_data, evaluate_vader, frozen_lr, frozen_mlp, finetune_model

def run_disambiguation_probe():
    print("\n" + "="*40)
    print("DISAMBIGUATION PROBE")
    print("="*40)
    sentences = [
        "The bank raised interest rates.",
        "She sat by the river bank."
    ]
    print(f"Sentence 1: '{sentences[0]}'")
    print(f"Sentence 2: '{sentences[1]}'\n")
    
    for pooling in ['cls', 'mean']:                                                                                                                                                                 
        encoder = FinancialEncoder(pooling_strategy=pooling)                                
        embeddings = encoder.encode(sentences)
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"Cosine Similarity ({pooling.upper()} Pooling): {sim:.4f}") 

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sentiment Pipeline")
    parser.add_argument("--pooling", choices=["cls", "mean"], help="Pooling strategy to evaluate")
    parser.add_argument("--finetune", action="store_true", help="Run end-to-end finetuning")
    args = parser.parse_args()
    
    if not args.pooling and not args.finetune:
        parser.error("Please specify either --pooling [cls|mean] or --finetune")
    
    # 1. Load data
    print("Loading dataset...")
    texts, labels = load_data()
    
    # 2. Stratified 80/20 train/test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Handle End-to-End Finetuning
    if args.finetune:
        print("\n" + "="*40)
        print("SETUP 4: END-TO-END FINE-TUNING")
        print("="*40)
        f1, cm, _ = finetune_model(X_train_text, y_train, X_test_text, y_test)
        print(f"\nFine-Tuned Model - Macro F1: {f1:.4f}")
        print("Confusion Matrix:\n", cm)
        return
        
    # Handle Frozen Evaluation
    if args.pooling:
        print("\n" + "="*40)
        print("EVALUATING VADER BASELINE")
        print("="*40)
        vader_f1, vader_cm = evaluate_vader(X_test_text, y_test)
        print(f"VADER - Macro F1: {vader_f1:.4f}")
        print("Confusion Matrix:\n", vader_cm)
        
        print("\n" + "="*40)
        print(f"SETUP: FROZEN {args.pooling.upper()} POOLING")
        print("="*40)
        encoder = FinancialEncoder(pooling_strategy=args.pooling)
        
        print("Extracting embeddings for training and test sets (this may take a moment)...")
        X_train_emb = encoder.encode(X_train_text)
        X_test_emb = encoder.encode(X_test_text)
        
        print(f"\n--- Frozen {args.pooling.upper()} + Logistic Regression ---")
        lr_f1, lr_cm = frozen_lr(X_train_emb, X_test_emb, y_train, y_test)
        print(f"Logistic Regression - Macro F1: {lr_f1:.4f}")
        print("Confusion Matrix:\n", lr_cm)
        
        if args.pooling == "mean":
            print("\n--- Frozen MEAN + MLP ---")
            mlp_f1, mlp_cm = frozen_mlp(X_train_emb, X_test_emb, y_train, y_test)
            print(f"MLP (256 -> 128 -> 3) - Macro F1: {mlp_f1:.4f}")
            print("Confusion Matrix:\n", mlp_cm)
            
        run_disambiguation_probe()

if __name__ == "__main__":
    main()
