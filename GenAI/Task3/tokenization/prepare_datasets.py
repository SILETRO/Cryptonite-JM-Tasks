import os
import json
import glob
import numpy as np
import tiktoken

def prepare_datasets():
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Tokenizer Vocabulary Size (gpt2): {vocab_size}")
    
    eot_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    print("Preparing Poems Dataset for Pre-training...")
    poems_dir = 'data/poems'
    
    poem_files = glob.glob(os.path.join(poems_dir, '**', '*.txt'), recursive=True)
    print(f"Found {len(poem_files)} poem files.")
    
    poems_tokens = []
    
    for file_path in poem_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
                if not text:
                    continue
                # Encode the text to token IDs
                tokens = enc.encode(text)
                poems_tokens.extend(tokens)
                # Append the special EOT token to separate poems
                poems_tokens.append(eot_token)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Create 90% train / 10% val split for poems
    n_poems = len(poems_tokens)
    train_poems_data = poems_tokens[:int(n_poems * 0.9)]
    val_poems_data = poems_tokens[int(n_poems * 0.9):]
    
    train_poems_ids = np.array(train_poems_data, dtype=np.uint16)
    val_poems_ids = np.array(val_poems_data, dtype=np.uint16)
    
    train_poems_ids.tofile('poems_train.bin')
    val_poems_ids.tofile('poems_val.bin')
    print(f"Poems train: {len(train_poems_ids):,} tokens | val: {len(val_poems_ids):,} tokens")
    print(f"Poems unique tokens: {len(np.unique(train_poems_ids)):,} | max token ID: {np.max(train_poems_ids)}")

    raps_json_path = 'data/raps/cleaned_combined.json'
    
    raps_tokens = []
    
    if os.path.exists(raps_json_path):
        with open(raps_json_path, 'r', encoding='utf-8') as f:
            rap_data = json.load(f)
            
        print(f"Found {len(rap_data)} tagged rap songs.")
            
        for song in rap_data:
            ann = song.get('annotations', {})
            genre = ann.get('genre', 'unknown')
            mood = ann.get('mood', 'unknown')
            rhyme = ann.get('rhyme_scheme', 'unknown')
            cadence = ann.get('cadence', 'unknown')
            
            lyrics = song.get('lyrics', '').strip()
            
            # Create the conditioning header prepended to the lyrics
            header = f"[Genre: {genre}] [Mood: {mood}] [Rhyme: {rhyme}] [Cadence: {cadence}]\n"
            full_text = header + lyrics
            
            tokens = enc.encode(full_text)
            raps_tokens.extend(tokens)
            raps_tokens.append(eot_token)

        # Create 90% train / 10% val split for raps
        n_raps = len(raps_tokens)
        train_raps_data = raps_tokens[:int(n_raps * 0.9)]
        val_raps_data = raps_tokens[int(n_raps * 0.9):]
        
        train_raps_ids = np.array(train_raps_data, dtype=np.uint16)
        val_raps_ids = np.array(val_raps_data, dtype=np.uint16)
        
        train_raps_ids.tofile('raps_train.bin')
        val_raps_ids.tofile('raps_val.bin')
        
        print(f"Raps train: {len(train_raps_ids):,} tokens | val: {len(val_raps_ids):,} tokens")
        print(f"Raps unique tokens: {len(np.unique(train_raps_ids)):,} | max token ID: {np.max(train_raps_ids)}")
    else:
        print(f"Warning: Raps JSON not found at {raps_json_path}. Skipping raps dataset preparation.")


if __name__ == '__main__':
    prepare_datasets()
