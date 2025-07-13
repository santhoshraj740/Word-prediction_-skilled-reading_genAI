import argparse
import os
import string
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(get_device())
    model.eval()
    return model,tokenizer

def computing_surprisal_multitoken(tokenizer, model, context, target_word):
    device = get_device()

    # Tokenize the context
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)

    # Tokenize the full target word (with leading space)
    target_ids = tokenizer.encode(" " + target_word.strip(), add_special_tokens=False)

    if not target_ids:
        return float("inf")  # skip broken tokenizations

    surprisal_sum = 0
    entropy = 0

    for tid in target_ids:
        with torch.no_grad():
            output = model(input_ids)
            logits = output.logits[:, -1, :]  # next-token prediction
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        p = probs[tid]
        surprisal_sum += -np.log2(p + 1e-20)  # accumulate surprisal
        entropy += -np.sum(probs * np.log2(probs + 1e-20))

        # Update context with the predicted token
        tid_tensor = torch.tensor([[tid]]).to(device)
        input_ids = torch.cat([input_ids, tid_tensor], dim=1)
        
    num_tokens = len(target_ids)
    mean_surprisal = surprisal_sum / num_tokens
    mean_entropy = entropy / num_tokens
    
    return surprisal_sum, entropy, num_tokens, mean_entropy, mean_surprisal

def file(input_text_path, model_name, output_csv_path):
    model, tokenizer = load_model(model_name)
    results = []

    with open(input_text_path, "r", encoding='utf-8') as f:
        lines = f.readlines()

    word_id = 0  # ← typo fix here
    for sentence_idx, line in enumerate(tqdm(lines)):
        words = line.strip().split()
        context = ""
        for word_idx, word in enumerate(words):
            clean_word = word.strip(string.punctuation)
            if not clean_word:
                continue
            if context == "":
                context = clean_word
                continue

            # FIX: Correct order of arguments
            surprisal,entropy, num_tokens, mean_entropy, mean_surprisal = computing_surprisal_multitoken(tokenizer, model, context, clean_word)

            results.append({
                "WordID": word_id,
                "SentenceNr": sentence_idx + 1,
                "WordNr": word_idx + 1,
                "Target": clean_word,
                "Num_Tokens": num_tokens,
                "Surprisal": surprisal,
                "Surprisal_Mean": mean_surprisal,
                "Entropy": entropy,
                "Entropy_Mean": mean_entropy,
            })
            context += " " + clean_word
            word_id += 1

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
    return df  

# Set model and file paths
model_name = "gpt2"
passage_path = "C:/Users/ssr17/next-token/passage.txt"
output_path = "C:/Users/ssr17/next-token/surprisal_gpt2_multitoken.csv"  

# Run processing
df = file(passage_path, model_name, output_path)

# Preview
df.head()