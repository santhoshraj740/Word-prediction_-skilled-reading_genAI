import argparse
import os
import string
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_device():
    return torch.device("cpu")

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(get_device())
    model.eval()
    return model,tokenizer

def computing_surprisal(tokenizer, model,context,target_word):
    device = get_device()
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits[0,-1,:]
        prob = torch.softmax(logits,dim=-1).cpu().numpy()


    target_ids = tokenizer.encode(" " + target_word.strip(),add_special_tokens=False)
    if not target_ids:
        return float("inf")
    token_id = target_ids[0]
    p = prob[token_id]
    surprisal = -np.log2(p + 1e-20)
    entropy = -np.sum(prob * np.log2(prob + 1e-20))
    return surprisal , entropy

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
            surprisal,entropy = computing_surprisal(tokenizer, model, context, clean_word)

            results.append({
                "WordID": word_id,
                "SentenceNr": sentence_idx + 1,
                "WordNr": word_idx + 1,
                "Target": clean_word,
                "Surprisal": surprisal,
                "Entropy": entropy,
            })
            context += " " + clean_word
            word_id += 1

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
    return df  

model_list = {
    "GPT2": "gpt2",
    "GPT2_XL": "gpt2-xl",
    "GPTNEO_1.3B": "EleutherAI/gpt-neo-1.3B",
}

output_dir = "C:/Users/ssr17/next-token/model_outputs"
os.makedirs(output_dir, exist_ok=True)

final_df = None


for model_label, model_name in model_list.items():
    print(f"\nRunning model: {model_label}")

    output_csv = os.path.join(output_dir, f"surprisal_{model_label}.csv")

    df = file(passage_path, model_name, output_csv)

    # Keep relevant columns and rename Surprisal
    df = df[["WordID", "SentenceNr", "WordNr", "Target", "Surprisal"]]
    df = df.rename(columns={"Surprisal": f"{model_label}_Surprisal"})

    # Merge with previous results
    if final_df is None:
        final_df = df
    else:
        final_df = pd.merge(final_df, df, on=["WordID", "SentenceNr", "WordNr", "Target"], how="outer")

final_df.to_csv("C:/Users/ssr17/next-token/all_model_surprisal_comparison.csv", index=False)
print("\n✅ All model surprisals saved!")