# Word-prediction_-skilled-reading_genAI
Aims to evaluate how different language models, each with varying capacity and architecture, estimate surprisal for a given passage. This is particularly relevant for research that investigates the alignment between computational models and human sentence processing, such as Cevoli et al. (2022).

# Environment Info:
## System:
- Python: 3.12.6 (conda-forge)
- Anaconda: 2.6.6
- Jupyter: 4.2.5

## Libraries:
- torch: 2.6.0+cu118
- transformers: 4.52.4
- pandas: 2.2.2
- numpy: 1.26.4
- matplotlib: 3.10.0
- seaborn: 0.13.2
- scikit-learn: 1.5.1
# Data:
The provo corpus data was collect from the cevoli et al.(2022). This is the link for the data https://osf.io/p6tkh. All the 56 passages where extracted from the original provo corpus data, the link for that is provided here, https://osf.io/e4a2m. These passages are saved in the passage.txt file. 

# Models:
Below are the three transformer models used in this project, all accessed via Hugging Face’s Transformers library: 

**GPT2**
- Model ID: gpt2
- Developer: OpenAI
- Size: ~117M parameters
- Architecture: 12 layers, 12 attention heads, 768 hidden units
- Notes: Original GPT-2 base model; lightweight and fast to run.

**GPT2-XL**
- Model ID: gpt2-xl
- Developer: OpenAI
- Size: ~1.5B parameters
- Architecture: 48 layers, 25 attention heads, 1600 hidden units
- Notes: Larger variant of GPT2 with more depth and representational power. Provides lower surprisals due to improved context modeling.

**GPT-Neo 1.3B**
- Model ID: EleutherAI/gpt-neo-1.3B
- Developer: EleutherAI
- Size: 1.3B parameters
- Architecture: GPT-style architecture (24 layers, 16 attention heads, 2048 hidden units)
- Notes: Open alternative to GPT-3. Slightly slower than GPT2 but good generalization.
All three models use byte-level BPE tokenization and were evaluated in inference mode using AutoModelForCausalLM and AutoTokenizer.

# Set up:
## Prerequisites
**Hardware**: A machine with a GPU (recommended for faster processing), but CPU fallback works for smaller models.
**Disk Space**: At least 10 GB for models and outputs on your C-drive, more if using large models like gpt2-xl or gpt-neo.
## Python Environment Setup
**Create and activate a new conda environment**:

	conda create -n surprisal_env python=3.12 -y
	conda activate surprisal_env

**Install core packages**:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

	pip install transformers pandas numpy matplotlib seaborn tqdm
If no GPU is available, skip the pytorch-cuda and just install CPU-only PyTorch.
			
   	pip install torch   
(Optional) Jupyter setup:
	
 	conda install jupyter notebook


#Model Pipeline:
The whole pipeline is sit in a way to easy swap between different LLM models available in HuggingFace Transformer library. The model can be loaded on CPU or GPU according to your preference.
## Import libraries 
	
	import argparse
	import os
	import string
	import pandas as pd
	import torch
	import numpy as np
	from transformers import AutoModelForCausalLM, AutoTokenizer
	from tqdm import tqdm


## Text preprocessing:
- Read passage.txt using open() and .readlines().
- Strip each line, split into words, and clean punctuation using word.strip(string.punctuation).
- Maintain WordID, SentenceNr, and WordNr counters to assign unique identifiers.

## Model and Tokenizer loading: 
- We have to load the desired language model and tokenizer for surprisal computation.  
- Load both using Hugging Face’s AutoModelForCausalLM and AutoTokenizer.
- Move model to device (cuda if available, otherwise cpu).
- Set the model to evaluation mode with model.eval() to avoid gradient tracking and dropout.
- Tokenizer ensures that input words and tokens align with the vocabulary used during model training.

		def load_model(model_name):
		    tokenizer = AutoTokenizer.from_pretrained(model_name)
		    model = AutoModelForCausalLM.from_pretrained(model_name).to(get_device())
		    model.eval()
		    return model,tokenizer


## Suprisal computation (single token):
- Compute surprisal using only the first token of the target word (traditional method).
- Encode the context (left-of-word text) into input_ids.
- Get model logits for next-token prediction.
- Apply softmax to extract probability distribution.
- Extract the probability of the target token (first token of word).
- Surprisal = -log2(p) where p is that probability.
- This method is simple but can underrepresent surprisal for multi-token words.

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


## Batch processing:
- Run the pipeline end-to-end for every word in the passage.
- Loop through each sentence and word.
- Build the context word-by-word.
- Compute surprisal and entropy using multi-token method.
- Append results to a list.
- Save as CSV for comparison and visualization.

		def file(input_text_path, model_name, output_csv_path):
		    model, tokenizer = load_model(model_name)
		    results = []
		    with open(input_text_path, "r", encoding='utf-8') as f:
		        lines = f.readlines()
		    word_id = 0 
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


## Model Loop Automation
- Efficiently evaluate multiple language models on the same passage with minimal changes.
- Define a dictionary (model_list) with model names and their corresponding Hugging Face IDs.
- Loop through each model, run the file() function, and save its output CSV.
- Rename surprisal columns to include the model name.
- Optionally, merge results into a single file for visual or statistical comparison

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
		
		    df = df[["WordID", "SentenceNr", "WordNr", "Target", "Surprisal"]]
		    df = df.rename(columns={"Surprisal": f"{model_label}_Surprisal"})
		
		    if final_df is None:
		        final_df = df
		    else:
		        final_df = pd.merge(final_df, df, on=["WordID", "SentenceNr", "WordNr", "Target"], how="outer")
		
		final_df.to_csv("C:/Users/ssr17/next-token/all_model_surprisal_comparison.csv", index=False)
		print("\n All model surprisals saved!")


## For mutitoken surprisal computation:
- Compute surprisal for target words that are split into multiple tokens.
- Tokenize the target word and context.
- For each target token:
- Feed current context to model
- Compute log-probability of the next token
- Update context with predicted token
- Sum surprisal values for each token.
- Also calculate mean surprisal and average entropy per token.
- Better represents surprisal for rare/complex words.

		def compute_surprisal_multitoken(model, tokenizer, context, target_word):
		    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)
		    target_ids = tokenizer.encode(" " + target_word, add_special_tokens=False)
		    if not target_ids:
		        return float('inf'), 0, 0
		
		    surprisal_sum = 0
		    entropy_total = 0
		
		    for tid in target_ids:
		        with torch.no_grad():
		            output = model(input_ids)
		            logits = output.logits[:, -1, :]
		            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
		
		        p = probs[tid]
		        surprisal_sum += -np.log2(p + 1e-20)
		        entropy_total += -np.sum(probs * np.log2(probs + 1e-20))
		
		        tid_tensor = torch.tensor([[tid]]).to(device)
		        input_ids = torch.cat([input_ids, tid_tensor], dim=1)
		
		    mean_surprisal = surprisal_sum / len(target_ids)
		    mean_entropy = entropy_total / len(target_ids)
		    return surprisal_sum, mean_surprisal, mean_entropy, len(target_ids)

The results are complied into the same CSV files,has columns with model name and their surprisal values next to each other for easier comparison. 


# Results
There was slight variations in the surprisal scores from that of cevoli’s. I believe that the reason might be due to varying input data (passage.txt), this passage data was not specified in the cevoli’s paper and was not available in the link as well. The passage that we used were extracted from the original provo corpus data. 
