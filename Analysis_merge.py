import pandas as pd 

surprisal_df = pd.read_csv("all_model_surprisal_comparison.csv")
et_df = pd.read_csv("PROVO_EyeTracking&PredictabilityData.csv")

et_df = et_df[et_df["Word_In_Sentence_Number"].notna()].copy()
et_df["WordNr"] = et_df["Word_In_Sentence_Number"].astype(int)

et_df["Target_clean"] = et_df["Target"].str.lower()
surprisal_df["Target_clean"] = surprisal_df["Target"].str.lower()

et_df_aligned = pd.merge(
    et_df,
    surprisal_df[["SentenceNr", "WordNr", "Target_clean", 
                  "GPT2_Surprisal", "GPT2_XL_Surprisal", "GPTNEO_1.3B_Surprisal"]],
    left_on=["Text_ID", "WordNr", "Target_clean"],
    right_on=["SentenceNr", "WordNr", "Target_clean"],
    how="inner"
)

from pathlib import Path  
filepath = Path('C:/Users/ssr17/Research project/merged_et_surprisal.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
et_df_aligned.to_csv(filepath) 

#sample distribution plot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.kdeplot(et_df_aligned["GPT2_Surprisal"], label="GPT2")
sns.kdeplot(et_df_aligned["GPT2_XL_Surprisal"], label="GPT2-XL")
sns.kdeplot(et_df_aligned["GPTNEO_1.3B_Surprisal"], label="GPT-Neo 1.3B")
plt.legend()
plt.title("Distribution of Surprisal Values")
plt.xlabel("Surprisal")
plt.show()
