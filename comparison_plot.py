import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/ssr17/next-token/all_model_surprisal_comparison.csv") #replace your csv path

df_long = df.melt(
    id_vars=["WordID", "Target"],
    value_vars=["GPT2_Surprisal", "GPT2_XL_Surprisal", "GPTNEO_1.3B_Surprisal"],
    var_name="Model",
    value_name="Surprisal"
)


plt.figure(figsize=(14, 6))
sns.lineplot(data=df_long, x="WordID", y="Surprisal", hue="Model", marker="o")
plt.title("Surprisal Score Across Passage")
plt.xlabel("Word Index (WordID)")
plt.ylabel("Surprisal (bits)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------

df["Avg_Surprisal"] = df[["GPT2_Surprisal", "GPT2_XL_Surprisal", "GPTNEO_1.3B_Surprisal"]].mean(axis=1)

top_df = df.sort_values("Avg_Surprisal", ascending=False).head(10)

top_long = top_df.melt(id_vars=["Target"], 
                       value_vars=["GPT2_Surprisal", "GPT2_XL_Surprisal", "GPTNEO_1.3B_Surprisal"],
                       var_name="Model", value_name="Surprisal")

plt.figure(figsize=(12, 6))
sns.barplot(data=top_long, x="Target", y="Surprisal", hue="Model")
plt.title("Top 10 Most Surprising Words by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------

import seaborn as sns

corr = df[["GPT2_Surprisal", "GPT2_XL_Surprisal", "GPTNEO_1.3B_Surprisal"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Model Surprisal Scores")
plt.show()

#-------------------------------------------------------------------------------------------------