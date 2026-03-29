"""
Script to organize the results CSV files from clean.py into a single CSV ready for annotation.
"""

import pandas as pd
import glob

# clean display names
MODEL_MAP = {
    "[groq:llama-3.3-70b-versatile] {{quest}}": "llama-3.3-70b",
    "[groq:openai/gpt-oss-120b] {{quest}}":      "gpt-oss-120b",
    "[groq:qwen/qwen3-32b] {{quest}}":            "qwen3-32b",
}

# Load and merge all results CSVs
csv_files = glob.glob("results*.csv")
if not csv_files:
    print("No results CSV files found in the current folder.")
    exit()
 
print(f"Found: {csv_files}")
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

id_cols = [c for c in df.columns if c not in MODEL_MAP]

df_long = df.melt(
    id_vars=id_cols,
    value_vars=list(MODEL_MAP.keys()),
    var_name="model",
    value_name="response"
)

# Replace long column names with clean model names
df_long["model"] = df_long["model"].map(MODEL_MAP)

# Add empty annotation columns ready to fill in
for col in ["attack_success", "refusal_type", "justification_quality", "persona_adoption", "hallucination"]:
    df_long[col] = ""
 
# Sort by technique then prompt id then model
df_long = df_long.sort_values(["technique", "id", "model"]).reset_index(drop=True)
 
df_long.to_csv("results_annotation.csv", index=False)
print(f"Done — {len(df_long)} rows saved to results_annotation.csv")
print(f"Columns: {df_long.columns.tolist()}")