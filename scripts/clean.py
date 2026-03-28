import pandas as pd

df = pd.read_csv("results.csv")

keep = [
    "id",
    "technique",
    "quest",
    "[groq:llama-3.3-70b-versatile] {{quest}}",
    "[groq:openai/gpt-oss-120b] {{quest}}",
    "[groq:qwen/qwen3-32b] {{quest}}"
]

df = df[keep].to_csv("resultsnameCHANGEME.csv", index=False)
