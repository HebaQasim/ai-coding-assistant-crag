"""Part 1: explore the HumanEval dataset (downloaded from Hugging Face and saved in data/humaneval.parquet)."""
import pandas as pd

df = pd.read_parquet("data/humaneval.parquet")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Print one full problem, column by column
first = df.iloc[0]
for column in df.columns:
    print(f"\n---------- {column} ----------")
    print(first[column])

# How long are the problems and the solutions? (number of characters)
print("\nPrompt length:")
print(df["prompt"].str.len().describe())
print("\nSolution length:")
print(df["canonical_solution"].str.len().describe())
