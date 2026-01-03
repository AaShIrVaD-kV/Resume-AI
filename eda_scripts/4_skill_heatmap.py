import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import config
from collections import Counter
from itertools import combinations

# Load Data
df = config.load_data()

# Common Skills list (Same as in training)
common_skills = {
    "python", "java", "c++", "sql", "html", "css", "javascript", "react", 
    "node", "aws", "azure", "docker", "kubernetes", "excel", "power bi", 
    "tableau", "machine learning", "tensorflow", "pytorch", "pandas", 
    "analysis", "management", "communication", "leadership", "sales", 
    "marketing", "finance", "accounting"
}

def extract_skills(text):
    text = str(text).lower()
    return [skill for skill in common_skills if skill in text]

print("Extracting skills for heatmap...")
df['Extracted_Skills'] = df['Resume_str'].apply(extract_skills)

# Count Co-occurrences
co_occurrence = Counter()
for skills in df['Extracted_Skills']:
    if len(skills) > 1:
        for pair in combinations(sorted(skills), 2):
            co_occurrence[pair] += 1

# Convert to Matrix
skills_list = sorted(list(common_skills))
matrix = pd.DataFrame(0, index=skills_list, columns=skills_list)

for (s1, s2), count in co_occurrence.items():
    matrix.loc[s1, s2] = count
    matrix.loc[s2, s1] = count # Symmetric

# Filter for better visualization (Top 15 most frequent skills)
top_skills = df['Extracted_Skills'].explode().value_counts().head(15).index
filtered_matrix = matrix.loc[top_skills, top_skills]

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(filtered_matrix, annot=True, cmap="YlGnBu", fmt='d', linewidths=.5)
plt.title("Skill Co-occurrence Heatmap (Top 15 Skills)", fontsize=16)
plt.tight_layout()

print("Showing plot... (Close window to continue)")
plt.show()
