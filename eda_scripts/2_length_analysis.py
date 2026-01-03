import matplotlib.pyplot as plt
import seaborn as sns
import config
import re

# Load Data
df = config.load_data()

# Calculate Word Count
def get_word_count(text):
    # Simple split by whitespace
    return len(str(text).split())

df['Word_Count'] = df['Resume_str'].apply(get_word_count)

# Set Theme
sns.set_theme(style="darkgrid")
plt.figure(figsize=(12, 6))

# Histogram
sns.histplot(df['Word_Count'], bins=30, kde=True, color='teal')

# Customize
plt.title("Distribution of Resume Lengths (Word Count)", fontsize=16)
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.axvline(df['Word_Count'].mean(), color='red', linestyle='--', label=f"Mean: {int(df['Word_Count'].mean())}")
plt.legend()
plt.tight_layout()

print("Showing plot... (Close window to continue)")
plt.show()

# Box Plot by Category
plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x="Category", y="Word_Count", palette="coolwarm", hue="Category", legend=False)
plt.xticks(rotation=45, ha='right')
plt.title("Resume Length by Category", fontsize=16)
plt.tight_layout()
plt.show()
