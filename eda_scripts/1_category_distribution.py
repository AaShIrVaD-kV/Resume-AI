import matplotlib.pyplot as plt
import seaborn as sns
import config

# Load Data
df = config.load_data()

# Set Theme
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 8))

# Count Plot
ax = sns.countplot(
    data=df, 
    x="Category", 
    order=df['Category'].value_counts().index, 
    palette="viridis",
    hue="Category",
    legend=False
)

# Customize
plt.xticks(rotation=45, ha='right')
plt.title("Distribution of Resumes by Category", fontsize=16)
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()

print("Showing plot... (Close window to continue)")
plt.show()
