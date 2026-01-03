import matplotlib.pyplot as plt
from wordcloud import WordCloud
import config
import nltk
from nltk.corpus import stopwords
import re

# Setup NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load Data
df = config.load_data()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Select Top 3 Categories
top_categories = df['Category'].value_counts().index[:3]

plt.figure(figsize=(20, 10))

for i, category in enumerate(top_categories):
    # Filter text for this category
    cat_text = " ".join(df[df['Category'] == category]['Resume_str'].apply(clean_text))
    
    # Generate Cloud
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        stopwords=stop_words,
        min_font_size=10
    ).generate(cat_text)
    
    plt.subplot(1, 3, i+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud: {category}", fontsize=20)

plt.tight_layout()
print("Showing plot... (Close window to continue)")
plt.show()
