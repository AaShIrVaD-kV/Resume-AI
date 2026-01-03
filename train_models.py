import pandas as pd
import numpy as np
import re
import joblib
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
from itertools import combinations  

# Setup NLTK
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

IMPORTANT_TERMS = {
    "python", "java", "c++", "sql", "html", "css", "javascript", "react", 
    "node", "aws", "azure", "docker", "kubernetes", "git", "linux", "excel",
    "power bi", "tableau", "machine learning", "deep learning", "tensorflow",
    "pytorch", "pandas", "numpy", "scikit-learn", "communication", "leadership",
    "management", "agile", "scrum", "marketing", "sales", "finance", "accounting"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', ' ', text)

    # Removed undefined IMPORTANT_TERMS loop
    # for term in IMPORTANT_TERMS:
    #     text = text.replace(term, term.replace(" ", "_"))

    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)



def train_category_model(df):
    """Trains a model to predict resume categories."""
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    
    print("Training Category Model (Enhanced)...")
    
    X = df['Clean_Resume']
    y = df['Category']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Improved TF-IDF
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2), # Bi-grams help ("Data Science", "Machine Learning")
        max_features=10000, # More features
        min_df=2, # Ignore unique typos
        max_df=0.85 # Ignore super common words
    )
    
    # 2. LinearSVC is often better for text, but needs calibration for probabilities
    svc = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
    clf = CalibratedClassifierCV(svc) # To get predict_proba
    
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf)
    ])
    
    print("Fitting model (this may take a moment)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # print(classification_report(y_test, y_pred)) # Reduce noise in output
    
    # Robustness Check (Sanity Check for Data Science vs Agriculture)
    test_ds = ["Data Science machine learning python sql pandas"]
    test_ag = ["Agriculture farming irrigation soil crops"]
    
    print(f"Sanity Check (DS): {pipeline.predict(test_ds)[0]}")
    print(f"Sanity Check (Ag): {pipeline.predict(test_ag)[0]}")

    # Retrain on Full Data
    print("Retraining on full dataset...")
    pipeline.fit(X, y)
    
    # Save Model
    joblib.dump(pipeline, 'category_model.pkl')
    print("Category Model saved to category_model.pkl")

def build_skill_graph(df):
    """
    Builds a co-occurrence graph of skills from the resume dataset.
    This allows us to suggest 'related skills'.
    """
    print("Building Skill Graph...")
    
    # We need to extract skills from the resumes first.
    # Since we don't have a labeled 'Skills' column, we'll use a heuristic 
    # or the same NER extraction logic if available. 
    # For speed/robustness here, we'll assume the text contains the skills.
    # A better approach (if 'Skills' column existed) would be to use that.
    # We will use a simple heuristic of common tech skills for this demonstration.
    
    # Helper: specific tech keywords to look for (expand this list for better results)
    # Expanded list of common skills across multiple domains
    common_skills = {
        # Tech / Data
        "python", "java", "c++", "sql", "html", "css", "javascript", "react", 
        "node", "aws", "azure", "docker", "kubernetes", "git", "linux", "excel",
        "power bi", "tableau", "machine learning", "deep learning", "tensorflow",
        "pytorch", "pandas", "numpy", "scikit-learn", "hadoop", "spark",
        
        # HR / Management
        "recruitment", "payroll", "employee relations", "onboarding", "compliance",
        "leadership", "management", "communication", "negotiation", "training",
        "performance management", "staffing", "sourcing", "interviewing",
        "conflict resolution", "scheduling", "coaching",
        
        # Finance / Accounting
        "financial analysis", "accounting", "auditing", "budgeting", "forecasting",
        "taxation", "bookkeeping", "reconciliation", "variance analysis", "sap",
        "quickbooks", "financial reporting", "investments", "risk management",
        "banking", "accounts payable", "accounts receivable",
        
        # Marketing / Sales
        "marketing", "sales", "social media", "seo", "sem", "content creation",
        "branding", "market research", "crm", "salesforce", "customer service",
        "lead generation", "advertising", "public relations", "campaign management",
        
        # Arts / Design
        "photoshop", "illustrator", "indesign", "graphic design", "ui/ux",
        "creativity", "typography", "sketch", "figma", "video editing",
        "photography", "animation", "visual design", "branding",
        
        # Engineering / Construction
        "autocad", "civil engineering", "project planning", "construction management",
        "blueprints", "quality control", "safety compliance", "estimation",
        "site management", "structural analysis", "surveying",
        
        # Education
        "teaching", "curriculum development", "classroom management", "mentoring",
        "lesson planning", "student assessment", "special education", "tutoring",
        
        # Healthcare / Fitness
        "nursing", "patient care", "medical terminology", "cpr", "first aid",
        "fitness training", "nutrition", "wellness", "personal training", "kinesiology",
        "rehabilitation", "clinical research"
    }
    
    co_occurrence = Counter()
    
    for text in df['Clean_Resume']:
        # simplistic checks
        found_skills = {skill for skill in common_skills if skill in text}
        
        # Count pairs
        if len(found_skills) > 1:
            for pair in combinations(sorted(found_skills), 2):
                co_occurrence[pair] += 1
                
    # Convert to adjacency list for JSON
    skill_graph = {}
    for (s1, s2), count in co_occurrence.most_common(1000): # Increased limit
        if s1 not in skill_graph: skill_graph[s1] = []
        if s2 not in skill_graph: skill_graph[s2] = []
        
        # Add bidirectional link if strong enough
        if count > 2: # Threshold
            skill_graph[s1].append(s2)
            skill_graph[s2].append(s1)
            
    # Deduplicate
    for k in skill_graph:
        skill_graph[k] = list(set(skill_graph[k]))
        
    with open('skill_graph.json', 'w') as f:
        json.dump(skill_graph, f)
    print("Skill Graph saved to skill_graph.json")

def extract_category_keywords(df):
    """Extracts top keywords for each category using TF-IDF."""
    print("Extracting Category Keywords...")
    
    # 1. Group text by Category
    grouped_df = df.groupby('Category')['Clean_Resume'].apply(lambda x: " ".join(x)).reset_index()
    
    # 2. TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(grouped_df['Clean_Resume'])
    feature_names = np.array(tfidf.get_feature_names_out())
    
    keywords_dict = {}
    
    for i, row in grouped_df.iterrows():
        category = row['Category']
        # Get top 20 keywords for this document (Category)
        # We sort by TF-IDF score
        row_vector = tfidf_matrix[i].toarray().flatten()
        top_indices = row_vector.argsort()[::-1][:20]
        top_keywords = feature_names[top_indices].tolist()
        keywords_dict[category] = top_keywords
        
    with open('category_keywords.json', 'w') as f:
        json.dump(keywords_dict, f)
    print("Category Keywords saved to category_keywords.json")

def main():
    # Load Data
    try:
        csv_path = r"Resume/Resume.csv"
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} resumes.")
    except FileNotFoundError:
        print("Error: Resume.csv not found in 'Resume' folder.")
        return

    # Preprocess
    print("Cleaning text...")
    df['Clean_Resume'] = df['Resume_str'].apply(clean_text)
    
    # Train
    train_category_model(df)
    build_skill_graph(df) # Updated with more skills
    extract_category_keywords(df) # New function
    print("Done! Artifacts generated.")

if __name__ == "__main__":
    main()
