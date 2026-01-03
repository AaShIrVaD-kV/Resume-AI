import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import textstat

# Load Spacy Model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Auto-download the model if missing
    import subprocess
    import sys
    print("⏳ Downloading Spacy model 'en_core_web_sm' (approx 12MB)...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ Model downloaded successfully. Loading...")
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        nlp = spacy.blank("en")

# NLTK Data Check
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    # Fallback if setup_nltk.py wasn't run, but this might still race if parallel
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return str(e)

import docx

def extract_text_from_docx(file):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return str(e)

def clean_text(text):
    """Cleans and preprocesses text."""
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization (split by whitespace)
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def calculate_similarity(resume_text, job_description, total_keywords=0, missing_keywords=0):
    """
    Calculates a hybrid match score:
    - 30% Semantic Similarity (TF-IDF Cosine)
    - 70% Keyword Match Ratio
    """
    
    # 1. Cosine Similarity
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_description)
    
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    except ValueError:
        # Handle cases with empty text
        cosine_sim = 0
    
    # 2. Keyword Match Ratio
    # If no keywords found in JD, fall back entirely to cosine sim
    if total_keywords == 0:
        return round(cosine_sim, 2)
        
    found_keywords = total_keywords - missing_keywords
    match_ratio = (found_keywords / total_keywords) * 100
    
    # Weighted Average (Heavily biased towards keywords as per user feedback)
    final_score = (cosine_sim * 0.30) + (match_ratio * 0.70)
    
    return round(final_score, 2)

def extract_entities(text):
    """Extracts entities using Spacy NER."""
    doc = nlp(text)
    entities = {
        "ORG": [],
        "PERSON": [],
        "GPE": [],
        "DATE": []
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

def extract_skills(text):
    """
    Extracts skills using Spacy and filters against known skill graph.
    This prevents random names/places from appearing as skills.
    """
    doc = nlp(text)
    # 1. Candidate Generation: Nouns and Proper Nouns
    candidates = set([token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 2])
    
    # 2. Filtering against Knowledge Base (Skill Graph)
    # If skill_graph is loaded, use it as a whitelist.
    if skill_graph:
        valid_skills = set()
        for candidate in candidates:
            # Check exact match or if candidate helps form a compound skill
            # (Simple check: is it in our known skills list?)
            if candidate in skill_graph:
                valid_skills.add(candidate)
        
        # Fallback: If filtered list is too empty (e.g. strict matching failed), 
        # return the raw candidates but maybe limit to common tech terms
        if len(valid_skills) < 3: 
             return candidates # Better to show something than nothing, but risk noise
        return valid_skills
    
    return candidates

def get_match_level(score):
    """Returns the match level, color, and message based on score."""
    if score >= 75:
        return "🌟 Excellent Match", "success", "Highly aligned with job requirements"
    elif score >= 65:
        return "🟢 Good Match", "success", "Strong alignment; minor tailoring recommended"
    elif score >= 50:
        return "🟡 Moderate Match", "warning", "Relevant profile, but needs focused tailoring"
    elif score >= 30:
        return "🟠 Low Match", "warning", "Partial relevance; significant tailoring required"
    elif score >= 10:
        return "🔴 Very Low Match", "error", "Major skill and experience gaps detected"
    else:
        return "❌ No Match", "error", "Resume does not align with the job description"

def calculate_readability(text):
    """Calculates Flesch Reading Ease score."""
    return textstat.flesch_reading_ease(text)

import json
import random

# Load Skill Graph
try:
    with open('skill_graph.json', 'r') as f:
        skill_graph = json.load(f)
        # Normalize keys to lower case for easier matching
        skill_graph = {k.lower(): v for k, v in skill_graph.items()}
except FileNotFoundError:
    skill_graph = {}

def recommend_skills(current_skills):
    """Recommends skills based on co-occurrence graph from real data."""
    recommendations = set()
    
    # Identify top skills to query
    # In a real graph, we prioritize skills that are "hubs" or highly connected
    for skill in current_skills:
        # fuzzy match or direct lookup
        # Simple direct lookup for now
        skill_lower = skill.lower()
        matches = [k for k in skill_graph if k in skill_lower]
        for match in matches:
            recommendations.update(skill_graph[match])
            
    # Remove skills already present
    current_lower = {s.lower() for s in current_skills}
    unique_recommendations = [r for r in recommendations if r not in current_lower]
    
    # Return top 10 random from the relevant set to vary suggestions
    if len(unique_recommendations) > 10:
        return random.sample(unique_recommendations, 10)
    return unique_recommendations[:10]

def generate_bullet_points(missing_skills):
    """
    Generates professional bullet points for missing skills.
    Uses simple templates.
    """
    templates = [
        "Proficient in {skill}, utilizing it to optimize data processing pipelines.",
        "Applied {skill} to enhance system performance and scalability.",
        "Developed robust solutions using {skill} in an agile environment.",
        "Demonstrated expertise in {skill} through successful project delivery.",
        "Integrated {skill} best practices to improve code quality and maintainability."
    ]
    
    suggestions = {}
    for skill in missing_skills:
        # Pick 2 random templates
        points = []
        for t in random.sample(templates, 2):
            points.append(t.format(skill=skill.title()))
        suggestions[skill] = points
        
    return suggestions
