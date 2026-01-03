import joblib
import re
import os

# Load Model if exists
try:
    category_pipeline = joblib.load('category_model.pkl')
except:
    category_pipeline = None

def predict_category(text):
    """Predicts the category of a resume using the trained model."""
    if not category_pipeline:
        return {"Error": 0.0}
    
    # Get probabilities
    probas = category_pipeline.predict_proba([text])[0]
    classes = category_pipeline.classes_
    
    # Sort and return top 5
    sorted_indices = probas.argsort()[::-1]
    top_results = {classes[i]: round(probas[i] * 100, 2) for i in sorted_indices[:5]}
    return top_results

def predict_ats_score(text, missing_skills_count, match_percentage):
    """
    Simulates a robust ATS scoring system.
    Score = Content (50%) + Formatting/Structure (30%) + Length/readability (20%)
    """
    # 1. Content Score (Base: Match Percentage & Missing Skills)
    # If many missing skills, penalize heavily provided match_percentage is high
    # but we will rely more on the match_percentage passed in (which is cosine sim)
    content_score = match_percentage
    if missing_skills_count > 0:
        penalty = min(20, missing_skills_count * 2)
        content_score -= penalty
    content_score = max(0, content_score)

    # 2. Structure/Formatting Score
    structure_score = 0
    # Checks
    if "@" in text: structure_score += 20 # Email
    if re.search(r'\d{10}', text): structure_score += 20 # Phone
    
    # Section Detection (Heuristic)
    sections = ['education', 'experience', 'skills', 'projects', 'summary']
    found_sections = sum(1 for s in sections if s in text.lower())
    structure_score += (found_sections * 12) # up to 60
    
    # Cap structure
    structure_score = min(100, structure_score)

    # 3. Data/Length Score
    length_score = 100
    word_count = len(text.split())
    if word_count < 200: length_score -= 40 # Too short
    if word_count > 2000: length_score -= 20 # Too long
    
    # Final Weighted Score
    final_score = (content_score * 0.5) + (structure_score * 0.3) + (length_score * 0.2)
    
    return round(final_score, 1)

def check_fake_skills(text, skills):
    """
    Heuristic to detect if skills are mentioned without context.
    Returns a list of suspicious skills.
    """
    suspicious = []
    text_lower = text.lower()
    
    # Context keywords indicating experience
    context_keywords = ['project', 'experience', 'worked', 'developed', 'created', 'using', 'proficient', 'knowledge', 'utilized']
    
    # If the text is very short, everything might be suspicious or nothing
    if len(text_lower.split()) < 100:
        return []

    # Check if context keywords appear in the resume AT ALL
    has_any_context = any(k in text_lower for k in context_keywords)
    
    if not has_any_context:
         # Extreme case: just a list of words?
         return list(skills)[:5]

    return []

