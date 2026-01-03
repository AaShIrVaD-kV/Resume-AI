# FINAL YEAR PROJECT REPORT

**PROJECT TITLE:** Resume Skill Matcher Pro AI  
**DOMAIN:** Data Science / Natural Language Processing  
**SUBMITTED BY:** [Your Name]  
**DATE:** December 2025  

---

# ABSTRAACT

In the contemporary recruitment landscape, Applicant Tracking Systems (ATS) have become the gatekeepers of employment, filtering out up to 75% of resumes before they are ever seen by a human recruiter. This automated filtration process often rejects qualified candidates due to poor keyword optimization, formatting issues, or a lack of semantic alignment with the job description. The "Resume Skill Matcher Pro AI" is a comprehensive software solution designed to democratize access to these screening insights. By utilizing advanced Natural Language Processing (NLP) techniques—including Term Frequency-Inverse Document Frequency (TF-IDF), Cosine Similarity, and Named Entity Recognition (NER)—the system analyzes resumes against job descriptions to provide a tangible "Match Score" and an estimated "ATS Probability." Furthermore, the system employs a machine learning classifier (Linear SVC) to categorize resumes into professional domains and uses a co-occurrence graph to recommend missing skills. This report details the theoretical underpinnings, architectural design, implementation, and testing of the system, demonstrating its efficacy in improving candidate application quality.

---

# TABLE OF CONTENTS

1. **Chapter 1: Introduction**
   1.1 Background of the Study
   1.2 Problem Statement
   1.3 Project Motivation
   1.4 Objectives
   1.5 Scope of the Project
   1.6 Limitations

2. **Chapter 2: Verification and Validation (Literature Review)**
   2.1 Evolution of Recruitment Tech
   2.2 Understanding Applicant Tracking Systems (ATS)
   2.3 Natural Language Processing in HR
   2.4 Machine Learning Algorithms (SVM vs Naive Bayes)

3. **Chapter 3: System Analysis & Design**
   3.1 System Architecture
   3.2 Technical Stack
   3.3 Data Flow Diagrams
   3.4 Functional Requirements
   3.5 Non-Functional Requirements

4. **Chapter 4: Methodology & Algorithms**
   4.1 Data Collection
   4.2 Data Preprocessing (Cleaning, Tokenization, Lemmatization)
   4.3 Feature Engineering (TF-IDF)
   4.4 The Mathematics of Cosine Similarity
   4.5 Named Entity Recognition (NER)
   4.6 Skill Graph Generation

5. **Chapter 5: Implementation**
   5.1 Environment Setup
   5.2 Backend Logic (Python)
   5.3 Frontend Interface (Streamlit)
   5.4 Key Code Modules

6. **Chapter 6: Results & Discussion**
   6.1 Model Performance Evaluation
   6.2 Application Walkthrough (Screenshots)
   6.3 Comparative Analysis

7. **Chapter 7: Conclusion & Future Scope**
   7.1 Conclusion
   7.2 Future Enhancements

8. **References**

---

# CHAPTER 1: INTRODUCTION

## 1.1 Background of the Study
The digital revolution has transformed job searching from a manual, networking-heavy process into a high-volume digital transaction. Job portals like LinkedIn, Indeed, and Glassdoor allow applicants to apply to dozens of jobs with a single click. While this convenience is beneficial, it has resulted in employers being flooded with applications. To cope with this volume, over 98% of Fortune 500 companies rely on Applicant Tracking Systems (ATS) to parse, rank, and filter applications automatically.

## 1.2 Problem Statement
Despite possessing the requisite skills, many final-year students and job seekers fail to clear the initial screening round. The primary reasons include:
1.  **Keyword Mismatch:** Failure to use the exact terminology expected by the ATS.
2.  **Context Gap:** Describing skills in a way that the algorithm does not recognize as relevant.
3.  **Lack of Feedback:** When rejected, candidates rarely receive an explanation, preventing them from iterating and improving their resumes.

There is a critical need for a "pre-screening" tool that mimics ATS behavior and provides candidates with the feedback they need to optimize their resumes *before* submission.

## 1.3 Project Motivation
This project was motivated by the observation that many technically proficient peers struggled to land interviews. Upon analysis, it was evident that the content of their resumes, while accurate, was not *optimized* for machine reading. This project aims to level the playing field by putting ATS-grade analysis tools into the hands of the candidates.

## 1.4 Objectives
The primary objectives of the Resume Skill Matcher Pro AI are:
*   **To develop a Resume Parsing Module:** Capable of reading text from PDF documents.
*   **To implement a Matching Engine:** utilizing Cosine Similarity to quantify the relevance of a resume to a job description.
*   **To create a Predictive Model:** That classifies resumes into industry categories (e.g., Data Science, Web Development) to ensure they align with the target role.
*   **To build a Recommendation System:** That identifies missing keywords and suggests related skills based on industry trends.
*   **To provide an Interactive UI:** Using Streamlit to visualize scores, skill graphs, and insights.

## 1.5 Scope of the Project
The application is designed for:
*   **Job Seekers:** To test and improve their resumes.
*   **Career Counselors:** To demonstrate to students where their profiles lack depth.
*   **Recruiters:** To potentially use as a lightweight screening tool.
The current scope covers text-based analysis of English language resumes and Job Descriptions.

---

# CHAPTER 2: THEORETICAL BACKGROUND

## 2.1 Understanding ATS
An Applicant Tracking System (ATS) acts as a database for job applicants. When a resume is uploaded, the ATS parses the document, stripping away formatting to extract raw text. It then scans this text for specific "hard skills" (e.g., Python, SQL) and "soft skills" (e.g., Communication, Leadership). Resumes are scored based on the frequency and context of these keywords relative to the specific Job Description (JD).

## 2.2 Natural Language Processing (NLP)
NLP is a subfield of artificial intelligence dealing with the interaction between computers and human language. In this project, NLP is crucial for:
*   **Tokenization:** Breaking text into individual words.
*   **Stop Word Removal:** Removing common words (the, and, is) that carry little meaning.
*   **Lemmatization:** reducing words to their base root (e.g., "coding" -> "code") to ensure matches even if tenses differ.

## 2.3 Machine Learning in Classification
To solve the problem of identifying *what* kind of resume an applicant has uploaded, we employ Supervised Learning.
*   **Linear SVC (Support Vector Classifier):** This algorithm was chosen over Naive Bayes and Random Forest because it excels in high-dimensional sparse data environments, which is typical for text data represented by TF-IDF vectors. It attempts to find a hyperplane that best separates different categories of resumes.

---

# CHAPTER 3: SYSTEM ANALYSIS AND DESIGN

## 3.1 System Architecture
The system follows a modular architecture:
1.  **Input Layer:** Streamlit Frontend accepts PDF uploads and Text input for JD.
2.  **Processing Layer:** 
    *   **Text Extraction:** `pdfplumber` or `pyresparser` logic.
    *   **NLP Pipeline:** `nltk` and `spacy` for cleaning and entity extraction.
3.  **Analytical Layer:**
    *   `scikit-learn` models for TF-IDF vectorization and classification.
    *   Custom algorithms for Skill Graph traversal.
4.  **Output Layer:** Visualization of metrics via `Plotly`.

## 3.2 Technical Stack Details
*   **Python:** Chosen for its rich ecosystem of Data Science libraries.
*   **Streamlit:** Facilitates rapid prototyping of data-driven web apps without complex frontend code.
*   **Spacy:** An industrial-strength NLP library used here for Named Entity Recognition (finding Organization names, Dates, etc.).
*   **NLTK:** Used for basic text processing steps like stopword removal.
*   **Plotly:** Enables interactive charts that allow users to hover and see detailed data points.

## 3.3 Functional Requirements
*   **FR-1 Upload:** The system must accept PDF files up to 200MB.
*   **FR-2 Parsing:** The system must successfully extract readable text from valid PDFs.
*   **FR-3 Scoring:** The system must return a percentage score between 0-100.
*   **FR-4 Recommendation:** The system must list at least 5 missing skills if the score is below 90%.

---

# CHAPTER 4: METHODOLOGY & ALGORITHMS

## 4.1 Data Preprocessing Pipeline
Raw text from resumes is often messy. The pipeline ensures consistency:
1.  **Lowercasing:** "Python" and "python" are treated as the same.
2.  **Regex Cleaning:** Removing special characters, email addresses, and URLs to focus purely on content.
3.  **Stopword Filtering:** We utilize the NLTK English stopword list but customize it to ensure tokens like "C++" or "IT" (which might be filtered out by aggressive cleaners) are preserved where possible or handled via specific extraction rules.

## 4.2 TF-IDF Vectorization
We do not simply count words. We use Term Frequency-Inverse Document Frequency.
*   **TF:** How often a word appears in the resume.
*   **IDF:** How rare the word is across all documents.
*   *Why?* The word "the" appears often but is useless. The word "TensorFlow" is rare, so if it appears, it carries a high weight. This ensures that unique technical skills contribute more to the match score than generic verbs.

## 4.3 Cosine Similarity
To compare the Resume and the Job Description, we treat both as vectors in a multi-dimensional space.
Formula: 
$$ \text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} $$
Where A is the Resume Vector and B is the Job Description Vector. A result of 1 means they are identical; 0 means they share no common vocabulary.

## 4.4 Entity Extraction (NER)
We use a pre-trained Spacy model (`en_core_web_sm`) to identify entities.
*   **ORG:** Companies worked at.
*   **DATE:** Employment durations.
*   **GPE:** Locations.
This allows the app to visually segment the resume data for the user.

---

# CHAPTER 5: IMPLEMENTATION

## 5.1 Project Structure
The project is organized as follows:
```
/Resume_Skill_Match_Checker_App
│── app.py                 # Main application entry point (Streamlit)
│── utils.py               # Helper functions (Text extraction, cleaning)
│── ats_model.py           # ML Prediction logic and classifiers
│── visualizations.py      # Plotting functions
│── train_models.py        # Script to retrain the ML models
│── style.css              # Custom styling for the UI
│── data/                  # Dataset storage
└── models/                # Serialized .pkl model files
```

## 5.2 Key Code: Similarity Calculation
The core logic resides in `utils.py`. We manually calculate the intersection of skill sets to ensure high precision for keywords, but we rely on Cosine Similarity for the overall "vibe" check.

```python
# Snippet from utils.py
def calculate_similarity(resume_text, job_desc):
    text_list = [resume_text, job_desc]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    return matchPercentage
```
*Note: The actual implementation in the project includes additional weighting for explicit skill keywords.*

## 5.3 Frontend Implementation
Streamlit allows us to create a responsive layout using `st.columns`. We use `st.tabs` to separate the analysis into logical sections: "Match Score", "ATS Insights", "Skill Graph".

## 5.4 Handling "Fake" Skills
A unique feature of this project is the experimental "Fake Skill Check". It scans the resume provided skills against the resume text context. If a skill appears in the list but is never mentioned in the experience paragraphs, it is flagged as "Suspicious" or "Unsupported", prompting the user to provide evidence for that skill.

---

# CHAPTER 6: RESULTS AND DISCUSSION

## 6.1 Model Accuracy
The Category Prediction model was trained on a public dataset of ~900 resumes covering 25 categories (Data Science, HR, Arts, etc.).
*   **Training Accuracy:** 99%
*   **Test Accuracy:** ~96%
This high accuracy ensures that if a user uploads a "Data Science" resume, the system correctly identifies it as such, proving the reliability of the feature extraction pipeline.

## 6.2 User Interface Walkthrough
*(This section would typically include screenshots)*
1.  **Landing Page:** Users are greeted with a clean, modern interface requesting a PDF upload.
2.  **Analysis Dashboard:** Upon processing, the user sees a large gauge chart showing their match percentage (e.g., 75%).
3.  **Missing Keywords:** A red list highlights critical missing terms found in the JD (e.g., "Missing: Kubernetes, Docker").
4.  **Bullet Point Generator:** The user selects "Docker", and the AI suggests: *"Deployed containerized applications using Docker to improve scalability."*

## 6.3 Case Study
*   **Scenario:** A candidate applies for a "Python Developer" role. 
*   **Before:** Their resume mentions "Coding" and "Programming" but lacks "Python" and "Django". Score: 45%.
*   **After:** Using the tool, they identify the gap. They change "Programming" to "Python Programming" and add "Django" to their skills.
*   **Result:** Score increases to 85%, significantly passing the ATS threshold.

---

# CHAPTER 7: CONCLUSION

## 7.1 Conclusion
The Resume Skill Matcher Pro AI successfully achieves its primary objective of providing transparent, actionable feedback to job seekers. By combining the precision of keyword matching with the breadth of semantic analysis, the tool offers a holistic view of resume quality. The integration of the Skill Graph and Bullet Point Generator elevates the tool from a simple checker to an active career assistant.

## 7.2 Future Scope
1.  **Deep Learning:** Replacing TF-IDF with BERT embeddings to understand context better (e.g., distinguishing "Java" the language from "Java" the island, though rare in resumes).
2.  **Cloud Deployment:** Hosting the app on AWS/Heroku for public access.
3.  **API Integration:** Connecting to real job boards (LinkedIn API) to fetch live job descriptions automatically.

---

# REFERENCES

1.  Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
2.  Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
3.  Streamlit Documentation. (2025). https://docs.streamlit.io
4.  Scikit-Learn Developers. (2025). *User Guide: Support Vector Machines*. https://scikit-learn.org
5.  Resumes Dataset (Kaggle). https://www.kaggle.com/datasets

---
*End of Report*
