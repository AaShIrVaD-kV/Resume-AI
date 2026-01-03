import streamlit as st
import utils
import ats_model
import visualizations
import os
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Resume Matcher Pro AI",
    page_icon="🤖",
    layout="wide"
)

# Load Custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    st.warning("style.css not found. Using default styles.")

# --- MAIN APP UI ---

# Header Section
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🤖 AI-Powered Resume Matcher & Career Assistant")
st.markdown("### Advanced Analysis: ATS Score, Skill Graph, & Recommendations")
st.markdown('</div>', unsafe_allow_html=True)

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 1. Upload Resume")
    input_type = st.radio("Format", ["PDF", "Word", "Text"], horizontal=True)
    
    uploaded_file = None
    text_input = ""
    
    if input_type == "PDF":
        uploaded_file = st.file_uploader("Drop PDF", type="pdf")
    elif input_type == "Word":
        uploaded_file = st.file_uploader("Drop DOCX", type="docx")
    else:
        text_input = st.text_area("Paste Resume Text", height=200)

with col2:
    st.markdown("#### 2. Job Description")
    job_description = st.text_area("Paste the job description here", height=200)

# Action Button
if st.button("Analyze Resume 🚀"):
    has_input = (input_type == "Text" and text_input) or (uploaded_file is not None)
    
    if has_input and job_description:
        with st.spinner("Running AI Analysis..."):
            # Extract Text
            resume_text = ""
            if input_type == "PDF":
                resume_text = utils.extract_text_from_pdf(uploaded_file)
            elif input_type == "Word":
                resume_text = utils.extract_text_from_docx(uploaded_file)
            else:
                resume_text = text_input
            
            if resume_text:
                # 1. Extract Skills First (Needed for score)
                resume_skills = utils.extract_skills(resume_text)
                jd_skills = utils.extract_skills(job_description)
                missing_keywords = list(jd_skills - resume_skills)
                total_jd_skills = len(jd_skills)
                
                # 2. Hybrid Match Score
                match_percentage = utils.calculate_similarity(
                    resume_text, 
                    job_description,
                    total_keywords=total_jd_skills,
                    missing_keywords=len(missing_keywords)
                )
                
                entities = utils.extract_entities(resume_text)
                
                # 3. ATS Score
                ats_score = ats_model.predict_ats_score(resume_text, len(missing_keywords), match_percentage)
                
                # 4. Readability
                readability = utils.calculate_readability(resume_text)
                
                # 5. Recommendations
                recommended_skills = utils.recommend_skills(resume_skills)
                
                # 6. Fake Skill Check
                suspicious_skills = ats_model.check_fake_skills(resume_text, resume_skills)
                
                # 7. Category Prediction
                category_probs = ats_model.predict_category(resume_text)

                # --- DISPLAY RESULTS ---
                
                # Tab Layout
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Match Score", "🎯 ATS & Insights", "🕸️ Skill Graph", "💡 Recommendations", "📝 Resume Builder"])
                
                with tab1:
                    st.subheader("Match Analysis")
                with tab1:
                    st.subheader("Match Analysis")
                    score_col, status_col = st.columns([1, 2])
                    
                    # Get Level Info
                    level_label, level_color, level_msg = utils.get_match_level(match_percentage)
                    
                    with score_col:
                        st.metric(label="Match Score", value=f"{match_percentage}%")
                        st.progress(int(match_percentage))
                    with status_col:
                        if level_color == "success":
                            st.success(f"**{level_label}**\n\n{level_msg}")
                        elif level_color == "warning":
                            st.warning(f"**{level_label}**\n\n{level_msg}")
                        else:
                            st.error(f"**{level_label}**\n\n{level_msg}")
                            
                    st.markdown("#### Keyword Matching")
                    if total_jd_skills > 0:
                        if missing_keywords:
                            st.error(f"❌ Missing {len(missing_keywords)} important keywords:")
                            st.write(", ".join(missing_keywords[:15]))
                        else:
                            st.success("✅ All key terms found!")
                    else:
                        st.warning("⚠️ No keywords extracted from Job Description. Try evaluating a longer description.")
                        st.caption("Using only semantic similarity for scoring.")

                with tab2:
                    st.subheader("ATS & Resume Quality")
                    
                    # ATS Gauge
                    st.plotly_chart(visualizations.create_ats_gauge(ats_score), use_container_width=True)
                    
                    # Category Prediction
                    if category_probs and "Error" not in category_probs:
                        st.markdown("#### Predicted Category")
                        # Create DataFrame for Plotly
                        import pandas as pd
                        import plotly.express as px
                        
                        df_prob = pd.DataFrame(list(category_probs.items()), columns=['Category', 'Probability'])
                        fig_cat = px.pie(
                            df_prob, 
                            names='Category', 
                            values='Probability', 
                            title='Resume Category Distribution',
                            hole=0.4
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)

                        # --- NEW: Display Top Keywords for Predicted Category ---
                        try:
                            import json
                            with open('category_keywords.json', 'r') as f:
                                all_category_keywords = json.load(f)
                            
                            # Get top category
                            top_category = df_prob.iloc[0]['Category']
                            if top_category in all_category_keywords:
                                st.markdown(f"#### 🔑 Top Keywords for {top_category}")
                                st.caption(f"These are the most important terms found in {top_category} resumes. Consider adding them if relevant.")
                                
                                keywords = all_category_keywords[top_category]
                                # Create a badge-like display
                                kw_html = '<div style="display: flex; flex-wrap: wrap; gap: 5px;">'
                                for kw in keywords:
                                    kw_html += f'<span style="background-color: #2b2b2b; color: #00d4ff; padding: 4px 8px; border-radius: 4px; border: 1px solid #00d4ff; font-size: 0.8em;">{kw}</span>'
                                kw_html += '</div>'
                                st.markdown(kw_html, unsafe_allow_html=True)
                                st.write("") # Spacer
                        except Exception as e:
                            # non-critical, just skip if file missing
                            pass
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.info(f"**Readability Score:** {readability} (Flesch Ease)")
                        st.caption("60-70 is standard. Higher is easier to read.")
                    
                    with col_b:
                        if suspicious_skills:
                            st.error(f"⚠️ Potential Unsupported Skills: {', '.join(suspicious_skills)}")
                            st.caption("These skills appear without context (projects/experience).")
                        else:
                            st.success("✅ Skill Consistency Check Passed")
                            
                    st.markdown("#### Extracted Entities (NER)")
                    
                    # Custom Entity Visualization
                    for label, items in entities.items():
                        if items:
                            st.markdown(f"**{label}**")
                            # Create HTML for badges
                            html = '<div class="entity-container">'
                            for item in items:
                                # Determine class based on label (defaulting to gray if unknown)
                                tag_class = "entity-tag"
                                if label == "ORG": tag_class += " entity-org"
                                elif label == "PERSON": tag_class += " entity-person"
                                elif label == "GPE": tag_class += " entity-gpe"
                                elif label == "DATE": tag_class += " entity-date"
                                
                                html += f'<span class="{tag_class}">{item}</span>'
                            html += '</div>'
                            st.markdown(html, unsafe_allow_html=True)

                with tab3:
                    st.subheader("Experience-Skill Network")
                    st.markdown("Visualizing connections between your profile and skills.")
                    st.plotly_chart(visualizations.create_skill_network(resume_skills), use_container_width=True)

                with tab4:
                    st.subheader("AI Recommendations (Data-Driven)")
                    st.markdown("Based on analyzing thousands of resumes, these skills are commonly seen with your profile:")
                    
                    if recommended_skills:
                        for skill in recommended_skills:
                            st.markdown(f"- **{skill.title()}**")
                        else:
                            st.info("No specific recommendations found. Try adding more skills to your resume!")
                            
                with tab5:
                    st.subheader("Resume Builder Helper")
                    st.markdown("Targeting missing keywords? Let AI help you write bullet points.")
                    
                    if missing_keywords:
                        selected_skill = st.selectbox("Select a missing skill to target:", list(missing_keywords)[:20])
                        
                        if st.button("Generate Bullet Points"):
                            suggestions = utils.generate_bullet_points([selected_skill])
                            st.markdown(f"**Suggestions for {selected_skill.title()}:**")
                            for point in suggestions[selected_skill]:
                                st.code(point, language="text")
                            st.caption("Copy these to your resume!")
                    else:
                        st.success("You have all the required keywords! Good luck!")

            else:
                st.error("Could not extract text from the PDF.")
    else:
        st.warning("Please upload a resume and enter a job description.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Spacy & Plotly")
