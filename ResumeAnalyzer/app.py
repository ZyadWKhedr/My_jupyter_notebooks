import streamlit as st
import spacy
import spacy.cli
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import PyPDF2
import re
from collections import defaultdict
import pandas as pd
from io import StringIO

# Load the English language model from spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm"))

# Universal skill categories (expandable)
SKILL_CATEGORIES = {
    'Programming Languages': ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'kotlin', 'swift', 'dart'],
    'Web Development': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'node', 'express', 'spring'],
    'Mobile Development': ['flutter', 'android', 'ios', 'react native', 'xamarin'],
    'Data Science/AI': ['pandas', 'numpy', 'tensorflow', 'pytorch', 'ml', 'ai', 'nlp', 'computer vision', 'scikit-learn'],
    'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'firebase', 'oracle'],
    'DevOps/Cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'terraform', 'ansible'],
    'Soft Skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'time management'],
    'Tools': ['git', 'jenkins', 'jira', 'figma', 'tableau', 'power bi']
}

def preprocess_text(text):
    """Preprocess text by removing stopwords, punctuation, and lemmatizing."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def extract_text_from_pdf(file):
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX file."""
    doc = Document(file)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def extract_skills(text):
    """Extract skills from text using dynamic pattern matching."""
    text_lower = text.lower()
    found_skills = defaultdict(list)
    
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found_skills[category].append(skill)
    
    # Extract additional skills not in predefined list
    doc = nlp(text_lower)
    nouns_chunks = [chunk.text for chunk in doc.noun_chunks]
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 3:
            if token.text not in str(SKILL_CATEGORIES.values()):
                found_skills['Other Skills'].append(token.text)
    
    return found_skills

def calculate_similarity(text1, text2):
    """Calculate cosine similarity between two texts."""
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1_processed, text2_processed])
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def generate_skill_chart(resume_skills, job_skills):
    """Generate a skill comparison chart."""
    chart_data = []
    all_categories = set(resume_skills.keys()).union(set(job_skills.keys()))
    
    for category in all_categories:
        resume_skill_count = len(resume_skills.get(category, []))
        job_skill_count = len(job_skills.get(category, []))
        
        if job_skill_count > 0:
            match_percentage = min(resume_skill_count / job_skill_count, 1) * 100
        else:
            match_percentage = 0
            
        chart_data.append({
            'Category': category,
            'Your Skills': resume_skill_count,
            'Required Skills': job_skill_count,
            'Match %': match_percentage
        })
    
    return pd.DataFrame(chart_data).sort_values('Match %', ascending=False)

def main():
    st.set_page_config(page_title="Universal Resume Analyzer", layout="wide")
    st.title("🔍 Universal Resume & Job Description Analyzer")
    st.write("Works for any job type - Software, Marketing, Finance, Healthcare, etc.")
    
    # Input columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Your Resume")
        resume_option = st.radio("Resume Input:", ("Paste Text", "Upload File"), horizontal=True)
        
        resume_text = ""
        if resume_option == "Paste Text":
            resume_text = st.text_area("Paste your resume content:", height=300)
        else:
            uploaded_file = st.file_uploader("Upload resume (PDF/DOCX)", type=["pdf", "docx"])
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = extract_text_from_docx(uploaded_file)
                st.text_area("Extracted Resume:", resume_text, height=300)
    
    with col2:
        st.subheader("📋 Job Description")
        job_desc_text = st.text_area("Paste the job description:", height=300)
    
    if st.button("🔎 Analyze Match", use_container_width=True):
        if resume_text and job_desc_text:
            # Calculate similarity
            similarity_score = calculate_similarity(resume_text, job_desc_text)
            
            # Extract skills
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_desc_text)
            
            # Generate skill chart
            skill_df = generate_skill_chart(resume_skills, job_skills)
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Overall Match", f"{similarity_score*100:.1f}%")
            with m2:
                strength = "Strong" if similarity_score > 0.7 else "Moderate" if similarity_score > 0.4 else "Weak"
                st.metric("Resume Strength", strength)
            with m3:
                missing_categories = len([cat for cat in job_skills if cat not in resume_skills])
                st.metric("Missing Categories", missing_categories)
            
            # Skill comparison chart
            st.subheader("🛠 Skills Comparison")
            st.dataframe(
                skill_df.style.background_gradient(cmap='Blues', subset=['Match %']),
                use_container_width=True
            )
            
            # Detailed analysis
            st.subheader("🔍 Detailed Breakdown")
            
            for index, row in skill_df.iterrows():
                category = row['Category']
                
                with st.expander(f"{category} ({row['Match %']:.1f}% match)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Job Requires:**")
                        if category in job_skills:
                            for skill in job_skills[category]:
                                if skill in resume_skills.get(category, []):
                                    st.success(f"✓ {skill}")
                                else:
                                    st.error(f"✗ {skill}")
                        else:
                            st.info("No specific skills required")
                    
                    with col2:
                        st.write("**You Have:**")
                        if category in resume_skills:
                            for skill in resume_skills[category]:
                                if skill in job_skills.get(category, []):
                                    st.success(f"✓ {skill} (Match)")
                                else:
                                    st.info(f"{skill} (Bonus)")
                        else:
                            st.warning("No skills in this category")
            
            # Recommendations
            st.subheader("💡 Improvement Suggestions")
            for category in job_skills:
                if category not in resume_skills:
                    st.error(f"**Add {category} skills**: The job requires {len(job_skills[category])} skills in this category")
                elif len(resume_skills[category]) < len(job_skills[category]):
                    st.warning(f"**Boost {category}**: You have {len(resume_skills[category])}/{len(job_skills[category])} required skills")
            
            # Generate report
            report = StringIO()
            report.write("UNIVERSAL RESUME ANALYSIS REPORT\n")
            report.write("===============================\n\n")
            report.write(f"Overall Match Score: {similarity_score*100:.2f}%\n\n")
            
            report.write("SKILLS ANALYSIS\n")
            report.write("---------------\n")
            for index, row in skill_df.iterrows():
                report.write(f"{row['Category']}: {row['Match %']:.1f}% match\n")
            
            st.download_button(
                label="📥 Download Full Report",
                data=report.getvalue(),
                file_name="resume_analysis_report.txt",
                mime="text/plain"
            )
            
        else:
            st.error("Please provide both resume and job description content")

if __name__ == "__main__":
    main()
