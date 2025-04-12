import streamlit as st
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re

nltk.download('punkt')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(uploaded_file):
    return extract_text(uploaded_file)

def calculate_similarity(resume_text, job_description):
    texts = [resume_text, job_description]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_score[0][0] * 100

st.title("SmartResumeAI")
st.subheader("Optimize your resume for the job you want")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the Job Description")

if st.button("Analyze"):
    if uploaded_file is not None and job_description:
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_text_clean = clean_text(resume_text)
        job_description_clean = clean_text(job_description)

        score = calculate_similarity(resume_text_clean, job_description_clean)
        st.success(f"Resume Match Score: {score:.2f}%")

        if score >= 80:
            st.info("Great match! Minor tweaks can further optimize your resume.")
        elif score >= 50:
            st.warning("Moderate match. You should improve your resume by adding relevant skills and keywords.")
        else:
            st.error("Low match. Customize your resume more towards the job description.")

    else:
        st.error("Please upload a resume and paste a job description.")
