import streamlit as st
import pypdf
import plotly.graph_objects as go
import os
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Set up page config
st.set_page_config(
    page_title="SmartResumeAI ✨ | The Premium Resume Optimizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Fidelity CSS styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Main application layout and color palette */
.stApp {
    background-color: #0b0f19;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #f8fafc;
}

/* Custom styled scrollbars */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0f172a;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

/* Sidebar styling overrides */
section[data-testid="stSidebar"] {
    background-color: #090d16 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

section[data-testid="stSidebar"] .stMarkdown h1, 
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f8fafc !important;
}

/* Modern Glassmorphic Container Cards */
.glass-card {
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    margin-bottom: 20px;
}

.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(6, 182, 212, 0.25);
    box-shadow: 0 12px 40px 0 rgba(6, 182, 212, 0.08);
}

/* Glowing text styling */
.glow-text-cyan {
    color: #06b6d4;
    text-shadow: 0 0 15px rgba(6, 182, 212, 0.3);
}

.glow-text-violet {
    color: #a78bfa;
    text-shadow: 0 0 15px rgba(139, 92, 246, 0.3);
}

/* Premium Navigation Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: rgba(15, 23, 42, 0.5);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 500;
    border: none;
    padding: 10px 24px;
    transition: all 0.3s ease;
    font-size: 0.95rem;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc;
    background-color: rgba(255, 255, 255, 0.02);
}

.stTabs [aria-selected="true"] {
    background-color: rgba(6, 182, 212, 0.08) !important;
    color: #06b6d4 !important;
    font-weight: 600 !important;
    border: 1px solid rgba(6, 182, 212, 0.2) !important;
}

/* Matching/Missing Badges */
.tag-matched {
    background-color: rgba(16, 185, 129, 0.09);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.2);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 500;
    display: inline-block;
    margin: 4px;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.05);
}

.tag-missing {
    background-color: rgba(244, 63, 94, 0.09);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.2);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 500;
    display: inline-block;
    margin: 4px;
    box-shadow: 0 2px 8px rgba(244, 63, 94, 0.05);
}

/* Styled text area and uploads */
.stTextArea textarea {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.15) !important;
}

/* Metric card specific styling */
.metric-container {
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
}

.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 5px 0 0 0;
}

/* Customize generic buttons */
div.stButton > button {
    background: linear-gradient(135deg, #06b6d4 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-weight: 600 !important;
    letter-spacing: 0.025em;
    font-size: 0.95rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 18px rgba(6, 182, 212, 0.2) !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.35) !important;
    filter: brightness(1.08);
}

div.stButton > button:active {
    transform: translateY(0) !important;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)

def extract_text_from_pdf(uploaded_file):
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text_content = page.extract_text()
            if text_content:
                text += text_content + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Gemini logic
def analyze_resume(resume_text, job_desc, api_key, model_name="gemini-2.5-flash"):
    client = get_gemini_client(api_key)
    prompt = f"""
    You are an expert Applicant Tracking System (ATS) recruiter and resume optimization expert.
    Thoroughly analyze the following resume against the provided job description.
    
    Resume Text:
    \"\"\"{resume_text}\"\"\"
    
    Job Description:
    \"\"\"{job_desc}\"\"\"
    
    You MUST provide your analysis in a strictly formatted JSON object with the following schema. Make sure the output contains ONLY the JSON block. Do not wrap in extra commentary.
    {{
      "match_score": <int between 0 and 100 representing semantic compatibility>,
      "ats_check": {{
        "verdict": "<Excellent / Good / Moderate / Poor based on match_score>",
        "explanation": "<1-2 sentence explanation of the ATS compatibility, mentioning formatting or keyword gaps>"
      }},
      "matching_keywords": ["keyword1", "keyword2", ...],
      "missing_keywords": ["keyword1", "keyword2", ...],
      "hard_skills_match": [
        {{"skill": "Skill Name", "status": "Matched" or "Missing"}}
      ],
      "soft_skills_match": [
        {{"skill": "Skill Name", "status": "Matched" or "Missing"}}
      ],
      "strengths": ["Highlight strength 1", "Highlight strength 2", ...],
      "improvements": ["Actionable improvement 1", "Actionable improvement 2", ...],
      "tailored_bullet_points": [
        {{
          "original": "Provide a summary of a major task or bullet point in their resume",
          "optimized": "Provide a high-impact optimized bullet point utilizing the STAR method (quantified achievements, active verbs, relevant keywords)",
          "rationale": "Briefly explain why this optimized version works better for ATS systems."
        }}
      ]
    }}
    """
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
    )
    return json.loads(response.text)

def optimize_bullet_point(bullet_point, job_desc, api_key, model_name="gemini-2.5-flash"):
    client = get_gemini_client(api_key)
    prompt = f"""
    You are a professional resume writer. Optimize the following single bullet point using the STAR (Situation, Task, Action, Result) method to make it highly aligned with the job description.
    Focus on adding active verbs, matching keywords, and quantifying the impact.
    
    Original Bullet Point:
    \"\"\"{bullet_point}\"\"\"
    
    Target Job Description:
    \"\"\"{job_desc}\"\"\"
    
    Provide the output in a strictly formatted JSON object with this schema:
    {{
      "optimized_versions": [
        {{"version": "Version 1 (Action-Focused)", "text": "high-impact text"}},
         {{"version": "Version 2 (Result & Metric-Focused)", "text": "high-impact text with metric placeholders if necessary"}},
         {{"version": "Version 3 (Keyword-Rich)", "text": "high-impact text loaded with JD keywords"}}
      ],
      "tips": [
        "Actionable tip on how to talk about this during an interview",
        "Actionable tip on what key metrics to gather to verify this bullet"
      ]
    }}
    """
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7,
        )
    )
    return json.loads(response.text)

def generate_cover_letter(resume_text, job_desc, api_key, model_name="gemini-2.5-flash"):
    client = get_gemini_client(api_key)
    prompt = f"""
    You are an expert career coach and corporate copywriter. Write a personalized, persuasive cover letter based on the applicant's resume and target job description.
    Ensure the cover letter is structured professionally, starts with a strong hook, weaves in key achievements that directly map to the job requirements, and closes with a proactive call to action.
    
    Resume Text:
    \"\"\"{resume_text}\"\"\"
    
    Job Description:
    \"\"\"{job_desc}\"\"\"
    
    Return the response in a JSON object with this schema:
    {{
      "subject": "Cover Letter Subject Line",
      "body": "Full text of the cover letter, complete with standard spacing, paragraphs, and placeholders like [Your Name], [Contact Info]."
    }}
    """
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.5,
        )
    )
    return json.loads(response.text)

# Custom FPDF Class for Report Generation
class PDFReport(FPDF):
    def __init__(self, data, resume_name):
        super().__init__()
        self.data = data
        self.resume_name = resume_name
        
    def header(self):
        # Draw background color
        self.set_fill_color(11, 15, 25) # Slate dark base
        self.rect(0, 0, 210, 297, "F")
        
        # Draw neon cyan banner
        self.set_fill_color(6, 182, 212)
        self.rect(0, 0, 210, 12, "F")
        
        # Header title
        self.set_y(15)
        self.set_font("Arial", "B", 10)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, "SMART RESUME AI  |  ATS MATCH REPORT", ln=True, align="R")
        self.ln(2)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}  |  Confidential Analysis", align="C")

def build_pdf_report(data, resume_name):
    # FPDF using standard system fonts
    pdf = PDFReport(data, resume_name)
    pdf.add_page()
    
    # Title Section
    pdf.set_y(30)
    pdf.set_font("Arial", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "RESUME OPTIMIZATION ANALYSIS", ln=True, align="L")
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(0, 6, f"Resume: {resume_name}", ln=True, align="L")
    pdf.ln(8)
    
    # Divider
    pdf.set_draw_color(30, 41, 59)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)
    
    # Score Widget
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(10, pdf.get_y(), 190, 32, "F")
    
    # Write inside widget
    current_y = pdf.get_y()
    pdf.set_y(current_y + 4)
    pdf.set_x(15)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 6, "ATS COMPATIBILITY SCORE", ln=True)
    
    pdf.set_x(15)
    pdf.set_font("Arial", "B", 26)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(0, 14, f"{data['match_score']}% Match", ln=True)
    pdf.ln(12)
    
    # Verdict Detail
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, f"Verdict: {data['ats_check']['verdict']}", ln=True)
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(203, 213, 225)
    
    # Clean non-latin characters from explanation
    explanation = data['ats_check']['explanation'].encode('ascii', 'ignore').decode('ascii')
    pdf.multi_cell(0, 6, explanation)
    pdf.ln(8)
    
    # Match vs Missing Skills
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(52, 211, 153) # Emerald green
    pdf.cell(0, 8, "Key Strengths & Matched Skills:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(203, 213, 225)
    
    strengths_text = ""
    for s in data['strengths']:
        clean_s = s.encode('ascii', 'ignore').decode('ascii')
        strengths_text += f"- {clean_s}\n"
    pdf.multi_cell(0, 5, strengths_text.strip())
    pdf.ln(6)
    
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(251, 113, 133) # Rose red
    pdf.cell(0, 8, "Missing Keywords & Key Gaps:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(203, 213, 225)
    
    missing_keywords = ", ".join(data['missing_keywords']).encode('ascii', 'ignore').decode('ascii')
    pdf.multi_cell(0, 5, missing_keywords if missing_keywords else "None detected! Excellent coverage.")
    pdf.ln(8)
    
    # Actionable suggestions - new page
    pdf.add_page()
    pdf.set_y(25)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Actionable Resume Tailoring Suggestions", ln=True)
    pdf.ln(4)
    
    for idx, bp in enumerate(data['tailored_bullet_points'][:3]):
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(167, 139, 250) # Violet title
        pdf.cell(0, 6, f"Suggestion #{idx+1}:", ln=True)
        
        pdf.set_font("Arial", "I", 9)
        pdf.set_text_color(148, 163, 184)
        orig_clean = bp['original'].encode('ascii', 'ignore').decode('ascii')
        pdf.multi_cell(0, 5, f"Original Text: \"{orig_clean}\"")
        pdf.ln(1)
        
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(6, 182, 212) # Cyan text
        opt_clean = bp['optimized'].encode('ascii', 'ignore').decode('ascii')
        pdf.multi_cell(0, 5, f"Optimized STAR Version: \"{opt_clean}\"")
        pdf.ln(1)
        
        pdf.set_font("Arial", "I", 9)
        pdf.set_text_color(203, 213, 225)
        rat_clean = bp['rationale'].encode('ascii', 'ignore').decode('ascii')
        pdf.multi_cell(0, 5, f"Rationale: {rat_clean}")
        pdf.ln(6)
        
    return pdf.output()

# Application Sidebar Configuration
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <h1 style='font-size: 1.8rem; font-weight: 800; margin: 0; color: #f8fafc;'>SmartResume<span style='color: #06b6d4;'>AI</span></h1>
    <p style='color: #64748b; font-size: 0.85rem;'>Modern Resume Engineering Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# API Configuration Panel
st.sidebar.subheader("🔑 API Configuration")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Grab an API key for free from Google AI Studio",
    value=""
)

# Retrieve key from variable fallback
gemini_api_key = api_key_input if api_key_input else os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    st.sidebar.warning("⚠️ No Gemini API Key found. Configure it above or in your local `.env` file to start.")
else:
    st.sidebar.success("⚡ Gemini Client Active!")

# Model Picker
model_choice = st.sidebar.selectbox(
    "🤖 Analytics Engine",
    ["gemini-2.5-flash", "gemini-2.5-pro"],
    help="Flash is incredibly fast and optimized for dashboards. Pro is suited for rigorous phrasing calculations."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px; font-size: 0.8rem; color: #94a3b8;'>
    <strong>Stack Details:</strong><br>
    - Python 3.11.0<br>
    - Streamlit Front-End<br>
    - Google Gemini AI Client<br>
    - Plotly Visual Graphics<br>
    - FPDF2 Report Generation
</div>
""", unsafe_allow_html=True)

# Main Application Frame
st.markdown("""
<div style='margin-bottom: 30px;'>
    <h1 style='font-size: 2.8rem; font-weight: 800; line-height: 1.1; margin: 0;'>Optimize Your <span class='glow-text-cyan'>Resume</span>.</h1>
    <h1 style='font-size: 2.8rem; font-weight: 800; line-height: 1.1; margin: 0;'>Land the <span class='glow-text-violet'>Interview</span>.</h1>
    <p style='color: #94a3b8; font-size: 1.1rem; margin-top: 10px;'>Leverage deep neural parsing to align your qualifications perfectly with any job description.</p>
</div>
""", unsafe_allow_html=True)

# Main Multi-Tab Layout
tab_dashboard, tab_bullet, tab_letter = st.tabs([
    "📊 ATS Analysis Dashboard",
    "🎯 Smart Bullet Optimizer",
    "✉️ Cover Letter Architect"
])

# Initialize session state for caching parsed data
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "resume_filename" not in st.session_state:
    st.session_state.resume_filename = ""
if "raw_resume_text" not in st.session_state:
    st.session_state.raw_resume_text = ""

# ----------------- TAB 1: ATS ANALYSIS DASHBOARD -----------------
with tab_dashboard:
    col_input, col_info = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📥 Profile Inputs")
        
        uploaded_pdf = st.file_uploader(
            "Upload Resume (PDF format)",
            type=["pdf"],
            help="Your PDF will be securely processed locally"
        )
        
        job_description_raw = st.text_area(
            "Target Job Description",
            height=260,
            placeholder="Paste the complete job details, required skills, and qualifications here..."
        )
        
        analyze_button = st.button("🚀 Analyze Compatibility")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if analyze_button:
            if not gemini_api_key:
                st.error("Please configure a Google Gemini API Key in the sidebar or `.env` file first.")
            elif not uploaded_pdf:
                st.warning("Please upload your PDF resume first.")
            elif not job_description_raw.strip():
                st.warning("Please paste a target job description.")
            else:
                with st.spinner("Extracting documents and parsing semantics..."):
                    resume_text = extract_text_from_pdf(uploaded_pdf)
                    
                    if resume_text:
                        st.session_state.raw_resume_text = resume_text
                        st.session_state.resume_filename = uploaded_pdf.name
                        try:
                            # Run Analysis
                            results = analyze_resume(
                                resume_text,
                                job_description_raw,
                                gemini_api_key,
                                model_choice
                            )
                            st.session_state.analysis_results = results
                            st.success("Analysis Complete!")
                        except Exception as e:
                            st.error(f"Failed to generate analysis: {e}")
                            
    with col_info:
        if st.session_state.analysis_results is not None:
            data = st.session_state.analysis_results
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 📈 Match Analysis Summary")
            
            col_chart, col_stats = st.columns([2, 3])
            
            with col_chart:
                score = data["match_score"]
                # Plotly Donut Chart
                fig = go.Figure(go.Pie(
                    values=[score, 100 - score],
                    labels=['Match', 'Gap'],
                    hole=0.78,
                    marker_colors=['#06b6d4', 'rgba(255, 255, 255, 0.05)'],
                    textinfo='none',
                    hoverinfo='none'
                ))
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    width=170,
                    height=170,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.add_annotation(
                    text=f"{score}%",
                    x=0.5, y=0.5,
                    font=dict(size=30, color='#f8fafc', family='Plus Jakarta Sans', weight='bold'),
                    showarrow=False
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
            with col_stats:
                st.markdown(f"""
                <div class='metric-container'>
                    <span style='font-size: 0.85rem; color: #94a3b8; font-weight: 500;'>ATS VERDICT</span>
                    <div class='metric-val glow-text-cyan'>{data['ats_check']['verdict']}</div>
                </div>
                <div style='margin-top: 15px; font-size: 0.9rem; color: #cbd5e1; line-height: 1.5;'>
                    {data['ats_check']['explanation']}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Matched & Missing Skills
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 🔑 Keyword & Competency Alignment")
            
            tab_hard, tab_soft = st.tabs(["💻 Technical Skills", "👥 Soft Competencies"])
            
            with tab_hard:
                matched_hard = [s["skill"] for s in data["hard_skills_match"] if s["status"] == "Matched"]
                missing_hard = [s["skill"] for s in data["hard_skills_match"] if s["status"] == "Missing"]
                
                st.markdown("##### **Matched Keywords**")
                if matched_hard:
                    for skill in matched_hard:
                        st.markdown(f"<span class='tag-matched'>✓ {skill}</span>", unsafe_allow_html=True)
                else:
                    st.write("No matching technical skills identified yet.")
                    
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                st.markdown("##### **Missing Keywords**")
                if missing_hard:
                    for skill in missing_hard:
                        st.markdown(f"<span class='tag-missing'>✗ {skill}</span>", unsafe_allow_html=True)
                else:
                    st.write("Perfect coverage! No missing technical skills identified.")
                    
            with tab_soft:
                matched_soft = [s["skill"] for s in data["soft_skills_match"] if s["status"] == "Matched"]
                missing_soft = [s["skill"] for s in data["soft_skills_match"] if s["status"] == "Missing"]
                
                st.markdown("##### **Matched Competencies**")
                if matched_soft:
                    for skill in matched_soft:
                        st.markdown(f"<span class='tag-matched'>✓ {skill}</span>", unsafe_allow_html=True)
                else:
                    st.write("No matching soft competencies identified yet.")
                    
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                st.markdown("##### **Missing Competencies**")
                if missing_soft:
                    for skill in missing_soft:
                        st.markdown(f"<span class='tag-missing'>✗ {skill}</span>", unsafe_allow_html=True)
                else:
                    st.write("Perfect coverage! No missing soft competencies identified.")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            # Empty State
            st.markdown("""
            <div style='background: rgba(30, 41, 59, 0.2); border: 1.5px dashed rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 60px; text-align: center;'>
                <span style='font-size: 3rem;'>📊</span>
                <h3 style='color: #94a3b8; font-weight: 500; margin-top: 15px;'>Analysis Dashboard Idle</h3>
                <p style='color: #64748b; font-size: 0.95rem; max-width: 400px; margin: 10px auto;'>Provide your resume and a target job description in the left-hand column to trigger our semantic match engine.</p>
            </div>
            """, unsafe_allow_html=True)
            
    # Expanded View for bullet points and strengths
    if st.session_state.analysis_results is not None:
        data = st.session_state.analysis_results
        
        col_st, col_imp = st.columns(2)
        with col_st:
            st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("### 🟢 Outstanding Strengths")
            for s in data["strengths"]:
                st.markdown(f"""
                <div style='display: flex; gap: 10px; margin-bottom: 12px;'>
                    <span style='color: #10b981; font-weight: bold;'>✓</span>
                    <span style='color: #e2e8f0; font-size: 0.95rem;'>{s}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_imp:
            st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("### 🟡 Core Areas of Improvement")
            for imp in data["improvements"]:
                st.markdown(f"""
                <div style='display: flex; gap: 10px; margin-bottom: 12px;'>
                    <span style='color: #f59e0b; font-weight: bold;'>!</span>
                    <span style='color: #e2e8f0; font-size: 0.95rem;'>{imp}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Actionable Suggestions
        st.markdown("<div class='glass-card' style='margin-top: 20px;'>", unsafe_allow_html=True)
        st.markdown("### ✏️ High-Impact STAR Rephrasing Suggestions")
        st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Instantly upgrade passive statements on your resume to quantifiable, results-driven bullets tailored to this JD.</p>", unsafe_allow_html=True)
        
        for idx, bp in enumerate(data["tailored_bullet_points"]):
            st.markdown(f"""
            <div style='background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.04); border-radius: 12px; padding: 20px; margin-bottom: 15px;'>
                <h5 style='color: #a78bfa; font-weight: 600; margin: 0 0 10px 0;'>Suggestion {idx+1}</h5>
                
                <div style='margin-bottom: 10px;'>
                    <span style='font-size: 0.8rem; color: #64748b; font-weight: 600; text-transform: uppercase;'>Original Text</span>
                    <p style='color: #94a3b8; font-size: 0.92rem; margin: 2px 0 0 0; font-style: italic;'>"{bp['original']}"</p>
                </div>
                
                <div style='margin-bottom: 10px;'>
                    <span style='font-size: 0.8rem; color: #06b6d4; font-weight: 600; text-transform: uppercase;'>Optimized (STAR Method)</span>
                    <p style='color: #f8fafc; font-size: 0.95rem; margin: 2px 0 0 0; font-weight: 500;'>"{bp['optimized']}"</p>
                </div>
                
                <div>
                    <span style='font-size: 0.8rem; color: #64748b; font-weight: 600; text-transform: uppercase;'>Optimization Rationale</span>
                    <p style='color: #cbd5e1; font-size: 0.9rem; margin: 2px 0 0 0;'>{bp['rationale']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Export PDF button
        try:
            pdf_bytes = build_pdf_report(data, st.session_state.resume_filename)
            st.download_button(
                label="📥 Download Complete Report (PDF)",
                data=pdf_bytes,
                file_name=f"SmartResumeAI_Analysis_{st.session_state.resume_filename.replace('.pdf', '')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.info(f"PDF download compiler preparing... ({e})")
            
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 2: SMART BULLET OPTIMIZER -----------------
with tab_bullet:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Single-Bullet Tailoring Assistant")
    st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Paste a single bullet point from your resume and get optimized, high-impact variations mapped directly to your target job description.</p>", unsafe_allow_html=True)
    
    col_opt_input, col_opt_out = st.columns([1, 1], gap="large")
    
    with col_opt_input:
        bullet_input = st.text_area(
            "Your Original Bullet Point",
            height=130,
            placeholder="e.g., I worked on the front end of the website and fixed bugs."
        )
        
        target_jd_snippet = st.text_area(
            "Target Job Details (Auto-filled if provided in Dashboard tab)",
            height=130,
            value=job_description_raw if job_description_raw else "",
            placeholder="Paste target role requirements here..."
        )
        
        optimize_bullet_button = st.button("✨ Rephrase with STAR")
        
    with col_opt_out:
        if optimize_bullet_button:
            if not gemini_api_key:
                st.error("Please configure a Google Gemini API Key in the sidebar or `.env` file first.")
            elif not bullet_input.strip():
                st.warning("Please paste a bullet point first.")
            elif not target_jd_snippet.strip():
                st.warning("Please paste a job description snippet first.")
            else:
                with st.spinner("Refactoring language mechanics..."):
                    try:
                        bullet_data = optimize_bullet_point(
                            bullet_input,
                            target_jd_snippet,
                            gemini_api_key,
                            model_choice
                        )
                        
                        for ver in bullet_data["optimized_versions"]:
                            st.markdown(f"""
                            <div style='background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(6, 182, 212, 0.15); border-radius: 12px; padding: 16px; margin-bottom: 12px;'>
                                <span style='font-size: 0.75rem; color: #06b6d4; font-weight: 700; text-transform: uppercase;'>{ver['version']}</span>
                                <p style='color: #f8fafc; font-size: 0.95rem; margin: 4px 0 0 0; font-weight: 500;'>"{ver['text']}"</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        st.markdown("##### 💡 Phrasing Tips")
                        for tip in bullet_data["tips"]:
                            st.markdown(f"- {tip}")
                            
                    except Exception as e:
                        st.error(f"Failed to optimize bullet: {e}")
        else:
            st.markdown("""
            <div style='background: rgba(30, 41, 59, 0.2); border: 1.5px dashed rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 50px; text-align: center; height: 100%;'>
                <span style='font-size: 2.2rem;'>🎯</span>
                <h4 style='color: #94a3b8; font-weight: 500; margin-top: 10px;'>Optimizer Idle</h4>
                <p style='color: #64748b; font-size: 0.85rem;'>Input your raw experience line on the left to trigger the refactoring process.</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 3: COVER LETTER ARCHITECT -----------------
with tab_letter:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### ✉️ Cover Letter Architect")
    st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Automatically orchestrate a professional, targeted cover letter integrating your resume accomplishments and matching the requirements of the job description.</p>", unsafe_allow_html=True)
    
    col_let_in, col_let_out = st.columns([1, 1], gap="large")
    
    with col_let_in:
        let_resume_text = st.text_area(
            "Resume Text (Auto-filled if uploaded in Dashboard tab)",
            height=200,
            value=st.session_state.raw_resume_text if st.session_state.raw_resume_text else "",
            placeholder="Paste your plain text resume details if not uploading a PDF..."
        )
        
        let_jd_text = st.text_area(
            "Job Details (Auto-filled if provided in Dashboard tab)",
            key="let_jd",
            height=200,
            value=job_description_raw if job_description_raw else "",
            placeholder="Paste complete job details..."
        )
        
        generate_letter_button = st.button("✉️ Synthesize Cover Letter")
        
    with col_let_out:
        if generate_letter_button:
            if not gemini_api_key:
                st.error("Please configure a Google Gemini API Key in the sidebar or `.env` file first.")
            elif not let_resume_text.strip():
                st.warning("Please upload a resume or paste details on the left first.")
            elif not let_jd_text.strip():
                st.warning("Please paste a job description on the left first.")
            else:
                with st.spinner("Weaving professional narrative..."):
                    try:
                        letter_data = generate_cover_letter(
                            let_resume_text,
                            let_jd_text,
                            gemini_api_key,
                            model_choice
                        )
                        
                        body_html = letter_data['body'].replace('\n', '<br>')
                        st.markdown(f"""
                        <div style='background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 24px; font-family: monospace; font-size: 0.9rem; line-height: 1.6; color: #f1f5f9;'>
                            <strong>Subject:</strong> {letter_data['subject']}<br><br>
                            {body_html}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Direct copy assistant
                        st.download_button(
                            label="📥 Download Draft (TXT)",
                            data=f"Subject: {letter_data['subject']}\n\n{letter_data['body']}",
                            file_name="Cover_Letter_Draft.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Failed to synthesize cover letter: {e}")
        else:
            st.markdown("""
            <div style='background: rgba(30, 41, 59, 0.2); border: 1.5px dashed rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 50px; text-align: center; height: 100%;'>
                <span style='font-size: 2.2rem;'>✉️</span>
                <h4 style='color: #94a3b8; font-weight: 500; margin-top: 10px;'>Architect Idle</h4>
                <p style='color: #64748b; font-size: 0.85rem;'>Configure the source materials on the left and click synthesize to compile your cover letter draft.</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)
