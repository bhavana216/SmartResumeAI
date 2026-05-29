---
title: SmartResumeAI
emoji: ✨
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
license: mit
---

# SmartResumeAI ✨ — Upgraded Premium Edition

🚀 **State-of-the-Art AI Resume Analyzer & Tailoring Suite** powered by **Google Gemini API** and an interactive, high-fidelity dark-mode dashboard.

SmartResumeAI analyzes your resume's semantic fit against any target job description using deep neural language understanding. It computes an ATS compatibility score, reveals critical keyword gaps, recommends bullet point rephrasings using the STAR method, synthesizes cover letters, and compiles a comprehensive PDF analysis report.

---

## 📄 Key Features

1. **📊 ATS Compatibility Dashboard**
   - **Semantic Scoring**: Matches your resume to a job description using Gemini models rather than raw word overlaps.
   - **Competency Mapping**: Compares hard and soft skills side-by-side to highlight matched and missing competencies.
   - **SWOT Profiler**: Lists outstanding strengths and core areas of improvement on your resume.
   - **PDF Report Exporter**: Downloads a beautifully formatted professional analysis report.
   
2. **🎯 Smart Bullet Point Optimizer**
   - Rephrases passive bullet points into results-oriented, STAR (Situation, Task, Action, Result) achievements packed with high-priority job keywords.
   
3. **✉️ Cover Letter Architect**
   - Synthesizes personalized, high-conversion cover letter drafts integrating your key experiences and matching the target role perfectly.

---

## 🛠️ Modern Tech Stack

- **Frontend**: Streamlit (with custom dark-mode glassmorphic CSS overrides)
- **Neural Engine**: Google Gemini API via official `google-genai` SDK
- **Data Visualizations**: Plotly Graph Objects
- **Document Management**: `pypdf` (text extraction) & `fpdf2` (custom PDF compilation)
- **Environment**: Python 3.11.0, `python-dotenv`

---

## 🚀 Getting Started & Local Installation

Follow these simple steps to run the premium dashboard locally:

### 1. Clone & Set Up Workspace
```bash
# Clone the repository (if not already cloned)
git clone https://github.com/bhavana216/SmartResumeAI.git
cd SmartResumeAI
```

### 2. Configure Virtual Environment & Install Dependencies
```bash
# Create python virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install the upgraded dependencies
pip install -r requirements.txt
```

### 3. Set Up API Key
1. Create a `.env` file in the root directory (based on `.env.example`).
2. Add your **Google Gemini API Key** (Get one for free at [Google AI Studio](https://aistudio.google.com/)):
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```
*Note: You can also enter the API key directly in the sidebar of the web interface at runtime.*

### 4. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing with Sample Materials
We have generated a mock candidate PDF resume (`Jane_Doe_Resume.pdf`) in the project root folder.
Upload `Jane_Doe_Resume.pdf` and paste any Software Engineer job description to instantly test the ATS grading, Plotly donut charts, bullet point rephraser, and PDF report downloads!

---

## 👩‍💻 Credits & Development
- **Original Developer**: Bhavana Battula
- **Premium Upgrade**: AI Agent Coding Partner
