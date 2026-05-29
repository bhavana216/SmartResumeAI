from fpdf import FPDF

def build_test_resume():
    pdf = FPDF()
    pdf.add_page()
    
    # Title / Name
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 10, "JANE DOE", ln=True, align="C")
    
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 5, "jane.doe@email.com | (123) 456-7890 | github.com/janedoe | linkedin.com/in/janedoe", ln=True, align="C")
    pdf.ln(10)
    
    # Professional Summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Professional Summary", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, "Highly motivated Software Engineer with 3+ years of experience designing and developing web applications using Python, Django, and JavaScript. Proficient in database management, REST API design, and cloud deployments. Strong problem-solving skills and a team player.")
    pdf.ln(5)
    
    # Skills
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Technical Skills", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, "Programming: Python, JavaScript, SQL, HTML, CSS\nFrameworks: Django, Flask, React, Bootstrap\nDatabases: PostgreSQL, MySQL, Redis\nTools & Platforms: Git, Docker, AWS, Heroku")
    pdf.ln(5)
    
    # Experience
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Professional Experience", ln=True)
    
    # Job 1
    pdf.set_font("Arial", "B", 11)
    pdf.cell(100, 6, "Software Developer - TechCorp Solutions")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 6, "June 2024 - Present", ln=True, align="R")
    
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, "- Led a team of 3 developers to build a custom customer dashboard in Python and Django, improving query response time by 25%.\n- Integrated multiple third-party REST APIs for payments and notifications.\n- Fixed bugs and refactored legacy database models in PostgreSQL.")
    pdf.ln(3)
    
    # Job 2
    pdf.set_font("Arial", "B", 11)
    pdf.cell(100, 6, "Junior Engineer - Innovate Web Inc.")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 6, "Jan 2023 - May 2024", ln=True, align="R")
    
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, "- Maintained web interface logic using Javascript and Flask.\n- Wrote unit tests in Python, increasing coverage from 60% to 85%.\n- Collaborated with product designers to implement responsive UI designs.")
    pdf.ln(5)
    
    # Education
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Education", ln=True)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(100, 6, "Bachelor of Science in Computer Science")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, "GPA: 3.8/4.0", ln=True, align="R")
    pdf.cell(0, 6, "State University - Graduated May 2023", ln=True)
    
    # Output PDF
    pdf.output("Jane_Doe_Resume.pdf")
    print("Jane_Doe_Resume.pdf generated successfully!")

if __name__ == "__main__":
    build_test_resume()
