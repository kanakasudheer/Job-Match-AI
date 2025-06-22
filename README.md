# Job Match AI

Job Match AI is an intelligent job matching platform built with Streamlit, spaCy, and Sentence Transformers. It connects candidates with the perfect jobs using AI-powered matching, resume analysis, and recruiter tools.

## Features

- **Candidate Portal**: Upload your resume, extract skills, and get AI-powered resume suggestions.
- **Job Recommendations**: Personalized job recommendations based on your profile and skills.
- **Recruiter Portal**: Post new jobs, upload job descriptions, and get top candidate recommendations.
- **AI Matching Engine**: Advanced matching using NLP, skill extraction, and semantic similarity.
- **Skill Gap Analysis**: Identify missing skills for candidates to improve their match.
- **Interview Preparation Kit**: Auto-generated technical and behavioral interview questions for each job.
- **Application Tracker**: Track job applications and statuses.
- **Salary Insights**: (Recruiter Portal) Enter salary in LPA and view monthly/annual breakdown.

## Tech Stack
- Python 3.8+
- Streamlit
- spaCy (NLP)
- Sentence Transformers (BERT embeddings)
- scikit-learn (TF-IDF, cosine similarity)
- pdfplumber, python-docx (resume/JD parsing)
- matplotlib, seaborn (visualizations)

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kanakasudheer/Job-Match-AI.git

```


### 2. Download commands
```bash
pip install streamlit pandas numpy spacy pdfplumber python-docx scikit-learn sentence-transformers matplotlib seaborn streamlit-tags
python -m spacy download en_core_web_lg
```

### 3. Run the app
```bash
streamlit run job_match.py  (OR)  python -m streamlit run job_match.py
```

## Usage
- Use the sidebar to navigate between Dashboard, Candidate Portal, Recruiter Portal, and Matching Engine.
- Candidates: Upload your resume, fill in your profile, and get job recommendations.
- Recruiters: Post jobs, upload JDs, and view top candidate matches.
- Use the Matching Engine for detailed AI-powered match analysis.

## File Structure
- `job_match.py` - Main Streamlit app
- `README.md` - Project documentation

## License
This project is for educational and demonstration purposes.

