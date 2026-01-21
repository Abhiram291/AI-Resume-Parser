"""
Streamlit Web Interface for Resume Parser
Provides a user-friendly GUI for resume parsing and skill matching
"""

import streamlit as st
import json
import os
from pathlib import Path
import tempfile
from resume_parser import ResumeParser
from skill_matcher import SkillMatcher
from utils import generate_summary_report, export_to_csv

# Page config
st.set_page_config(
    page_title="AI Resume Parser",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'parsed_resumes' not in st.session_state:
        st.session_state.parsed_resumes = []
    if 'current_resume' not in st.session_state:
        st.session_state.current_resume = None
    if 'match_results' not in st.session_state:
        st.session_state.match_results = None


@st.cache_resource
def load_models():
    """Load models with caching"""
    parser = ResumeParser()
    matcher = SkillMatcher()
    return parser, matcher


def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Resume Parser</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.radio(
            "Select Mode",
            ["Single Resume Parser", "Job Matching", "Batch Processing"]
        )
        st.markdown("---")
        st.info("💡 **Tip**: Upload PDF, DOCX, or TXT files")
    
    # Load models
    with st.spinner("Loading AI models..."):
        parser, matcher = load_models()
    
    # Main content based on mode
    if mode == "Single Resume Parser":
        single_resume_mode(parser)
    elif mode == "Job Matching":
        job_matching_mode(parser, matcher)
    else:
        batch_processing_mode(parser, matcher)


def single_resume_mode(parser):
    """Single resume parsing mode"""
    st.header("📄 Parse Single Resume")
    
    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=['pdf', 'docx', 'txt'],
        help="Upload a resume in PDF, DOCX, or TXT format"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Parse Resume", type="primary"):
                with st.spinner("Parsing resume..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Parse resume
                        resume_data = parser.parse(tmp_path)
                        st.session_state.current_resume = resume_data
                        st.success("✅ Resume parsed successfully!")
                    except Exception as e:
                        st.error(f"❌ Error parsing resume: {str(e)}")
                    finally:
                        os.unlink(tmp_path)
        
        # Display results
        if st.session_state.current_resume:
            display_resume_data(st.session_state.current_resume)


def job_matching_mode(parser, matcher):
    """Job matching mode"""
    st.header("🎯 Job Matching & Skill Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Resume")
        resume_file = st.file_uploader(
            "Resume File",
            type=['pdf', 'docx', 'txt'],
            key="resume_match"
        )
    
    with col2:
        st.subheader("Job Description")
        job_input_method = st.radio(
            "Input Method",
            ["Upload JSON", "Paste Text"]
        )
        
        if job_input_method == "Upload JSON":
            job_file = st.file_uploader(
                "Job Description JSON",
                type=['json'],
                key="job_json"
            )
            job_desc = None
            if job_file:
                job_desc = json.load(job_file)
        else:
            job_text = st.text_area(
                "Paste Job Description",
                height=200,
                placeholder="Paste the job description here..."
            )
            job_desc = {"description": job_text} if job_text else None
    
    if st.button("🎯 Match Skills", type="primary"):
        if not resume_file or not job_desc:
            st.warning("⚠️ Please upload both resume and job description")
        else:
            with st.spinner("Analyzing and matching..."):
                # Save resume temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(resume_file.name).suffix) as tmp_file:
                    tmp_file.write(resume_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Parse and match
                    resume_data = parser.parse(tmp_path)
                    match_results = matcher.match(resume_data, job_desc)
                    
                    st.session_state.current_resume = resume_data
                    st.session_state.match_results = match_results
                    
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    os.unlink(tmp_path)
    
    # Display results
    if st.session_state.match_results:
        display_match_results(
            st.session_state.current_resume,
            st.session_state.match_results
        )


def batch_processing_mode(parser, matcher):
    """Batch processing mode"""
    st.header("📦 Batch Processing")
    
    uploaded_files = st.file_uploader(
        "Upload Multiple Resumes",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    job_file = st.file_uploader(
        "Job Description (optional)",
        type=['json'],
        key="batch_job"
    )
    
    if st.button("🚀 Process All", type="primary"):
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one resume")
        else:
            progress_bar = st.progress(0)
            results = []
            
            job_desc = None
            if job_file:
                job_desc = json.load(job_file)
            
            for idx, file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    resume_data = parser.parse(tmp_path)
                    
                    if job_desc:
                        match_result = matcher.match(resume_data, job_desc)
                        resume_data['skill_match'] = match_result
                    
                    results.append(resume_data)
                except Exception as e:
                    st.warning(f"Failed to process {file.name}: {str(e)}")
                finally:
                    os.unlink(tmp_path)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            st.session_state.parsed_resumes = results
            st.success(f"✅ Processed {len(results)} resumes!")
    
    # Display batch results
    if st.session_state.parsed_resumes:
        display_batch_results(st.session_state.parsed_resumes)


def display_resume_data(resume_data):
    """Display parsed resume data"""
    st.markdown("---")
    st.subheader("📊 Parsed Information")
    
    # Personal Info
    with st.expander("👤 Personal Information", expanded=True):
        personal_info = resume_data.get('personal_info', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Name", personal_info.get('name', 'N/A'))
            st.metric("Email", personal_info.get('email', 'N/A'))
        with col2:
            st.metric("Phone", personal_info.get('phone', 'N/A'))
            st.metric("Location", personal_info.get('location', 'N/A'))
        with col3:
            linkedin = personal_info.get('linkedin', 'N/A')
            github = personal_info.get('github', 'N/A')
            st.metric("LinkedIn", "✓" if linkedin != 'N/A' else "✗")
            st.metric("GitHub", "✓" if github != 'N/A' else "✗")
    
    # Skills
    with st.expander("💼 Skills", expanded=True):
        skills = resume_data.get('skills', [])
        if skills:
            st.metric("Total Skills", len(skills))
            st.write(" • ".join(skills))
        else:
            st.info("No skills extracted")
    
    # Experience
    with st.expander("🏢 Work Experience"):
        experiences = resume_data.get('experience', [])
        if experiences:
            for exp in experiences:
                st.markdown(f"**{exp.get('role', 'N/A')}** at {exp.get('company', 'N/A')}")
                st.caption(exp.get('duration', 'N/A'))
                responsibilities = exp.get('responsibilities', [])
                if responsibilities:
                    for resp in responsibilities[:3]:
                        st.write(f"• {resp}")
                st.markdown("---")
        else:
            st.info("No experience found")
    
    # Education
    with st.expander("🎓 Education"):
        education = resume_data.get('education', [])
        if education:
            for edu in education:
                st.markdown(f"**{edu.get('degree', 'N/A')}**")
                st.write(f"{edu.get('institution', 'N/A')} - {edu.get('year', 'N/A')}")
                if edu.get('GPA'):
                    st.caption(f"GPA: {edu.get('GPA')}")
                st.markdown("---")
        else:
            st.info("No education found")
    
    # Download JSON
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(resume_data, indent=2),
            file_name="parsed_resume.json",
            mime="application/json"
        )
    with col2:
        report = generate_summary_report(resume_data)
        st.download_button(
            "📄 Download Report",
            data=report,
            file_name="resume_report.txt",
            mime="text/plain"
        )


def display_match_results(resume_data, match_results):
    """Display job matching results"""
    st.markdown("---")
    st.subheader("🎯 Matching Results")
    
    # Score card
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = match_results.get('overall_score', 0)
        st.metric("Match Score", f"{score}%", delta=None)
    with col2:
        matched = len(match_results.get('matched_skills', []))
        st.metric("Matched Skills", matched)
    with col3:
        missing = len(match_results.get('missing_skills', []))
        st.metric("Missing Skills", missing)
    with col4:
        recommendation = match_results.get('recommendation', 'N/A')
        st.metric("Status", recommendation.split('-')[0].strip())
    
    # Progress bar
    st.progress(score / 100)
    
    # Matched Skills
    with st.expander("✅ Matched Skills", expanded=True):
        matched_skills = match_results.get('matched_skills', [])
        if matched_skills:
            cols = st.columns(3)
            for idx, skill in enumerate(matched_skills):
                cols[idx % 3].success(f"✓ {skill}")
        else:
            st.info("No matched skills")
    
    # Missing Skills
    with st.expander("❌ Skills to Learn", expanded=True):
        missing_skills = match_results.get('missing_skills', [])
        if missing_skills:
            cols = st.columns(3)
            for idx, skill in enumerate(missing_skills):
                cols[idx % 3].error(f"✗ {skill}")
        else:
            st.success("All required skills are matched!")
    
    # Download
    combined_data = {**resume_data, 'skill_match': match_results}
    report = generate_summary_report(resume_data, match_results)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Full Report",
            data=report,
            file_name="match_report.txt",
            mime="text/plain"
        )
    with col2:
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(combined_data, indent=2),
            file_name="matched_resume.json",
            mime="application/json"
        )


def display_batch_results(results):
    """Display batch processing results"""
    st.markdown("---")
    st.subheader(f"📊 Batch Results ({len(results)} resumes)")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    total_skills = sum(len(r.get('skills', [])) for r in results)
    avg_skills = total_skills / len(results) if results else 0
    
    with col1:
        st.metric("Total Resumes", len(results))
    with col2:
        st.metric("Avg Skills", round(avg_skills, 1))
    with col3:
        has_match = sum(1 for r in results if 'skill_match' in r)
        st.metric("Matched Resumes", has_match)
    
    # Rankings (if job matching was done)
    if results and 'skill_match' in results[0]:
        st.subheader("🏆 Candidate Rankings")
        
        # Sort by match score
        sorted_results = sorted(
            results,
            key=lambda x: x.get('skill_match', {}).get('overall_score', 0),
            reverse=True
        )
        
        for idx, result in enumerate(sorted_results[:10], 1):
            name = result.get('personal_info', {}).get('name', f'Candidate {idx}')
            score = result.get('skill_match', {}).get('overall_score', 0)
            
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.markdown(f"**#{idx}**")
                with col2:
                    st.markdown(f"**{name}**")
                with col3:
                    st.metric("Score", f"{score}%")
                st.progress(score / 100)
    
    # Export options
    st.markdown("---")
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Download All as JSON",
            data=json.dumps(results, indent=2),
            file_name="batch_results.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()