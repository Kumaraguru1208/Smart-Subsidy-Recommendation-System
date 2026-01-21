import streamlit as st
import pandas as pd
import google.generativeai as genai
from PIL import Image
import json
import re
from datetime import datetime
import os

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY") 

if not API_KEY:
    # This message appears in the app if the key is missing in the environment
    st.error("🚨 GEMINI_API_KEY not found! Please set it in your terminal before running the app.")
    st.stop()
else:
    genai.configure(api_key=API_KEY)

# Initialize session state
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Load schemes database
@st.cache_data
def load_schemes():
    """Load government schemes from CSV or create sample data"""
    schemes_data = {
        'scheme_name': [
            'National Scholarship Scheme',
            'PM Kisan Scheme',
            'TN Education Grant',
            'Post Matric Scholarship for SC/ST',
            'Chief Minister Relief Fund',
            'Minority Scholarship Scheme',
            'Merit-cum-Means Scholarship',
            'Girl Child Education Support',
            'Vocational Training Grant',
            'Startup Support Scheme'
        ],
        'income_limit': [200000, 250000, 300000, 250000, 100000, 200000, 150000, 250000, 180000, 300000],
        'education_requirement': ['12th Pass', 'Any', '12th Pass', 'Graduate', 'Any', '12th Pass', 'Graduate', '10th Pass', '12th Pass', 'Graduate'],
        'target_group': ['Students', 'Farmers', 'TN Residents', 'SC/ST Students', 'BPL Families', 'Minority Students', 'Meritorious Students', 'Female Students', 'Youth', 'Entrepreneurs'],
        'benefits': ['₹10,000/year', '₹6,000/year', 'Free tuition', '₹15,000/year', '₹5,000 one-time', '₹12,000/year', '₹20,000/year', '₹8,000/year', '₹15,000 training', '₹50,000 grant'],
        'eligibility_criteria': [
            'Annual income < ₹2L, Pursuing higher education',
            'Small/marginal farmers with land < 2 hectares',
            'Tamil Nadu resident, pursuing UG/PG',
            'SC/ST category, pursuing post-graduation',
            'BPL family, medical/education emergency',
            'Minority community, academic excellence',
            'Rank in top 10%, family income < ₹1.5L',
            'Female student, rural background',
            'Age 18-30, interest in vocational training',
            'Age < 35, innovative business idea'
        ]
    }
    return pd.DataFrame(schemes_data)

def extract_text_with_gemini(image):
    """Extract text from image using Google Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        Analyze this document image and extract ALL text accurately.
        Return the extracted text in a clean, structured format.
        Include all numbers, names, addresses, and important information.
        """
        
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        st.error(f"Error in OCR: {str(e)}")
        return ""

def extract_structured_data(aadhaar_text, income_text, marksheet_text):
    """Use Gemini to extract structured data from OCR text"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an expert at extracting information from government documents.
        
        I have text extracted from three documents:
        
        AADHAAR CARD:
        {aadhaar_text}
        
        INCOME CERTIFICATE:
        {income_text}
        
        MARKSHEET:
        {marksheet_text}
        
        Extract the following information and return ONLY a valid JSON object:
        {{
            "name": "full name",
            "age": age as integer,
            "gender": "Male/Female/Other",
            "income": annual income as integer,
            "education": "10th Pass/12th Pass/Graduate/Post Graduate",
            "category": "General/SC/ST/OBC/Minority",
            "state": "state name",
            "marks": total percentage/marks as integer,
            "occupation": "Student/Farmer/Employed/Self-employed"
        }}
        
        If any field is not found, use reasonable defaults or "Not Found".
        Return ONLY the JSON object, no other text.
        """
        
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        json_text = response.text.strip()
        if json_text.startswith('```json'):
            json_text = json_text[7:-3].strip()
        elif json_text.startswith('```'):
            json_text = json_text[3:-3].strip()
        
        data = json.loads(json_text)
        return data
        
    except Exception as e:
        st.error(f"Error extracting structured data: {str(e)}")
        # Return default data
        return {
            "name": "Not Found",
            "age": 0,
            "gender": "Not Found",
            "income": 0,
            "education": "Not Found",
            "category": "General",
            "state": "Not Found",
            "marks": 0,
            "occupation": "Not Found"
        }

def calculate_match_score(user_data, scheme):
    """Calculate match score between user data and scheme"""
    score = 0
    
    # Income eligibility (40 points)
    if user_data['income'] <= scheme['income_limit']:
        score += 40
        # Bonus for significantly lower income
        income_ratio = user_data['income'] / scheme['income_limit']
        score += (1 - income_ratio) * 20
    
    # Education match (25 points)
    if scheme['education_requirement'] == 'Any':
        score += 25
    elif user_data['education'] == scheme['education_requirement']:
        score += 25
    elif ('Graduate' in scheme['education_requirement'] and 
          user_data['education'] in ['12th Pass', '10th Pass']):
        score += 12  # Partial match for pursuing higher education
    
    # Category/Target match (20 points)
    target_keywords = scheme['target_group'].lower().split()
    if user_data['category'].lower() in scheme['target_group'].lower():
        score += 20
    elif 'students' in target_keywords and user_data['occupation'].lower() == 'student':
        score += 15
    elif user_data['state'].lower() in scheme['target_group'].lower():
        score += 15
    
    # Merit-based bonus (15 points)
    if user_data['marks'] >= 85:
        score += 15
    elif user_data['marks'] >= 70:
        score += 10
    elif user_data['marks'] >= 60:
        score += 5
    
    return min(100, round(score))

def recommend_schemes(user_data, schemes_df):
    """Generate scheme recommendations based on user data"""
    recommendations = []
    
    for _, scheme in schemes_df.iterrows():
        score = calculate_match_score(user_data, scheme)
        
        recommendations.append({
            'scheme_name': scheme['scheme_name'],
            'score': score,
            'benefits': scheme['benefits'],
            'target_group': scheme['target_group'],
            'income_limit': scheme['income_limit'],
            'education_requirement': scheme['education_requirement'],
            'eligibility_criteria': scheme['eligibility_criteria']
        })
    
    # Sort by score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return recommendations[:5]  # Return top 5

# Streamlit UI
st.set_page_config(page_title="Smart Subsidy Recommender", page_icon="🎯", layout="wide")

st.title("🎯 Smart Subsidy Recommendation System")
st.markdown("*AI-powered government scheme recommendations based on your documents*")

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("Enter Gemini API Key", type="password", value=API_KEY if API_KEY else "")
    if api_key_input and api_key_input != "YOUR_GEMINI_API_KEY_HERE":
        genai.configure(api_key=api_key_input)
    
    st.markdown("---")
    st.markdown("### 📋 How to use:")
    st.markdown("""
    1. Upload Aadhaar Card
    2. Upload Income Certificate
    3. Upload Marksheet
    4. Click 'Analyze Documents'
    5. View recommendations!
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Documents")
    
    aadhaar_file = st.file_uploader("Aadhaar Card", type=['png', 'jpg', 'jpeg'], key='aadhaar')
    income_file = st.file_uploader("Income Certificate", type=['png', 'jpg', 'jpeg'], key='income')
    marksheet_file = st.file_uploader("Marksheet", type=['png', 'jpg', 'jpeg'], key='marksheet')
    
    if st.button("🔍 Analyze Documents", type="primary"):
        if not all([aadhaar_file, income_file, marksheet_file]):
            st.error("⚠️ Please upload all three documents!")
        else:
            with st.spinner("🔄 Processing documents with AI..."):
                # Load images
                aadhaar_img = Image.open(aadhaar_file)
                income_img = Image.open(income_file)
                marksheet_img = Image.open(marksheet_file)
                
                # Extract text using Gemini OCR
                st.info("📄 Extracting text from documents...")
                aadhaar_text = extract_text_with_gemini(aadhaar_img)
                income_text = extract_text_with_gemini(income_img)
                marksheet_text = extract_text_with_gemini(marksheet_img)
                
                # Extract structured data
                st.info("🧠 Analyzing extracted information...")
                user_data = extract_structured_data(aadhaar_text, income_text, marksheet_text)
                st.session_state.extracted_data = user_data
                
                # Generate recommendations
                st.info("🎯 Finding best matching schemes...")
                schemes_df = load_schemes()
                recommendations = recommend_schemes(user_data, schemes_df)
                st.session_state.recommendations = recommendations
                
                st.success("✅ Analysis complete!")

with col2:
    if st.session_state.extracted_data:
        st.header("📊 Extracted Information")
        
        data = st.session_state.extracted_data
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Name", data['name'])
            st.metric("Age", data['age'])
            st.metric("Income", f"₹{data['income']:,}")
            st.metric("Marks", f"{data['marks']}%")
        
        with col_b:
            st.metric("Gender", data['gender'])
            st.metric("Education", data['education'])
            st.metric("Category", data['category'])
            st.metric("State", data['state'])

# Recommendations section
if st.session_state.recommendations:
    st.markdown("---")
    st.header("🏆 Top Recommended Schemes")
    
    for idx, rec in enumerate(st.session_state.recommendations):
        with st.expander(f"#{idx+1} {rec['scheme_name']} - Match Score: {rec['score']}%", expanded=(idx==0)):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Benefits:** {rec['benefits']}")
                st.markdown(f"**Eligibility:** {rec['eligibility_criteria']}")
            
            with col2:
                st.markdown(f"**Target Group:**")
                st.info(rec['target_group'])
            
            with col3:
                st.markdown(f"**Requirements:**")
                st.info(f"Income: ₹{rec['income_limit']:,}")
                st.info(f"Education: {rec['education_requirement']}")
            
            # Score visualization
            st.progress(rec['score'] / 100)

# Footer
st.markdown("---")
st.markdown("*Developed with Google Gemini AI | For demonstration purposes*")