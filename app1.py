import numpy as np
import pandas as pd
import pickle as pkl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Smart Insurance Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with vibrant color scheme
st.markdown("""
    <style>
        /* Global styles */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 2rem;
        }
        
        /* Card styling */
        .stCard {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Input container styling */
        .input-container {
            background: rgba(255, 255, 255, 0.1);
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
        }
        
        /* Result container styling */
        .result-container {
            background: rgba(255, 255, 255, 0.15);
            padding: 2rem;
            border-radius: 20px;
            margin-top: 2rem;
            text-align: center;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            color: white;
            font-weight: bold;
            padding: 1rem 2rem;
            border-radius: 50px;
            border: none;
            width: 100%;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .stButton button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        
        /* Select box styling */
        .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        
        /* Slider styling */
        .stSlider div[data-baseweb="slider"] {
            background-color: #4ECDC4;
        }
        
        /* Custom cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 15px;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.2);
        }
        
        /* Animation keyframes */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .floating {
            animation: float 3s ease-in-out infinite;
        }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    return pkl.load(open('gradient_boosting_regressor_model.pkl', 'rb'))

model = load_model()

# Helper functions
def calculate_health_metrics(age, bmi, smoker):
    risk_score = 0
    risk_score += (age // 10) * 5  # Age factor
    
    # BMI factor
    if bmi < 18.5:
        risk_score += 10
    elif 18.5 <= bmi < 25:
        risk_score += 0
    elif 25 <= bmi < 30:
        risk_score += 15
    else:
        risk_score += 25
        
    # Smoker factor
    if smoker == "Yes":
        risk_score += 50
        
    return min(risk_score, 100)

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'color': "white", 'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#4ECDC4"},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 33], 'color': '#4ECDC4'},
                {'range': [33, 66], 'color': '#FFD93D'},
                {'range': [66, 100], 'color': '#FF6B6B'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "white", 'family': "Arial"}
    )
    
    return fig

def create_cost_breakdown(prediction):
    labels = ['Base Cost', 'Age Factor', 'BMI Factor', 'Smoking Factor', 'Region Factor']
    base_cost = prediction * 0.4
    values = [
        base_cost,
        prediction * 0.15,  # Age factor
        prediction * 0.15,  # BMI factor
        prediction * 0.2,   # Smoking factor
        prediction * 0.1    # Region factor
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=['#4ECDC4', '#FFD93D', '#FF6B6B', '#FF9A8B', '#764BA2'])
    )])
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.1)",
            bordercolor="white",
            borderwidth=1
        )
    )
    
    return fig

# Main app
st.markdown('<h1 class="main-header floating">Smart Insurance Cost Predictor</h1>', unsafe_allow_html=True)

# Create three columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Personal Information Section
    st.subheader("📋 Personal Information")
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 100, 25)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
    
    # Lifestyle Section
    st.subheader("🌟 Lifestyle Factors")
    smoker = st.radio("Smoking Status", ["No", "Yes"])
    
    # Location and Family Section
    st.subheader("📍 Location & Family")
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    children = st.number_input("Number of Children", 0, 10, 0)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Data Processing
gender_mapping = {"Female": 0, "Male": 1}
smoker_mapping = {"Yes": 1, "No": 0}
region_mapping = {
    "northeast": [1, 0, 0],
    "northwest": [0, 1, 0],
    "southeast": [0, 0, 1],
    "southwest": [0, 0, 0]
}

input_data = [
    gender_mapping[gender],
    age,
    bmi,
    smoker_mapping[smoker],
    *region_mapping[region],
    children
]

# Prediction and Analysis
if st.button("Calculate Insurance Cost 🚀"):
    try:
        prediction = model.predict([input_data])[0]
        risk_score = calculate_health_metrics(age, bmi, smoker == "Yes")
        
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        
        # Main prediction display
        st.markdown(f"""
        <h2 style='color: #4ECDC4; font-size: 2.5rem; margin-bottom: 1rem;'>
            ${prediction:,.2f}
        </h2>
        <p style='font-size: 1.2rem; color: #FFD93D;'>Estimated Annual Insurance Cost</p>
        """, unsafe_allow_html=True)
        
        # Create three columns for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Monthly Cost", f"${prediction/12:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Risk Score", f"{risk_score}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Cost Rating", "Average" if prediction < 10000 else "High")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk Gauge
        st.plotly_chart(create_gauge_chart(risk_score, "Health Risk Assessment"), use_container_width=True)
        
        # Cost Breakdown
        st.subheader("💰 Cost Breakdown Analysis")
        st.plotly_chart(create_cost_breakdown(prediction), use_container_width=True)
        
        # Recommendations
        st.subheader("🎯 Personalized Recommendations")
        recommendations = []
        
        if bmi > 25:
            recommendations.append("• Consider a wellness program to maintain a healthy BMI")
        if smoker == "Yes":
            recommendations.append("• Quitting smoking could significantly reduce your premium")
        if age > 50:
            recommendations.append("• Regular health check-ups recommended")
        if not recommendations:
            recommendations.append("• Continue maintaining your healthy lifestyle")
            
        for rec in recommendations:
            st.markdown(f'<div class="metric-card">{rec}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("""
<div style='text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);'>
    <p>Last updated: {}</p>
    <p>This is a prediction model. Actual insurance costs may vary.</p>
</div>
""".format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)