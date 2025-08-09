import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="📊 Smart Stock Advisor - US Stocks",
    layout="wide",
    page_icon="💹"
)

# --------- HEADER SECTION ---------
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>📈 SMART STOCK ADVISOR </h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>US-Based Stock Price Analysis Powered by AI</h4>", unsafe_allow_html=True)
st.markdown("---")

# --------- INTRO SECTION ---------
st.markdown("""
<div style="background-color: #F2F3F4; padding: 20px; border-radius: 10px;">
    <h3 style="color: #1A5276;">🧠 Ready to Predict the Market?</h3>
    <p style="font-size: 16px; color: #34495E;">
        Whether you're new to investing or a seasoned analyst, our AI-powered dashboard helps you make smarter investment decisions.
        Choose your level to begin your journey with US-based stock predictions, visual analytics, and actionable insights.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Choose Your Experience Level")

# --------- CHOICE BUTTONS ---------
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div style="background-color: #D6EAF8; padding: 20px; border-radius: 10px;">
            <h4 style="color: #21618C;">🟢 Beginner Mode</h4>
            <p style="color: #1B2631;">For new investors who want clear, simple stock insights and easy predictions.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<a href="https://github.com/longway2go-ai/AI-Stock-Advisor-for-Beginner" target="_blank">'
        '<button style="background-color:#21618C;color:white;padding:10px 20px;border:none;border-radius:8px;font-size:16px;cursor:pointer;">Enter Beginner Mode</button>'
        '</a>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown("""
        <div style="background-color: #D5F5E3; padding: 20px; border-radius: 10px;">
            <h4 style="color: #117A65;">🔵 Expert Mode</h4>
            <p style="color: #1B2631;">Dive deep into technical indicators, forecasting models, and customizable visualizations.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<a href="https://github.com/Sayan-ML/Time-Series-for-Experts" target="_blank">'
        '<button style="background-color:#117A65;color:white;padding:10px 20px;border:none;border-radius:8px;font-size:16px;cursor:pointer;">Enter Expert Mode</button>'
        '</a>',
        unsafe_allow_html=True
    )

# --------- TEASER PROMPT ---------
st.markdown("---")
st.markdown("""
<div style="background-color: #FEF9E7; padding: 20px; border-radius: 10px;">
    <h4 style="color: #B7950B;">🔮 Curious about tomorrow’s market?</h4>
    <p style="color: #7D6608;">
        Use our AI engine to predict whether your favorite stock will <strong>rise 📈 or fall 📉</strong> tomorrow based on historical trends, news sentiment, and technical signals.
    </p>
    <p style="color: #7D6608;">
        Ready to try it out? Choose a mode above and start analyzing!
    </p>
</div>
""", unsafe_allow_html=True)

# --------- FOOTER ---------
st.markdown("---")
st.caption("© 2025 Smart Stock Advisor | Powered by AI & Streamlit | 📍 US Markets Only")
