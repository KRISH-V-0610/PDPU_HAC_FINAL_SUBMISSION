import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re

# Set page config
st.set_page_config(
    page_title="Compliance Audit Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Global text color */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown ul, .stMarkdown ol {
        color: #ffffff !important;
    }
    
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 20px;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        margin-top: 30px;
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
    .card {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border: 2px solid #333333;
        color: #ffffff;
    }
    .card h3 {
        color: #ffffff;
        margin-bottom: 15px;
        font-weight: bold;
    }
    .card p {
        color: #ffffff;
    }
    .card ul {
        color: #ffffff;
    }
    .card li {
        color: #ffffff;
    }
    .score-card {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        text-align: center;
        color: white;
    }
    .score-value {
        font-size: 48px;
        font-weight: bold;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .score-label {
        font-size: 18px;
        color: #ffffff;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #ffffff;
        font-size: 14px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
    /* New styles for Q&A section */
    .qa-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border: 2px solid #404040;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .qa-card p {
        color: #ffffff;
        font-size: 16px;
        line-height: 1.6;
    }
    .stButton button {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    /* Additional styles for markdown content */
    .stMarkdown {
        color: #ffffff;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff;
        font-weight: bold;
    }
    .stMarkdown p {
        color: #ffffff;
    }
    .stMarkdown ul, .stMarkdown ol {
        color: #ffffff;
    }
    .stMarkdown li {
        color: #ffffff;
    }
    /* Special styling for important text */
    .highlight {
        color: #00ff00 !important;
        font-weight: bold;
    }
    .warning {
        color: #ff9900 !important;
        font-weight: bold;
    }
    .danger {
        color: #ff4444 !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "https://fingenius-ai-fastapi-0rad.onrender.com"

# Function to extract compliance scores from the text
def extract_scores(text):
    # Look for patterns like "KYC Verification Rate: 72% → Standard: 100%"
    pattern = r'([^:]+):\s*(\d+(?:\.\d+)?)%\s*→\s*[^:]+:\s*(\d+(?:\.\d+)?)%'
    
    matches = re.findall(pattern, text)
    
    scores = {}
    for match in matches:
        category = match[0].strip()
        actual = float(match[1])
        target = float(match[2])
        scores[category] = {
            'actual': actual,
            'target': target,
            'compliance_rate': (actual / target) * 100 if target > 0 else 0
        }
    
    return scores

# Function to create gauge chart
def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FF4B4B'},
                {'range': [50, 75], 'color': '#FFA500'},
                {'range': [75, 100], 'color': '#00B050'}
            ],
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_radar_chart(scores_dict):
    categories = list(scores_dict.keys())
    actual_values = [scores_dict[cat]['actual'] for cat in categories]
    target_values = [scores_dict[cat]['target'] for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=actual_values,
        theta=categories,
        fill='toself',
        name='Actual',
        line_color='rgba(31, 119, 180, 0.8)',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=target_values,
        theta=categories,
        fill='toself',
        name='Target',
        line_color='rgba(44, 160, 44, 0.8)',
        fillcolor='rgba(44, 160, 44, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title={
            'text': 'Compliance Metrics Comparison',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=500
    )
    
    return fig

def create_category_compliance_chart(scores_dict):
    categories = list(scores_dict.keys())
    compliance_rates = [scores_dict[cat]['compliance_rate'] for cat in categories]
    
    colors = ['#FF4B4B' if rate < 50 else '#FFA500' if rate < 75 else '#00B050' for rate in compliance_rates]
    
    fig = px.bar(
        x=categories,
        y=compliance_rates,
        labels={'x': 'Category', 'y': 'Compliance Rate (%)'},
        title='Compliance Rate by Category'
    )
    
    fig.update_traces(marker_color=colors)
    
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},
        yaxis={'range': [0, 100]},
        height=400
    )
    
    return fig

# Sidebar
with st.sidebar:
    # st.image(r"", width=150)
    # st.markdown("")
    st.title("ComplySmart Dashboard")
    st.markdown("---")
    
    # Navigation
    page = st.radio("Navigation", ["Dashboard", "Detailed Audit", "Compliance Q&A"])
    
    # Run Audit Button
    if st.button("Run New Audit", type="primary"):
        with st.spinner("Running compliance audit..."):
            try:
                response = requests.post(f"{API_URL}/query")
                if response.status_code == 200:
                    audit_data = response.json()
                    st.session_state['audit_data'] = audit_data
                    st.success("Audit completed successfully!")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("CompAudit helps organizations monitor compliance with regulatory standards and identify areas for improvement.")

# Main content
if page == "Dashboard":
    st.markdown("<div class='main-header'>Compliance Audit Dashboard</div>", unsafe_allow_html=True)
    
    if 'audit_data' not in st.session_state:
        # Display placeholder content
        st.info("Run an audit to view your compliance dashboard. Click 'Run New Audit' in the sidebar to get started.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Sample Compliance Metrics")
            st.markdown("The dashboard will display key compliance metrics including:")
            st.markdown("- Overall compliance score")
            st.markdown("- Compliance by regulatory body")
            st.markdown("- Key metrics and indicators")
            st.markdown("- Areas needing improvement")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("How It Works")
            st.markdown("1. Click 'Run New Audit' to analyze your compliance data")
            st.markdown("2. Review your compliance dashboard")
            st.markdown("3. Explore detailed findings in the 'Detailed Audit' tab")
            st.markdown("4. Ask questions about your compliance in the 'Compliance Q&A' tab")
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Extract the score from the score text
        score_text = st.session_state['audit_data']['score']
        score_match = re.search(r'Score:\s*(\d+)', score_text)
        overall_score = int(score_match.group(1)) if score_match else 65  # Default score if not found
        
        # Raw compliance data from text
        compliance_data = """
        KYC Verification Rate: 72% → Standard: 100%
        AML Transaction Monitoring: 85% → Standard: 100%
        Credit Information Reporting: 75% → Standard: 100%
        Loan-to-Value (LTV) Ratio: 65% → Standard: 75%
        NPA (Non-Performing Assets): 83% → Standard: 100%
        AIF Due Diligence: 65% → Standard: 100%
        Financial Disclosure Timeliness: 80% → Standard: 100%
        Large Investor Verification: 50% → Standard: 100%
        Capital Adequacy Ratio: 83% → Standard: 100%
        GST Filing Timeliness: 88% → Standard: 100%
        TDS Compliance Rate: 92% → Standard: 100%
        """
        
        # Extract scores
        scores = extract_scores(compliance_data)
        
        # Display Overall Score
        st.markdown("<div class='score-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='score-value'>{overall_score}</div>", unsafe_allow_html=True)
        st.markdown("<div class='score-label'>Overall Compliance Score</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Overview", "By Regulatory Body", "Key Metrics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(overall_score, "Overall Compliance"), use_container_width=True)
            
            with col2:
                # Create dummy data for regulatory bodies
                reg_bodies = {
                    "RBI": 78,
                    "SEBI": 68,
                    "Income Tax": 85,
                    "GST": 88
                }
                
                reg_df = pd.DataFrame({
                    "Regulatory Body": list(reg_bodies.keys()),
                    "Compliance Score": list(reg_bodies.values())
                })
                
                fig = px.bar(
                    reg_df, 
                    x="Regulatory Body", 
                    y="Compliance Score",
                    color="Compliance Score",
                    color_continuous_scale=[(0, "red"), (0.5, "yellow"), (1, "green")],
                    range_color=[0, 100],
                    labels={"Compliance Score": "Score (%)"}
                )
                
                fig.update_layout(title="Compliance by Regulatory Body")
                st.plotly_chart(fig, use_container_width=True)
            
            # Display radar chart
            st.plotly_chart(create_radar_chart(scores), use_container_width=True)
        
        with tab2:
            # RBI Compliance
            st.markdown("<div class='sub-header'>RBI Compliance</div>", unsafe_allow_html=True)
            rbi_col1, rbi_col2 = st.columns(2)
            
            with rbi_col1:
                st.plotly_chart(create_gauge_chart(78, "RBI Compliance"), use_container_width=True)
            
            with rbi_col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Key Findings")
                st.markdown("- KYC Verification Rate below standard (72% vs 100%)")
                st.markdown("- AML Transaction Monitoring needs improvement (85% vs 100%)")
                st.markdown("- NPA slightly above threshold (4.8% vs 4%)")
                st.markdown("- Loan-to-Value ratio within acceptable range (65% vs ≤75%)")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # SEBI Compliance
            st.markdown("<div class='sub-header'>SEBI Compliance</div>", unsafe_allow_html=True)
            sebi_col1, sebi_col2 = st.columns(2)
            
            with sebi_col1:
                st.plotly_chart(create_gauge_chart(68, "SEBI Compliance"), use_container_width=True)
            
            with sebi_col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Key Findings")
                st.markdown("- AIF Due Diligence below standard (65% vs 100%)")
                st.markdown("- Large Investor Verification significantly below requirement (50% vs 100%)")
                st.markdown("- Capital Adequacy Ratio below threshold (12.5% vs ≥15%)")
                st.markdown("- Financial Disclosure delayed by 20 days (standard: on-time)")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Tax Compliance
            st.markdown("<div class='sub-header'>Tax Compliance</div>", unsafe_allow_html=True)
            tax_col1, tax_col2 = st.columns(2)
            
            with tax_col1:
                st.plotly_chart(create_gauge_chart(85, "Tax Compliance"), use_container_width=True)
            
            with tax_col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Key Findings")
                st.markdown("- GST Filing mostly on time (88% vs 100%)")
                st.markdown("- TDS Compliance Rate near standard (92% vs 100%)")
                st.markdown("- Tax Payment Delays of 15-20 days (standard: no delays)")
                st.markdown("- Tax Penalties of ₹3.2 lakh in the last year (standard: no fines)")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.plotly_chart(create_category_compliance_chart(scores), use_container_width=True)
            
            # Risk areas
            st.markdown("<div class='sub-header'>Risk Areas</div>", unsafe_allow_html=True)
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### High Risk")
                st.markdown("- Large Investor Verification (50%)")
                st.markdown("- AIF Due Diligence (65%)")
                st.markdown("- Capital Adequacy Ratio (12.5%)")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with risk_col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Medium Risk")
                st.markdown("- KYC Verification Rate (72%)")
                st.markdown("- NPA Rate (4.8%)")
                st.markdown("- Financial Disclosure Timeliness")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with risk_col3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Low Risk")
                st.markdown("- TDS Compliance Rate (92%)")
                st.markdown("- GST Filing Timeliness (88%)")
                st.markdown("- AML Transaction Monitoring (85%)")
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='sub-header'>Improvement Recommendations</div>", unsafe_allow_html=True)
        
        with st.expander("View Recommendations", expanded=False):
            st.markdown("""
            ### Priority Actions
            1. **Improve Large Investor Verification** (Current: 50%)
               - Implement automated verification systems
               - Create dedicated team for large investor due diligence
               - Conduct quarterly verification audits
               
            2. **Enhance AIF Due Diligence** (Current: 65%)
               - Update due diligence protocols to align with latest SEBI requirements
               - Implement regular training for staff on due diligence procedures
               - Create standardized due diligence checklist for all investors
               
            3. **Address Capital Adequacy Ratio** (Current: 12.5%, Threshold: ≥15%)
               - Develop capital enhancement plan
               - Review asset weightings and risk classifications
               - Consider equity infusion to meet required threshold
               
            ### Secondary Actions
            4. **Improve KYC Verification Rate** (Current: 72%)
               - Streamline KYC process with digital verification
               - Implement automated reminders for incomplete KYC
               - Conduct monthly KYC compliance reviews
               
            5. **Address NPA Rate** (Current: 4.8%, Threshold: ≤4%)
               - Enhance credit assessment procedures
               - Implement early warning system for potential defaults
               - Develop specialized recovery strategies for different asset classes
            """)

elif page == "Detailed Audit":
    st.markdown("<div class='main-header'>Detailed Compliance Audit Report</div>", unsafe_allow_html=True)
    
    if 'audit_data' not in st.session_state:
        st.info("No audit data available. Please run an audit from the Dashboard.")
    else:
        audit_report = st.session_state['audit_data']['audit']
        
        # Format and display the report
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(audit_report, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Audit Report (PDF)",
                data=audit_report,
                file_name="compliance_audit_report.pdf",
                mime="application/pdf",
            )
        
        with col2:
            st.download_button(
                label="Download Audit Report (Excel)",
                data="Sample Excel content",
                file_name="compliance_audit_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

elif page == "Compliance Q&A":
    st.markdown("<div class='main-header'>Compliance Q&A</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='qa-card'>
    <p>Ask any questions about your compliance status, regulatory requirements, or recommendations for improvement.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for user question if not exists
    if 'user_question' not in st.session_state:
        st.session_state['user_question'] = ""
    
    # User input
    user_question = st.text_input("Enter your question:", value=st.session_state['user_question'], placeholder="e.g., What are our biggest compliance risks?")
    
    # Suggested questions
    st.markdown("### Suggested Questions")
    
    question_cols = st.columns(2)
    
    suggested_questions = [
        "What are our biggest compliance risks?",
        "How can we improve our KYC verification rate?",
        "What regulatory changes should we prepare for?",
        "How does our compliance compare to industry standards?",
        "What immediate actions should we take to improve compliance?",
        "What penalties might we face for our current compliance gaps?"
    ]
    
    for i, question in enumerate(suggested_questions):
        col = question_cols[i % 2]
        if col.button(question, key=f"q_{i}"):
            st.session_state['user_question'] = question
            st.rerun()
    
    # Process the question (either from input or suggestion)
    if user_question:
        with st.spinner("Generating answer..."):
            try:
                # Prepare the request payload
                payload = {"q": user_question}
                
                # Send the request to the API
                response = requests.post(f"{API_URL}/ans", json=payload)
                
                if response.status_code == 200:
                    answer = response.json()["ans"]
                    
                    # Display the answer
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("### Answer")
                    st.markdown(answer)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Clear the question after processing
                    st.session_state['user_question'] = ""
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")

# Footer
st.markdown("""
<div class="footer">
    ComplySmart Dashboard v1.0 | © 2025 | Powered by Team FinGenius
</div>
""", unsafe_allow_html=True)