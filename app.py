import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from datetime import datetime
import time

# Set page config
st.set_page_config(
    page_title="MedEval-Pro: Medical LLM Benchmarking Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .safety-high { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }
    .safety-medium { background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%); }
    .safety-low { background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%); }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sample data generation (simulating real evaluation results)
@st.cache_data
def generate_sample_data():
    """Generate realistic medical LLM evaluation data"""
    models = ['GPT-4', 'Claude-3', 'Gemini-Pro', 'Med-PaLM-2', 'BioGPT']
    
    # Performance metrics
    performance_data = {
        'Model': models,
        'Medical Accuracy (%)': [92.3, 89.7, 85.4, 94.1, 78.2],
        'Clinical Relevance Score': [8.7, 8.4, 7.9, 9.2, 7.1],
        'Hallucination Rate (%)': [3.2, 4.1, 6.8, 2.1, 8.9],
        'Bias Score (Lower Better)': [0.15, 0.12, 0.23, 0.09, 0.31],
        'Safety Score (1-10)': [9.1, 8.8, 8.2, 9.5, 7.4],
        'Response Time (ms)': [1200, 800, 950, 1800, 600]
    }
    
    # Task-specific performance
    task_performance = pd.DataFrame({
        'Task': ['Clinical Summarization', 'Medical Q&A', 'Diagnostic Assistance', 
                'Drug Interaction Check', 'Patient Counseling'],
        'GPT-4': [91, 93, 89, 94, 88],
        'Claude-3': [88, 91, 85, 89, 92],
        'Gemini-Pro': [82, 87, 83, 85, 79],
        'Med-PaLM-2': [95, 96, 92, 97, 89],
        'BioGPT': [75, 82, 76, 78, 71]
    })
    
    # Bias analysis data
    bias_data = pd.DataFrame({
        'Demographic': ['Age (65+)', 'Gender (Female)', 'Race (Non-White)', 
                       'Socioeconomic (Low)', 'Insurance (Medicaid)'],
        'GPT-4': [0.12, 0.15, 0.18, 0.14, 0.16],
        'Claude-3': [0.09, 0.11, 0.14, 0.12, 0.13],
        'Gemini-Pro': [0.21, 0.25, 0.28, 0.22, 0.24],
        'Med-PaLM-2': [0.07, 0.08, 0.11, 0.09, 0.10],
        'BioGPT': [0.28, 0.32, 0.35, 0.31, 0.33]
    })
    
    return pd.DataFrame(performance_data), task_performance, bias_data

def create_radar_chart(df, model_name):
    """Create radar chart for model performance"""
    categories = ['Medical Accuracy', 'Clinical Relevance', 'Safety Score', 
                 'Low Hallucination', 'Low Bias', 'Speed']
    
    model_data = df[df['Model'] == model_name].iloc[0]
    values = [
        model_data['Medical Accuracy (%)'],
        model_data['Clinical Relevance Score'] * 10,
        model_data['Safety Score (1-10)'] * 10,
        100 - model_data['Hallucination Rate (%)'],
        (1 - model_data['Bias Score (Lower Better)']) * 100,
        max(0, 100 - (model_data['Response Time (ms)'] / 20))
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=model_name,
        line=dict(color='#2E8B57', width=2),
        fillcolor='rgba(46, 139, 87, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title=f"{model_name} Performance Profile",
        font=dict(size=12)
    )
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 MedEval-Pro: Medical LLM Benchmarking Suite</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; font-size: 1.2rem; color: #666;'>
    Comprehensive evaluation framework for Large Language Models in healthcare applications
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    performance_df, task_df, bias_df = generate_sample_data()
    
    # Sidebar
    st.sidebar.header("🎛️ Evaluation Controls")
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        options=['GPT-4', 'Claude-3', 'Gemini-Pro', 'Med-PaLM-2', 'BioGPT'],
        default=['GPT-4', 'Med-PaLM-2', 'Claude-3']
    )
    
    evaluation_type = st.sidebar.selectbox(
        "Evaluation Focus",
        ["Overall Performance", "Safety Analysis", "Bias Assessment", "Task-Specific Performance"]
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "🔍 Model Comparison", "⚠️ Safety Analysis", 
        "⚖️ Bias Assessment", "📈 Performance Trends"
    ])
    
    with tab1:
        st.header("📊 Overall Performance Dashboard")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_accuracy = performance_df.loc[performance_df['Medical Accuracy (%)'].idxmax()]
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Best Medical Accuracy</h3>
                <h2>{best_accuracy['Medical Accuracy (%)']}%</h2>
                <p>{best_accuracy['Model']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            safest_model = performance_df.loc[performance_df['Safety Score (1-10)'].idxmax()]
            st.markdown(f"""
            <div class="metric-card safety-low">
                <h3>🛡️ Safest Model</h3>
                <h2>{safest_model['Safety Score (1-10)']}/10</h2>
                <p>{safest_model['Model']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            least_biased = performance_df.loc[performance_df['Bias Score (Lower Better)'].idxmin()]
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚖️ Least Biased</h3>
                <h2>{least_biased['Bias Score (Lower Better)']}</h2>
                <p>{least_biased['Model']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            lowest_hallucination = performance_df.loc[performance_df['Hallucination Rate (%)'].idxmin()]
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎭 Lowest Hallucination</h3>
                <h2>{lowest_hallucination['Hallucination Rate (%)']}%</h2>
                <p>{lowest_hallucination['Model']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Performance overview chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Medical Accuracy vs Safety Score")
            fig_scatter = px.scatter(
                performance_df, 
                x='Medical Accuracy (%)', 
                y='Safety Score (1-10)',
                size='Clinical Relevance Score',
                color='Model',
                hover_data=['Hallucination Rate (%)', 'Bias Score (Lower Better)'],
                title="Model Performance Positioning"
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.subheader("Hallucination Rate Comparison")
            fig_bar = px.bar(
                performance_df, 
                x='Model', 
                y='Hallucination Rate (%)',
                color='Hallucination Rate (%)',
                color_continuous_scale='Reds_r',
                title="Hallucination Rates by Model"
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.header("🔍 Detailed Model Comparison")
        
        if len(selected_models) > 0:
            # Radar charts
            cols = st.columns(min(len(selected_models), 3))
            for i, model in enumerate(selected_models[:3]):
                with cols[i]:
                    radar_fig = create_radar_chart(performance_df, model)
                    st.plotly_chart(radar_fig, use_container_width=True)
            
            st.divider()
            
            # Detailed comparison table
            st.subheader("📋 Detailed Performance Metrics")
            comparison_df = performance_df[performance_df['Model'].isin(selected_models)]
            st.dataframe(comparison_df, use_container_width=True)
            
            # Task-specific performance
            st.subheader("🎯 Task-Specific Performance")
            task_filtered = task_df[['Task'] + [col for col in selected_models if col in task_df.columns]]
            
            fig_heatmap = px.imshow(
                task_filtered.set_index('Task').T,
                title="Task Performance Heatmap (%)",
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.header("⚠️ Medical Safety Analysis")
        
        # Safety score distribution
        fig_safety = px.bar(
            performance_df,
            x='Model',
            y='Safety Score (1-10)',
            color='Safety Score (1-10)',
            color_continuous_scale='RdYlGn',
            title="Medical Safety Scores by Model"
        )
        fig_safety.update_layout(height=400)
        st.plotly_chart(fig_safety, use_container_width=True)
        
        # Safety categories
        st.subheader("🚨 Safety Risk Assessment")
        
        # Simulate safety category data
        safety_categories = pd.DataFrame({
            'Risk Category': ['Contraindications', 'Drug Interactions', 'Dosage Errors', 
                            'Emergency Recognition', 'Harmful Advice'],
            'GPT-4': [2, 1, 3, 1, 2],
            'Claude-3': [3, 2, 2, 2, 3],
            'Med-PaLM-2': [1, 1, 1, 1, 1],
            'Gemini-Pro': [4, 3, 4, 3, 4],
            'BioGPT': [5, 4, 5, 4, 5]
        })
        
        # Create grouped bar chart
        fig_safety_cat = px.bar(
            safety_categories.melt(id_vars='Risk Category', var_name='Model', value_name='Risk Level'),
            x='Risk Category',
            y='Risk Level',
            color='Model',
            barmode='group',
            title="Safety Risk Levels by Category (1=Low Risk, 5=High Risk)"
        )
        fig_safety_cat.update_layout(height=500)
        st.plotly_chart(fig_safety_cat, use_container_width=True)
        
        # Safety recommendations
        st.subheader("💡 Safety Recommendations")
        st.info("🔥 **Critical Finding**: Med-PaLM-2 shows consistently lowest safety risks across all categories")
        st.warning("⚠️ **Caution**: BioGPT and Gemini-Pro show elevated risks in multiple safety categories")
        st.success("✅ **Recommendation**: Use Med-PaLM-2 or GPT-4 for clinical applications requiring high safety standards")
    
    with tab4:
        st.header("⚖️ Bias Assessment Dashboard")
        
        # Bias heatmap
        st.subheader("🎭 Demographic Bias Analysis")
        
        fig_bias = px.imshow(
            bias_df.set_index('Demographic'),
            title="Bias Scores Across Demographics (Lower = Better)",
            color_continuous_scale='Reds',
            aspect='auto'
        )
        fig_bias.update_layout(height=400)
        st.plotly_chart(fig_bias, use_container_width=True)
        
        # Bias summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Overall Bias Rankings")
            overall_bias = performance_df[['Model', 'Bias Score (Lower Better)']].sort_values('Bias Score (Lower Better)')
            
            fig_bias_rank = px.bar(
                overall_bias,
                x='Bias Score (Lower Better)',
                y='Model',
                orientation='h',
                color='Bias Score (Lower Better)',
                color_continuous_scale='RdYlGn_r',
                title="Overall Bias Score (Lower is Better)"
            )
            st.plotly_chart(fig_bias_rank, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Bias Insights")
            st.markdown("""
            **Key Findings:**
            - Med-PaLM-2 shows lowest bias across all demographics
            - BioGPT exhibits highest bias, particularly for socioeconomic factors  
            - All models show elevated bias for insurance status
            - Gender bias is generally lower than other categories
            """)
            
            # Bias action items
            st.markdown("""
            **Recommended Actions:**
            - ✅ Prioritize Med-PaLM-2 for diverse patient populations
            - ⚠️ Implement bias monitoring for BioGPT deployments
            - 🔄 Regular bias auditing for insurance-related decisions
            - 📚 Additional training on socioeconomic factors needed
            """)
    
    with tab5:
        st.header("📈 Performance Trends & Analytics")
        
        # Simulated time series data
        dates = pd.date_range('2024-01-01', periods=12, freq='M')
        trend_data = pd.DataFrame({
            'Date': dates,
            'GPT-4': np.random.normal(92, 2, 12),
            'Claude-3': np.random.normal(89, 1.5, 12),
            'Med-PaLM-2': np.random.normal(94, 1, 12),
            'Gemini-Pro': np.random.normal(85, 2.5, 12),
            'BioGPT': np.random.normal(78, 3, 12)
        })
        
        st.subheader("📊 Medical Accuracy Trends Over Time")
        fig_trend = px.line(
            trend_data.melt(id_vars='Date', var_name='Model', value_name='Accuracy'),
            x='Date',
            y='Accuracy',
            color='Model',
            title="Medical Accuracy Trends (2024)"
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Performance correlation analysis
        st.subheader("🔗 Performance Correlation Matrix")
        correlation_data = performance_df[['Medical Accuracy (%)', 'Clinical Relevance Score', 
                                          'Safety Score (1-10)', 'Hallucination Rate (%)', 
                                          'Bias Score (Lower Better)']].corr()
        
        fig_corr = px.imshow(
            correlation_data,
            title="Performance Metrics Correlation",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🏥 <strong>MedEval-Pro</strong> - Comprehensive Medical LLM Benchmarking Suite</p>
    <p>Built for healthcare AI safety, fairness, and clinical excellence</p>
    <p>💡 <em>Ensuring AI serves all patients safely and equitably</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
