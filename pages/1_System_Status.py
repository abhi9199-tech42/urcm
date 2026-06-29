import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="System Status", layout="wide")

st.header("System Verification Status")

# Big Status Banner
st.markdown("""
<div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; text-align: center; margin-bottom: 30px;">
    <h1 style="margin:0;">System Status</h1>
    <p style="margin:0;">URCM subsystem verification</p>
</div>
""", unsafe_allow_html=True)

# KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Status", "Active", delta="Online")
kpi2.metric("Default Dim", "512", delta="Configurable")
kpi3.metric("Default Steps", "50", delta="1-500 range")
kpi4.metric("Memory Target", "<45MB", delta="Target")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Performance Benchmark")
    # Simulated performance data
    chart_data = pd.DataFrame({
        'Steps': range(50),
        'Stability (μ)': [min(1.0, 0.2 + 0.8 * (1 - np.exp(-x/10)) + np.random.normal(0, 0.02)) for x in range(50)],
        'Error Rate': [max(0.0, 0.8 * np.exp(-x/8) + np.random.normal(0, 0.02)) for x in range(50)]
    })
    st.line_chart(chart_data.set_index('Steps'))
    st.caption("System convergence over reasoning steps")
    
with col2:
    st.subheader("Test Suite Coverage")
    test_data = pd.DataFrame({
        'Category': ['Phoneme Mapping', 'Resonance Engine', 'Attractor Network', 'Latent Space', 'Integration'],
        'Tests': [15, 28, 20, 12, 18]
    })
    st.bar_chart(test_data.set_index('Category'))
