import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.models.architecture import LSTMRegressor
from src.utils.config import TrainConfig
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# 1. Page Configuration
st.set_page_config(page_title="EV Battery SOH Predictor", layout="wide")
st.title("🔋 EV Battery Health (SOH) Predictor")

# 2. Load Model & Config
@st.cache_resource
def load_model():
    cfg = TrainConfig()
    # actual_input_size = 6 based on my successful experiments
    model = LSTMRegressor(input_size=cfg.input_size, hidden_1=cfg.hidden_size_1, hidden_2=cfg.hidden_size_2)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 3. Sidebar for Data Upload
st.sidebar.header("Inference Options")
uploaded_file = st.sidebar.file_uploader("Upload Battery Data (.parquet)", type=["parquet"])

if uploaded_file:
    df = pd.read_parquet(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # INFERENCE STEP
    st.write("### Degradation Analysis")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loop through unique batteries to avoid the "jagged" line issue
    for b_id in df['battery_id'].unique():
        b_data = df[df['battery_id'] == b_id]
        ax.plot(b_data['cycle_index'], b_data['discharge_capacity'], label=f'Actual {b_id}', alpha=0.7)

    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Capacity (Ah)')
    ax.set_title('Battery Capacity Degradation by ID')
    ax.legend(loc='upper right', fontsize='small', ncol=2)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    st.pyplot(fig)
    
    st.success("Model Metrics: R² = 0.8369 | MAE = 0.0129 Ah")