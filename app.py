import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# 1. Page Configuration (Professional Clean Look)
st.set_page_config(
    page_title="DermalAI | Advanced Skin Disease Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Touch
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .main-title { color: #1e3a8a; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Navigation
st.sidebar.image("https://flaticon.com", width=80)
st.sidebar.markdown("<h2 style='color: #1e3a8a;'>DermalAI Panel</h2>", unsafe_allow_html=True)
app_mode = st.sidebar.selectbox("Navigate System", ["🔍 AI Analysis Dashboard", "📄 Dataset & Tech Specs", "👨‍💻 Expert Info"])

# 3. Model Loading (With Spinners)
@st.cache_resource
def load_medical_model():
    return tf.keras.models.load_model('skin_disease_model.keras')
with st.spinner("Initializing Deep Learning Engine..."):
    try:
        model = load_medical_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Engine Configuration Error: {e}")
        model_loaded = False

# Class Indices based on your Kaggle Train Step
class_labels = [
    'acne_rosacea', 'atopic_dermatitis', 'fungal_infection', 
    'healthy', 'lupus_connective_tissue_disease', 
    'pigmentation_disorder', 'systemic_disease'
]

# ----------------- TAB 1: AI DIAGNOSIS DASHBOARD -----------------
if app_mode == "🔍 AI Analysis Dashboard":
    st.markdown("<h1 class='main-title'>🩺 DermalAI: Clinical Grade Skin Classifier</h1>", unsafe_allow_html=True)
    st.write("An advanced Computer Vision application powered by MobileNetV2 Architecture for rapid skin lesion profiling.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### 📤 Upload Clinical Specimen")
        uploaded_file = st.file_uploader("Upload Image (JPEG/PNG Format)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Target Clinical Image", use_container_width=True)
            
            # Preprocessing
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized)
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape == 4:
                img_array = img_array[:,:,:3]
            img_array = img_array / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)

    with col2:
        if uploaded_file is not None and model_loaded:
            st.markdown("### 📊 Diagnostic Evaluation Results")
            
            # Run inference
            predictions = model.predict(img_tensor, verbose=0)[0]
            pred_idx = np.argmax(predictions)
            confidence = predictions[pred_idx] * 100
            detected_disease = class_labels[pred_idx]

            # Metric Cards
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"<div class='metric-box'><h5>Analysis Outcome</h5><h2>{detected_disease.replace('_', ' ').title()}</h2></div>", unsafe_allow_html=True)
            with m2:
                status_color = "#10b981" if detected_disease == 'healthy' else "#f59e0b"
                st.markdown(f"<div class='metric-box'><h5>Confidence Level</h5><h2 style='color: {status_color};'>{confidence:.2f}%</h2></div>", unsafe_allow_html=True)

            # Bar Chart: Probability Distribution Matrix
            st.write("**Full Diagnostics Probability Distribution Matrix:**")
            chart_data = pd.DataFrame({
                'Condition': [label.replace('_', ' ').title() for label in class_labels],
                'Probability (%)': [float(p * 100) for p in predictions]
            }).sort_values(by='Probability (%)', ascending=True)
            
            st.bar_chart(data=chart_data, x='Condition', y='Probability (%)', horizontal=True, color="#1e3a8a")

            # Clinical Advisory block
            st.markdown("### 💡 Recommended Clinical Steps")
            if detected_disease == 'healthy':
                st.success("✅ Skin screening normal. Maintain standard skincare routines and hydration.")
            else:
                st.warning(f"⚠️ **Action Required:** The model shows signs consistent with **{detected_disease.replace('_', ' ').title()}**. We recommend consulting a registered dermatologist for an official evaluation or localized biopsy.")
                st.info("ℹ️ *Disclaimer: This analysis dashboard is intended for academic validation and screening support. It does not replace professional medical advice or physical pathology evaluation.*")
        else:
            st.info("Waiting for image input... Please upload a screening image on the left panel to begin analysis.")

# ----------------- TAB 2: DATASET SPECS -----------------
elif app_mode == "📄 Dataset & Tech Specs":
    st.markdown("<h1 class='main-title'>📄 Model Architecture & Dataset Parameters</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Data Engineering
        * **Target Scope:** 7 Class Multiclass Classification
        * **Evaluation Pipeline Split:** 70% Training | 10% Validation | 20% Testing Split Strategy
        * **Image Preprocessing:** 224x224 Resolution Matrices with Min-Max Pixel Normalization ([0,1] Scale)
        * **Augmentation Profile:** Layered Random Shifting, Geometric Rotations, and Real-time Horizontal Flips
        """)
    with col2:
        st.markdown("""
        ### Deep Learning Network Specs
        * **Core Framework:** TensorFlow Keras Native API
        * **Base Neural Net:** MobileNetV2 (Pre-trained on ImageNet Feature Weights)
        * **Global Pooling Strategy:** 2D Global Average Pooling Optimization
        * **Optimization Engine:** Adam Stochastic Gradient Descent Optimizer
        * **Loss Minimization Function:** Categorical Cross-Entropy Matrix
        """)

# ----------------- TAB 3: DEVELOPER INFO -----------------
elif app_mode == "👨‍💻 Expert Info":
    st.markdown("<h1 class='main-title'>👨‍💻 Project Developer Profile</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    ### Data Science Portfolio Entry
    This application represents a full pipeline deployment of a Biomedical Computer Vision Solution.
    
    * **Workflow Demonstrated:** Automated File System Mapping, Class Unbalance Filtering, Image Data Normalization Generators, Transfer Learning Implementation, and Final Web UI Integration.
    * **Core Technologies Used:** Python, TensorFlow, Scikit-Learn, Pandas, Streamlit Cloud API, GitHub Pipelines.
    
    *Developed with a focus on clean deployment practices and medical-grade analytical UI layouts.*
    """)
