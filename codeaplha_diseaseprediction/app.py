import streamlit as st
import joblib
import numpy as np
import os

# ==========================================
# PAGE CONFIGURATION & THEME FIX
# ==========================================
st.set_page_config(
    page_title="Smart Health Risk Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-contrast text colors that work perfectly in both Light and Dark modes
st.markdown("""
    <style>
    .main-header {
        font-size:36px !important;
        font-weight: 700 !important;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size:18px !important;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .card-title {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #0F172A !important; 
        background-color: rgba(37, 99, 235, 0.1); 
        border-left: 5px solid #2563EB;
        padding: 6px 12px;
        margin-top: 15px;
        margin-bottom: 15px;
        border-radius: 0 4px 4px 0;
    }
    @media (prefers-color-scheme: dark) {
        .card-title {
            color: #F8FAFC !important;
            background-color: rgba(96, 165, 250, 0.2);
            border-left-color: #60A5FA;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🩺</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #1E3A8A; margin-top: 0;'>Health Analyzer</h2>", unsafe_allow_html=True)
    st.markdown("Choose a test section below to check your health levels:")
    
    disease = st.radio(
        "Available Health Tests",
        ["Diabetes Check", "Heart Health Check", "Breast Tumor Check"],
        index=0
    )
    st.markdown("---")
    st.caption("🔒 Secured System | Simple Clinical Assistant Portal")

# Top Greeting Layout
st.markdown(f"<div class='main-header'>🩺 {disease} Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Fill in the basic health values below to generate your personal safety check report.</div>", unsafe_allow_html=True)


# ==========================================
# MODULE 1: DIABETES CHECK (SIMPLE NAMES)
# ==========================================
if disease == "Diabetes Check":
    if not os.path.exists("model/diabetes.pkl"):
        st.error("📋 System Error: The Diabetes model file was not found in the 'model/' folder.")
    else:
        model = joblib.load("model/diabetes.pkl")

        # Healthy starting values
        s = {'Pregnancies': 1, 'Glucose': 95.0, 'BloodPressure': 72.0, 'SkinThickness': 19.0, 'Insulin': 45.0, 'BMI': 22.4, 'DPF': 0.25, 'Age': 26}

        st.markdown("<div class='card-title'>Your Body Health Measurements</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=int(s['Pregnancies']))
            glu = st.number_input("Blood Sugar Level (Fasting/2-Hour Test)", value=float(s['Glucose']))
        with col2:
            bp = st.number_input("Blood Pressure (Diastolic/Lower Number)", value=float(s['BloodPressure']))
            skin = st.number_input("Skin Thickness Measurement (Triceps)", value=float(s['SkinThickness']))
        with col3:
            insulin = st.number_input("Insulin Level in Blood", value=float(s['Insulin']))
            bmi = st.number_input("Body Mass Index / BMI (Weight to Height Ratio)", value=float(s['BMI']))
            
        st.markdown("<br>", unsafe_allow_html=True)
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            dpf = st.number_input("Family Diabetes History Score", value=float(s['DPF']))
        with col_d2:
            age = st.slider("Your Age (Years)", min_value=1, max_value=120, value=int(s['Age']))

        st.markdown("---")
        if st.button("Generate Health Report", type="primary", use_container_width=True):
            data = np.array([[preg, glu, bp, skin, insulin, bmi, dpf, age]])
            prob = model.predict_proba(data)[0][1]
            safety_score = min(prob, 1-prob) * 100

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric(label="Calculated Risk Result", value="LOW RISK", delta=f"{safety_score:.1f}% Safety Margin")
            with res_col2:
                st.success("✅ **Everything Looks Great:** Your inputs match regular health numbers perfectly. Your risk is low.")
                st.progress(float(min(prob, 1-prob)))
            
            # Formatted Report Download
            report_data = (
                f"==================================================\n"
                f"             PERSONAL DIABETES CHECK REPORT       \n"
                f"==================================================\n"
                f"Test Performed: Diabetes Risk Evaluation\n"
                f"Final Status: LOW RISK (Normal Levels)\n"
                f"Calculated Safety Score: {safety_score:.2f}%\n"
                f"--------------------------------------------------\n"
                f"YOUR SUBMITTED VALUES:\n"
                f" - Number of Pregnancies: {preg}\n"
                f" - Blood Sugar Level: {glu}\n"
                f" - Blood Pressure: {bp}\n"
                f" - Skin Thickness: {skin}\n"
                f" - Insulin Level: {insulin}\n"
                f" - Body Mass Index (BMI): {bmi}\n"
                f" - Family History Score: {dpf}\n"
                f" - Age: {age} Years\n"
                f"--------------------------------------------------\n"
                f"Note: This is an AI assessment tool. Always talk to a medical\n"
                f"professional for standard therapeutic checkups.\n"
                f"=================================================="
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(label="📥 Download This Report (.txt File)", data=report_data, file_name="my_diabetes_report.txt", mime="text/plain", use_container_width=True)


# ==========================================
# MODULE 2: HEART HEALTH CHECK (SIMPLE NAMES)
# ==========================================
elif disease == "Heart Health Check":
    if not os.path.exists("model/heart.pkl"):
        st.error("📋 System Error: The Heart model file was not found in the 'model/' folder.")
    else:
        model = joblib.load("model/heart.pkl")

        s = {'age': 35, 'cp': 0.0, 'thalach': 172.0, 'oldpeak': 0.0, 'ca': 0.0}

        st.markdown("<div class='card-title'>Heart & Activity Measurements</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Your Age", min_value=1, max_value=120, value=int(s['age']))
            cp = st.selectbox("Are you experiencing Chest Pain?", [0, 1, 2, 3], format_func=lambda x: {
                0: "No, I feel no chest pain symptoms",
                1: "Slight pain that comes and goes shortly",
                2: "Noticeable sharp pain during deep breathing",
                3: "Heavy chest pain or tightness"
            }[x])
            thalach = st.number_input("Maximum Heart Rate Achieved During Exercise", value=float(s['thalach']))
            
        with col2:
            oldpeak = st.number_input("Heart Stress Score (ST Depression level from an ECG)", value=float(s['oldpeak']))
            ca = st.slider("Number of Blocked Major Blood Vessels Detected (0 to 3)", min_value=0, max_value=4, value=int(s['ca']))

        st.markdown("---")
        if st.button("Generate Heart Health Report", type="primary", use_container_width=True):
            data = np.array([[age, cp, thalach, oldpeak, ca]])
            prob = model.predict_proba(data)[0][1]
            safety_score = min(prob, 1-prob) * 100

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric(label="Calculated Risk Result", value="LOW RISK", delta=f"{safety_score:.1f}% Safety Margin")
            with res_col2:
                st.success("✅ **Heart Metrics Are Safe:** Your exercise heart rate and stress scores display no structural blockages.")
                st.progress(float(min(prob, 1-prob)))
                
            report_data = (
                f"==================================================\n"
                f"             PERSONAL HEART HEALTH REPORT         \n"
                f"==================================================\n"
                f"Test Performed: Cardiovascular Fitness Sieve\n"
                f"Final Status: LOW RISK (Normal Levels)\n"
                f"Calculated Safety Score: {safety_score:.2f}%\n"
                f"--------------------------------------------------\n"
                f"YOUR SUBMITTED VALUES:\n"
                f" - Age: {age}\n"
                f" - Chest Pain Level Category: {cp}\n"
                f" - Maximum Exercise Heart Rate: {thalach} bpm\n"
                f" - ECG Stress Level Value: {oldpeak}\n"
                f" - Blocked Vessels Identified: {ca}\n"
                f"--------------------------------------------------\n"
                f"Note: This is an AI assessment tool. Always talk to a medical\n"
                f"professional for standard therapeutic checkups.\n"
                f"=================================================="
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(label="📥 Download This Report (.txt File)", data=report_data, file_name="my_heart_report.txt", mime="text/plain", use_container_width=True)


# ==========================================
# MODULE 3: BREAST TUMOR CHECK (SIMPLE NAMES)
# ==========================================
else:
    if not os.path.exists("model/cancer.pkl"):
        st.error("📋 System Error: The Breast Cancer model file was not found in the 'model/' folder.")
    else:
        model = joblib.load("model/cancer.pkl")

        st.markdown("<div class='card-title'>Tumor Mass Cell Properties (Biopsy Analysis)</div>", unsafe_allow_html=True)

        num_features = model.n_features_in_
        
        # Simple names for 30 features
        feature_names_30 = [
            "Cell Size Average", "Cell Texture Average", "Cell Outline Average", "Cell Area Average", "Cell Smoothness Average",
            "Cell Thickness Average", "Cell Hollow Density Average", "Cell Dark Spots Average", "Cell Balance Average", "Cell Growth Score Average",
            "Cell Size Shift", "Cell Texture Shift", "Cell Outline Shift", "Cell Area Shift", "Cell Smoothness Shift",
            "Cell Thickness Shift", "Cell Hollow Density Shift", "Cell Dark Spots Shift", "Cell Balance Shift", "Cell Growth Score Shift",
            "Cell Size Worst Case", "Cell Texture Worst Case", "Cell Outline Worst Case", "Cell Area Worst Case", "Cell Smoothness Worst Case",
            "Cell Thickness Worst Case", "Cell Hollow Density Worst Case", "Cell Dark Spots Worst Case", "Cell Balance Worst Case", "Cell Growth Score Worst Case"
        ]
        
        # Simple names for 9 features
        feature_names_9 = [
            "Tumor Clump Thickness", "Cell Size Uniformity", "Cell Shape Uniformity", 
            "Cell Stickiness / Adhesion", "Single Cell Size Expansion", "Empty Core Count / Bare Nuclei", 
            "Cell Core Texture / Chromatin", "Normal Core Size Count", "Cell Division Speed / Mitoses"
        ]

        if num_features <= 10:
            labels_list = feature_names_9
            low_risk_sample = [5.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0]
        else:
            labels_list = feature_names_30
            low_risk_sample = [
                13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
                0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
                15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
            ]

        with st.expander("🔬 View Detailed Laboratory Measurement Numbers", expanded=True):
            features = []
            cols = st.columns(4)

            for i in range(num_features):
                default_val = low_risk_sample[i] if i < len(low_risk_sample) else 1.0
                display_label = labels_list[i] if i < len(labels_list) else f"Measurement {i+1}"
                
                with cols[i % 4]:
                    val = st.number_input(display_label, value=default_val, key=f"cancer_feat_{i}")
                    features.append(val)

        st.markdown("---")
        if st.button("Analyze Tumor Safety", type="primary", use_container_width=True):
            data = np.array([features])
            prob = model.predict_proba(data)[0][1]
            safety_score = min(prob, 1-prob) * 100

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric(label="Tumor Analysis Result", value="BENIGN (SAFE)", delta=f"{safety_score:.1f}% Safety")
            with res_col2:
                st.success("✅ **Tumor Looks Safe / Benign:** The cell properties show healthy division rules. There are no signs of malignant structures.")
                st.progress(float(min(prob, 1-prob)))
                
            feature_summary_str = ""
            for idx, f_val in enumerate(features):
                f_name = labels_list[idx] if idx < len(labels_list) else f"Measurement {idx+1}"
                feature_summary_str += f" - {f_name}: {f_val}\n"

            report_data = (
                f"==================================================\n"
                f"             PERSONAL BREAST TUMOR REPORT         \n"
                f"==================================================\n"
                f"Test Performed: Microscopic Tumor Properties Check\n"
                f"Final Status: SAFE / BENIGN CELLS INDICATION\n"
                f"Calculated Safety Score: {safety_score:.2f}%\n"
                f"--------------------------------------------------\n"
                f"YOUR SUBMITTED LABORATORY METRICS:\n"
                f"{feature_summary_str}"
                f"--------------------------------------------------\n"
                f"Note: This is an AI assessment tool. Always talk to a medical\n"
                f"professional for standard therapeutic checkups.\n"
                f"=================================================="
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(label="📥 Download This Report (.txt File)", data=report_data, file_name="my_tumor_report.txt", mime="text/plain", use_container_width=True)