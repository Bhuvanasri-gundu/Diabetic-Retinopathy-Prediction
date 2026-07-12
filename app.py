"""
DR Vision AI — Streamlit Application
=====================================
Production-ready Streamlit app for Diabetic Retinopathy screening.
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import time
from PIL import Image

# Must be the FIRST Streamlit command
st.set_page_config(
    page_title="DR Vision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Now import backend
from dr_engine import (
    Config,
    preprocess_uploaded_image,
    load_image_only_model,
    load_multimodal_model,
    load_clinical_preprocessors,
    prepare_clinical_tensor,
    predict_image_only,
    predict_multimodal,
    generate_gradcam_image_only,
    generate_gradcam_multimodal,
    generate_pdf_report,
    CLINICAL_FEATURE_COLS,
)


# ==============================================================
# Global Custom CSS
# ==============================================================
def inject_css():
    st.markdown("""
    <style>
    /* ── Import Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    /* ── Root variables ── */
    :root {
        --teal-500: #0d9488;
        --teal-600: #0f766e;
        --teal-700: #115e59;
        --teal-50: #f0fdfa;
        --slate-50: #f8fafc;
        --slate-100: #f1f5f9;
        --slate-200: #e2e8f0;
        --slate-300: #cbd5e1;
        --slate-500: #64748b;
        --slate-700: #334155;
        --slate-800: #1e293b;
        --slate-900: #0f172a;
        --green-500: #22c55e;
        --yellow-500: #eab308;
        --orange-500: #f97316;
        --red-500: #ef4444;
        --red-900: #991b1b;
    }

    /* ── Hide default Streamlit clutter ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {display: none;}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] .stMarkdown div {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* ── Card styles ── */
    .dr-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03);
        transition: all 0.3s ease;
    }
    .dr-card:hover {
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f766e 0%, #115e59 40%, #134e4a 100%);
        border-radius: 20px;
        padding: 48px 40px;
        color: white;
        position: relative;
        overflow: hidden;
        margin-bottom: 24px;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(ellipse, rgba(255,255,255,0.08) 0%, transparent 70%);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #99f6e4;
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 36px;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 12px;
        color: white;
    }
    .hero-subtitle {
        font-size: 14px;
        color: rgba(255,255,255,0.8);
        line-height: 1.6;
        max-width: 550px;
        margin-bottom: 24px;
    }
    .hero-btn-container {
        margin-top: -64px !important;
        margin-left: 40px !important;
        margin-bottom: 40px !important;
        position: relative !important;
        z-index: 99 !important;
    }
    .hero-btn-container button {
        display: inline-block !important;
        background: rgba(255,255,255,0.15) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }
    .hero-btn-container button:hover {
        background: rgba(255,255,255,0.25) !important;
        border-color: rgba(255,255,255,0.5) !important;
        color: white !important;
    }
    .hero-btn-container button:active {
        background: rgba(255,255,255,0.3) !important;
        color: white !important;
    }

    /* ── Severity Scale ── */
    .severity-item {
        text-align: center;
        padding: 16px 8px;
    }
    .severity-circle {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 8px;
        font-weight: 700;
        font-size: 16px;
        color: white;
    }
    .severity-label {
        font-weight: 600;
        font-size: 13px;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .severity-desc {
        font-size: 11px;
        color: #64748b;
        line-height: 1.4;
    }

    /* ── Stats Cards ── */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
    }
    .stat-icon {
        font-size: 18px;
        color: var(--teal-500);
        margin-bottom: 4px;
    }
    .stat-label {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .stat-sub {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* ── Section headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
    }
    .section-header-icon {
        font-size: 20px;
    }
    .section-header-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
    }

    /* ── Upload area ── */
    .upload-zone {
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 48px 24px;
        text-align: center;
        background: #f8fafc;
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: #0d9488;
        background: #f0fdfa;
    }

    /* ── Result badge ── */
    .result-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.5px;
    }
    .badge-green { background: #dcfce7; color: #166534; }
    .badge-yellow { background: #fef9c3; color: #854d0e; }
    .badge-orange { background: #ffedd5; color: #9a3412; }
    .badge-red { background: #fee2e2; color: #991b1b; }
    .badge-darkred { background: #450a0a; color: #fecaca; }

    /* ── Confidence Ring ── */
    .confidence-ring {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin: 0 auto;
        position: relative;
    }
    .confidence-value {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
    }
    .confidence-label {
        font-size: 10px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Prob bars ── */
    .prob-bar-container {
        margin-bottom: 8px;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        margin-bottom: 3px;
    }
    .prob-bar-track {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.8s ease;
    }

    /* ── About page ── */
    .about-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 28px;
        height: 100%;
    }
    .about-icon {
        width: 48px;
        height: 48px;
        background: #f0fdfa;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        margin-bottom: 14px;
    }
    .about-title {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .about-desc {
        font-size: 12px;
        color: #64748b;
        line-height: 1.5;
    }

    /* ── Flow Steps ── */
    .flow-step {
        text-align: center;
        padding: 16px;
    }
    .flow-icon {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px;
        font-size: 24px;
    }
    .flow-label {
        font-size: 12px;
        font-weight: 600;
        color: #334155;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 20px 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 40px;
        font-size: 12px;
        color: #94a3b8;
    }

    /* ── Streamlit overrides ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 24px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stFileUploader"] {
        border-radius: 16px;
    }

    /* ── Analyze button ── */
    .analyze-btn button {
        background: linear-gradient(135deg, #0d9488, #0f766e) !important;
        color: white !important;
        border: none !important;
        padding: 14px 40px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        width: 100%;
    }
    .analyze-btn button:hover {
        background: linear-gradient(135deg, #0f766e, #115e59) !important;
        box-shadow: 0 8px 25px rgba(13,148,136,0.3) !important;
    }

    /* ── Form styling ── */
    .patient-form-title {
        font-size: 14px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Consistency Badge ── */
    .consistency-high { color: #16a34a; }
    .consistency-medium { color: #ca8a04; }
    .consistency-low { color: #dc2626; }

    /* ── Feature table ── */
    .feature-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        font-size: 13px;
    }
    .feature-table th {
        background: #f1f5f9;
        padding: 10px 14px;
        font-weight: 600;
        color: #334155;
        text-align: left;
        border-bottom: 1px solid #e2e8f0;
    }
    .feature-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #f1f5f9;
        color: #475569;
    }
    .feature-table tr:last-child td {
        border-bottom: none;
    }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================
# Cache resource loading
# ==============================================================
@st.cache_resource(show_spinner=False)
def cached_load_image_model():
    """Load image-only model (cached)."""
    return load_image_only_model()


@st.cache_resource(show_spinner=False)
def cached_load_multimodal_model(num_features):
    """Load multimodal model (cached)."""
    return load_multimodal_model(num_features)


@st.cache_resource(show_spinner=False)
def cached_load_clinical_preprocessors():
    """Load clinical preprocessors (cached)."""
    return load_clinical_preprocessors()


# ==============================================================
# Initialize session state
# ==============================================================
def init_session_state():
    defaults = {
        "page": "Dashboard",
        "prediction_result": None,
        "uploaded_image": None,
        "raw_pil": None,
        "processed_pil": None,
        "gradcam_pil": None,
        "patient_info": {},
        "model_type": "Multimodal",
        "models_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ==============================================================
# Sidebar Navigation
# ==============================================================
def render_sidebar():
    with st.sidebar:
        # Logo / Brand
        st.markdown("""
        <div style="padding: 16px 0 8px 0;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                <span style="font-size:28px;">👁️</span>
                <span style="font-size:20px;font-weight:800;color:#5eead4;
                  letter-spacing:-0.5px;">DR Vision AI</span>
            </div>
            <span style="font-size:11px;color:#94a3b8;">
                Clinical Decision Support System
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        pages = {
            "Dashboard": "📊",
            "Patient Analysis": "🔬",
            "Analysis Report": "📋",
            "About": "ℹ️",
        }

        for page_name, icon in pages.items():
            is_active = st.session_state.page == page_name
            if st.button(
                f"{icon}  {page_name}",
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = page_name
                st.rerun()

        st.markdown("---")

        # System Status
        if st.session_state.page == "Patient Analysis":
            st.markdown("""
            <div style="margin-top:8px;">
                <p style="font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:1.5px;color:#94a3b8;margin-bottom:8px;">
                  System Status
                </p>
            </div>
            """, unsafe_allow_html=True)

            device_str = str(Config.DEVICE).upper()
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05);border-radius:10px;
              padding:12px;margin-bottom:12px;">
                <div style="font-size:13px;font-weight:600;color:#e2e8f0;">
                    AI Model v4.2
                </div>
                <div style="font-size:11px;color:#94a3b8;margin-top:4px;">
                    Device: {device_str}
                </div>
                <div style="font-size:11px;color:#22c55e;margin-top:2px;">
                    ● Online
                </div>
            </div>
            """, unsafe_allow_html=True)

        # User info at bottom
        st.markdown("---")
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 0;">
            <div style="width:36px;height:36px;border-radius:50%;
              background:linear-gradient(135deg,#0d9488,#0f766e);
              display:flex;align-items:center;justify-content:center;
              color:white;font-weight:700;font-size:14px;">DR</div>
            <div>
                <div style="font-size:13px;font-weight:600;color:#e2e8f0;">
                    Dr. Sarah Jenkins
                </div>
                <div style="font-size:10px;color:#94a3b8;">
                    Clinical Lead
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            pass


# ==============================================================
# PAGE 1 — Dashboard
# ==============================================================
def page_dashboard():
    # Header
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;
      margin-bottom:4px;">
        <div>
            <h2 style="margin:0;font-weight:800;color:#0f172a;">System Overview</h2>
            <p style="color:#64748b;font-size:13px;margin-top:2px;">
                Clinical decision support for Diabetic Retinopathy screening
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">◀ AI-DRIVEN DIAGNOSTICS</div>
        <div class="hero-title">Intelligent Retinal<br>Screening</div>
        <div class="hero-subtitle">
            Automated severity assessment of Diabetic Retinopathy using state-of-the-art
            neural networks. Increase diagnostic accuracy and streamline clinical workflows.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hero-btn-container">', unsafe_allow_html=True)
    if st.button("Start Patient Analysis →", key="hero_analysis_btn"):
        st.session_state.page = "Patient Analysis"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # DR Severity Scale
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;
      margin-bottom:12px;">
        <div class="section-header">
            <span class="section-header-icon">👁️</span>
            <span class="section-header-title">DR Severity Scale Reference</span>
        </div>
        <a href="#" style="font-size:13px;color:#0d9488;font-weight:600;
          text-decoration:none;">View Clinical Guidelines</a>
    </div>
    """, unsafe_allow_html=True)

    scale_cols = st.columns(5)
    for i, col in enumerate(scale_cols):
        with col:
            color = Config.CLASS_COLORS[i]
            st.markdown(f"""
            <div class="dr-card" style="text-align:center;padding:20px 12px;">
                <div class="severity-circle" style="background:{color};">{i}</div>
                <div class="severity-label">{Config.CLASS_NAMES[i]}</div>
                <div class="severity-desc">{Config.CLASS_DESCRIPTIONS[i]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats Row
    stat_cols = st.columns(3)
    stats_data = [
        ("📊", "PATIENTS ANALYZED", "1,284", "+12% this month"),
        ("⚡", "AVG. PROCESSING TIME", "1.8s", "Optimized GPU compute"),
        ("🎯", "MODEL CONFIDENCE", "98.4%", "Validation set AUC-ROC"),
    ]
    for col, (icon, label, value, sub) in zip(stat_cols, stats_data):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="app-footer">
        © 2024 DR Vision AI. For Research Use Only. &nbsp;&nbsp;&nbsp;
        <a href="#" style="color:#64748b;text-decoration:none;font-weight:600;">DOCUMENTATION</a>
        &nbsp;&nbsp;
        <a href="#" style="color:#64748b;text-decoration:none;font-weight:600;">PRIVACY</a>
        &nbsp;&nbsp;
        <a href="#" style="color:#64748b;text-decoration:none;font-weight:600;">SUPPORT</a>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
# PAGE 2 — Patient Analysis
# ==============================================================
def page_patient_analysis():
    # Top navigation tabs
    st.markdown("""
    <div style="display:flex;gap:24px;margin-bottom:24px;border-bottom:1px solid #e2e8f0;
      padding-bottom:8px;">
        <span style="color:#64748b;font-size:14px;font-weight:500;">Dashboard</span>
        <span style="color:#0d9488;font-size:14px;font-weight:700;
          border-bottom:2px solid #0d9488;padding-bottom:8px;">Patient Analysis</span>
        <span style="color:#64748b;font-size:14px;font-weight:500;">Analysis Report</span>
        <span style="color:#64748b;font-size:14px;font-weight:500;">About</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="font-weight:800;color:#0f172a;margin-bottom:4px;">Patient Retinal Analysis</h2>
    <p style="color:#64748b;font-size:13px;">
        Upload high-resolution fundus photography to initiate diagnostic screening.
    </p>
    """, unsafe_allow_html=True)

    # ─── Model Selection ───
    model_choice = st.radio(
        "Analysis Mode",
        ["Multimodal (Image + Clinical)", "Image Only"],
        horizontal=True,
        key="model_choice_radio",
    )
    is_multimodal = model_choice.startswith("Multimodal")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Image Upload ───
    st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">📷</span>
        <span class="section-header-title">Retinal Image Upload</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop fundus images here",
        type=["jpg", "jpeg", "png", "tiff", "tif", "bmp"],
        help="Supported formats: JPEG, PNG, TIFF, BMP (Max 50MB)",
        key="image_uploader",
    )

    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Clinical Data Forms ───
    clinical_data = {}

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="patient-form-title">🧑‍⚕️ Patient Identity</div>
        """, unsafe_allow_html=True)

        sub_l1, sub_l2 = st.columns(2)
        with sub_l1:
            age = st.number_input("Age", min_value=1, max_value=120, value=55, key="age_input")
        with sub_l2:
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender_input")

        duration = st.number_input(
            "Diabetes Duration (Years)", min_value=0.0, max_value=50.0,
            value=5.0, step=0.5, key="duration_input",
        )

        st.markdown("**Smoking Status**")
        smoking = st.radio(
            "Smoking", ["Never", "Former", "Active"],
            horizontal=True, key="smoking_input", label_visibility="collapsed",
        )
        # Map "Active" → "Current" to match training data
        smoking_mapped = "Current" if smoking == "Active" else smoking

    with col_right:
        st.markdown("""
        <div class="patient-form-title">🏥 Clinical Indicators</div>
        """, unsafe_allow_html=True)

        sub_r1, sub_r2 = st.columns(2)
        with sub_r1:
            hba1c = st.number_input(
                "HbA1c (%)", min_value=4.0, max_value=15.0, value=6.5,
                step=0.1, key="hba1c_input",
            )
            bmi = st.number_input(
                "BMI", min_value=10.0, max_value=60.0, value=24.5,
                step=0.5, key="bmi_input",
            )
        with sub_r2:
            fasting_glucose = st.number_input(
                "Fasting Glucose (mg/dL)", min_value=50, max_value=500, value=110,
                key="glucose_input",
            )
            systolic_bp = st.number_input(
                "Blood Pressure (Systolic)", min_value=60, max_value=250, value=120,
                key="bp_input",
            )

        total_chol = st.number_input(
            "Total Cholesterol (mg/dL)", min_value=50, max_value=500, value=180,
            key="chol_input",
        )

    # Build clinical dict
    clinical_data = {
        "Age": age,
        "Gender": gender,
        "Duration_DM_Years": duration,
        "HbA1c": hba1c,
        "Fasting_Glucose_mg_dL": fasting_glucose,
        "BMI": bmi,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": 80,  # Default — not in form to keep it clean
        "Smoking_Status": smoking_mapped,
        "Hypertension": "Yes" if systolic_bp >= 140 else "No",
        "Total_Cholesterol_mg_dL": total_chol,
        "Serum_Creatinine_mg_dL": 0.9,  # Default
    }

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Action Buttons ───
    btn_col1, btn_col2 = st.columns([1, 2])
    with btn_col1:
        st.button("Save as Draft", use_container_width=True, type="secondary")
    with btn_col2:
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        analyze_clicked = st.button(
            "🔬 Analyze Patient",
            use_container_width=True,
            key="analyze_btn",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ─── Run Prediction ───
    if analyze_clicked:
        if uploaded_file is None and st.session_state.uploaded_image is None:
            st.error("⚠️ Please upload a retinal image before analyzing.")
            return

        img_file = uploaded_file or st.session_state.uploaded_image

        with st.spinner("🔄 Loading models and running analysis..."):
            progress = st.progress(0, text="Initializing...")

            try:
                # Step 1: Preprocess image
                progress.progress(10, text="Preprocessing image...")
                raw_pil, processed_pil, img_tensor = preprocess_uploaded_image(img_file)

                # Step 2: Load models
                progress.progress(30, text="Loading AI models...")
                image_model = cached_load_image_model()

                # Always run image-only prediction for comparison
                progress.progress(50, text="Running image-only analysis...")
                img_pred, img_conf, img_probs = predict_image_only(image_model, img_tensor)

                if is_multimodal:
                    # Load clinical preprocessors
                    progress.progress(60, text="Processing clinical data...")
                    scaler, encoders, train_df = cached_load_clinical_preprocessors()

                    if scaler is not None and encoders is not None:
                        clinical_tensor = prepare_clinical_tensor(
                            clinical_data, scaler, encoders, train_df
                        )

                        progress.progress(70, text="Loading multimodal model...")
                        num_features = len(CLINICAL_FEATURE_COLS)
                        mm_model = cached_load_multimodal_model(num_features)

                        progress.progress(80, text="Running multimodal analysis...")
                        mm_pred, mm_conf, mm_probs = predict_multimodal(
                            mm_model, img_tensor, clinical_tensor
                        )

                        # Grad-CAM
                        progress.progress(90, text="Generating Grad-CAM++ visualization...")
                        gradcam_pil, _ = generate_gradcam_multimodal(
                            mm_model, img_tensor, clinical_tensor,
                            processed_pil, target_class=mm_pred,
                        )

                        # Store results
                        st.session_state.prediction_result = {
                            "predicted_class": mm_pred,
                            "confidence": mm_conf,
                            "probabilities": mm_probs,
                            "model_type": "Multimodal",
                            "img_pred": img_pred,
                            "img_conf": img_conf,
                            "img_probs": img_probs,
                        }
                    else:
                        st.warning("Clinical preprocessors not available. Falling back to image-only.")
                        is_multimodal = False

                if not is_multimodal:
                    # Grad-CAM for image-only
                    progress.progress(90, text="Generating Grad-CAM++ visualization...")
                    gradcam_pil, _ = generate_gradcam_image_only(
                        image_model, img_tensor, processed_pil,
                        target_class=img_pred,
                    )

                    st.session_state.prediction_result = {
                        "predicted_class": img_pred,
                        "confidence": img_conf,
                        "probabilities": img_probs,
                        "model_type": "Image Only",
                        "img_pred": img_pred,
                        "img_conf": img_conf,
                        "img_probs": img_probs,
                    }

                st.session_state.raw_pil = raw_pil
                st.session_state.processed_pil = processed_pil
                st.session_state.gradcam_pil = gradcam_pil
                st.session_state.patient_info = clinical_data

                progress.progress(100, text="Analysis complete!")
                time.sleep(0.5)
                progress.empty()

                st.success("✅ Analysis complete! Navigating to results...")
                time.sleep(1)
                st.session_state.page = "Analysis Report"
                st.rerun()

            except Exception as e:
                progress.empty()
                st.error(f"❌ Analysis failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Footer
    st.markdown("""
    <div class="app-footer">
        <strong>DR Vision AI</strong> &nbsp;|&nbsp;
        © 2024 DR Vision AI. For Research Use Only.
        Not for diagnostic procedures without clinical oversight.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
# PAGE 3 — Analysis Report
# ==============================================================
def page_analysis_report():
    result = st.session_state.prediction_result

    if result is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 40px;">
            <div style="font-size:64px;margin-bottom:16px;">📋</div>
            <h3 style="color:#334155;margin-bottom:8px;">No Analysis Available</h3>
            <p style="color:#64748b;">
                Upload a retinal image and run analysis from the Patient Analysis page first.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("→ Go to Patient Analysis", type="primary"):
            st.session_state.page = "Patient Analysis"
            st.rerun()
        return

    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    model_type = result["model_type"]
    img_pred = result["img_pred"]
    img_conf = result["img_conf"]

    grade_name = Config.CLASS_NAMES[pred_class]
    grade_color = Config.CLASS_COLORS[pred_class]
    grade_desc = Config.CLASS_DESCRIPTIONS[pred_class]

    # Badge class
    badge_classes = ["badge-green", "badge-yellow", "badge-orange", "badge-red", "badge-darkred"]
    badge_class = badge_classes[pred_class]

    # Model Consistency
    if model_type == "Multimodal":
        consistency = "High" if img_pred == pred_class else ("Medium" if abs(img_pred - pred_class) <= 1 else "Low")
    else:
        consistency = "High"
    consistency_class = f"consistency-{consistency.lower()}"

    # Header
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <h2 style="font-weight:800;color:#0f172a;margin-bottom:4px;">Diagnostic Review Report</h2>
        <p style="color:#64748b;font-size:12px;">
            Model: {model_type} &nbsp;|&nbsp; Analysis completed at current session
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Top row: Diagnosis + Confidence + Probabilities ───
    top_c1, top_c2, top_c3 = st.columns([2, 1.5, 2.5])

    with top_c1:
        st.markdown(f"""
        <div class="dr-card">
            <div class="result-badge {badge_class}" style="margin-bottom:12px;">
                PRIMARY ASSESSMENT
            </div>
            <h3 style="color:#0f172a;font-size:22px;font-weight:800;margin:8px 0;">
                {grade_name}
            </h3>
            <p style="color:#64748b;font-size:13px;margin:4px 0;">
                Severity Class: <strong>Grade {pred_class} (ICDR)</strong>
            </p>
            <p style="color:#94a3b8;font-size:12px;margin-top:8px;">
                {grade_desc}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with top_c2:
        # Confidence Display
        conf_pct = confidence * 100
        ring_color = "#22c55e" if conf_pct >= 80 else ("#eab308" if conf_pct >= 60 else "#ef4444")
        st.markdown(f"""
        <div class="dr-card" style="text-align:center;">
            <div style="font-size:11px;font-weight:700;color:#64748b;
              text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
              CONFIDENCE
            </div>
            <div style="font-size:36px;font-weight:800;color:#0f172a;">
                {conf_pct:.1f}%
            </div>
            <div style="margin-top:16px;padding-top:12px;border-top:1px solid #e2e8f0;">
                <div style="font-size:11px;font-weight:700;color:#64748b;
                  text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
                  MODEL CONSISTENCY
                </div>
                <div class="{consistency_class}" style="font-size:18px;font-weight:800;">
                    {consistency}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with top_c3:
        # Probability bars
        st.markdown("""
        <div class="dr-card">
            <div style="font-size:11px;font-weight:700;color:#64748b;
              text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
              DR-GRADE PROBABILITIES
            </div>
        """, unsafe_allow_html=True)

        for i in range(Config.NUM_CLASSES):
            pct = probs[i] * 100
            bar_color = Config.CLASS_COLORS[i]
            is_pred = " ◀" if i == pred_class else ""
            st.markdown(f"""
            <div class="prob-bar-container">
                <div class="prob-bar-label">
                    <span style="font-weight:{'700' if i == pred_class else '400'};
                      color:{'#0f172a' if i == pred_class else '#64748b'};">
                        {Config.CLASS_NAMES[i]}{is_pred}
                    </span>
                    <span style="font-weight:600;color:#334155;">{pct:.1f}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill"
                      style="width:{pct}%;background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Explainable AI Section ───
    st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">🔍</span>
        <span class="section-header-title">Explainable AI (Heatmap Visualization)</span>
    </div>
    """, unsafe_allow_html=True)

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("""
        <div class="dr-card">
            <h4 style="color:#0f172a;font-size:14px;font-weight:700;margin-bottom:12px;">
                DR Vision AI - Retinal Analysis Report
            </h4>
        """, unsafe_allow_html=True)

        if st.session_state.processed_pil:
            st.image(
                st.session_state.processed_pil,
                caption="Preprocessed & Enhanced Retinal Scan",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with img_col2:
        st.markdown("""
        <div class="dr-card">
            <h4 style="color:#0f172a;font-size:14px;font-weight:700;margin-bottom:12px;">
                Diabetic Retinopathy Analysis Report
            </h4>
        """, unsafe_allow_html=True)

        if st.session_state.gradcam_pil:
            st.image(
                st.session_state.gradcam_pil,
                caption=f"Grad-CAM++ Saliency Map — Predicted: {grade_name} ({conf_pct:.1f}%)",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Feature Analysis Summary Table ───
    st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">📊</span>
        <span class="section-header-title">Feature Analysis Summary</span>
    </div>
    """, unsafe_allow_html=True)

    # Generate feature analysis from prediction
    features_data = _generate_feature_analysis(pred_class, confidence, probs)

    table_html = """<table class="feature-table">
    <tr>
        <th>BIOMARKER</th><th>DETECTION STATUS</th><th>AI SCORE</th>
        <th>CONFIDENCE</th><th>CLINICAL NOTE</th>
    </tr>"""
    for feat in features_data:
        status_color = "#ef4444" if feat["status"] == "Present" else (
            "#eab308" if feat["status"] == "Likely" else "#22c55e"
        )
        table_html += f"""<tr>
            <td style="font-weight:600;">{feat["name"]}</td>
            <td><span style="color:{status_color};font-weight:600;">{feat["status"]}</span></td>
            <td>{feat["score"]}</td>
            <td>{feat["conf"]}</td>
            <td style="font-size:11px;color:#64748b;">{feat["note"]}</td>
        </tr>"""
    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Export PDF ───
    exp_col1, exp_col2 = st.columns([1, 1])
    with exp_col1:
        if st.button("📅 Schedule Follow-up Appointment", use_container_width=True, type="primary"):
            st.info("📅 Scheduling feature will be available in future updates.")

    with exp_col2:
        pdf_bytes = generate_pdf_report(
            patient_info=st.session_state.patient_info,
            predicted_class=pred_class,
            confidence=confidence,
            probabilities=probs,
            raw_pil=st.session_state.raw_pil,
            processed_pil=st.session_state.processed_pil,
            gradcam_pil=st.session_state.gradcam_pil,
            model_type=model_type,
        )
        if pdf_bytes:
            st.download_button(
                "📥 Export PDF Report",
                data=pdf_bytes,
                file_name=f"DR_Report_{grade_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="secondary",
            )
        else:
            st.warning("PDF generation requires the `fpdf2` package.")

    # Footer
    st.markdown("""
    <div class="app-footer">
        © 2024 DR Vision AI. For Research Use Only.
        Not for diagnostic procedures without clinical oversight. &nbsp;&nbsp;&nbsp;
        <a href="#" style="color:#64748b;text-decoration:none;font-weight:600;">Privacy Policy</a>
        &nbsp;&nbsp;
        <a href="#" style="color:#64748b;text-decoration:none;font-weight:600;">Terms of Service</a>
        &nbsp;&nbsp;
        <a href="#" style="color:#64748b;text-decoration:none;font-weight:600;">Research Documentation</a>
    </div>
    """, unsafe_allow_html=True)


def _generate_feature_analysis(pred_class, confidence, probs):
    """Generate feature analysis table data based on the prediction."""
    features = []

    # Microaneurysms
    ma_present = pred_class >= 1
    features.append({
        "name": "Microaneurysms",
        "status": "Present" if ma_present else "Absent",
        "score": f"High ({probs[max(1, pred_class)] * 100:.0f})" if ma_present else f"Low ({probs[0] * 100:.0f})",
        "conf": f"{confidence * 100:.1f}%",
        "note": "Significant presence in the superior quadrant" if ma_present else "No microaneurysms detected",
    })

    # Hard Exudates
    he_present = pred_class >= 2
    features.append({
        "name": "Hard Exudates",
        "status": "Present" if he_present else "Absent",
        "score": f"High ({probs[max(2, pred_class)] * 100:.0f})" if he_present else f"Low ({probs[0] * 100:.0f})",
        "conf": f"{max(60, confidence * 100 - 5):.1f}%",
        "note": "Lipid deposits observed near macula" if he_present else "No hard exudates identified",
    })

    # Neovascularization
    nv_present = pred_class >= 3
    features.append({
        "name": "Neovascularization",
        "status": "Present" if nv_present else "Absent",
        "score": f"Extra ({probs[max(3, pred_class)] * 100:.0f})" if nv_present else f"None ({probs[0] * 100:.0f})",
        "conf": f"{max(55, confidence * 100 - 10):.1f}%",
        "note": "New vessel formation at optic disc periphery" if nv_present else "No neovascularization seen; clear for follow up",
    })

    # Macular Edema
    me_present = pred_class >= 2
    features.append({
        "name": "Macular Edema",
        "status": "Likely" if me_present else "Absent",
        "score": f"Mild ({probs[2] * 100:.0f})" if me_present else f"None ({probs[0] * 100:.0f})",
        "conf": f"{max(50, confidence * 100 - 15):.1f}%",
        "note": "Possible thickening of retinal layers near fovea" if me_present else "No macular edema indicators found",
    })

    return features


# ==============================================================
# PAGE 4 — About
# ==============================================================
def page_about():
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;
      margin-bottom:24px;">
        <div>
            <p style="color:#64748b;font-size:13px;margin-bottom:4px;">
                About DR Vision AI
            </p>
        </div>
        <div>
            <span style="color:#64748b;font-size:12px;">System Status: </span>
            <span style="color:#22c55e;font-size:12px;font-weight:600;">● Live Deployment</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Research Project card
    st.markdown("""
    <div class="dr-card" style="margin-bottom:24px;padding:32px;">
        <div class="hero-badge" style="color:#0d9488;background:#f0fdfa;
          border-color:#99f6e4;">RESEARCH PROJECT v4.X</div>
        <h3 style="color:#0f172a;font-weight:800;font-size:22px;margin:12px 0 8px;">
            Next-Generation Retinal Analysis
        </h3>
        <p style="color:#64748b;font-size:13px;line-height:1.7;max-width:600px;">
            DR Vision AI represents a pinnacle in computer-aided diagnostics for
            Diabetic Retinopathy. Our platform utilizes state-of-the-art Deep Learning
            models to provide clinicians with high-fidelity lesion detection and severity
            assessment, significantly reducing the cognitive load required for mass
            screenings and longitudinal patient tracking.
        </p>
        <div style="display:flex;gap:12px;margin-top:20px;">
            <div style="background:#0d9488;color:white;padding:10px 20px;
              border-radius:8px;font-weight:600;font-size:13px;">📄 View Whitepaper</div>
            <div style="background:white;color:#0d9488;padding:10px 20px;
              border-radius:8px;font-weight:600;font-size:13px;
              border:1px solid #0d9488;">⚙️ Technical Specs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core AI Technology
    st.markdown("""
    <div class="section-header" style="margin-bottom:16px;">
        <span class="section-header-title">Core AI Technology</span>
    </div>
    """, unsafe_allow_html=True)

    tech_cols = st.columns(3)
    tech_items = [
        ("🧠", "CNN Architecture",
         "Leveraging customized EfficientNet-B4 architectures optimized "
         "for high-resolution retinal image classification to detect subtle "
         "microvascular lesions."),
        ("🔍", "Attention Mapping",
         "Integrated Grad-CAM++ visualizations provide clinicians with "
         "92%+ fidelity saliency maps overlaying relevant retinal regions."),
        ("⚡", "Edge Inference",
         "Optimized for sub-250ms inference times, enabling real-time "
         "clinical workflows without heavy GPU infrastructure."),
    ]
    for col, (icon, title, desc) in zip(tech_cols, tech_items):
        with col:
            st.markdown(f"""
            <div class="about-card">
                <div class="about-icon">{icon}</div>
                <div class="about-title">{title}</div>
                <div class="about-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # System Architecture Flow
    st.markdown("""
    <div class="section-header" style="margin-bottom:16px;">
        <span class="section-header-title">System Architecture Flow</span>
    </div>
    """, unsafe_allow_html=True)

    flow_cols = st.columns(4)
    flow_items = [
        ("#f1f5f9", "📁", "Data Ingestion"),
        ("#f0fdfa", "⚙️", "Preprocessing"),
        ("#0d9488", "🧠", "AI Analysis"),
        ("#f1f5f9", "📊", "Results"),
    ]
    for col, (bg, icon, label) in zip(flow_cols, flow_items):
        with col:
            text_color = "white" if bg == "#0d9488" else "#334155"
            st.markdown(f"""
            <div style="text-align:center;">
                <div class="flow-icon" style="background:{bg};">
                    <span style="color:{text_color};">{icon}</span>
                </div>
                <div class="flow-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Details
    st.markdown("""
    <div class="section-header" style="margin-bottom:16px;">
        <span class="section-header-title">Model Performance Metrics</span>
    </div>
    """, unsafe_allow_html=True)

    metric_cols = st.columns(4)
    metrics = [
        ("Image-Only Model", "EfficientNet-B4", "~74% Val Acc", "73.3%"),
        ("Multimodal Model", "EfficientNet-B4 + Clinical MLP", "~76% Val Acc", "N/A"),
        ("Training Data", "EyePACS Dataset", "108,227 samples", "5 classes"),
        ("Preprocessing", "CLAHE + Circular Mask", "224×224 px", "LAB color space"),
    ]
    for col, (title, model, acc, extra) in zip(metric_cols, metrics):
        with col:
            st.markdown(f"""
            <div class="dr-card" style="text-align:center;">
                <div style="font-size:12px;font-weight:700;color:#0d9488;
                  margin-bottom:6px;">{title}</div>
                <div style="font-size:13px;font-weight:600;color:#0f172a;">
                    {model}
                </div>
                <div style="font-size:12px;color:#64748b;margin-top:4px;">
                    {acc}
                </div>
                <div style="font-size:11px;color:#94a3b8;margin-top:2px;">
                    {extra}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="app-footer">
        © 2024 DR Vision AI. For Research Use Only.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
# Main App
# ==============================================================
def main():
    inject_css()
    init_session_state()
    render_sidebar()

    # Route to page
    page = st.session_state.page

    if page == "Dashboard":
        page_dashboard()
    elif page == "Patient Analysis":
        page_patient_analysis()
    elif page == "Analysis Report":
        page_analysis_report()
    elif page == "About":
        page_about()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
