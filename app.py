import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ==========================
# Config
# ==========================
SHEETDB_URL    = "https://sheetdb.io/api/v1/qnp2lhzc94mrn"
MODEL_PATH     = "random_forest_smote.pkl"
THRESHOLD_PATH = "best_threshold.pkl"

# ==========================
# Load model (cached)
# ==========================
@st.cache_resource
def load_model():
    model     = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, threshold

# ==========================
# Risk tier from probability
# ==========================
def get_risk_tier(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"

def get_recommendations(tier):
    base = [
        "Maintain a healthy lifestyle and regular exercise.",
        "Avoid smoking and limit alcohol consumption.",
        "Get vaccinated against HPV if not already done.",
    ]
    if tier == "Low Risk":
        return base + [
            "Continue routine cervical screening every 3–5 years as advised by your doctor.",
            "Stay informed about cervical health and screening options.",
        ]
    elif tier == "Medium Risk":
        return base + [
            "Schedule a cervical screening (Pap smear) within the next 6 months.",
            "Discuss your risk factors with a gynaecologist.",
            "Avoid unprotected intercourse with multiple partners.",
        ]
    else:
        return base + [
            "Consult a gynaecologist as soon as possible.",
            "Request a colposcopy or further diagnostic tests.",
            "Do not delay — early detection significantly improves outcomes.",
            "Inform close family members to get screened as well.",
        ]

# ==========================
# Page setup
# ==========================
st.set_page_config(
    page_title="Cervical Health Risk Assessment",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Cervical Health Risk Assessment System")
st.write("Early awareness and preventive healthcare platform")
st.info(
    "This tool provides a preliminary risk assessment only. "
    "It is **not a medical diagnosis**. Always consult a qualified healthcare professional.",
    icon="ℹ️",
)
st.divider()

# ==========================
# Survey Form
# ==========================
st.subheader("Personal Information")

with st.form("survey_form"):

    name = st.text_input("Full Name *")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age *", min_value=1, max_value=100, value=None)
        pregnancies = st.number_input(
            "Number of pregnancies *", min_value=0, max_value=20, value=None
        )
    with col2:
        smoking = st.selectbox("Do you smoke? *", ["Select", "Yes", "No"])
        smokes_years = st.number_input(
            "If yes, how many years?", min_value=0, max_value=60, value=0
        )
    smokes_packs_year = st.number_input(
    "Smokes (packs/year)",
    min_value=0.0,
    value=0.0
     )
    st.divider()
    st.subheader("Sexual History")

    # --- Key checkbox for no sexual intercourse ---
    no_intercourse = st.checkbox("I have never had sexual intercourse")

    if no_intercourse:
        st.info("Sexual history fields skipped.")
        sexual_partners   = 0
        first_intercourse = 0.0
    else:
        col3, col4 = st.columns(2)
        with col3:
            sexual_partners = st.number_input(
                "Number of sexual partners *", min_value=0, max_value=50, value=None
            )
        with col4:
            first_intercourse = st.number_input(
                "Age at first sexual intercourse *", min_value=1, max_value=60, value=None
            )

    st.divider()
    st.subheader("Contraceptive History")

    col5, col6 = st.columns(2)
    with col5:
        hc = st.selectbox(
            "Currently using hormonal contraceptives? *", ["Select", "Yes", "No"]
        )
        hc_years = st.number_input(
            "If yes, for how many years?", min_value=0, max_value=40, value=0
        )
    with col6:
        iud = st.selectbox("Do you use an IUD? *", ["Select", "Yes", "No"])
        iud_years = st.number_input(
            "If yes, for how many years?", min_value=0, max_value=30, value=0,
            key="iud_years"
        )

    st.divider()
    st.subheader("Medical History")

    if no_intercourse:
        # STDs not applicable if no intercourse
        st.info("STD fields skipped (not applicable).")
        stds           = "No"
        stds_number    = 0
        condylomatosis = "No"
        hiv            = "No"
        stds_diagnoses = 0
    else:
        col7, col8 = st.columns(2)
        with col7:
            stds = st.selectbox("History of STDs? *", ["Select", "Yes", "No"])
            stds_number = st.number_input(
                "If yes, how many STDs diagnosed?", min_value=0, max_value=10, value=0
            )
            stds_diagnoses = st.number_input(
                "Number of STD diagnoses (total)", min_value=0, max_value=10, value=0
            )
        with col8:
            condylomatosis = st.selectbox("STD: Condylomatosis?", ["No", "Yes"])
            hiv            = st.selectbox("STD: HIV?", ["No", "Yes"])

    st.divider()
    st.subheader("Previous Diagnoses")
    st.caption("Answer Yes only if a doctor has previously told you.")

    col9, col10 = st.columns(2)
    with col9:
        dx_cancer = st.selectbox("Diagnosed with cancer?",            ["No", "Yes"])
        dx_cin    = st.selectbox("Diagnosed with CIN?",               ["No", "Yes"])
        dx_hpv    = st.selectbox("Diagnosed with HPV?",               ["No", "Yes"])
        dx        = st.selectbox("Any other cervical diagnosis?",      ["No", "Yes"])
    with col10:
        hinselmann = st.selectbox("Hinselmann test positive?",         ["No", "Yes"])
        schiller   = st.selectbox("Schiller test positive?",           ["No", "Yes"])
        citology   = st.selectbox("Citology test positive?",           ["No", "Yes"])

    submitted = st.form_submit_button("Get Risk Assessment", use_container_width=True)

# ==========================
# On Submit
# ==========================
if submitted:

    # --- Validation ---
    errors = []
    if not name.strip():
        errors.append("Full name is required.")
    if age is None:
        errors.append("Age is required.")
    if pregnancies is None:
        errors.append("Number of pregnancies is required.")
    if smoking == "Select":
        errors.append("Smoking field is required.")
    if hc == "Select":
        errors.append("Hormonal contraceptives field is required.")
    if iud == "Select":
        errors.append("IUD field is required.")

    # Only validate sexual history fields if intercourse has occurred
    if not no_intercourse:
        if sexual_partners is None:
            errors.append("Number of sexual partners is required.")
        if first_intercourse is None:
            errors.append("Age at first sexual intercourse is required.")
        if stds == "Select":
            errors.append("STDs field is required.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # --- Safe defaults if no intercourse ---
    if no_intercourse:
        sexual_partners   = 0
        first_intercourse = 0.0
        stds              = "No"
        stds_number       = 0
        condylomatosis    = "No"
        hiv               = "No"
        stds_diagnoses    = 0

    # --- Convert Yes/No to 1/0 ---
    def yn(val):
        return 1 if val == "Yes" else 0

    # --- Build input for model ---
    input_data = pd.DataFrame([{
        "Age":                             float(age),
        "Number of sexual partners":       float(sexual_partners),
        "First sexual intercourse":        float(first_intercourse),
        "Num of pregnancies":              float(pregnancies),
        "Smokes":                          float(yn(smoking)),
        "Smokes (years)":                  float(smokes_years),
        "Smokes (packs/year)":             float(smokes_packs_year),
        "Hormonal Contraceptives":         float(yn(hc)),
        "Hormonal Contraceptives (years)": float(hc_years),
        "IUD":                             float(yn(iud)),
        "IUD (years)":                     float(iud_years),
        "STDs":                            float(yn(stds)),
        "STDs (number)":                   float(stds_number),
        "STDs:condylomatosis":             float(yn(condylomatosis)),
        "STDs:HIV":                        float(yn(hiv)),
        "STDs: Number of diagnosis":       float(stds_diagnoses),
        "Dx:Cancer":                       float(yn(dx_cancer)),
        "Dx:CIN":                          float(yn(dx_cin)),
        "Dx:HPV":                          float(yn(dx_hpv)),
        "Dx":                              float(yn(dx)),
        "Hinselmann":                      float(yn(hinselmann)),
        "Schiller":                        float(yn(schiller)),
        "Citology":                        float(yn(citology)),
    }])

    # --- Run model ---
    try:
        model, threshold = load_model()
        prob  = model.predict_proba(input_data)[0][1]
    except FileNotFoundError:
        st.error(
            "Model files not found. Make sure `random_forest_smote.pkl` and "
            "`best_threshold.pkl` are in the same folder as app.py."
        )
        st.stop()

    tier             = get_risk_tier(prob)
    recommendations  = get_recommendations(tier)

    # --- Save to Google Sheets ---
    sheet_data = {
        "data": {
            "Timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name":                    name.strip(),
            "Age":                     int(age),
            "Pregnancies":             int(pregnancies),
            "No intercourse":          str(no_intercourse),
            "Smokes":                  smoking,
            "Smokes (years)":          smokes_years,
            "Sexual partners":         int(sexual_partners),
            "First intercourse age":   int(first_intercourse),
            "Hormonal Contraceptives": hc,
            "HC years":                hc_years,
            "IUD":                     iud,
            "IUD years":               iud_years,
            "STDs":                    stds,
            "STDs count":              stds_number,
            "Risk Probability":        round(float(prob), 4),
            "Risk Tier":               tier,
        }
    }
    try:
        requests.post(SHEETDB_URL, json=sheet_data, timeout=5)
    except Exception:
        pass

    # ==========================
    # Display Result
    # ==========================
    st.divider()
    st.subheader(f"Risk Assessment Result for {name.strip()}")

    tier_colors = {"Low Risk": "#EAF3DE", "Medium Risk": "#FAEEDA", "High Risk": "#FCEBEB"}
    text_colors = {"Low Risk": "#27500A", "Medium Risk": "#633806", "High Risk": "#791F1F"}

    st.markdown(
        f"""
        <div style="
            background:{tier_colors[tier]};
            border-radius:12px;
            padding:24px;
            text-align:center;
            margin-bottom:16px;
        ">
            <p style="font-size:14px;color:{text_colors[tier]};margin:0 0 6px;">
                Estimated cancer probability
            </p>
            <p style="font-size:40px;font-weight:600;color:{text_colors[tier]};margin:0 0 6px;">
                {prob*100:.1f}%
            </p>
            <p style="font-size:22px;font-weight:500;color:{text_colors[tier]};margin:0;">
                {tier}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk scale bar
    st.markdown("**Risk scale**")
    col_l, col_m, col_h = st.columns(3)
    col_l.markdown(
        "<div style='background:#EAF3DE;border-radius:6px;padding:8px;text-align:center;"
        "font-size:13px;color:#27500A;'>Low Risk<br><b>&lt; 30%</b></div>",
        unsafe_allow_html=True,
    )
    col_m.markdown(
        "<div style='background:#FAEEDA;border-radius:6px;padding:8px;text-align:center;"
        "font-size:13px;color:#633806;'>Medium Risk<br><b>30% – 60%</b></div>",
        unsafe_allow_html=True,
    )
    col_h.markdown(
        "<div style='background:#FCEBEB;border-radius:6px;padding:8px;text-align:center;"
        "font-size:13px;color:#791F1F;'>High Risk<br><b>&gt; 60%</b></div>",
        unsafe_allow_html=True,
    )

    # Recommendations
    st.divider()
    st.subheader("Recommendations")
    for rec in recommendations:
        st.markdown(f"- {rec}")

    # Disclaimer
    st.divider()
    st.caption(
        "⚠️ This assessment is generated by a machine learning model trained on a public dataset. "
        "It is intended for awareness purposes only and does not replace professional medical advice, "
        "diagnosis, or treatment. Please consult a qualified gynaecologist for proper evaluation."
    )