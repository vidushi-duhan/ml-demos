import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ML Demos", page_icon="🫀", layout="centered")

# ── Design layer ──────────────────────────────────────────────────────────────
# Light, contained styling: accent colour, clearer hierarchy, chip pills.
st.markdown(
    """
    <style>
      :root {
        --accent: #e11d48;        /* rose-600  */
        --accent-soft: #fff1f2;   /* rose-50   */
        --ink: #0f172a;           /* slate-900 */
        --muted: #64748b;         /* slate-500 */
        --line: #e2e8f0;          /* slate-200 */
      }

      /* App background */
      .stApp { background: #ffffff; }

      /* Hero banner */
      .hero {
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.4rem;
      }
      .hero h1 {
        font-size: 1.7rem; font-weight: 750; color: var(--ink);
        margin: 0 0 0.2rem 0; line-height: 1.2;
      }
      .hero p { color: var(--muted); font-size: 0.92rem; margin: 0; }

      /* Section headings */
      .section-head {
        font-size: 1.15rem; font-weight: 700; color: var(--ink);
        margin: 0.4rem 0 0.1rem 0;
      }

      /* Chip / pill row */
      .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.6rem 0 0.2rem 0; }
      .chip {
        background: #eff6ff; color: #1d4ed8;
        border: 1px solid #bfdbfe; border-radius: 999px;
        padding: 0.18rem 0.7rem; font-size: 0.76rem; font-weight: 600;
        white-space: nowrap;
      }

      /* Tighten metric cards */
      div[data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid var(--line);
        border-radius: 10px; padding: 0.6rem 0.4rem; text-align: center;
      }
      div[data-testid="stMetricLabel"] { justify-content: center; }
      div[data-testid="stMetricValue"] { color: var(--accent); font-size: 1.35rem; }

      /* Primary button */
      div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--accent); border: none; font-weight: 650;
        border-radius: 10px;
      }
      div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #be123c;
      }

      /* Footer note */
      .builtwith {
        text-align: center; color: var(--muted); font-size: 0.78rem;
        margin-top: 0.6rem; padding-top: 0.6rem; border-top: 1px dashed var(--line);
      }
      .builtwith b { color: var(--accent); }
      .builtwith .sklearn { color: #f97316; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    return joblib.load("models/heart_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

model = load_model()
df = load_data()

tab1, tab2, tab3 = st.tabs(["Heart Disease", "Demo 2", "Demo 3"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:

    # SECTION 1: HEADER
    st.markdown(
        """
        <div class="hero">
          <h1>🫀 Heart Disease Risk Predictor</h1>
          <p>Predicts the likelihood of heart disease from a few basic health readings.
          Built with logistic regression on the UCI Cleveland dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chips = [
        "Logistic Regression", "Binary Classification", "Train/Test Split",
        "Feature Scaling", "Cross-Validation", "ROC-AUC",
    ]
    st.markdown(
        '<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # SECTION 2: INTERACTIVE PREDICTION
    st.markdown('<div class="section-head">Try it yourself</div>', unsafe_allow_html=True)
    st.caption(
        "Enter the values below and see the model's prediction. "
        "This is a learning demo, not medical advice."
    )

    age_mean = int(round(df["age"].mean()))
    trestbps_mean = int(round(df["trestbps"].mean()))
    thalach_mean = int(round(df["thalach"].mean()))
    chol_mean = int(round(df["chol"].mean()))

    col_left, col_right = st.columns(2)

    with col_left:
        age = st.slider("Age", min_value=29, max_value=77, value=age_mean)
        trestbps = st.slider(
            "Resting Blood Pressure (mm Hg)", min_value=94, max_value=200, value=trestbps_mean
        )
        thalach = st.slider(
            "Max Heart Rate Achieved (bpm)", min_value=71, max_value=202, value=thalach_mean
        )
        sex_input = st.selectbox("Sex", options=["Male", "Female"])

    with col_right:
        chol = st.slider(
            "Cholesterol (mg/dl)", min_value=126, max_value=564, value=chol_mean
        )
        exang_input = st.radio(
            "Exercise-induced Angina", options=["No", "Yes"], horizontal=True
        )
        fbs_input = st.radio(
            "Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"], horizontal=True
        )

    sex = 1 if sex_input == "Male" else 0
    exang = 1 if exang_input == "Yes" else 0
    fbs = 1 if fbs_input == "Yes" else 0

    if st.button("Predict", type="primary"):
        input_array = np.array([[age, sex, trestbps, chol, thalach, exang, fbs]])
        prediction = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0][1]

        if prediction == 1:
            st.error("### ⚠️ Likely at risk")
        else:
            st.success("### ✅ Unlikely at risk")

        st.progress(float(prob))
        st.markdown(
            f"Model gives this a **:red[{prob * 100:.1f}%]** likelihood of heart disease."
        )
        st.caption(
            "Model is 77.05% accurate on unseen data. Treat this as indicative only."
        )

    st.divider()

    # SECTION 3: MODEL PERFORMANCE
    st.markdown('<div class="section-head">Model Performance</div>', unsafe_allow_html=True)
    st.write("")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "77.05%")
    m2.metric("Precision", "76.92%")
    m3.metric("Recall", "71.43%")
    m4.metric("ROC-AUC", "87.55%")

    st.info(
        "In a screening tool, recall matters more than accuracy. "
        "At 71.43%, the model correctly identifies about 7 in 10 at-risk patients. "
        "The remaining 3 in 10 are missed. "
        "The threshold is kept at the default 0.5. In a real product you would lower it "
        "to catch more true cases at the cost of more false alarms. "
        "ROC-AUC of 87.55% shows the model separates the two classes well overall."
    )

    st.divider()

    # SECTION 4: ABOUT THE DATA
    with st.expander("About the data", expanded=False):
        st.markdown(
            "**Dataset:** UCI Cleveland Heart Disease. 303 patients, "
            "Cleveland Clinic Foundation. CC BY 4.0 license."
        )
        st.markdown(
            "**Class balance:** 139 with disease (45.9%), "
            "164 without (54.1%). Reasonably balanced."
        )
        st.markdown(
            "**Target:** Originally scored 0 to 4 by severity. "
            "Collapsed to binary: 0 = no disease, 1 = disease present."
        )
        st.markdown(
            "**Features used:** 7 of the 13 available features, "
            "chosen for being intuitive to enter. The model trains on exactly these 7. "
            "Nothing hidden."
        )

    # SECTION 5: DATA PROCESSING
    with st.expander("Data processing", expanded=False):
        st.markdown(
            "**Missing values:** None in the 7 features used. "
            "A median imputer is included in the pipeline as a safety net for inference."
        )

        missing_table = pd.DataFrame({
            "Feature": [
                "Age", "Sex", "Resting BP", "Cholesterol",
                "Max Heart Rate", "Exercise-induced Angina", "Fasting Blood Sugar"
            ],
            "Present": [303, 303, 303, 303, 303, 303, 303],
            "Missing": [0, 0, 0, 0, 0, 0, 0],
        })
        st.dataframe(missing_table, hide_index=True, use_container_width=True)

        st.markdown(
            "**Scaling:** Features standardized using StandardScaler (mean=0, std=1). "
            "Necessary because logistic regression is sensitive to feature scale. "
            "Without it, cholesterol (range 126 to 564) would dominate age (range 29 to 77) "
            "purely due to magnitude."
        )
        st.markdown(
            "**Split:** 80/20 train/test split, stratified to preserve class balance. "
            "Cross-validated over 5 folds: 75.92% plus or minus 4.22%."
        )
        st.markdown(
            "**Pipeline:** SimpleImputer → StandardScaler → LogisticRegression. "
            "The pipeline ensures the scaler is fit only on training data and applied "
            "consistently to test data, preventing data leakage."
        )

    # SECTION 6: WHY I BUILT THIS
    with st.expander("Why I built this", expanded=False):
        st.markdown(
            "A hands-on way to apply classification end to end: cleaning data, "
            "choosing an interpretable model, scaling, splitting, and reading the right "
            "metrics for the context. A learning project, not a clinical tool."
        )

    # SECTION 7: DISCLAIMER
    st.caption(
        "Educational demo using a public 1980s research dataset. "
        "Not for real heart disease assessment. Consult a doctor for health concerns."
    )

    st.markdown(
        '<div class="builtwith">A lightweight frontend built with '
        '<b>Streamlit</b> &nbsp;•&nbsp; model trained in <span class="sklearn">scikit-learn</span></div>',
        unsafe_allow_html=True,
    )

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.info("Coming soon.")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.info("Coming soon.")
