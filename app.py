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

      /* ── Mobile overrides (phones up to 640px) ── */
      @media (max-width: 640px) {
        /* Hero title smaller so it doesn't wrap badly */
        .hero h1 { font-size: 1.3rem; }

        /* Stack Streamlit's 2-col and 4-col grids into single column */
        div[data-testid="stHorizontalBlock"] {
          flex-direction: column !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
          width: 100% !important;
          flex: 1 1 100% !important;
          min-width: 100% !important;
        }

        /* Metric cards: 2-per-row grid instead of 4-in-a-row */
        div[data-testid="stMetric"] {
          min-width: 0;
        }

        /* Radio options: allow wrapping on very small screens */
        div[data-testid="stRadio"] > div {
          flex-wrap: wrap !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_heart_model():
    return joblib.load("models/heart_model.pkl")

@st.cache_resource
def load_student_model():
    return joblib.load("models/student_model.pkl")

@st.cache_resource
def load_bank_model():
    return joblib.load("models/bank_model.pkl")

@st.cache_data
def load_heart_data():
    return pd.read_csv("data/heart.csv")

@st.cache_data
def load_student_data():
    return pd.read_csv("data/student.csv")

heart_model   = load_heart_model()
student_model = load_student_model()
bank_model    = load_bank_model()
df_heart      = load_heart_data()
df_student    = load_student_data()

# ── App-level intro ───────────────────────────────────────────────────────────
st.markdown(
    """
    Three small ML demos, each showing something different:

    <span style="color:#1d4ed8;font-weight:700;">Heart Disease</span> - straightforward classification, near-balanced data, default model settings.

    <span style="color:#1d4ed8;font-weight:700;">Student Performance</span> - regression, and what it looks like when the signal is genuinely weak.

    <span style="color:#1d4ed8;font-weight:700;">Bank Marketing</span> - imbalanced classes and catching a data leak before it inflated the results.

    Each tab shows the data, the model, and the reasoning. Pick one to start.
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Heart Disease", "Student Performance", "Bank Marketing"])

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

    age_mean = int(round(df_heart["age"].mean()))
    trestbps_mean = int(round(df_heart["trestbps"].mean()))
    thalach_mean = int(round(df_heart["thalach"].mean()))
    chol_mean = int(round(df_heart["chol"].mean()))

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
        prediction = heart_model.predict(input_array)[0]
        prob = heart_model.predict_proba(input_array)[0][1]

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

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    r1c1.metric("Accuracy", "77.05%")
    r1c2.metric("Precision", "76.92%")
    r2c1.metric("Recall", "71.43%")
    r2c2.metric("ROC-AUC", "87.55%")

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

    # SECTION 1: HEADER
    st.markdown(
        """
        <div class="hero">
          <h1>&#127891; Student Performance Predictor</h1>
          <p>Predicts a student's final grade from study habits and family background.
          Built with Ridge Regression on the UCI Student Performance dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chips2 = [
        "Multiple Linear Regression", "Ridge & Lasso",
        "R-squared & RMSE", "Train/Test Split",
        "Cross-Validation", "Feature Scaling",
    ]
    st.markdown(
        '<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips2) + "</div>",
        unsafe_allow_html=True,
    )

    # SECTION 1b: HONEST NOTE
    st.info(
        "A note of honesty: this model is not winning any awards. "
        "R-squared of 0.20 is weak. I kept it only to learn through a good analysis "
        "of a weak model than a lucky result on a strong one. Consider this a small "
        "museum exhibit on what 'not enough signal' looks like."
    )

    st.divider()

    # SECTION 2: INTERACTIVE PREDICTION
    st.markdown('<div class="section-head">Try it yourself</div>', unsafe_allow_html=True)
    st.caption(
        "Enter a student's details and see the predicted final grade (0 to 20). "
        "Read the note below the result. This model is deliberately a rough estimate, "
        "as explained above."
    )

    s_left, s_right = st.columns(2)

    with s_left:
        studytime = st.slider(
            "Weekly study time", min_value=1, max_value=4, value=2,
            help="1 = under 2 hrs, 2 = 2 to 5 hrs, 3 = 5 to 10 hrs, 4 = over 10 hrs"
        )
        failures = st.slider("Past class failures", min_value=0, max_value=4, value=0)
        absences = st.slider("School absences", min_value=0, max_value=32, value=4)
        goout = st.slider(
            "Going out with friends", min_value=1, max_value=5, value=3,
            help="1 = very low, 5 = very high"
        )

    with s_right:
        Medu = st.slider(
            "Mother's education", min_value=0, max_value=4, value=2,
            help="0 = none, 1 = primary, 2 = 5th-9th grade, 3 = secondary, 4 = higher education"
        )
        Fedu = st.slider(
            "Father's education", min_value=0, max_value=4, value=2,
            help="0 = none, 1 = primary, 2 = 5th-9th grade, 3 = secondary, 4 = higher education"
        )
        higher_input = st.radio(
            "Wants higher education", options=["Yes", "No"], horizontal=True, index=0
        )

    higher = 1 if higher_input == "Yes" else 0

    if st.button("Predict Grade", type="primary"):
        # Feature order must match training: [studytime, failures, absences, Medu, Fedu, goout, higher]
        s_input = np.array([[studytime, failures, absences, Medu, Fedu, goout, higher]])
        raw_pred = student_model.predict(s_input)[0]
        grade = float(np.clip(raw_pred, 0, 20))
        st.metric("Predicted Final Grade", f"{grade:.1f} / 20")
        st.caption(
            "This is a rough estimate. "
            "The model explains only about 20% of grade variation. See why below."
        )

    st.divider()

    # SECTION 3: MODEL PERFORMANCE
    st.markdown('<div class="section-head">Model Performance</div>', unsafe_allow_html=True)
    st.write("")

    perf_df = pd.DataFrame({
        "Model":        ["Linear Regression", "Ridge", "Lasso"],
        "R²":           [0.20, 0.20, 0.03],
        "RMSE":         [2.79, 2.79, 3.07],
        "CV R²":        [0.12, 0.12, -0.06],
    })
    st.dataframe(perf_df, hide_index=True, use_container_width=True)

    st.markdown(
        "- **Linear Regression:** Weak fit. Explains only ~20% of grade variation.\n"
        "- **Ridge:** Same as Linear, slightly more stable. Used for prediction.\n"
        "- **Lasso:** Poor fit. Discarded almost all features."
    )

    st.markdown(
        "**How to read the numbers above:**\n\n"
        "- **R-squared:** 0 means the model explains nothing, 1 means it explains everything. "
        "Here 0.20 is very low. With prior grades included this would jump above 0.85, "
        "but that wouldn't be a real prediction.\n"
        "- **RMSE:** lower is better. 2.79 means predictions are off by about 2.8 grade points "
        "on average, on a 20-point scale.\n"
        "- **CV R-squared:** the model tested across 5 different splits. Close to the main "
        "R-squared means the result is stable. A negative value (Lasso) means it did worse "
        "than just guessing the average."
    )

    st.divider()

    # SECTION 4: THE LASSO INSIGHT
    st.markdown('<div class="section-head">What the model tells us about what matters</div>', unsafe_allow_html=True)
    st.write("")

    st.success(
        "Lasso technique automatically discards features it finds unhelpful. "
        "Applied here, it threw away 6 of the 7 inputs and kept only one: past failures.\n\n"
        "In plain terms: out of everything we gave the model - study time, parents' education, "
        "going out, ambition - the only factor with a strong reliable link to the final grade "
        "was how many classes the student had failed before. The rest carry weak signal.\n\n"
        "A student's track record predicts their future more reliably than study habits or "
        "family background, at least in this data."
    )

    st.markdown(
        "**The directional effects:**\n"
        "- Past failures: strong negative effect (more failures, lower grade)\n"
        "- Wanting higher education: positive effect\n"
        "- Study time: positive effect\n"
        "- Going out a lot: small negative effect"
    )

    st.divider()

    # SECTION 5: ABOUT THE DATA
    with st.expander("About the data", expanded=False):
        st.markdown(
            "**Dataset:** UCI Student Performance (Portuguese course), "
            "649 students, by Cortez & Silva. CC BY 4.0.\n\n"
            "**Target:** Final grade G3, scored 0 to 20.\n\n"
            "**Grade spread:** Most students score between 10 and 14. "
            "There is a cluster of 15 students who scored 0.\n\n"
            "**About those zeros:** The 15 students scoring 0 are likely dropouts rather than "
            "students who genuinely scored nothing. I tested removing them, but it made "
            "R-squared worse, so I kept all 649 records.\n\n"
            "**Features used:** 7 of the 30+ available, chosen for being intuitive. "
            "Prior grades (G1, G2) deliberately excluded because predicting a final grade "
            "from earlier grades is trivial."
        )

    # SECTION 6: DATA PROCESSING
    with st.expander("Data processing", expanded=False):
        st.markdown(
            "**Missing values:** None.\n\n"
            "**Encoding:** 'Wants higher education' converted from yes/no to 1/0.\n\n"
            "**Scaling:** All features standardized (StandardScaler) so no single feature "
            "dominates due to its range.\n\n"
            "**Split and validation:** 80/20 train/test split. Cross-validated over 5 folds.\n\n"
            "**Three models compared:** Linear, Ridge, Lasso, each in a scaler + model pipeline. "
            "Ridge saved for prediction."
        )

    # SECTION 7: WHY I BUILT THIS
    with st.expander("Why I built this", expanded=False):
        st.markdown(
            "To apply regression end to end and practice reading a model, including when "
            "it tells you the features are weak."
        )

    # SECTION 8: DISCLAIMER
    st.caption(
        "Educational demo using a public research dataset. "
        "Not a real assessment of any student's ability."
    )

    st.markdown(
        '<div class="builtwith">A lightweight frontend built with '
        '<b>Streamlit</b> &nbsp;•&nbsp; model trained in <span class="sklearn">scikit-learn</span></div>',
        unsafe_allow_html=True,
    )

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:

    # SECTION 1: HEADER
    st.markdown(
        "<div class=\"hero\">"
        "<h1>&#127968; Bank Marketing Predictor</h1>"
        "<p>Predicts whether a customer will subscribe to a term deposit. "
        "Built with logistic regression on the UCI Bank Marketing dataset (45,211 customers).</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    chips3 = [
        "Logistic Regression", "Class Imbalance",
        "Precision/Recall Tradeoff", "Cross-Validation",
        "ROC-AUC", "Feature Scaling",
    ]
    st.markdown(
        '<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips3) + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # SECTION 2: INTERACTIVE PREDICTION
    st.markdown('<div class="section-head">Try it yourself</div>', unsafe_allow_html=True)
    st.caption(
        "Enter a customer's profile and see whether the model would flag them as a likely subscriber."
    )

    b_left, b_right = st.columns(2)

    JOB_OPTIONS      = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                        "retired", "self-employed", "services", "student", "technician", "unemployed"]
    MARITAL_OPTIONS  = ["divorced", "married", "single"]
    EDUCATION_OPTIONS = ["primary", "secondary", "tertiary"]
    POUTCOME_MAP     = {
        "No prior contact": np.nan,   # imputed to 'failure' (most frequent)
        "Failure":          "failure",
        "Other":            "other",
        "Success":          "success",
    }

    with b_left:
        b_age      = st.slider("Age", min_value=18, max_value=95, value=40)
        b_job      = st.selectbox("Job", options=JOB_OPTIONS, index=1)
        b_marital  = st.selectbox("Marital status", options=MARITAL_OPTIONS, index=1)
        b_edu      = st.selectbox("Education", options=EDUCATION_OPTIONS, index=1)

    with b_right:
        b_housing_input  = st.radio("Has housing loan",   options=["No", "Yes"], horizontal=True)
        b_loan_input     = st.radio("Has personal loan",  options=["No", "Yes"], horizontal=True)
        b_poutcome_input = st.selectbox(
            "Outcome of previous campaign",
            options=list(POUTCOME_MAP.keys()),
            index=0,
        )

    b_housing  = "yes" if b_housing_input  == "Yes" else "no"
    b_loan     = "yes" if b_loan_input     == "Yes" else "no"
    b_poutcome = POUTCOME_MAP[b_poutcome_input]  # NaN or string

    if st.button("Predict", type="primary", key="bank_predict"):
        # Feature order must match training: age, job, marital, education, housing, loan, poutcome
        b_input = pd.DataFrame([{
            "age":      b_age,
            "job":      b_job,
            "marital":  b_marital,
            "education": b_edu,
            "housing":  b_housing,
            "loan":     b_loan,
            "poutcome": b_poutcome,
        }])

        b_pred = bank_model.predict(b_input)[0]
        b_prob = bank_model.predict_proba(b_input)[0][1]

        if b_pred == 1:
            st.success("### Likely to subscribe")
        else:
            st.warning("### Unlikely to subscribe")

        st.progress(float(b_prob))
        st.markdown(f"Model gives this a **{b_prob * 100:.1f}%** likelihood of subscribing.")
        st.caption(
            "This model is tuned to catch more potential subscribers, even at the cost of "
            "more false alarms. See why below."
        )

    st.divider()

    # SECTION 3: MODEL PERFORMANCE
    st.markdown('<div class="section-head">Model Performance</div>', unsafe_allow_html=True)
    st.write("")

    b1, b2 = st.columns(2)
    b3, b4 = st.columns(2)
    b1.metric("Accuracy",  "69.93%")
    b2.metric("Precision", "21.61%")
    b3.metric("Recall",    "59.74%")
    b4.metric("ROC-AUC",   "71.41%")

    st.info(
        "This dataset is imbalanced: only about 1 in 8 customers actually subscribe. "
        "A model that just predicted 'no' for everyone would score about 88% accuracy "
        "while being completely useless. That's why accuracy is not the metric to trust here.\n\n"
        "The model was deliberately trained with class_weight set to balanced, which tells it "
        "to pay more attention to the rare 'yes' cases. The tradeoff: precision drops to 21.6%, "
        "meaning most flagged customers will say no. But recall rises to 59.7%, meaning it "
        "catches 6 out of 10 real subscribers it would otherwise miss.\n\n"
        "Why this tradeoff makes sense for marketing: missing a real subscriber costs more than "
        "making an extra phone call. A bank would rather call 5 people and get 1 yes, than only "
        "call sure bets and miss most of the actual subscribers."
    )

    st.divider()

    # SECTION 4: THE POUTCOME INSIGHT
    st.markdown('<div class="section-head">The one feature that matters most</div>', unsafe_allow_html=True)
    st.write("")

    st.success(
        "One factor stood out far above the rest: whether the customer said yes to a previous campaign."
    )

    poutcome_df = pd.DataFrame({
        "Previous outcome":    ["Said yes before", "Other outcome", "Said no before", "Never contacted before"],
        "Subscription rate":   ["64.7%", "16.7%", "12.6%", "9.2%"],
    })
    st.table(poutcome_df)

    st.markdown(
        "A customer who subscribed before is roughly **7 times more likely** to subscribe again "
        "than someone never contacted. This single factor carries more predictive weight than "
        "age, job, education, marital status, and loans combined.\n\n"
        "What this means practically: the best leads for a campaign like this are not new "
        "customers - they are past customers who already said yes once."
    )

    st.divider()

    # SECTION 5: ABOUT THE DATA
    with st.expander("About the data", expanded=False):
        st.markdown(
            "**Dataset:** UCI Bank Marketing, 45,211 customers, Portuguese bank telemarketing "
            "campaigns. CC BY 4.0. Moro, Cortez and Rita.\n\n"
            "**Class balance:** about 1 in 8 customers (12%) subscribed; 88% did not.\n\n"
            "**A note on data quality:** an earlier 10% sample of this dataset showed a much "
            "more extreme imbalance (32x) due to sampling artifacts. The full 45,211-row dataset "
            "shows the true ratio (7.5x), which is why it was used here instead of the smaller "
            "sample.\n\n"
            "**Target leakage caught and removed:** the dataset includes a duration column "
            "(length of the sales call). This almost perfectly predicts the outcome, since a "
            "call of 0 seconds means it never happened, and duration is only known after the "
            "call. It would inflate accuracy while being useless in practice. Excluded from training."
        )

    # SECTION 6: DATA PROCESSING
    with st.expander("Data processing", expanded=False):
        st.markdown(
            "**Missing values:** job missing in 0.6% of rows, education in 4.1%, and poutcome "
            "(previous campaign outcome) missing in 81.7% of rows, since most customers simply "
            "had no prior campaign contact. The model imputes missing poutcome with the most "
            "common known value, which is a simplification worth noting: customers with no prior "
            "contact get treated similarly to customers whose last campaign failed, since the two "
            "groups have a similar subscription rate (9.2% vs 12.6%).\n\n"
            "**Class imbalance handling:** class_weight set to balanced in logistic regression, "
            "which automatically up-weights the minority class during training.\n\n"
            "**Scaling:** numerical features standardized.\n\n"
            "**Split and validation:** 80/20 train/test split, stratified. Cross-validated "
            "ROC-AUC: 70.8% plus or minus 1.2%, confirming the result is stable across "
            "different splits.\n\n"
            "**Why ROC-AUC over accuracy:** ROC-AUC measures how well the model separates the "
            "two classes regardless of class balance, making it the more honest metric here."
        )

    # SECTION 7: WHY I BUILT THIS
    with st.expander("Why I built this", expanded=False):
        st.markdown(
            "To work through a real imbalanced classification problem: catching a data leak "
            "before it inflated the results, choosing a deliberate precision/recall tradeoff, "
            "and explaining why that tradeoff is the right business call, not just a technical one."
        )

    # SECTION 8: DISCLAIMER
    st.caption(
        "Educational demo using a public research dataset. "
        "Not a real marketing or credit decision tool."
    )

    st.markdown(
        '<div class="builtwith">A lightweight frontend built with '
        '<b>Streamlit</b> &nbsp;•&nbsp; model trained in <span class="sklearn">scikit-learn</span></div>',
        unsafe_allow_html=True,
    )
