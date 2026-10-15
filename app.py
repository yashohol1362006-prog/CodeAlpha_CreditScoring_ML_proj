
from pathlib import Path

import joblib
import pandas as pd 
import streamlit as st


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="CreditIQ | Credit Risk Analytics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =====================================================
# 2. LOAD THE TRAINED MODEL
# =====================================================
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "random_forest.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}. "
            "Run python src/train.py first."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# =====================================================
# 3. CUSTOM STYLING
# =====================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%, #19294b 0%, transparent 42%),
            linear-gradient(135deg, #080e1b, #10182b);
        color: #edf2ff;
        font-family: 'DM Sans', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #f4f7ff !important;
    }

    .hero {
        padding: 32px;
        border: 1px solid rgba(137, 164, 255, .24);
        border-radius: 24px;
        background: linear-gradient(
            120deg,
            rgba(36, 57, 106, .78),
            rgba(20, 29, 53, .88)
        );
        margin-bottom: 25px;
        box-shadow: 0 18px 55px rgba(0, 0, 0, .18);
    }

    .eyebrow {
        color: #8db7ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(30px, 4vw, 44px);
        font-weight: 700;
        line-height: 1.15;
        margin: 15px 0;
        color: white;
    }

    .hero-description {
        color: #bac7e2;
        font-size: 15px;
        line-height: 1.7;
        max-width: 750px;
    }

    .pill {
        display: inline-block;
        padding: 8px 13px;
        margin-top: 12px;
        border-radius: 30px;
        color: #a7f3d0;
        background: rgba(16, 185, 129, .10);
        border: 1px solid rgba(52, 211, 153, .24);
        font-size: 12px;
    }

    .section-label {
        color: #8da8d8;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.6px;
        margin: 22px 0 15px;
    }

    .metric-card, .result-card {
        background: linear-gradient(145deg, #172744, #111a2d);
        border: 1px solid #304568;
        border-radius: 18px;
        padding: 23px;
        min-height: 155px;
        margin-bottom: 12px;
    }

    .result-card {
        min-height: 190px;
    }

    .result-label {
        color: #9bb0d4;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-weight: 700;
    }

    .result-number {
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 36px;
        font-weight: 700;
        margin: 14px 0 8px;
    }

    .muted {
        color: #a9b8d4;
        font-size: 13px;
        line-height: 1.7;
    }

    div[data-testid="stForm"] {
        background: rgba(18, 28, 49, .82);
        border: 1px solid #293754;
        border-radius: 20px;
        padding: 24px;
    }

    div[data-testid="stWidgetLabel"] label {
        color: #cbd6ee !important;
        font-size: 13px !important;
    }

    div[data-baseweb="input"] > div {
        background: #0c1528 !important;
        border: 1px solid #30415f !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
        color: #f4f7ff !important;
    }

    div[data-testid="stNumberInput"] button {
        background: #192642 !important;
        color: #b9ccf5 !important;
        border: none !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(100deg, #5278ff, #8866f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 11px !important;
        font-weight: 700 !important;
        min-height: 48px;
        width: 100%;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        box-shadow: 0 0 24px rgba(103, 124, 255, .35);
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        color: #7183a4;
        font-size: 12px;
        padding-top: 28px;
    }

    hr {
        border-color: #273550;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 4. HERO SECTION
# =====================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            ✦ MACHINE LEARNING · RISK INTELLIGENCE
        </div>

        <div class="hero-title">
            CreditIQ<br>Credit Risk Analytics
        </div>

        <div class="hero-description">
            AI-powered credit risk insights from financial
            indicators and repayment history.
        </div>

        <div class="pill">
            ● MODEL READY &nbsp; | &nbsp; RANDOM FOREST
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 5. MODEL PERFORMANCE
# =====================================================
st.markdown(
    '<div class="section-label">MODEL PERFORMANCE · HELD-OUT TEST SET</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="result-label">Selected Model</div>
            <div style="font-size:25px;font-weight:700;margin:14px 0 8px">
                Random Forest
            </div>
            <div class="muted">Ensemble classification model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="result-label">ROC-AUC</div>
            <div class="result-number">0.8595</div>
            <div class="muted">Test-set ranking performance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="result-label">Class 1 Recall</div>
            <div class="result-number">67.55%</div>
            <div class="muted">Recall for serious delinquency</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# 6. FINANCIAL INPUT FORM
# =====================================================
st.markdown(
    '<div class="section-label">01 / CUSTOMER FINANCIAL PROFILE</div>',
    unsafe_allow_html=True,
)

with st.form("credit_form"):
    st.subheader("Financial indicators")
    st.caption(
        "Enter values from a financial profile to generate a prediction."
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Personal & credit information**")

        age = st.number_input(
            "Age (years)", min_value=18, max_value=100, value=35
        )

        income = st.number_input(
            "Monthly income",
            min_value=0.0,
            value=5000.0,
            step=500.0,
        )

        dependents = st.number_input(
            "Number of dependents",
            min_value=0,
            value=1,
        )

        open_credit = st.number_input(
            "Open credit lines and loans",
            min_value=0,
            value=8,
        )

        real_estate = st.number_input(
            "Real estate loans or lines",
            min_value=0,
            value=1,
        )

    with right:
        st.markdown("**Debt & repayment history**")

        utilization = st.number_input(
            "Revolving utilization ratio",
            min_value=0.0,
            value=0.25,
            step=0.05,
        )

        debt_ratio = st.number_input(
            "Debt ratio",
            min_value=0.0,
            value=0.30,
            step=0.05,
        )

        late_30_59 = st.number_input(
            "Times 30–59 days past due",
            min_value=0,
            value=0,
        )

        late_60_89 = st.number_input(
            "Times 60–89 days past due",
            min_value=0,
            value=0,
        )

        late_90 = st.number_input(
            "Times 90+ days late",
            min_value=0,
            value=0,
        )

    submitted = st.form_submit_button(
        "✦  Analyze Credit Profile",
        use_container_width=True,
    )


# =====================================================
# 7. PREDICTION AND RESULTS
# =====================================================
if submitted:
    # Feature names and order match X in train.py.
    input_data = pd.DataFrame(
        [
            {
                "RevolvingUtilizationOfUnsecuredLines": utilization,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
                "DebtRatio": debt_ratio,
                "MonthlyIncome": income,
                "NumberOfOpenCreditLinesAndLoans": open_credit,
                "NumberOfTimes90DaysLate": late_90,
                "NumberRealEstateLoansOrLines": real_estate,
                "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
                "NumberOfDependents": dependents,
            }
        ]
    )

    try:
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0, 1])

        st.markdown("---")
        st.markdown(
            '<div class="section-label">02 / MODEL OUTPUT</div>',
            unsafe_allow_html=True,
        )

        result_col1, result_col2 = st.columns(2, gap="large")

        with result_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Estimated probability · Class 1
                    </div>
                    <div class="result-number">{probability:.1%}</div>
                    <div class="muted">
                        Model-estimated probability of serious delinquency.
                        This is not a calibrated guarantee.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(probability)

        with result_col2:
            if prediction == 1:
                result_title = "Higher predicted risk"
                result_color = "#ff9b9b"
                result_description = (
                    "The model predicts serious delinquency "
                    "within the target period."
                )
            else:
                result_title = "Lower predicted risk"
                result_color = "#7ce7bd"
                result_description = (
                    "The model predicts no serious delinquency "
                    "within the target period."
                )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted class</div>
                    <div style="font-size:25px;font-weight:700;
                                color:{result_color};margin:14px 0">
                        {result_title}
                    </div>
                    <div class="muted">{result_description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(
            "Educational demonstration only. Predictions can be wrong "
            "and must not be used to approve, reject, price, or otherwise "
            "make decisions about real credit applications."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# =====================================================
# 8. FOOTER
# =====================================================
st.markdown(
    """
    <div class="footer">
        CREDITIQ · MACHINE LEARNING PROJECT<br>
        Built with Python · Pandas · Scikit-learn · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
