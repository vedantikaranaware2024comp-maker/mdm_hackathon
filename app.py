import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    load_model = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ScholarTrack | Student Success Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_PATH = MODELS_DIR / "model_lstm.keras"
SEQ_SCALER_PATH = MODELS_DIR / "sc_seq.pkl"
STATIC_SCALER_PATH = MODELS_DIR / "sc_sta.pkl"
MEDIANS_PATH = MODELS_DIR / "train_medians.pkl"
COLUMNS_PATH = MODELS_DIR / "columns.json"
DEFAULT_DATA_PATH = BASE_DIR / "data.csv"
EVALUATION_PATH = RESULTS_DIR / "test_predictions.csv"

RANDOM_STATE = 42


# ============================================================
# ATTRACTIVE UI THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(
            140deg,
            #f4f7ff 0%,
            #f8faff 48%,
            #eef5ff 100%
        );
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111c3d 0%,
            #1c3264 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding: 7px 0;
        font-weight: 500;
    }

    .hero {
        padding: 30px 32px;
        border-radius: 24px;
        color: white;
        background: linear-gradient(
            120deg,
            #172554 0%,
            #2349a5 58%,
            #3977e8 100%
        );
        box-shadow: 0 12px 35px rgba(35, 73, 165, 0.18);
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: 34px;
        font-weight: 800;
        margin: 0 0 10px 0;
        color: white;
    }

    .hero p {
        font-size: 15px;
        line-height: 1.7;
        color: #e0eaff;
        margin: 0;
    }

    .eyebrow {
        color: #bfdbfe;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 800;
        color: #172554;
        margin-top: 12px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #e1e9f8;
        box-shadow: 0 5px 18px rgba(28, 50, 100, 0.05);
        min-height: 132px;
    }

    .metric-label {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #172554;
        margin-top: 9px;
    }

    .metric-note {
        font-size: 12px;
        color: #64748b;
        margin-top: 6px;
    }

    .panel {
        background: #ffffff;
        padding: 21px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 18px rgba(28, 50, 100, 0.04);
        margin-bottom: 14px;
    }

    .risk-high {
        color: #b91c1c;
        background: #fee2e2;
        padding: 6px 12px;
        border-radius: 100px;
        font-weight: 700;
        font-size: 12px;
    }

    .risk-medium {
        color: #92400e;
        background: #fef3c7;
        padding: 6px 12px;
        border-radius: 100px;
        font-weight: 700;
        font-size: 12px;
    }

    .risk-low {
        color: #166534;
        background: #dcfce7;
        padding: 6px 12px;
        border-radius: 100px;
        font-weight: 700;
        font-size: 12px;
    }

    .info-note {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 14px 17px;
        border-radius: 8px;
        color: #1e3a8a;
        font-size: 13px;
        line-height: 1.7;
        margin: 12px 0;
    }

    .success-note {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 14px 17px;
        border-radius: 8px;
        color: #065f46;
        font-size: 13px;
        line-height: 1.7;
        margin: 12px 0;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    .footer {
        color: #94a3b8;
        font-size: 11px;
        text-align: center;
        padding: 25px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# REUSABLE UI COMPONENTS
# ============================================================

def show_hero(title, description, eyebrow="SCHOLARTRACK"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_section(title, subtitle=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_risk_badge(risk_level):
    classes = {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
    }

    css_class = classes.get(risk_level, "risk-low")

    st.markdown(
        f'<span class="{css_class}">{risk_level} Risk</span>',
        unsafe_allow_html=True,
    )


def clean_column_names(df):
    df = df.copy()

    df.columns = [
        str(column).replace("\ufeff", "").replace("\t", " ").strip()
        for column in df.columns
    ]

    return df


def read_csv_flexible(file_or_path):
    try:
        df = pd.read_csv(file_or_path, sep=None, engine="python")
    except Exception:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)

        df = pd.read_csv(file_or_path, sep=";")

    return clean_column_names(df)


def normalize_target(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return int(value)

    text = str(value).strip().lower()

    if text in {"dropout", "dropped out", "1", "true"}:
        return 1

    if text in {"graduate", "graduated", "enrolled", "0", "false"}:
        return 0

    return np.nan


def get_risk_level(probability, medium_threshold, high_threshold):
    if probability >= high_threshold:
        return "High"

    if probability >= medium_threshold:
        return "Medium"

    return "Low"


def recommended_action(risk_level):
    actions = {
        "High": (
            "Priority support: personal meeting, academic review, "
            "and financial or wellbeing support assessment."
        ),
        "Medium": (
            "Schedule a check-in, review recent academic progress, "
            "and offer appropriate tutoring or mentoring."
        ),
        "Low": (
            "Continue regular mentoring and monitor progress "
            "at the next academic review."
        ),
    }

    return actions[risk_level]


# ============================================================
# LOAD MODEL AND PREPROCESSING FILES
# ============================================================

@st.cache_resource(show_spinner="Loading the trained LSTM model...")
def load_resources():
    required_files = [
        MODEL_PATH,
        SEQ_SCALER_PATH,
        STATIC_SCALER_PATH,
        MEDIANS_PATH,
        COLUMNS_PATH,
    ]

    missing_files = [
        str(path) for path in required_files if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The following required files are missing:\n"
            + "\n".join(missing_files)
        )

    if load_model is None:
        raise ImportError(
            "TensorFlow is not installed. Run: pip install tensorflow"
        )

    lstm_model = load_model(MODEL_PATH, compile=False)

    seq_scaler = joblib.load(SEQ_SCALER_PATH)
    static_scaler = joblib.load(STATIC_SCALER_PATH)
    medians = joblib.load(MEDIANS_PATH)

    with open(COLUMNS_PATH, "r", encoding="utf-8") as file:
        schema = json.load(file)

    for key in ["sem1", "sem2", "static"]:
        if key not in schema:
            raise ValueError(
                f"columns.json must contain the '{key}' feature list."
            )

    schema = {
        key: [
            str(column)
            .replace("\ufeff", "")
            .replace("\t", " ")
            .strip()
            for column in schema[key]
        ]
        for key in ["sem1", "sem2", "static"]
    }

    if not schema["sem1"] or not schema["sem2"]:
        raise ValueError("Semester feature lists cannot be empty.")

    if len(schema["sem1"]) != len(schema["sem2"]):
        raise ValueError(
            "The sem1 and sem2 feature lists must have the same length "
            "for this two-step LSTM sequence."
        )

    return lstm_model, seq_scaler, static_scaler, medians, schema


# ============================================================
# FEATURE PREPARATION
# ============================================================

def median_for_column(medians, column):
    if isinstance(medians, dict):
        value = medians.get(column, 0)

    elif isinstance(medians, pd.Series):
        value = medians.get(column, 0)

    else:
        value = 0

    try:
        value = float(value)

        if not np.isfinite(value):
            return 0.0

        return value

    except (TypeError, ValueError):
        return 0.0


def prepare_features(df, seq_scaler, static_scaler, medians, schema):
    df = clean_column_names(df)

    sem1_columns = schema["sem1"]
    sem2_columns = schema["sem2"]
    static_columns = schema["static"]

    all_columns = list(
        dict.fromkeys(
            sem1_columns + sem2_columns + static_columns
        )
    )

    missing_columns = [
        column for column in all_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The uploaded dataset is missing model features:\n"
            + ", ".join(missing_columns)
            + "\n\nUpload the same type of dataset used to train the model."
        )

    def numeric_frame(columns):
        frame = df[columns].copy()

        for column in columns:
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            )

            frame[column] = frame[column].replace(
                [np.inf, -np.inf], np.nan
            )

            frame[column] = frame[column].fillna(
                median_for_column(medians, column)
            )

        return frame.astype(float)

    sem1_raw = numeric_frame(sem1_columns)
    sem2_raw = numeric_frame(sem2_columns)
    static_raw = numeric_frame(static_columns)

    expected_seq_features = getattr(
        seq_scaler, "n_features_in_", len(sem1_columns)
    )

    expected_static_features = getattr(
        static_scaler, "n_features_in_", len(static_columns)
    )

    if expected_seq_features != len(sem1_columns):
        raise ValueError(
            "Sequence scaler mismatch: sc_seq.pkl expects "
            f"{expected_seq_features} features, but columns.json lists "
            f"{len(sem1_columns)} semester features."
        )

    if expected_static_features != len(static_columns):
        raise ValueError(
            "Static scaler mismatch: sc_sta.pkl expects "
            f"{expected_static_features} features, but columns.json lists "
            f"{len(static_columns)} static features."
        )

    sem1_scaled = seq_scaler.transform(sem1_raw)
    sem2_scaled = seq_scaler.transform(sem2_raw)

    sequence_3d = np.stack(
        [sem1_scaled, sem2_scaled],
        axis=1,
    ).astype(np.float32)

    static_scaled = static_scaler.transform(
        static_raw
    ).astype(np.float32)

    return {
        "sequence_3d": sequence_3d,
        "static_scaled": static_scaled,
        "sequence_columns": sem1_columns + sem2_columns,
        "static_columns": static_columns,
        "sem1_raw": sem1_raw,
        "sem2_raw": sem2_raw,
        "static_raw": static_raw,
        "feature_names": (
            sem1_columns + sem2_columns + static_columns
        ),
    }


# ============================================================
# LSTM PREDICTION
# ============================================================

def get_lstm_probabilities(model, sequence_3d, static_scaled):
    output = model.predict(
        [sequence_3d, static_scaled],
        verbose=0,
    )

    output = np.asarray(output)

    if output.ndim == 2 and output.shape[1] == 2:
        probabilities = output[:, 1]
    else:
        probabilities = output.reshape(-1)

    probabilities = np.asarray(probabilities, dtype=float)

    if len(probabilities) != len(sequence_3d):
        raise ValueError(
            "The LSTM returned a different number of predictions "
            "than the number of students."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "The model returned invalid probability values."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        probabilities = 1 / (
            1 + np.exp(-np.clip(probabilities, -50, 50))
        )

    return np.clip(probabilities, 0, 1)


def make_predictions(
    df,
    model,
    seq_scaler,
    static_scaler,
    medians,
    schema,
    medium_threshold=0.30,
    high_threshold=0.60,
):
    df = clean_column_names(df)

    prepared = prepare_features(
        df,
        seq_scaler,
        static_scaler,
        medians,
        schema,
    )

    probabilities = get_lstm_probabilities(
        model,
        prepared["sequence_3d"],
        prepared["static_scaled"],
    )

    results = pd.DataFrame(index=df.index)

    if "Student ID" in df.columns:
        results["Student ID"] = df["Student ID"].astype(str)

    elif "id" in df.columns:
        results["Student ID"] = df["id"].astype(str)

    else:
        results["Student ID"] = [
            f"STU-{i + 1:04d}" for i in range(len(df))
        ]

    results["LSTM Risk (%)"] = np.round(probabilities * 100, 2)
    results["Risk Probability"] = probabilities

    results["Risk Level"] = [
        get_risk_level(
            probability,
            medium_threshold,
            high_threshold,
        )
        for probability in probabilities
    ]

    results["Recommended Action"] = results["Risk Level"].map(
        recommended_action
    )

    results["LSTM Prediction"] = [
        "Potential Dropout"
        if probability >= 0.5
        else "Not Predicted as Dropout"
        for probability in probabilities
    ]

    if "Target" in df.columns:
        results["Actual Outcome"] = df["Target"].astype(str)

    return results, prepared


# ============================================================
# DATA LOADING
# ============================================================

def load_default_dataset():
    if not DEFAULT_DATA_PATH.exists():
        return None

    try:
        return read_csv_flexible(DEFAULT_DATA_PATH)

    except Exception as error:
        st.warning(f"Could not load data.csv: {error}")
        return None


def get_current_data():
    uploaded_file = st.session_state.get("uploaded_dataset")

    if uploaded_file is not None:
        uploaded_file.seek(0)
        return read_csv_flexible(uploaded_file)

    return load_default_dataset()


# ============================================================
# CHART HELPERS
# ============================================================

def display_bar_chart(
    data,
    x,
    y,
    title,
    xlabel=None,
    ylabel=None,
):
    fig, ax = plt.subplots(figsize=(9, 4.2))

    sns.barplot(data=data, x=x, y=y, ax=ax)

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def display_donut_chart(labels, values, title):
    fig, ax = plt.subplots(figsize=(5, 4.2))

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={
            "width": 0.42,
            "edgecolor": "white",
        },
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def display_risk_distribution(predictions):
    order = ["Low", "Medium", "High"]

    counts = (
        predictions["Risk Level"]
        .value_counts()
        .reindex(order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))

    bars = ax.bar(
        counts.index,
        counts.values,
        width=0.55,
    )

    ax.set_title(
        "Student Risk Distribution",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_ylabel("Number of students")
    ax.set_xlabel("Risk category")

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# OVERVIEW DASHBOARD
# ============================================================

def show_overview_page(df, predictions):
    show_hero(
        "Student Success Intelligence",
        "An AI-assisted early-warning dashboard to identify students "
        "who may benefit from timely academic, mentoring, and wellbeing support.",
        "LSTM • EARLY WARNING • STUDENT SUPPORT",
    )

    st.markdown(
        """
        <div class="info-note">
        <b>How ScholarTrack works:</b> The trained LSTM processes semester-wise
        academic features together with static student features and produces
        a dropout-risk estimate. Staff can use this estimate to prioritize
        supportive conversations—not to make automatic decisions about students.
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_students = len(predictions)

    high_count = int(
        (predictions["Risk Level"] == "High").sum()
    )

    medium_count = int(
        (predictions["Risk Level"] == "Medium").sum()
    )

    low_count = int(
        (predictions["Risk Level"] == "Low").sum()
    )

    average_risk = predictions["LSTM Risk (%)"].mean()

    cols = st.columns(4)

    with cols[0]:
        metric_card(
            "Students analyzed",
            f"{total_students:,}",
            "Records in the active dataset",
        )

    with cols[1]:
        metric_card(
            "High risk",
            f"{high_count:,}",
            "Priority for human review",
        )

    with cols[2]:
        metric_card(
            "Medium risk",
            f"{medium_count:,}",
            "Consider a supportive check-in",
        )

    with cols[3]:
        metric_card(
            "Average risk estimate",
            f"{average_risk:.1f}%",
            "Model output; not a confirmed outcome",
        )

    st.write("")

    left, right = st.columns([1.15, 0.85])

    with left:
        display_risk_distribution(predictions)

    with right:
        display_donut_chart(
            ["Low", "Medium", "High"],
            [low_count, medium_count, high_count],
            "Risk Category Share",
        )

    show_section(
        "Priority review list",
        "Students with the highest estimated risk appear first. Review these "
        "records with appropriate context before taking action.",
    )

    priority = predictions.sort_values(
        "Risk Probability",
        ascending=False,
    ).head(10)

    columns_to_show = [
        "Student ID",
        "LSTM Risk (%)",
        "Risk Level",
        "Recommended Action",
    ]

    st.dataframe(
        priority[columns_to_show],
        use_container_width=True,
        hide_index=True,
    )

    show_section("Project workflow")

    steps = st.columns(4)

    workflow = [
        (
            "01",
            "Upload data",
            "Use the supplied CSV or upload a compatible dataset.",
        ),
        (
            "02",
            "Analyze",
            "The LSTM processes semester and static features.",
        ),
        (
            "03",
            "Prioritize",
            "Review the estimated risk and supporting context.",
        ),
        (
            "04",
            "Support",
            "Record interventions and follow up with students.",
        ),
    ]

    for col, (number, title, description) in zip(steps, workflow):
        with col:
            st.markdown(
                f"""
                <div class="panel">
                    <div style="color:#3b82f6;font-size:12px;font-weight:800;">
                        STEP {number}
                    </div>
                    <div style="font-size:16px;font-weight:800;color:#172554;
                    margin:8px 0;">{title}</div>
                    <div style="font-size:12px;color:#64748b;line-height:1.6;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# STUDENT PREDICTIONS PAGE
# ============================================================

def show_predictions_page(df, predictions, prepared):
    show_hero(
        "Student Risk Explorer",
        "Explore individual model estimates, filter the student list, "
        "and export records for a human-led support review.",
    )

    show_section("Filter students")

    c1, c2, c3 = st.columns(3)

    with c1:
        risk_filter = st.multiselect(
            "Risk category",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

    with c2:
        minimum_risk = st.slider(
            "Minimum estimated risk (%)",
            min_value=0,
            max_value=100,
            value=0,
        )

    with c3:
        search_id = st.text_input(
            "Search Student ID",
            placeholder="Enter an ID...",
        )

    filtered = predictions[
        predictions["Risk Level"].isin(risk_filter)
        & (predictions["LSTM Risk (%)"] >= minimum_risk)
    ].copy()

    if search_id.strip():
        filtered = filtered[
            filtered["Student ID"]
            .astype(str)
            .str.contains(
                search_id.strip(),
                case=False,
                na=False,
            )
        ]

    st.caption(
        f"{len(filtered):,} students match your filters."
    )

    st.dataframe(
        filtered[
            [
                "Student ID",
                "LSTM Risk (%)",
                "Risk Level",
                "LSTM Prediction",
                "Recommended Action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇ Download filtered predictions (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="scholartrack_lstm_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()

    show_section(
        "Individual student profile",
        "Choose a student to review the model estimate and available academic "
        "features. A risk estimate alone cannot explain the student's situation.",
    )

    if predictions.empty:
        st.info("No student records are available.")
        return

    student_options = list(range(len(predictions)))

    selected_position = st.selectbox(
        "Select a student",
        student_options,
        format_func=lambda i: (
            f"{predictions.iloc[i]['Student ID']} — "
            f"{predictions.iloc[i]['LSTM Risk (%)']:.1f}% estimated risk"
        ),
    )

    student_result = predictions.iloc[selected_position]
    original_index = predictions.index[selected_position]

    try:
        student_source = df.loc[original_index]
    except KeyError:
        student_source = df.iloc[selected_position]

    left, right = st.columns([0.8, 1.2])

    with left:
        st.markdown(
            f"""
            <div class="panel">
                <div style="font-size:12px;color:#64748b;font-weight:700;">
                    STUDENT IDENTIFIER
                </div>
                <div style="font-size:27px;font-weight:800;color:#172554;
                margin:8px 0 15px 0;">
                    {student_result['Student ID']}
                </div>
                <div style="font-size:12px;color:#64748b;font-weight:700;">
                    LSTM DROPOUT-RISK ESTIMATE
                </div>
                <div style="font-size:42px;font-weight:800;color:#1d4ed8;
                margin:6px 0;">
                    {student_result['LSTM Risk (%)']:.1f}%
                </div>
                <div style="font-size:12px;color:#64748b;">
                    Model estimate, not a certainty or a diagnosis
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        show_risk_badge(student_result["Risk Level"])

        st.write("")
        st.markdown("**Suggested next step**")
        st.write(student_result["Recommended Action"])

    with right:
        show_section("Available student information")

        display_student = (
            student_source.to_frame(name="Value")
            .reset_index()
        )

        display_student.columns = ["Feature", "Value"]

        st.dataframe(
            display_student,
            use_container_width=True,
            hide_index=True,
            height=330,
        )

    st.markdown(
        """
        <div class="info-note">
        <b>Responsible use:</b> Use the model estimate as a prompt for a
        confidential, supportive conversation. Consider the student's own
        circumstances, recent changes, and data quality before recommending
        any intervention.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# NEW STUDENT INFORMATION PAGE
# ============================================================

def show_new_student_page(
    model,
    seq_scaler,
    static_scaler,
    medians,
    schema,
    medium_threshold,
    high_threshold,
):
    show_hero(
        "New Student Information",
        "Enter details for a new student or upload a CSV containing new "
        "student records. ScholarTrack will use the existing trained LSTM "
        "to estimate dropout risk without retraining the model.",
        "NEW STUDENT • LSTM PREDICTION",
    )

    st.markdown(
        """
        <div class="info-note">
        <b>How this page works:</b> Enter the student's features required by
        the trained model. The saved scalers and trained LSTM are reused.
        The new student's record does not need to be added to the training
        dataset. Any missing numeric feature is filled using the saved
        training median.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_manual, tab_csv = st.tabs(
        [
            "✍️ Enter Student Details",
            "📁 Upload Student CSV",
        ]
    )

    # --------------------------------------------------------
    # TAB 1: MANUAL ENTRY
    # --------------------------------------------------------

    with tab_manual:
        show_section(
            "Enter a new student's details",
            "Complete the feature fields using the same units and numeric "
            "encoding used during model training.",
        )

        student_id = st.text_input(
            "Student ID",
            value="NEW001",
            key="new_student_id",
            help="Enter a new identifier for this student.",
        )

        feature_values = {}

        with st.form("new_student_details_form"):
            st.markdown("### Semester 1 features")

            sem1_columns = schema["sem1"]

            sem1_layout = st.columns(2)

            for i, column in enumerate(sem1_columns):
                with sem1_layout[i % 2]:
                    feature_values[column] = st.number_input(
                        column,
                        value=float(
                            median_for_column(medians, column)
                        ),
                        key=f"new_sem1_{i}_{column}",
                        help=(
                            "Use the same numeric representation as "
                            "the training dataset."
                        ),
                    )

            st.markdown("### Semester 2 features")

            sem2_columns = schema["sem2"]

            sem2_layout = st.columns(2)

            for i, column in enumerate(sem2_columns):
                with sem2_layout[i % 2]:
                    feature_values[column] = st.number_input(
                        column,
                        value=float(
                            median_for_column(medians, column)
                        ),
                        key=f"new_sem2_{i}_{column}",
                        help=(
                            "Use the same numeric representation as "
                            "the training dataset."
                        ),
                    )

            st.markdown("### Static student features")

            static_columns = schema["static"]

            static_layout = st.columns(2)

            for i, column in enumerate(static_columns):
                with static_layout[i % 2]:
                    feature_values[column] = st.number_input(
                        column,
                        value=float(
                            median_for_column(medians, column)
                        ),
                        key=f"new_static_{i}_{column}",
                        help=(
                            "Use the same numeric representation as "
                            "the training dataset."
                        ),
                    )

            submitted = st.form_submit_button(
                "Predict New Student Risk",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            if not student_id.strip():
                st.error("Please enter a Student ID.")

            else:
                # Build a single row using exactly the model's features.
                # The Student ID is not included in the model inputs.
                new_student = pd.DataFrame(
                    [
                        {
                            column: feature_values[column]
                            for column in (
                                schema["sem1"]
                                + schema["sem2"]
                                + schema["static"]
                            )
                        }
                    ]
                )

                try:
                    with st.spinner(
                        "Analyzing the new student with the trained LSTM..."
                    ):
                        results, _ = make_predictions(
                            df=new_student,
                            model=model,
                            seq_scaler=seq_scaler,
                            static_scaler=static_scaler,
                            medians=medians,
                            schema=schema,
                            medium_threshold=medium_threshold,
                            high_threshold=high_threshold,
                        )

                    results["Student ID"] = student_id.strip()

                    result = results.iloc[0]
                    risk_probability = float(
                        result["Risk Probability"]
                    )
                    risk_percent = float(
                        result["LSTM Risk (%)"]
                    )
                    risk_level = result["Risk Level"]

                    st.divider()

                    show_section(
                        "Prediction result",
                        "This result is generated by the existing trained LSTM.",
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        metric_card(
                            "Student ID",
                            student_id.strip(),
                            "New student record",
                        )

                    with c2:
                        metric_card(
                            "Estimated dropout risk",
                            f"{risk_percent:.2f}%",
                            "LSTM model probability estimate",
                        )

                    with c3:
                        metric_card(
                            "Risk category",
                            risk_level,
                            "Based on sidebar thresholds",
                        )

                    st.write("")

                    show_risk_badge(risk_level)

                    st.progress(
                        max(0.0, min(1.0, risk_probability))
                    )

                    st.markdown("### Model prediction")

                    st.write(result["LSTM Prediction"])

                    st.markdown("### Recommended next step")

                    st.write(result["Recommended Action"])

                    st.markdown(
                        """
                        <div class="info-note">
                        This prediction is an estimate, not a confirmed
                        outcome. Review the student's circumstances and
                        verify the entered information before planning
                        any support.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("### Entered student information")

                    st.dataframe(
                        new_student.assign(
                            **{"Student ID": student_id.strip()}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "⬇ Download this student's prediction",
                        data=results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name=(
                            f"scholartrack_prediction_"
                            f"{student_id.strip()}.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True,
                    )

                except Exception as error:
                    st.error(
                        "Could not generate the new student's prediction. "
                        "Check that the entered features match the trained "
                        "model's saved preprocessing configuration."
                    )
                    st.code(str(error))

    # --------------------------------------------------------
    # TAB 2: UPLOAD CSV
    # --------------------------------------------------------

    with tab_csv:
        show_section(
            "Upload new student records",
            "Upload a separate CSV containing new students. This uploader "
            "is independent of the dataset uploader used by the existing "
            "dashboard pages.",
        )

        st.markdown(
            """
            <div class="info-note">
            <b>CSV requirements:</b> Include all feature columns listed in
            <code>models/columns.json</code>. Column names must match exactly.
            Student ID is optional. The CSV does not need a Target column.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_new_students = st.file_uploader(
            "Choose a CSV file containing new students",
            type=["csv"],
            key="new_students_csv_upload",
        )

        if uploaded_new_students is not None:
            try:
                uploaded_new_students.seek(0)

                new_students_df = read_csv_flexible(
                    uploaded_new_students
                )

                if new_students_df.empty:
                    st.error("The uploaded CSV contains no student records.")

                else:
                    st.markdown("### Uploaded data preview")

                    st.caption(
                        f"Rows: {len(new_students_df):,} | "
                        f"Columns: {len(new_students_df.columns):,}"
                    )

                    st.dataframe(
                        new_students_df.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                    expected_features = list(
                        dict.fromkeys(
                            schema["sem1"]
                            + schema["sem2"]
                            + schema["static"]
                        )
                    )

                    missing_features = [
                        column
                        for column in expected_features
                        if column not in new_students_df.columns
                    ]

                    if missing_features:
                        st.error(
                            "The uploaded CSV is missing these required "
                            "model features:"
                        )

                        st.write(missing_features)

                        st.info(
                            "Add the missing columns using the exact names "
                            "shown in Project Insights, then upload the CSV again."
                        )

                    else:
                        st.success(
                            "All required model feature columns are present."
                        )

                        if st.button(
                            "Predict Risk for Uploaded Students",
                            type="primary",
                            use_container_width=True,
                            key="predict_uploaded_new_students",
                        ):
                            try:
                                with st.spinner(
                                    "Predicting risk for uploaded students..."
                                ):
                                    uploaded_predictions, _ = make_predictions(
                                        df=new_students_df,
                                        model=model,
                                        seq_scaler=seq_scaler,
                                        static_scaler=static_scaler,
                                        medians=medians,
                                        schema=schema,
                                        medium_threshold=medium_threshold,
                                        high_threshold=high_threshold,
                                    )

                                st.session_state[
                                    "new_students_prediction_results"
                                ] = uploaded_predictions

                                st.session_state[
                                    "new_students_prediction_filename"
                                ] = uploaded_new_students.name

                            except Exception as error:
                                st.error(
                                    "Prediction failed for the uploaded CSV. "
                                    "Check its data types, column names, "
                                    "saved scalers, and model inputs."
                                )
                                st.code(str(error))

                # Display predictions saved for the uploaded CSV.
                saved_predictions = st.session_state.get(
                    "new_students_prediction_results"
                )

                saved_filename = st.session_state.get(
                    "new_students_prediction_filename"
                )

                if (
                    saved_predictions is not None
                    and saved_filename == uploaded_new_students.name
                ):
                    st.divider()

                    show_section(
                        "New student prediction results",
                        "These estimates were generated using the existing "
                        "trained LSTM model.",
                    )

                    uploaded_results = saved_predictions.copy()

                    high_count = int(
                        (uploaded_results["Risk Level"] == "High").sum()
                    )

                    medium_count = int(
                        (uploaded_results["Risk Level"] == "Medium").sum()
                    )

                    low_count = int(
                        (uploaded_results["Risk Level"] == "Low").sum()
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        metric_card(
                            "Students analyzed",
                            f"{len(uploaded_results):,}",
                            "Uploaded records",
                        )

                    with c2:
                        metric_card(
                            "High risk",
                            f"{high_count:,}",
                            "Priority for human review",
                        )

                    with c3:
                        metric_card(
                            "Medium risk",
                            f"{medium_count:,}",
                            "Consider a supportive check-in",
                        )

                    with c4:
                        metric_card(
                            "Low risk",
                            f"{low_count:,}",
                            "Continue routine monitoring",
                        )

                    st.write("")

                    display_risk_distribution(uploaded_results)

                    st.dataframe(
                        uploaded_results[
                            [
                                "Student ID",
                                "LSTM Risk (%)",
                                "Risk Level",
                                "LSTM Prediction",
                                "Recommended Action",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "⬇ Download all new student predictions",
                        data=uploaded_results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="scholartrack_new_student_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.markdown(
                        """
                        <div class="info-note">
                        Review high and medium risk estimates with care.
                        They are intended to support appropriate human
                        follow-up, not to make automatic decisions about
                        a student's education.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as error:
                st.error(
                    "The uploaded CSV could not be read. Check the file "
                    "format and column headers."
                )
                st.code(str(error))


# ============================================================
# EARLY WARNING SYSTEM
# ============================================================

def show_early_warning_page(predictions, df):
    show_hero(
        "Early Warning & Intervention Center",
        "Turn model estimates into structured, human-reviewed follow-up actions.",
        "EARLY INTERVENTION",
    )

    high_students = predictions[
        predictions["Risk Level"] == "High"
    ].copy()

    medium_students = predictions[
        predictions["Risk Level"] == "Medium"
    ].copy()

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "High priority",
            len(high_students),
            "Review first",
        )

    with c2:
        metric_card(
            "Medium priority",
            len(medium_students),
            "Plan a check-in",
        )

    with c3:
        metric_card(
            "All flagged students",
            len(high_students) + len(medium_students),
            "High and medium categories",
        )

    show_section(
        "Intervention planning",
        "Select a student, choose support activities, and record a follow-up "
        "note. The checklist is a session-level planning aid, not a permanent "
        "student record system.",
    )

    candidates = predictions[
        predictions["Risk Level"].isin(["High", "Medium"])
    ].copy()

    if candidates.empty:
        st.success(
            "No students are currently in the high or medium categories."
        )
        return

    candidates = candidates.sort_values(
        "Risk Probability",
        ascending=False,
    )

    chosen_index = st.selectbox(
        "Select student for intervention planning",
        list(candidates.index),
        format_func=lambda idx: (
            f"{candidates.loc[idx, 'Student ID']} — "
            f"{candidates.loc[idx, 'Risk Level']} — "
            f"{candidates.loc[idx, 'LSTM Risk (%)']:.1f}%"
        ),
    )

    student = candidates.loc[chosen_index]

    st.markdown(
        f"""
        <div class="panel">
            <div style="font-size:12px;color:#64748b;font-weight:700;">
                SELECTED STUDENT
            </div>
            <div style="font-size:23px;color:#172554;font-weight:800;
            margin:6px 0 12px 0;">
                {student['Student ID']}
            </div>
            <div style="font-size:14px;color:#334155;">
                Estimated risk: <b>{student['LSTM Risk (%)']:.1f}%</b>
                &nbsp; | &nbsp; Category: <b>{student['Risk Level']}</b>
            </div>
            <div style="font-size:13px;color:#64748b;margin-top:12px;">
                {student['Recommended Action']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    checklist_items = [
        "Arrange a confidential student check-in",
        "Review academic progress and attendance context",
        "Offer tutoring or academic mentoring",
        "Ask whether financial or logistical support is needed",
        "Discuss wellbeing support if appropriate",
        "Agree on a follow-up date",
    ]

    session_key = f"intervention_{student['Student ID']}"

    if session_key not in st.session_state:
        st.session_state[session_key] = {
            "checked": [],
            "note": "",
            "status": "Not started",
        }

    state = st.session_state[session_key]

    st.markdown("### Support checklist")

    for item in checklist_items:
        checked = item in state["checked"]

        if st.checkbox(
            item,
            value=checked,
            key=f"{session_key}_{item}",
        ):
            if item not in state["checked"]:
                state["checked"].append(item)

        elif item in state["checked"]:
            state["checked"].remove(item)

    note = st.text_area(
        "Follow-up notes",
        value=state["note"],
        placeholder="Record a brief, respectful action plan. Avoid sensitive details.",
        key=f"{session_key}_note",
    )

    status_options = [
        "Not started",
        "Contact planned",
        "In progress",
        "Follow-up completed",
    ]

    status = st.selectbox(
        "Intervention status",
        status_options,
        index=status_options.index(state["status"]),
        key=f"{session_key}_status",
    )

    state["note"] = note
    state["status"] = status

    st.caption(
        f"{len(state['checked'])} of {len(checklist_items)} "
        "support activities selected."
    )

    if st.button(
        "Save intervention plan in this session",
        type="primary",
    ):
        st.session_state[session_key] = state

        st.success(
            "Plan updated for this active session. "
            "It is not stored in a permanent database."
        )

    plan = pd.DataFrame(
        [
            {
                "Student ID": student["Student ID"],
                "LSTM Risk (%)": student["LSTM Risk (%)"],
                "Risk Level": student["Risk Level"],
                "Intervention Status": state["status"],
                "Completed Activities": "; ".join(state["checked"]),
                "Follow-up Notes": state["note"],
            }
        ]
    )

    st.download_button(
        "⬇ Download this intervention plan",
        data=plan.to_csv(index=False).encode("utf-8"),
        file_name=f"intervention_{student['Student ID']}.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        <div class="info-note">
        <b>Privacy note:</b> This prototype keeps checklist state in the active
        Streamlit session. Downloaded notes should be handled according to
        your institution's student-data privacy rules.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def load_evaluation_data():
    if not EVALUATION_PATH.exists():
        return None

    try:
        return read_csv_flexible(EVALUATION_PATH)
    except Exception:
        return None


def show_evaluation_page():
    show_hero(
        "LSTM Model Evaluation",
        "Review held-out predictions using standard classification metrics. "
        "This page does not generate or invent test results.",
        "MODEL VALIDATION",
    )

    evaluation_df = load_evaluation_data()

    if evaluation_df is None:
        st.warning(
            "Evaluation data was not found. Add "
            "`results/test_predictions.csv` with `actual` and "
            "`lstm_probability` columns to display held-out test metrics."
        )

        st.markdown(
            """
            <div class="panel">
                <b>Expected CSV structure</b>

                | actual | lstm_probability |
                |---|---:|
                | Dropout | 0.82 |
                | Graduate | 0.14 |
                | Enrolled | 0.27 |

                <p style="color:#64748b;font-size:12px;">
                These values illustrate the required format only. They are
                not actual results and must not be used as model metrics.
                Use probabilities produced by your trained model on an
                untouched test set.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    required = {"actual", "lstm_probability"}

    if not required.issubset(set(evaluation_df.columns)):
        st.error(
            "The evaluation CSV must contain columns named "
            "`actual` and `lstm_probability`."
        )
        st.write("Columns found:", list(evaluation_df.columns))
        return

    actual = evaluation_df["actual"].apply(normalize_target)

    probability = pd.to_numeric(
        evaluation_df["lstm_probability"],
        errors="coerce",
    )

    valid = actual.notna() & probability.notna()

    actual = actual[valid].astype(int)
    probability = probability[valid].clip(0, 1)

    if len(actual) == 0:
        st.error("No valid actual labels and probabilities were found.")
        return

    predicted = (probability >= 0.5).astype(int)

    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(
        actual, predicted, zero_division=0
    )
    recall = recall_score(
        actual, predicted, zero_division=0
    )
    f1 = f1_score(
        actual, predicted, zero_division=0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Accuracy",
            f"{accuracy * 100:.1f}%",
            "Correct classifications",
        )

    with c2:
        metric_card(
            "Precision",
            f"{precision * 100:.1f}%",
            "Dropout-positive predictions",
        )

    with c3:
        metric_card(
            "Recall",
            f"{recall * 100:.1f}%",
            "Dropout cases detected",
        )

    with c4:
        metric_card(
            "F1 score",
            f"{f1 * 100:.1f}%",
            "Precision-recall balance",
        )

    st.caption(f"Valid evaluation records: {len(actual):,}")

    if actual.nunique() == 2:
        try:
            auc = roc_auc_score(actual, probability)
            st.metric("ROC-AUC", f"{auc:.3f}")

        except ValueError:
            st.info(
                "ROC-AUC could not be calculated for this evaluation set."
            )

    else:
        st.info(
            "ROC-AUC requires both actual classes to be represented "
            "in the evaluation set."
        )

    show_section("Confusion matrix")

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(3.8, 3.0))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Dropout", "Dropout"],
        yticklabels=["Not Dropout", "Dropout"],
        ax=ax,
        annot_kws={"size": 10},
        cbar=False,
    )

    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("Actual label", fontsize=9)
    ax.set_title(
        "LSTM Confusion Matrix",
        fontsize=11,
        fontweight="bold",
    )

    ax.tick_params(axis="both", labelsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    show_section("Classification report")

    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Not Dropout", "Dropout"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose().round(3)

    st.dataframe(
        report_df,
        use_container_width=True,
    )

    st.download_button(
        "⬇ Download evaluation report",
        data=report_df.to_csv().encode("utf-8"),
        file_name="scholartrack_lstm_evaluation.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        <div class="info-note">
        <b>Interpretation:</b> Recall measures how many actual dropout cases
        were detected. Precision measures how many predicted dropout cases
        were actual dropout cases. Review both alongside false positives,
        false negatives, data quality, and subgroup performance.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EXPLAINABLE AI
# ============================================================

def show_explainability_page(
    df,
    model,
    prepared,
    selected_student_id=None,
):
    show_hero(
        "Explainable AI",
        "Explore which input features influence an individual model estimate "
        "using SHAP when the required package and model interface are supported.",
        "MODEL INTERPRETABILITY",
    )

    st.markdown(
        """
        <div class="info-note">
        <b>Important:</b> Feature attribution describes how the model responds
        to input features. It does not establish why a student is at risk,
        prove causation, or replace a conversation with the student.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not SHAP_AVAILABLE:
        st.warning(
            "SHAP is not installed. Run `pip install shap` and restart the app "
            "to enable the experimental explanation section."
        )
        return

    sequence = prepared["sequence_3d"]
    static = prepared["static_scaled"]
    feature_names = prepared["feature_names"]

    if len(sequence) == 0:
        st.info("No records are available for explanation.")
        return

    max_student_index = min(len(sequence) - 1, 100)

    student_index = st.number_input(
        "Student row to explain (index in current dataset)",
        min_value=0,
        max_value=max_student_index,
        value=0,
        step=1,
    )

    student_index = int(student_index)

    st.caption(
        "Kernel SHAP can be computationally expensive. "
        "This prototype uses a small background sample."
    )

    if not st.button("Generate SHAP explanation", type="primary"):
        st.info("Click the button to calculate an explanation.")
        return

    try:
        seq_feature_count = sequence.shape[2]
        static_feature_count = static.shape[1]

        flattened = np.concatenate(
            [
                sequence.reshape(len(sequence), -1),
                static,
            ],
            axis=1,
        )

        expected_features = len(feature_names)

        if flattened.shape[1] != expected_features:
            st.error(
                "The flattened input does not match the feature-name list. "
                "Check columns.json and the saved model preprocessing."
            )

            st.write("Flattened input shape:", flattened.shape)
            st.write("Expected feature count:", expected_features)
            st.write("Feature names:", feature_names)
            return

        def predict_flattened(values):
            values = np.asarray(values, dtype=np.float32)

            if values.ndim == 1:
                values = values.reshape(1, -1)

            seq_end = 2 * seq_feature_count

            sequence_part = values[:, :seq_end].reshape(
                -1, 2, seq_feature_count
            )

            static_part = values[
                :,
                seq_end:seq_end + static_feature_count,
            ]

            return get_lstm_probabilities(
                model,
                sequence_part,
                static_part,
            ).reshape(-1, 1)

        rng = np.random.default_rng(RANDOM_STATE)

        background_count = min(10, len(flattened))

        background_indices = rng.choice(
            len(flattened),
            size=background_count,
            replace=False,
        )

        background = flattened[background_indices]

        student_vector = flattened[
            student_index:student_index + 1
        ]

        with st.spinner("Calculating SHAP values..."):
            explainer = shap.KernelExplainer(
                predict_flattened,
                background,
            )

            raw_shap_values = explainer.shap_values(
                student_vector,
                nsamples=50,
            )

        if hasattr(raw_shap_values, "values"):
            raw_shap_values = raw_shap_values.values

        if isinstance(raw_shap_values, list):
            if len(raw_shap_values) == 0:
                st.error("SHAP returned an empty result.")
                return

            raw_shap_values = raw_shap_values[0]

        shap_array = np.asarray(raw_shap_values)
        shap_array = np.squeeze(shap_array)

        if shap_array.ndim == 0:
            shap_values = shap_array.reshape(1)

        elif shap_array.ndim == 1:
            shap_values = shap_array

        elif shap_array.ndim == 2:
            if shap_array.shape[1] == expected_features:
                shap_values = shap_array[0]

            elif shap_array.shape[0] == expected_features:
                shap_values = shap_array[:, 0]

            else:
                st.error(
                    "Cannot match SHAP dimensions to the input features."
                )
                st.write("SHAP output shape:", shap_array.shape)
                st.write("Expected feature count:", expected_features)
                st.write("Feature names:", feature_names)
                return

        elif shap_array.ndim == 3:
            if shap_array.shape[1] == expected_features:
                shap_values = shap_array[0, :, 0]

            elif shap_array.shape[2] == expected_features:
                shap_values = shap_array[0, 0, :]

            else:
                st.error(
                    "The SHAP output contains dimensions that do not "
                    "match the input features."
                )
                st.write("SHAP output shape:", shap_array.shape)
                st.write("Expected feature count:", expected_features)
                st.write("Feature names:", feature_names)
                return

        else:
            st.error(
                "The SHAP output has an unsupported number of dimensions."
            )
            st.write("SHAP output shape:", shap_array.shape)
            st.write("Expected feature count:", expected_features)
            st.write("Feature names:", feature_names)
            return

        shap_values = np.asarray(shap_values).reshape(-1)

        if len(shap_values) != expected_features:
            st.error(
                "SHAP returned a different number of values than expected."
            )

            st.write(
                "Raw SHAP output shape:",
                np.asarray(raw_shap_values).shape,
            )

            st.write(
                "Processed SHAP shape:",
                shap_values.shape,
            )

            st.write(
                "Expected feature count:",
                expected_features,
            )

            st.write(
                "Flattened model input shape:",
                flattened.shape,
            )

            st.write(
                "Sequence input shape:",
                sequence.shape,
            )

            st.write(
                "Static input shape:",
                static.shape,
            )

            st.write(
                "Feature names:",
                feature_names,
            )

            return

        explanation_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP value": shap_values,
                "Absolute impact": np.abs(shap_values),
            }
        ).sort_values(
            "Absolute impact",
            ascending=False,
        )

        st.success("Explanation generated.")

        show_section("Top influential features")

        top = explanation_df.head(12).sort_values("SHAP value")

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.barh(
            top["Feature"],
            top["SHAP value"],
        )

        ax.axvline(0, linewidth=1)

        ax.set_title(
            "SHAP Feature Contributions for Selected Student",
            fontweight="bold",
        )

        ax.set_xlabel(
            "SHAP value (direction relative to model output)"
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            explanation_df.drop(
                columns=["Absolute impact"]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇ Download SHAP explanation",
            data=explanation_df.to_csv(index=False).encode("utf-8"),
            file_name=f"scholartrack_shap_student_{student_index}.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.error(
            "SHAP could not explain this model with the current inputs. "
            "The prediction dashboard can still work. Check the model input "
            "signature and preprocessing before using this feature."
        )

        st.code(str(error))


# ============================================================
# PROJECT INSIGHTS
# ============================================================

def show_project_insights(df, predictions, schema):
    show_hero(
        "Project Insights & Data Quality",
        "Understand the dataset, model inputs, missing values, and the "
        "responsible-use boundaries of ScholarTrack.",
        "PROJECT TRANSPARENCY",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Rows",
            f"{len(df):,}",
            "Records loaded",
        )

    with c2:
        metric_card(
            "Dataset columns",
            f"{len(df.columns):,}",
            "Input fields and any labels",
        )

    with c3:
        metric_card(
            "Model input features",
            f"{len(schema['sem1']) + len(schema['sem2']) + len(schema['static'])}",
            "Sequence plus static inputs",
        )

    show_section("Model input design")

    design_df = pd.DataFrame(
        [
            {
                "Input group": "Semester 1",
                "Number of features": len(schema["sem1"]),
                "Purpose": "First semester academic sequence",
            },
            {
                "Input group": "Semester 2",
                "Number of features": len(schema["sem2"]),
                "Purpose": "Second semester academic sequence",
            },
            {
                "Input group": "Static features",
                "Number of features": len(schema["static"]),
                "Purpose": "Other student attributes used during training",
            },
        ]
    )

    st.dataframe(
        design_df,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("View exact features used by the model"):
        st.markdown("**Semester 1**")
        st.write(schema["sem1"])

        st.markdown("**Semester 2**")
        st.write(schema["sem2"])

        st.markdown("**Static features**")
        st.write(schema["static"])

    show_section("Dataset quality")

    missing = df.isna().sum().sort_values(ascending=False)

    missing_df = (
        missing[missing > 0]
        .rename("Missing values")
        .reset_index()
        .rename(columns={"index": "Column"})
    )

    if missing_df.empty:
        st.success(
            "No missing values were detected in the loaded raw dataset."
        )

    else:
        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Preview of the active dataset**")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇ Download current dataset preview",
        data=df.head(1000).to_csv(index=False).encode("utf-8"),
        file_name="scholartrack_dataset_preview.csv",
        mime="text/csv",
    )

    show_section("Responsible deployment checklist")

    responsible_items = [
        "Validate predictions on a separate held-out test set.",
        "Review false positives and false negatives before deployment.",
        "Check data quality, drift, and subgroup performance.",
        "Restrict access to student records and exported reports.",
        "Allow staff to review and override model-based prioritization.",
        "Never use model risk alone to deny education or impose penalties.",
        "Provide a process for students and staff to correct inaccurate data.",
    ]

    for item in responsible_items:
        st.markdown(f"- {item}")


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.sidebar.markdown(
        """
        <div style="padding:8px 0 20px 0;">
            <div style="font-size:28px;font-weight:800;">
                🎓 ScholarTrack
            </div>
            <div style="font-size:11px;color:#bfdbfe;">
                STUDENT SUCCESS INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # NEW: Added New Student Information to navigation.
    page = st.sidebar.radio(
        "NAVIGATION",
        [
            "Overview",
            "Student Predictions",
            "New Student Information",
            "Early Warning System",
            "Model Evaluation",
            "Explainable AI",
            "Project Insights",
        ],
    )

    st.sidebar.markdown("---")

    # The existing dataset uploader remains available for existing pages.
    # New Student Information has its own separate CSV uploader.
    st.sidebar.markdown("### Dataset")

    st.sidebar.file_uploader(
        "Upload compatible CSV",
        type=["csv"],
        key="uploaded_dataset",
        help=(
            "The file must contain the features specified in "
            "models/columns.json. Use the same data schema as the training dataset."
        ),
    )

    st.sidebar.caption(
        "If no file is uploaded, the existing dashboard tries to use data.csv "
        "from the project folder. New Student Information uses its own input."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Risk thresholds")

    medium_threshold_percent = st.sidebar.slider(
        "Medium risk starts at (%)",
        min_value=10,
        max_value=70,
        value=30,
        step=5,
    )

    high_threshold_percent = st.sidebar.slider(
        "High risk starts at (%)",
        min_value=30,
        max_value=95,
        value=60,
        step=5,
    )

    if high_threshold_percent <= medium_threshold_percent:
        st.sidebar.error(
            "The high-risk threshold must be greater than the medium-risk threshold."
        )

        st.error(
            "Please adjust the risk thresholds in the sidebar so the "
            "high-risk threshold is greater than the medium-risk threshold."
        )

        return

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        """
        <div style="font-size:11px;line-height:1.7;color:#dbeafe;">
            <b>Model:</b> LSTM only<br>
            <b>Purpose:</b> Early support prioritization<br>
            <b>Decision:</b> Human review required
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load the same trained model and preprocessing resources.
    try:
        (
            model,
            seq_scaler,
            static_scaler,
            medians,
            schema,
        ) = load_resources()

    except Exception as error:
        st.error("ScholarTrack could not load its LSTM resources.")

        st.markdown(
            "Check that all five model files exist in the `models` folder."
        )

        st.code(str(error))
        st.stop()

    medium_threshold = medium_threshold_percent / 100
    high_threshold = high_threshold_percent / 100

    # NEW: Route to the new page before loading the existing dashboard
    # dataset. This means manual entry and the separate new-student CSV
    # work even when data.csv is not present.
    if page == "New Student Information":
        show_new_student_page(
            model=model,
            seq_scaler=seq_scaler,
            static_scaler=static_scaler,
            medians=medians,
            schema=schema,
            medium_threshold=medium_threshold,
            high_threshold=high_threshold,
        )

        st.markdown(
            """
            <div class="footer">
                SCHOLARTRACK • LSTM-BASED STUDENT SUPPORT PROTOTYPE<br>
                Model estimates are not certainties. Human review and student privacy
                remain essential.
            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    # Existing pages continue to use the existing dataset.
    try:
        df = get_current_data()

    except Exception as error:
        st.error("The dataset could not be read.")
        st.code(str(error))
        st.stop()

    if df is None:
        st.warning(
            "No dataset is loaded. Place `data.csv` in the project folder "
            "or upload a compatible CSV from the sidebar."
        )
        st.stop()

    if df.empty:
        st.error("The loaded dataset contains no rows.")
        st.stop()

    try:
        predictions, prepared = make_predictions(
            df=df,
            model=model,
            seq_scaler=seq_scaler,
            static_scaler=static_scaler,
            medians=medians,
            schema=schema,
            medium_threshold=medium_threshold,
            high_threshold=high_threshold,
        )

    except Exception as error:
        st.error(
            "The LSTM prediction step failed. Check the input schema, "
            "saved scalers, and the trained model's expected input shapes."
        )

        st.code(str(error))
        st.stop()

    if page == "Overview":
        show_overview_page(df, predictions)

    elif page == "Student Predictions":
        show_predictions_page(df, predictions, prepared)

    elif page == "Early Warning System":
        show_early_warning_page(predictions, df)

    elif page == "Model Evaluation":
        show_evaluation_page()

    elif page == "Explainable AI":
        show_explainability_page(
            df,
            model,
            prepared,
        )

    elif page == "Project Insights":
        show_project_insights(df, predictions, schema)

    st.markdown(
        """
        <div class="footer">
            SCHOLARTRACK • LSTM-BASED STUDENT SUPPORT PROTOTYPE<br>
            Model estimates are not certainties. Human review and student privacy
            remain essential.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()