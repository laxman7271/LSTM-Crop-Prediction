
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


# =========================================================
# 2. CUSTOM UI
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f4f8f2;
        color: #20352a;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp p,
    .stApp li,
    .stApp label {
        color: #20352a;
    }

    

/* Force readable sidebar text */
section[data-testid="stSidebar"] {
    background-color: #173b2b !important;
}

/* All sidebar text elements */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] * {
    color: #ffffff !important;
}

/* Sidebar navigation */
section[data-testid="stSidebar"] [role="radiogroup"] label,
section[data-testid="stSidebar"] [role="radiogroup"] label p,
section[data-testid="stSidebar"] [role="radiogroup"] label span {
    color: #ffffff !important;
}

/* Sidebar captions and footer */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #d5e6da !important;
}


/* Sidebar navigation labels */
[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: #ffffff !important;
}

/* Keep selected navigation text readable */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #ffffff !important;
}


    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #174c35, #36865b);
        color: #ffffff;
        margin-bottom: 24px;
    }

    .hero h1,
    .hero p {
        color: #ffffff !important;
    }

    .hero p {
        font-size: 16px;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dce7dc;
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #20352a !important;
    }

    [data-testid="stNumberInput"] label {
        color: #20352a !important;
    }

    [data-testid="stNumberInput"] input {
        color: #20352a !important;
        background-color: #ffffff !important;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #247348;
        color: #ffffff !important;
        border: none;
        border-radius: 9px;
        min-height: 42px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #174c35;
        color: #ffffff !important;
    }

    button[data-baseweb="tab"] {
        color: #20352a !important;
    }

    [data-testid="stCaptionContainer"] {
        color: #53665a !important;
    }

    [data-testid="stAlert"] p {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD TRAINED MODELS
# =========================================================
@st.cache_resource
def load_crop_resources():
    model = load_model(MODEL_DIR / "crop_classifier.keras")
    scaler = joblib.load(MODEL_DIR / "crop_scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "crop_label_encoder.pkl")
    return model, scaler, encoder


@st.cache_resource
def load_temperature_model():
    return load_model(MODEL_DIR / "temperature_lstm.keras")


# =========================================================
# 4. DISPLAY SAVED GRAPHS
# =========================================================
def show_graph(filename, title, description):
    st.subheader(title)
    st.caption(description)

    graph_path = DATA_DIR / filename

    if graph_path.exists():
        st.image(str(graph_path), use_container_width=True)
    else:
        st.warning(
            f"Graph not found: data/{filename}. "
            "Run the corresponding plotting script."
        )


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🌱 Smart Agriculture")
st.sidebar.caption("Deep Learning Project")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Crop Recommendation",
        "Temperature Forecasting",
        "Model Graphs",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Models: TensorFlow / Keras")
st.sidebar.caption("Interface: Streamlit")


# =========================================================
# 6. DASHBOARD
# =========================================================
if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🌱 Smart Agriculture AI</h1>
        <p>
            Crop classification and historical temperature
            forecasting using deep learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Crop Categories", "22")
    col2.metric("Crop Test Accuracy", "99.32%")
    col3.metric("Temperature Test MAE", "0.67 °C")

    st.subheader("Explore the system")

    crop_col, temp_col = st.columns(2)

    with crop_col:
        st.markdown("### 🌾 Crop Recommendation")
        st.write(
            "Enter soil nutrients and environmental conditions "
            "to predict a crop category."
        )

    with temp_col:
        st.markdown("### 🌡️ Temperature Forecasting")
        st.write(
            "Test the LSTM model using historical temperature "
            "sequences."
        )

    st.info(
        "Choose Crop Recommendation or Temperature Forecasting "
        "from the navigation menu in the left sidebar."
    )

    st.subheader("Model performance")

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        show_graph(
            "crop_accuracy.png",
            "Crop Classifier Accuracy",
            "Training and validation accuracy across epochs.",
        )

    with graph_col2:
        show_graph(
            "actual_vs_predicted.png",
            "Temperature Predictions",
            "Actual versus predicted temperatures on test data.",
        )


# =========================================================
# 7. CROP RECOMMENDATION
# =========================================================
elif page == "Crop Recommendation":

    st.title("🌾 Crop Recommendation")
    st.write(
        "Enter the soil and weather values used by the trained "
        "crop classification model."
    )

    with st.form("crop_prediction_form"):

        left, right = st.columns(2)

        with left:
            n = st.number_input(
                "Nitrogen (N)", min_value=0.0, value=90.0
            )
            p = st.number_input(
                "Phosphorus (P)", min_value=0.0, value=42.0
            )
            k = st.number_input(
                "Potassium (K)", min_value=0.0, value=43.0
            )
            temperature = st.number_input(
                "Temperature (°C)", value=20.88
            )

        with right:
            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=82.0,
            )
            ph = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=6.5,
            )
            rainfall = st.number_input(
                "Rainfall (mm)", min_value=0.0, value=202.94
            )

        submitted = st.form_submit_button(
            "🌱 Recommend Crop",
            use_container_width=True,
        )

    if submitted:
        try:
            model, scaler, encoder = load_crop_resources()

            feature_names = [
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall",
            ]

            input_data = pd.DataFrame(
                [[n, p, k, temperature, humidity, ph, rainfall]],
                columns=feature_names,
            )

            scaled_input = scaler.transform(input_data)

            probabilities = model.predict(
                scaled_input, verbose=0
            )[0]

            best_index = int(np.argmax(probabilities))
            predicted_crop = encoder.inverse_transform(
                [best_index]
            )[0]

            confidence = float(
                probabilities[best_index] * 100
            )

            st.success(
                f"Recommended crop: {predicted_crop.title()}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%",
            )

            st.caption(
                "Model confidence is not a guarantee of farming success."
            )

            st.subheader("Top 5 Crop Predictions")

            top_indices = np.argsort(probabilities)[-5:][::-1]

            chart_data = pd.DataFrame({
                "Crop": encoder.inverse_transform(top_indices),
                "Probability (%)": probabilities[top_indices] * 100,
            })

            st.bar_chart(
                chart_data.set_index("Crop")
            )

        except Exception as error:
            st.error(f"Crop prediction failed: {error}")


# =========================================================
# 8. TEMPERATURE FORECASTING
# =========================================================
elif page == "Temperature Forecasting":

    st.title("🌡️ LSTM Temperature Forecasting")

    st.write(
        "Select a held-out historical test sample to compare "
        "the recorded temperature with the model prediction."
    )

    try:
        weather_data = np.load(
            DATA_DIR / "weather_sequences.npz"
        )

        X_test = weather_data["X_test"]
        y_test = weather_data["y_test"]

        scaler_min = float(
            weather_data["scaler_min"].item()
        )
        scaler_scale = float(
            weather_data["scaler_scale"].item()
        )

        sample_number = st.number_input(
            "Test Sample Number",
            min_value=1,
            max_value=len(X_test),
            value=1,
            step=1,
        )

        if st.button(
            "🌡️ Predict Temperature",
            type="primary",
        ):

            model = load_temperature_model()

            sequence = X_test[
                int(sample_number) - 1:int(sample_number)
            ]

            scaled_prediction = float(
                model.predict(sequence, verbose=0)[0][0]
            )

            predicted_temperature = (
                scaled_prediction / scaler_scale + scaler_min
            )

            actual_temperature = (
                float(y_test[int(sample_number) - 1][0])
                / scaler_scale
                + scaler_min
            )

            absolute_error = abs(
                actual_temperature - predicted_temperature
            )

            a, b, c = st.columns(3)

            a.metric(
                "Actual Temperature",
                f"{actual_temperature:.2f} °C",
            )

            b.metric(
                "Predicted Temperature",
                f"{predicted_temperature:.2f} °C",
            )

            c.metric(
                "Absolute Error",
                f"{absolute_error:.2f} °C",
            )

            st.info(
                "This is a historical test example, not a live "
                "forecast for tomorrow."
            )

            show_graph(
                "actual_vs_predicted.png",
                "Actual vs Predicted Temperatures",
                "Overall performance on the held-out test period.",
            )

    except Exception as error:
        st.error(f"Temperature prediction failed: {error}")


# =========================================================
# 9. MODEL GRAPHS
# =========================================================
elif page == "Model Graphs":

    st.title("📊 Model Performance Dashboard")

    st.write(
        "Explore the training and evaluation graphs generated "
        "by your project scripts."
    )

    temperature_tab, crop_tab = st.tabs([
        "🌡️ Temperature LSTM",
        "🌾 Crop Classifier",
    ])

    with temperature_tab:

        show_graph(
            "training_loss.png",
            "Temperature Training Loss",
            "Training loss across epochs.",
        )

        show_graph(
            "actual_vs_predicted.png",
            "Actual vs Predicted Temperature",
            "Comparison on the held-out test dataset.",
        )

    with crop_tab:

        show_graph(
            "crop_accuracy.png",
            "Crop Classifier Accuracy",
            "Training and validation accuracy.",
        )

        show_graph(
            "crop_loss.png",
            "Crop Classifier Loss",
            "Training and validation loss.",
        )


# =========================================================
# 10. ABOUT PROJECT
# =========================================================
elif page == "About Project":

    st.title("📘 About the Project")

    st.subheader("Problem Statement")

    st.write(
        "This project demonstrates two deep learning applications: "
        "historical temperature forecasting using LSTM and crop "
        "classification using a neural network."
    )

    st.subheader("Datasets")

    st.markdown("""
    **Weather dataset**
    - Source: NASA POWER
    - Location: Warangal
    - Period: 2001–2020
    - Daily records: 7,305
    - Input: Previous 30 days of temperature

    **Crop recommendation dataset**
    - Records: 2,200
    - Crop categories: 22
    - Features: N, P, K, temperature, humidity, pH and rainfall
    """)

    st.subheader("Model Results")

    a, b = st.columns(2)

    a.metric("Temperature MAE", "0.67 °C")
    b.metric("Temperature RMSE", "0.90 °C")

    a, b = st.columns(2)

    a.metric("Crop Test Accuracy", "99.32%")
    b.metric("Crop Test Samples", "440")

    st.subheader("Limitations")

    st.markdown("""
    - The temperature model uses previous temperature values.
    - The crop classifier uses its own crop dataset.
    - These are independent models.
    - Test results do not guarantee real-world performance.
    - Agricultural decisions should consider local conditions
      and expert advice.
    """)

    st.subheader("Technologies Used")

    st.write(
        "Python · TensorFlow/Keras · NumPy · Pandas · "
        "scikit-learn · Streamlit"
    )


# =========================================================
# 11. FOOTER
# =========================================================
st.divider()

st.caption(
    "Smart Agriculture AI | Academic Demonstration | "
    "Python • TensorFlow • Streamlit"
)
