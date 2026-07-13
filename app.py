"""
Streamlit web app for making predictions with a trained Random Forest pipeline.
=============================================================================

HOW TO RUN
----------
1. Install the requirements (once):

       pip install streamlit pandas scikit-learn joblib

2. Put your trained pipeline file next to this script (see MODEL_PATH below).
   It should be a scikit-learn Pipeline saved with joblib, e.g.:

       import joblib
       joblib.dump(my_pipeline, "model.joblib")

3. From a terminal, in this folder, run:

       streamlit run app.py

   Your browser will open automatically at http://localhost:8501

-----------------------------------------------------------------------------
WHAT YOU NEED TO EDIT
---------------------
- MODEL_PATH        -> the filename of your saved pipeline.
- NUMERIC_FEATURES  -> your numeric input columns (with sensible defaults).
- CATEGORICAL_FEATURES -> your categorical input columns (with the choices).
The app uses these to build the input widgets and to order the columns
correctly before calling the model.
=============================================================================
"""


import streamlit as st
import pandas as pd
import joblib

# =============================================================================
# 1. CONFIGURATION  --  EDIT THIS SECTION TO MATCH YOUR MODEL
# =============================================================================

# Path to your saved Random Forest pipeline (joblib or pickle file).
MODEL_PATH = "model.joblib"

# --- Numeric features -------------------------------------------------------
# For each numeric input the model expects, provide a default value.
# Format:  "column_name": default_value
# Example below assumes a housing-style model -- REPLACE with your own columns.
NUMERIC_FEATURES = {
    "feature_1": 0.0,
    "feature_2": 0.0,
    "feature_3": 0.0,
    
}

# --- Categorical features ---------------------------------------------------
# For each categorical input, list the allowed choices (they become a dropdown).
# Format:  "column_name": ["choice_a", "choice_b", ...]
# REPLACE with your own columns and categories.
CATEGORICAL_FEATURES = {
    "category_1": ["option_a", "option_b", "option_c"],
}

# A friendly label for whatever your model predicts (shown in the UI).
TARGET_LABEL = "Predicted value"

# =============================================================================
# 2. LOAD THE MODEL  (cached so it only loads once, not on every interaction)
# =============================================================================


@st.cache_resource
def load_model(path):
    """Load and cache the trained pipeline from disk."""
    return joblib.load(path)


# =============================================================================
# 3. PAGE SETUP
# =============================================================================

st.set_page_config(page_title="ML Predictor", page_icon="🌲", layout="centered")
st.title("🌲 Random Forest Predictor")
st.write(
    "Make a prediction by **uploading a CSV** or **entering values manually** below."
)

# Try to load the model up front and show a clear error if it isn't there yet.
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find the model file **'{MODEL_PATH}'**.\n\n"
        "Save your trained pipeline with `joblib.dump(pipeline, \"model.joblib\")` "
        "and place it in the same folder as this app, then reload the page."
    )
    st.stop()
except Exception as e:  # noqa: BLE001 - surface any load error to the user
    st.error(f"Failed to load the model: {e}")
    st.stop()

# The full, ordered list of columns the model expects.
FEATURE_ORDER = list(NUMERIC_FEATURES.keys()) + list(CATEGORICAL_FEATURES.keys())


def predict(input_df):
    """Run the pipeline on a DataFrame and return the prediction array."""
    # Ensure columns are in the exact order/selection the model was trained on.
    input_df = input_df[FEATURE_ORDER]
    return model.predict(input_df)


# =============================================================================
# 4. TWO WAYS TO PROVIDE INPUT  (tabs keep the UI clean)
# =============================================================================

tab_manual, tab_csv = st.tabs(["✍️ Enter values", "📄 Upload CSV"])

# ---- 4a. Manual entry with interactive widgets -----------------------------
with tab_manual:
    st.subheader("Enter feature values")

    # Collect user input into a dict, one widget per feature.
    user_input = {}

    # Number inputs for the numeric features.
    for name, default in NUMERIC_FEATURES.items():
        user_input[name] = st.number_input(name, value=float(default))

    # Dropdowns (selectboxes) for the categorical features.
    for name, choices in CATEGORICAL_FEATURES.items():
        user_input[name] = st.selectbox(name, choices)

    # Predict when the button is clicked.
    if st.button("Predict", type="primary"):
        # Build a single-row DataFrame from the collected inputs.
        row = pd.DataFrame([user_input])
        try:
            result = predict(row)[0]
            # Show the result prominently.
            st.success("Prediction complete!")
            st.metric(label=TARGET_LABEL, value=f"{result:,.2f}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")

# ---- 4b. CSV upload for batch predictions ----------------------------------
with tab_csv:
    st.subheader("Upload a CSV file")
    st.caption(
        "The CSV must contain these columns: " + ", ".join(f"`{c}`" for c in FEATURE_ORDER)
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame.
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head())

        # Check that all required columns are present before predicting.
        missing = [c for c in FEATURE_ORDER if c not in data.columns]
        if missing:
            st.error("Your CSV is missing these required columns: " + ", ".join(missing))
        elif st.button("Predict for all rows", type="primary"):
            try:
                predictions = predict(data)
                # Attach predictions as a new column and show them.
                results = data.copy()
                results[TARGET_LABEL] = predictions
                st.success(f"Made {len(results)} prediction(s).")
                st.dataframe(results)

                # Let the user download the results as a CSV.
                st.download_button(
                    "Download results as CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")
