import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="XAI Clinical Dashboard", layout="wide")

# =========================
# Load model and data
# =========================
model = joblib.load("rf")  # Make sure this is a Pipeline with 'preprocess' and 'clf'

df = pd.read_csv("Heart-disease-.csv")

# Binary target
df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
X = df.drop(columns=["num", "target"])
y = df["target"]

# =========================
# Data Cleaning
# =========================
# Replace newline and empty strings with NaN
X = X.replace(r'^\s*$', np.nan, regex=True)
X = X.replace('\n', np.nan)

# Convert numeric columns to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='ignore')

# Drop rows with NaN
X = X.dropna().reset_index(drop=True)
y = y.loc[X.index]  # Ensure target aligns

# =========================
# Sidebar: Patient Selection
# =========================
st.sidebar.title("Patient Selection")
patient_id = st.sidebar.slider(
    "Select patient index",
    0, len(X) - 1, 0
)
patient = X.iloc[[patient_id]]

# =========================
# Model Prediction
# =========================
pred_proba = model.predict_proba(patient)[0][1]
prediction = model.predict(patient)[0]

st.subheader("Model Prediction")
st.metric(
    label="Heart Disease Probability",
    value=f"{pred_proba:.2f}"
)
st.write("Prediction:", "🟥 Disease" if prediction == 1 else "🟩 No Disease")

# =========================
# Global SHAP
# =========================
st.subheader("Global Feature Importance (SHAP)")

preprocessor = model.named_steps["preprocess"]
X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

explainer = shap.Explainer(model.named_steps["clf"], X_transformed)
shap_values = explainer(X_transformed, check_additivity=False)

fig, ax = plt.subplots()
shap.summary_plot(
    shap_values[:, :, 1],
    X_transformed,
    feature_names=feature_names,
    show=False
)
st.pyplot(fig)
plt.close(fig)

# =========================
# LIME: Local Explanation
# =========================
st.subheader("Local Explanation (LIME)")

# Step 1: Use preprocessed data (numeric + one-hot encoded)
X_lime = X_transformed  # preprocessed features the model actually sees

# Step 2: Specify categorical feature indices (columns that are categorical)
categorical_features = [
    X.columns.get_loc(col)
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
]

# Step 3: Create the LIME explainer
lime_explainer = LimeTabularExplainer(
    training_data=X_lime,                 # numeric dataset for LIME to perturb
    feature_names=feature_names,          # human-readable feature names
    categorical_features=categorical_features,  # indices of categorical columns
    class_names=["No Disease", "Disease"],       # names of target classes
    mode="classification",                # classification problem
    discretize_continuous=True            # simplify numeric features into intervals
)

# Step 4: Define wrapper predict function
def predict_fn_lime(data):
    """
    LIME generates numeric perturbations. Our model expects a DataFrame.
    This function converts LIME's numeric array into a DataFrame and
    returns class probabilities from the trained pipeline.
    """
    df_temp = pd.DataFrame(data, columns=feature_names)
    return model.named_steps["clf"].predict_proba(df_temp)



# Display the patient’s feature values in a table
st.markdown("**Patient Feature Values:**")
patient_df = pd.DataFrame(
    X.iloc[patient_id].values.reshape(1, -1),
    columns=X.columns
)
st.dataframe(patient_df.T.rename(columns={0: "Value"}))  # Transpose to show features vertically

# Step 6: Explain this patient with LIME
lime_exp = lime_explainer.explain_instance(
    data_row=X_lime[patient_id],   # preprocessed features
    predict_fn=predict_fn_lime,    # wrapper to call model on perturbed data
    num_features=10                # top 10 most important features
)

# Step 7: Visualize LIME explanation in Streamlit
fig3 = lime_exp.as_pyplot_figure()  # convert LIME explanation to a figure
st.pyplot(fig3)                      # display in Streamlit
plt.close(fig3)
                     



