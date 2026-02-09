# Explainable AI using SHAP and LIME
Explainable AI project using SHAP and LIME to interpret predictions of a Random Forest classifier.
## Overview
This project demonstrates how **SHAP** and **LIME** can be used to explain
predictions made by a machine learning model.

The goal is to improve **model transparency and trust** by understanding
feature contributions at both global and local levels.

## Dataset
- Dataset used: (Kaggle heart disease.CSV)
- Features: numerical and categorical
- Target variable: (classification

## Model
- Algorithm: (Random Forest)
- Evaluation metric: Accuracy / Precision / Recall

## Explainability Techniques
### SHAP
- Global feature importance
- Local explanations for individual predictions

### LIME
- Local interpretable explanations
- Instance-level feature impact

## Results
- SHAP revealed the most influential features affecting predictions
- LIME provided human-readable explanations for individual samples
- 
## Interactive Visualization with Streamlit
The project includes a **Streamlit-based web application** that enables
interactive exploration of a Random Forest classifier. Users can input feature
values and view corresponding predictions along with **SHAP** and **LIME**
explanations, improving accessibility and interpretability of the model.

## Tools & Libraries
- Python
- scikit-learn
- SHAP
- LIME
- NumPy, Pandas, Matplotlib
- streamlit

## What I Learned
- How to interpret black-box models
- Differences between SHAP and LIME
- Importance of explainability in ML systems
- Visualisation of the interpreted Results
