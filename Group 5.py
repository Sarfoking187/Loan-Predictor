# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- HOME PAGE ---
def Home():
    st.title("🏠 Loan Default Prediction App")
    st.markdown("""
    Welcome to the **Loan Default Prediction Web App**. This tool uses machine learning techniques to
    help predict loan default risk based on uploaded data or a sample dataset.

    Navigate through the pages on the sidebar to:
    - Upload and preview your data
    - Perform data preprocessing
    - Select the best features
    - Train and evaluate Ridge Regression models
    - Make interactive predictions
    """)

# --- DATA IMPORT PAGE ---
def Data_Import():
    st.title("📂 Data Import")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        st.info("No file uploaded. Using sample dataset.")
        df = pd.read_csv("Loan_Default_sample.csv")

    os.makedirs("saved_data", exist_ok=True)
    df.to_csv("saved_data/1_raw_data.csv", index=False)
    st.dataframe(df.head())
    return df

# --- PREPROCESSING PAGE ---
def Preprocess_Data():
    st.title("🧹 Data Preprocessing")
    df = pd.read_csv("saved_data/1_raw_data.csv")

    # Example cleaning
    df_cleaned = df.dropna()
    df_cleaned[df_cleaned.columns[1:]] = df_cleaned[df_cleaned.columns[1:]].apply(pd.to_numeric, errors='coerce')
    df_cleaned = df_cleaned.dropna()

    st.write("After cleaning:")
    st.dataframe(df_cleaned.head())
    df_cleaned.to_csv("saved_data/2_cleaned_data.csv", index=False)
    return df_cleaned

# --- FEATURE SELECTION PAGE ---
def Feature_Selection():
    st.title("📊 Feature Selection")
    df = pd.read_csv("saved_data/2_cleaned_data.csv")

    X = df.drop(columns=['loan_amount'])
    y = df['loan_amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save for later use
    with open("saved_data/X_train_scaled.pkl", "wb") as f:
        pickle.dump(X_train_scaled, f)
    with open("saved_data/X_test_scaled.pkl", "wb") as f:
        pickle.dump(X_test_scaled, f)
    with open("saved_data/y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open("saved_data/y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)

    st.write("Features and target prepared. Proceed to model training.")

# --- MODEL TRAINING PAGE ---
def Train_Model():
    st.title("🧠 Ridge Regression Model")

    with open("saved_data/X_train_scaled.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("saved_data/X_test_scaled.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("saved_data/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open("saved_data/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    alpha = st.slider("Choose Ridge Alpha (regularization)", 0.01, 10.0, 1.0)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.metric("Mean Squared Error", f"{mse:.2f}")
    st.metric("R2 Score", f"{r2:.2f}")

    # Save model
    with open("saved_data/ridge_model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.success("✅ Model trained and saved.")

# --- RESULT INTERPRETATION PAGE ---
def Interpret_Results():
    st.title("📈 Model Results Interpretation")

    with open("saved_data/ridge_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("saved_data/X_test_scaled.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("saved_data/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    y_pred = model.predict(X_test)
    results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    st.subheader("📊 Actual vs Predicted")
    st.dataframe(results.head())

    st.subheader("🔍 Error Distribution")
    errors = y_test - y_pred
    plt.figure(figsize=(8, 4))
    sns.histplot(errors, kde=True)
    st.pyplot(plt.gcf())

# --- CONCLUSION PAGE ---
def Conclusion():
    st.title("📌 Conclusion")
    st.markdown("""
    This loan default prediction app demonstrates how Ridge Regression and Best Subset Feature Selection
    can be used to build predictive models for financial risk. 

    **Key Takeaways:**
    - Regularization reduces overfitting and improves generalization.
    - Scaling and cleaning data significantly improves performance.
    - Ridge Regression performs well when multicollinearity is present.

    You can extend this app by testing other models (like Random Forest or XGBoost), tuning hyperparameters,
    or deploying with real-time APIs.
    """)

# --- SIDEBAR NAVIGATION ---
pages = {
    "Home": Home,
    "Data Import": Data_Import,
    "Preprocessing": Preprocess_Data,
    "Feature Selection": Feature_Selection,
    "Train Model": Train_Model,
    "Results Interpretation": Interpret_Results,
    "Conclusion": Conclusion
}

st.sidebar.title("📚 Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
