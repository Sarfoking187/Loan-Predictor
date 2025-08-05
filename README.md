
````markdown
# Loan Default Prediction Web App

This interactive web application, developed using Python, Streamlit, and Scikit-learn, provides a complete machine learning pipeline to predict loan default amounts. The app walks users through the data science workflow—from data import and preprocessing to feature selection, model training, evaluation, and real-time predictions.

> Developed by Group 5 as part of an Applied Regression and Machine Learning course.

---

## App Overview

The project simulates a real-world use case where a data science team is tasked with building a predictive system for financial institutions. Users can:

- Explore loan application datasets.
- Preprocess, clean, and encode data.
- Select features via best subset selection.
- Train and evaluate a Ridge Regression model.
- Make interactive predictions using custom inputs.

---

## Machine Learning Pipeline

| Step | Description |
|------|-------------|
| **1. Data Import and Overview** | Upload and explore the dataset with visual summaries |
| **2. Data Preprocessing** | Handle missing values, encode categoricals, scale numericals |
| **3. Feature Selection** | Sequential Forward Selection with Ridge Regression |
| **4. Model Training** | Ridge Regression with adjustable `alpha`, cross-validation |
| **5. Evaluation** | RMSE, R², actual vs predicted plots, feature importance |
| **6. Prediction** | Real-time prediction form with preprocessing pipeline |
| **7. Conclusion** | Summary of results, limitations, and future improvements |

---

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend / ML**: [Scikit-learn](https://scikit-learn.org/)
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: Pandas, NumPy
- **Persistence**: Pickle for storing artifacts

---

## 📂 Project Structure

```bash
├── Loan_Default.csv              # Sample dataset (user uploaded in app)
├── LDP.jpg                       # Logo
├── saved_data/                   # Folder for storing intermediate artifacts
│   ├── 1_raw_data.csv
│   ├── 2_column_types.pkl
│   ├── 3_preprocessor.pkl
│   ├── 4_processed_data.csv
│   ├── 5_best_subset_features.pkl
│   ├── 6_cv_results.pkl
│   ├── 7_trained_model.pkl
│   ├── 8_predictions.csv
├── loan_default_ridge_finalapp.py # Main Streamlit script
└── README.md                      # Project documentation
````

---

## Dataset Information

* **Source**: [Kaggle - Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
* **Target Variable**: `loan_amount`
* **Features**: Demographic and financial indicators

---

## Team Members

| Name                   | Student ID | Role                | Deployment                                                                        |
| ---------------------- | ---------- | ------------------- | --------------------------------------------------------------------------------- |
| Kingsley Sarfo         | 22252461   | Project Coordinator | [App Link](https://loan-predictor-hbbz24vwfzaue2qx4hwcat.streamlit.app)           |
| Francisca Manu Sarpong | 22255796   | Data Preprocessing  | [App Link](https://kftalde5ypwd5a3qqejuvo.streamlit.app)                          |
| George Owell           | 22256146   | Model Evaluation    | [App Link](https://loandefaultpredictionapp-utmbic9znd7uzqqhs9zgo6.streamlit.app) |
| Barima Owiredu Addo    | 22254055   | UI Testing          | [App Link](https://loandefaultapp-ky4yy9kmt6ehsq8jqdcgs2.streamlit.app)           |
| Akrobettoe Marcus      | 11410687   | Feature Selection   | [App Link](https://models-loan-default-prediction.streamlit.app)                  |

---

## Instructions

1. Run the app using `loan_default_ridge_finalapp.py`
2. Use the sidebar to navigate between pages.
3. Start from **Data Import** and proceed sequentially for best results.

---

## Features

* End-to-end ML pipeline with persistent storage
* Real-time prediction interface
* Custom preprocessing with numerical/categorical handling
* Ridge regression with feature selection and cross-validation
* Visual analytics and correlation summaries

---

## Future Improvements

* Integrate additional predictive models (e.g., Lasso, Gradient Boosting)
* Deploy via Docker or Streamlit Cloud for production use
* Add automated unit tests for reproducibility
* Incorporate fairness metrics for financial decisioning

---

## License

This project is for academic use only. Refer to your institution’s policy on academic integrity before reuse.

---




