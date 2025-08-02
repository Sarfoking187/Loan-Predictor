Here's a professional and comprehensive `README.md` file for your Streamlit-based Loan Default Prediction project:

---

````markdown
# Loan Default Prediction System

This is an interactive **Loan Default Prediction Web Application** built with **Streamlit**, leveraging machine learning and advanced data visualization techniques to assess the likelihood of loan default. It is designed for use by financial institutions, loan officers, and analysts to support data-driven decision-making.

---

## Key Features

### Data Upload & EDA
- Upload your own CSV loan dataset.
- View descriptive statistics, missing value analysis, and feature summaries.
- Visualize:
  - Distributions of numerical and categorical features.
  - Target variable balance (default vs non-default).
  - Correlation heatmaps.
  - Interactive scatterplots (e.g., Debt-to-Income vs Credit Score).

### Data Preprocessing
- Intelligent pipeline for:
  - Missing value imputation (median for numeric, mode for categorical).
  - Standardization of numerical data.
  - One-hot encoding of categorical data.
- Saves preprocessed dataset for reuse and model training.

### Feature Selection
- Correlation-based filtering.
- Sequential Feature Selector using Logistic Regression.
- Interactive evaluation using metrics like accuracy, precision, recall, and F1.
- Feature importance bar plots.

### Model Training
- Trains a Random Forest Classifier.
- Hyperparameter tuning (number of trees, depth, bootstrap settings).
- 5-fold Cross-Validation.
- Model is saved and reused for prediction and evaluation.

### Model Evaluation
- Displays confusion matrix.
- Shows precision, recall, accuracy, and F1 Score.
- Bar plot of feature importances.
- Saves evaluation results for auditing.

### Interactive Predictions
- Input new applicant data via form.
- Real-time default probability predictions.
- Risk classification:
  - Low Risk
  - ⚠Medium Risk
  - High Risk
- Debt-to-income and collateral coverage ratios.
- Risk factor detection and warning alerts.

### Results Interpretation & Team Credits
- Summary of model performance.
- Business recommendations.
- Limitations and future directions.
- Group contributions and Streamlit deployment links.

---

## 🛠Tech Stack

| Technology     | Role                        |
|----------------|-----------------------------|
| Python         | Core programming language   |
| Streamlit      | Web application framework   |
| pandas, NumPy  | Data processing             |
| seaborn, matplotlib | Visualizations         |
| scikit-learn   | ML modeling and pipelines   |
| pickle         | Saving and loading models   |
| Git/GitHub     | Version control and hosting |

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/loan-default-prediction-app.git
cd loan-default-prediction-app
````

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## Project Structure

```
📁 saved_data/
    └── Preprocessed data, model artifacts, and evaluation results

📄 app.py
📄 Loan_Default.csv (https://www.kaggle.com/datasets/yasserh/loan-default-dataset))
📄 requirements.txt
📄 README.md
```

---

## Team Members (Group 5)

    | Name                     | Student ID | Role                                             | Deployment link                                              |
    |--------------------------|------------|--------------------------------------------------|--------------------------------------------------------------|
    | Kingsley Sarfo           | 22252461   | Project Coordination, App Design & Preprocessing | https://loan-predictor-hbbz24vwfzaue2qx4hwcat.streamlit.app  |                   
    | Francisca Manu Sarpong   | 22255796   | Documentation & Deployment                       | https://kftalde5ypwd5a3qqejuvo.streamlit.app |               
    | George Owell             | 22256146   | Model Evaluation & Cross-validation              | loandefaultpredictionapp-utmbic9znd7uzqqhs9zgo6.streamlit.app|
    | Barima Owiredu Addo      | 22254055   | UI & Prediction Testing                          | https://loandefaultapp-ky4yy9kmt6ehsq8jqdcgs2.streamlit.app/ |
    | Akrobettoe Marcus        | 11410687   | Feature Selection & Model Training               | https://models-loan-default-prediction.streamlit.app/        |

---

## Future Improvements

* Add support for other ML models (e.g., XGBoost, Neural Networks).
* Include additional features like employment history and credit history length.
* Integrate a model monitoring dashboard.
* Add support for batch predictions and PDF report generation.

---

## License

MIT License. Feel free to use and modify with attribution.

---

## Contact

For inquiries or suggestions, please contact any member of **Group 5**.






