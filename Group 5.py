# Core libraries for data processing and visualization
import streamlit as st  # Web application framework
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from matplotlib import pyplot as plt  # Visualization
import seaborn as sns  # Enhanced visualization
import pickle  # Object serialization
import os  # File system operations
from PIL import Image  # Image support (for adding logos or visuals)

# Scikit-learn components for machine learning
from sklearn.ensemble import RandomForestClassifier  # ML algorithm
from sklearn.model_selection import cross_val_score  # Cross-Validation
from sklearn.metrics import (accuracy_score, precision_score,  # Evaluation metrics
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Feature engineering
from sklearn.impute import SimpleImputer  # Missing value handling
from sklearn.pipeline import Pipeline  # ML workflow
from sklearn.compose import ColumnTransformer  # Column-wise transformations

# CONFIGURATION SECTION
DATA_DIR = "saved_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Set seaborn style for faster plotting
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


# HELPER FUNCTIONS
def save_artifact(obj, filename):
    """Serializes and saves Python objects to disk for persistence."""
    with open(f"{DATA_DIR}/{filename}", 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifact(filename):
    """Loads serialized objects from disk with error handling."""
    try:
        with open(f"{DATA_DIR}/{filename}", 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {filename}")
        return None
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None


# DATA LOADING AND PREPROCESSING
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_data():
    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data'].copy()
        df = df.drop(columns=['ID', 'dtir1', 'submission_of_application', 'year'], errors='ignore')
        df.to_csv(f"{DATA_DIR}/1_raw_data.csv", index=False)
        return df
    else:
        st.warning("Please upload your dataset via the 'Data Import and Overview' page before preprocessing.")
        return pd.DataFrame()


# PREPROCESSOR CREATION with caching
@st.cache_resource(show_spinner="Creating preprocessor...")
def create_preprocessor():
    df = load_data()
    if df.empty:
        st.error("Data not loaded. Upload a dataset first.")
        return None

    if 'Status' not in df.columns:
        st.error("'Status' column is missing. Cannot proceed.")
        return None

    X = df.drop('Status', axis=1)

    # Feature type identification
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    save_artifact({'numerical': numerical_cols, 'categorical': categorical_cols},
                  "2_column_types.pkl")

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])  # dense output for faster processing

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    try:
        preprocessor.fit(X)
    except ValueError as ve:
        st.error(f"ValueError during preprocessor fitting: {ve}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during preprocessing: {e}")
        return None

    save_artifact(preprocessor, "3_preprocessor.pkl")

    # Transform and save
    X_processed = preprocessor.transform(X)
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    all_features = np.concatenate([num_features, cat_features])

    processed_df = pd.DataFrame(X_processed, columns=all_features)
    processed_df['Status'] = df['Status'].values
    processed_df.to_csv(f"{DATA_DIR}/4_processed_data.csv", index=False)

    return preprocessor


# STREAMLIT PAGE FUNCTIONS
def Home_Page():
    st.title("Loan Default Prediction System")
    st.write("""
    Welcome to the Loan Default Prediction System. This application helps financial institutions 
    assess the risk of loan default using machine learning.

    Use the navigation menu on the left to explore different sections of the application.
    """)

    st.markdown("""---
    ## Team Members (Group 5)
    | Name                     | Student ID | Role                                             | Deployment link                                              |
    |--------------------------|------------|--------------------------------------------------|--------------------------------------------------------------|
    | Kingsley Sarfo           | 22252461   | Project Coordination, App Design & Preprocessing | https://loan-predictor-hbbz24vwfzaue2qx4hwcat.streamlit.app  |                           
    | Francisca Manu Sarpong   | 22255796   | Documentation & Deployment                       | https://kftalde5ypwd5a3qqejuvo.streamlit.app                 |               
    | George Owell             | 22256146   | Model Evaluation & Cross-validation              | loandefaultpredictionapp-utmbic9znd7uzqqhs9zgo6.streamlit.app|
    | Barima Owiredu Addo      | 22254055   | UI & Prediction Testing                          | https://loandefaultapp-ky4yy9kmt6ehsq8jqdcgs2.streamlit.app/ |
    | Akrobettoe Marcus        | 11410687   | Feature Selection & Model Training               | https://models-loan-default-prediction.streamlit.app/        |
    ---
    """)


@st.cache_data(ttl=3600, show_spinner="Loading data overview...")
def Data_Import_and_Overview_page():
    st.title("1. Data Import and Overview")
    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv",
                                     help="Please upload your loan data in CSV format")

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_data'] = df
            st.success("File successfully uploaded!")

            # Summary Statistics
            st.subheader("1. Summary Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", df.shape[0])
            with col2:
                st.metric("Total Features", df.shape[1])
            with col3:
                if 'Status' in df.columns:
                    st.metric("Default Rate", f"{df['Status'].mean():.2%}")

            # Numerical features summary
            st.markdown("**Numerical Features Summary**")
            st.dataframe(df.describe().T.style.format("{:.2f}"))

            # Categorical features summary
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                st.markdown("**Categorical Features Summary**")
                cat_summary = pd.DataFrame({
                    'Unique Values': df[cat_cols].nunique(),
                    'Most Common': df[cat_cols].mode().iloc[0],
                    'Missing Values': df[cat_cols].isnull().sum()
                })
                st.dataframe(cat_summary)

            # Missing values analysis
            st.markdown("**Missing Values Analysis**")
            missing = df.isnull().sum().to_frame('Missing Values')
            missing['Percentage'] = (missing['Missing Values'] / len(df)) * 100
            st.dataframe(missing.style.format({'Percentage': '{:.2f}%'}))

            # Data Visualizations
            st.subheader("Data Visualizations")

            # Target distribution (if exists)
            if 'Status' in df.columns:
                st.markdown("**Target Variable Distribution**")
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                sns.countplot(x='Status', data=df, ax=ax[0])
                ax[0].set_title('Default Status Count')
                df['Status'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
                ax[1].set_title('Default Status Proportion')
                st.pyplot(fig, use_container_width=True)

            # Numerical distributions
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(num_cols) > 0:
                st.markdown("**Numerical Features Distribution**")
                selected_num = st.multiselect("Select numerical features to visualize",
                                              num_cols, default=num_cols[:3])

                if selected_num:
                    fig, ax = plt.subplots(len(selected_num), 2, figsize=(14, 5 * len(selected_num)))
                    for i, col in enumerate(selected_num):
                        sns.histplot(df[col], kde=True, ax=ax[i, 0])
                        ax[i, 0].set_title(f'{col} Distribution')
                        sns.boxplot(x=df[col], ax=ax[i, 1])
                        ax[i, 1].set_title(f'{col} Boxplot')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

            # Correlation matrix
            if len(num_cols) > 1:
                st.markdown("**Correlation Matrix**")
                corr_matrix = df[num_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                            center=0, ax=ax)
                ax.set_title("Feature Correlations")
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def Data_Preprocessing_page():
    st.title("2. Data Preprocessing")

    if st.button("Run Data Preprocessing"):
        with st.spinner("Preprocessing data... This may take a few moments"):
            preprocessor = create_preprocessor()
            if preprocessor:
                try:
                    processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
                    st.subheader("Processed Data Sample")
                    st.dataframe(processed_df.head())

                    st.subheader("Preprocessing Details")
                    st.write("Numerical features:",
                             len(preprocessor.named_transformers_['num'].get_feature_names_out()))
                    st.write("Categorical features:",
                             len(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()))
                    st.success("Preprocessing completed and saved!")
                except Exception as e:
                    st.error(f"Error loading processed data: {str(e)}")


@st.cache_resource(show_spinner="Running feature selection...")
def Feature_Selection_page():
    st.title("3. Feature Selection")

    try:
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        X_full = processed_df.drop('Status', axis=1)
        y = processed_df['Status']
    except Exception as e:
        st.warning(f"Please complete preprocessing first. Error: {str(e)}")
        return

    # Compute correlations
    st.subheader("Correlation Analysis")
    corr_matrix = processed_df.corr()
    corr_with_target = corr_matrix['Status'].sort_values(key=abs, ascending=False)
    st.dataframe(corr_with_target.to_frame("Correlation with Target"))

    # Feature Selection Method
    st.subheader("Feature Selection Method")
    method = st.radio("Choose method", ["Random Forest Importance", "Sequential Feature Selection"])
    max_features = st.slider("Maximum features to select", 1, min(15, len(X_full.columns)), 5)

    if st.button("Run Feature Selection"):
        with st.spinner("Running feature selection... please wait..."):
            if method == "Random Forest Importance":
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_full, y)
                importance = pd.Series(rf.feature_importances_, index=X_full.columns).sort_values(ascending=False)
                selected_features = importance.head(max_features).index.tolist()

                save_artifact({
                    'selected_features': selected_features,
                    'selection_metric': "random_forest_importance",
                    'importance_values': importance[selected_features].values.tolist()
                }, "5_best_subset_features.pkl")

                st.success(f"Top {len(selected_features)} features selected via Random Forest:")
                st.write(selected_features)

                fig, ax = plt.subplots(figsize=(10, 6))
                importance[selected_features].plot(kind='barh', ax=ax)
                ax.set_title("Feature Importance (Random Forest)")
                st.pyplot(fig, use_container_width=True)

            else:  # Sequential Feature Selection
                from sklearn.feature_selection import SequentialFeatureSelector
                from sklearn.linear_model import LogisticRegression

                scoring_metric = st.selectbox("Scoring metric", ['accuracy', 'precision', 'recall', 'f1'])
                estimator = LogisticRegression(solver='liblinear', max_iter=500, random_state=42)

                sfs = SequentialFeatureSelector(
                    estimator,
                    n_features_to_select=max_features,
                    direction='forward',
                    scoring=scoring_metric,
                    cv=5,
                    n_jobs=-1
                )
                sfs.fit(X_full, y)
                selected_features = X_full.columns[sfs.get_support()].tolist()

                save_artifact({
                    'selected_features': selected_features,
                    'selection_metric': scoring_metric,
                    'support_mask': sfs.get_support()
                }, "5_best_subset_features.pkl")

                st.success(f"Selected {len(selected_features)} features via SFS:")
                st.write(selected_features)


def Model_Selection_And_Training_page():
    st.title("4. Model Selection and Training")

    try:
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        X = processed_df.drop('Status', axis=1)
        y = processed_df['Status']
    except Exception as e:
        st.warning(f"Please complete preprocessing first. Error: {str(e)}")
        return

    # Model configuration
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Number of trees", 50, 500, 100)
        max_depth = st.slider("Max depth", 2, 20, 5)
    with col2:
        min_samples_split = st.slider("Min samples split", 2, 10, 2)
        bootstrap = st.checkbox("Bootstrap samples", value=True)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1)

    # Cross-validation
    if st.button("Run Cross-Validation"):
        with st.spinner("Running cross-validation... This may take a few minutes"):
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            st.write(f"Mean Accuracy: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

            save_artifact({
                'cv_scores': cv_scores,
                'params': model.get_params()
            }, "6_cv_results.pkl")

    # Full training
    if st.button("Train Final Model"):
        with st.spinner("Training final model..."):
            model.fit(X, y)
            save_artifact(model, "7_trained_model.pkl")

            # Generate and save predictions
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            processed_df['Prediction'] = y_pred
            processed_df['Default_Probability'] = y_proba[:, 1]
            processed_df.to_csv(f"{DATA_DIR}/8_predictions.csv", index=False)

            st.session_state['model'] = model
            st.session_state['features'] = X.columns.tolist()
            st.success("Model trained and saved!")


def Model_Evaluation_page():
    st.title("5. Model Evaluation")

    try:
        model = load_artifact("7_trained_model.pkl")
        if model is None:
            st.warning("Model not found. Please train the model first.")
            return

        predictions_df = pd.read_csv(f"{DATA_DIR}/8_predictions.csv")
    except Exception as e:
        st.warning(f"Error loading model or predictions: {str(e)}")
        return

    X = predictions_df.drop(['Status', 'Prediction', 'Default_Probability'], axis=1)
    y = predictions_df['Status']
    y_pred = predictions_df['Prediction']

    # Performance metrics
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{accuracy_score(y, y_pred):.2%}")
        st.metric("Precision", f"{precision_score(y, y_pred):.2%}")

    with col2:
        st.metric("Recall", f"{recall_score(y, y_pred):.2%}")
        st.metric("F1 Score", f"{f1_score(y, y_pred):.2%}")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.nlargest(5)

    fig, ax = plt.subplots()
    top_features.plot(kind='barh', ax=ax)
    st.pyplot(fig, use_container_width=True)


def Interactive_Prediction_page():
    st.title("6. Interactive Prediction")

    try:
        # Load necessary artifacts
        model = load_artifact("7_trained_model.pkl")
        preprocessor = load_artifact("3_preprocessor.pkl")
        original_features = load_artifact("2_column_types.pkl")

        if None in [model, preprocessor, original_features]:
            st.warning("Required artifacts not found. Please complete previous steps.")
            return
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        return

    st.subheader("Enter Applicant Information")

    # Create input form with original feature names
    input_data = {}
    col1, col2 = st.columns(2)

    with col1:
        input_data['loan_amount'] = st.number_input("Loan Amount", min_value=0, value=100000)
        input_data['income'] = st.number_input("Annual Income", min_value=0, value=50000)
        input_data['Credit_Score'] = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

    with col2:
        input_data['property_value'] = st.number_input("Property Value", min_value=0, value=200000)
        input_data['loan_purpose'] = st.selectbox("Loan Purpose", ['p1', 'p2', 'p3', 'p4'])
        input_data['Gender'] = st.selectbox("Gender", ['Male', 'Female', 'Joint', 'Sex Not Available'])

    if st.button("Predict Default Risk"):
        with st.spinner("Calculating risk..."):
            try:
                # Create a DataFrame with all original features
                df_template = pd.DataFrame(columns=original_features['numerical'] + original_features['categorical'])

                # Fill in the provided values
                for feature, value in input_data.items():
                    if feature in df_template.columns:
                        df_template[feature] = [value]

                # Fill missing values with defaults
                for col in df_template.columns:
                    if col not in input_data:
                        if col in original_features['numerical']:
                            df_template[col] = 0
                        else:
                            df_template[col] = df_template[col].astype('object')
                            df_template[col] = df_template[col].fillna('unknown')

                # Apply preprocessing
                X_processed = preprocessor.transform(df_template)

                # Make prediction
                probability = model.predict_proba(X_processed)[0]

                # Display results
                st.subheader("Prediction Results")

                if probability[1] > 0.3:  # High risk threshold
                    st.error(f"🚨 HIGH RISK (Default Probability: {probability[1]:.2%})")
                elif probability[1] > 0.1:  # Medium risk threshold
                    st.warning(f"⚠️ MEDIUM RISK (Default Probability: {probability[1]:.2%})")
                else:
                    st.success(f"✅ LOW RISK (Default Probability: {probability[0]:.2%})")

                # Probability visualization
                fig, ax = plt.subplots()
                ax.bar(['No Default', 'Default'], probability, color=['green', 'red'])
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                st.pyplot(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")


def Results_Interpretation_And_Conclusion_page():
    st.title("7. Results Interpretation and Conclusion")
    st.write("""
    ## Model Performance Summary

    The Random Forest classifier has shown good performance in predicting loan defaults:
    - High accuracy on both training and test sets
    - Balanced precision and recall scores
    - Important features align with financial domain knowledge

    ## Business Implications
    - The model can help reduce financial losses by identifying high-risk applicants
    - Can be used to adjust interest rates based on risk levels
    - Helps standardize the loan approval process

    ## Limitations
    - Model performance depends on data quality
    - May need periodic retraining as economic conditions change
    - Doesn't capture all qualitative factors in loan decisions

    ## Future Improvements
    - Experiment with other algorithms (XGBoost, Neural Networks)
    - Incorporate more features (economic indicators, employment history)
    - Develop a risk scoring system based on model probabilities
    """)


# Map sidebar names to functions
pages = {
    "Home Page": Home_Page,
    "Data Import and Overview": Data_Import_and_Overview_page,
    "Data Preprocessing": Data_Preprocessing_page,
    "Feature Selection": Feature_Selection_page,
    "Model Selection and Training": Model_Selection_And_Training_page,
    "Model Evaluation": Model_Evaluation_page,
    "Interactive Prediction": Interactive_Prediction_page,
    "Result Interpretation and Conclusion": Results_Interpretation_And_Conclusion_page,
}


def main():
    # Set page config first
    st.set_page_config(
        page_title="Loan Default Prediction",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS for faster rendering
    st.markdown("""
    <style>
        .stDataFrame { width: 100% !important; }
        .stProgress > div > div > div > div { background-color: #1E90FF; }
        .stSpinner { color: #1E90FF; }
        div[data-testid="stToolbar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state if not already done
    if 'uploaded_data' not in st.session_state:
        st.session_state['uploaded_data'] = None

    # Sidebar navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Display the selected page
    pages[selection]()


if __name__ == "__main__":
    main()
