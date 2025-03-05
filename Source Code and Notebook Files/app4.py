import xgboost as xgb
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshaping data for MLP
    X_mlp = X_scaled.astype('float32')
    y_mlp = y.astype('float32')

    # Reshaping data for CNN
    y_cnn = y_mlp
    X_cnn = X_mlp.reshape(-1, X.shape[1], 1)

    return df, X_scaled, y, X_cnn, y_cnn, X_mlp, y_mlp

# This function returns a dmatrix for xgboost
def create_dmatrix(X, y):
    return xgb.DMatrix(X_scaled, label=y)

# Loading all the models to be used
rf_clf = joblib.load('random_forest_model.pkl')
knn_clf = joblib.load('knn_model.pkl')
cnn_model = tf.keras.models.load_model('cnn_model3.h5')
mlp_model = tf.keras.models.load_model('mlp_model.h5')
xgboost_model = joblib.load('xgboost_model.pkl')
logistic_model = joblib.load('logistic_regression_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Title for the Dashboard
st.title('Credit Card Fraud Detection')

# CSV gile uploading
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df, X_scaled, y, X_cnn, y_cnn, X_mlp, y_mlp = load_and_preprocess_data(uploaded_file)
    test = create_dmatrix(X_scaled, y)

    # Dropdown menu for model selection
    model_name = st.selectbox(
        'Select a model to evaluate',
        ('Random Forest', 'KNN', 'MLP', 'CNN', 'Logistic Regression', 'SVM', 'XGBoost')
    )

    # loading the selected model for use
    model = None
    if model_name == 'Random Forest':
        model = rf_clf
    elif model_name == 'KNN':
        model = knn_clf
    elif model_name == 'MLP':
        model = mlp_model
    elif model_name == 'CNN':
        model = cnn_model
    elif model_name == 'XGBoost':
        model = xgboost_model
    elif model_name == 'Logistic Regression':
        model = logistic_model
    elif model_name == 'SVM':
        model = svm_model

    # Prediction and Prediction Probability variable for model output
    predictions = None
    prediction_probs = None

    # Model Predictions and Evaluations
    if model_name in ['Random Forest', 'KNN', 'Logistic Regression', 'SVM']:
        # for RF, SVM, LR and KNN
        if hasattr(model, "predict_proba"):
            prediction_probs = model.predict_proba(X_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            prediction_probs = model.decision_function(X_scaled)
        else:
            st.error("Selected model does not support probability predictions.")
            st.stop()

        predictions = model.predict(X_scaled)
    
    elif model_name == 'MLP':
        # MLP
        prediction_probs = model.predict(X_mlp).ravel()
        predictions = (prediction_probs > 0.5).astype("int32")
    
    elif model_name == 'CNN':
        # CNN
        prediction_probs = model.predict(X_cnn).ravel()
        predictions = (prediction_probs > 0.5).astype("int32")
    
    elif model_name == 'XGBoost':
        # XGBoost
        prediction_probs = model.predict(test)
        predictions = [round(value) for value in prediction_probs]

    # Classification Report
    st.subheader('Classification Report')
    st.text(classification_report(y, predictions))

    # Confusion matrix
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Genuine', 'Fraudulent'])
    fig_cm, ax_cm = plt.subplots()
    disp.plot(cmap='Blues', ax=ax_cm)
    st.pyplot(fig_cm)

    # PRC Calculation and Plotting
    precision, recall, thresholds_pr = precision_recall_curve(y, prediction_probs)
    pr_auc = auc(recall, precision)
    st.subheader('Precision-Recall Curve')
    st.write(f"Precision-Recall AUC: **{pr_auc:.2f}**")

    fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
    ax_pr.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend(loc='best')
    st.pyplot(fig_pr)

    # ROC-AUC Plotting and Calcuations
    fpr, tpr, thresholds_roc = roc_curve(y, prediction_probs)
    roc_auc = roc_auc_score(y, prediction_probs)
    st.subheader('Receiver Operating Characteristic (ROC) Curve')
    st.write(f"ROC AUC: **{roc_auc:.2f}**")

    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)

    # Mean Time and Mean Amount
    fraud_transactions = df[df['Class'] == 1]
    avg_time_fraud = fraud_transactions['Time'].mean()
    avg_amount_fraud = fraud_transactions['Amount'].mean()

    st.subheader('Additional Metrics for Fraudulent Transactions')
    st.write(f'**Average Time:** {avg_time_fraud:.2f} milliseconds')
    st.write(f'**Average Amount:** €{avg_amount_fraud:.2f}')

    # Predicted Fruads
    df['Predicted Fraud'] = predictions

    fraudulent_transactions = df[df['Predicted Fraud'] == 1]
    st.subheader('Predicted Fraudulent Transactions')
    st.write(fraudulent_transactions)

    # To download the fraudulent transactions
    st.download_button(
        label='Download Fraudulent Transactions',
        data=fraudulent_transactions.to_csv(index=False).encode('utf-8'),
        file_name='fraudulent_transactions.csv',
        mime='text/csv'
    )

    # Getting the counts of each class
    class_counts = df['Predicted Fraud'].value_counts()

    # Plotting the bar graph for genuine vs fraudulent transactions
    fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
    class_counts.plot(kind='bar', color=['#66b3ff', '#ff9999'], ax=ax_dist)
    ax_dist.set_title('Distribution of Genuine and Fraudulent Transactions')
    ax_dist.set_xlabel('Transaction Type')
    ax_dist.set_ylabel('Number of Transactions')
    ax_dist.set_xticks(ticks=[0, 1])
    ax_dist.set_xticklabels(['Genuine', 'Fraudulent'], rotation=0)
    st.pyplot(fig_dist)

else:
    st.write("Please upload a CSV file to proceed.")
