import pandas as pd
import joblib
import xgboost as xgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Class'])
    y = df['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Model predictions and probabilities
def model_predictions(model, X, model_type='sklearn'):
    if model_type == 'keras':
        preds = model.predict(X).ravel()
        return (preds > 0.5).astype(int), preds
    elif model_type == 'xgboost':
        preds = model.predict(xgb.DMatrix(X))
        return (preds > 0.5).astype(int), preds
    else:
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)[:, 1]
        else:
            preds = model.decision_function(X)
        return model.predict(X), preds

# Load the models
def load_models():
    models = {
        'Random Forest': joblib.load('random_forest_model.pkl'),
        'KNN': joblib.load('knn_model.pkl'),
        'MLP': tf.keras.models.load_model('mlp_model.h5'),
        'CNN': tf.keras.models.load_model('cnn_model3.h5'),
        'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
        'SVM': joblib.load('svm_model.pkl'),
        'XGBoost': joblib.load('xgboost_model.pkl')
    }
    return models

# Evaluate all models and return tabular results
def evaluate_models(models, X, y):
    results = []
    
    for name, model in models.items():
        if name in ['MLP', 'CNN']:
            preds, _ = model_predictions(model, X, model_type='keras')
        elif name == 'XGBoost':
            preds, _ = model_predictions(model, X, model_type='xgboost')
        else:
            preds, _ = model_predictions(model, X)
        
        precision = precision_score(y, preds)
        recall = recall_score(y, preds)
        f1 = f1_score(y, preds)
        accuracy = accuracy_score(y, preds)
        
        results.append({
            'Model': name,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': accuracy
        })
    
    df_results = pd.DataFrame(results).set_index('Model')
    return df_results

if __name__ == "__main__":
    # Load data and models
    X, y = load_data('creditcard.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state=42)
    models = load_models()

    # Evaluate models and output results
    df_results = evaluate_models(models, X_test, y_test)
    print(df_results)
    
    # Save the results to an Excel file
    df_results.to_excel('Preliminary Evaluation.xlsx')
