import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DATA_DIR = 'D:/VS Code/fraud-detection/data/'
MODEL_SAVE_PATH = 'D:/VS Code/fraud-detection/models/fraud_detection_model_v2.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(data_dir):
    """Loads all .pkl files from a directory and concatenates them."""
    all_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    df_list = [pd.read_pickle(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    return df

def feature_engineer(df):
    """
    Creates new features based on transaction history and time.
    """
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df['TX_DATE'] = df['TX_DATETIME'].dt.date
    
    # Sort data by time and set as index for rolling windows
    df.sort_values(by=['TX_DATETIME'], inplace=True)
    df.set_index('TX_DATETIME', inplace=True)
    
    df['CUST_AVG_AMT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling('7D').mean().shift(1)
    )
    df['TERM_AVG_AMT_7D'] = df.groupby('TERMINAL_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling('7D').mean().shift(1)
    )
    df['CUST_TX_COUNT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling('7D').count().shift(1)
    )
    df['CUST_MAX_AMT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling('7D').max().shift(1)
    )
    df['CUST_MIN_AMT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling('7D').min().shift(1)
    )
    df['TERM_FRAUD_COUNT_28D'] = df.groupby('TERMINAL_ID')['TX_FRAUD'].transform(
        lambda x: x.rolling('28D').sum().shift(1)
    )

    df.reset_index(inplace=True)
    
    df['TX_DAY_OF_WEEK'] = df['TX_DATETIME'].dt.dayofweek
    df['TX_HOUR_OF_DAY'] = df['TX_DATETIME'].dt.hour
    
    df.fillna(0, inplace=True)
    
    le_customer = LabelEncoder()
    le_terminal = LabelEncoder()
    df['CUSTOMER_ID_ENC'] = le_customer.fit_transform(df['CUSTOMER_ID'])
    df['TERMINAL_ID_ENC'] = le_terminal.fit_transform(df['TERMINAL_ID'])

    return df

def evaluate_model():
    """
    Loads the saved model and evaluates its performance on the test set.
    """
    print("--- Starting Model Evaluation Script ---")

    # Step 1: Load data and apply feature engineering
    print(f"Loading data from {DATA_DIR}...")
    try:
        df = load_data(DATA_DIR)
        df_features = feature_engineer(df.copy())
    except FileNotFoundError:
        print(f"❌ Error: Data directory '{DATA_DIR}' not found.")
        return

    # Step 2: Define features and target, and load the model
    features = [
        'TX_AMOUNT', 'CUST_AVG_AMT_7D', 'TERM_AVG_AMT_7D', 
        'CUST_TX_COUNT_7D', 'CUST_MAX_AMT_7D', 'CUST_MIN_AMT_7D',
        'TERM_FRAUD_COUNT_28D', 'TX_DAY_OF_WEEK', 'TX_HOUR_OF_DAY',
        'CUSTOMER_ID_ENC', 'TERMINAL_ID_ENC'
    ]
    target = 'TX_FRAUD'

    X = df_features[features]
    y = df_features[target]
    
    # Split data to get the same test set as in training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ Error: Model file not found at {MODEL_SAVE_PATH}.")
        print("Please run train.py first to save the model.")
        return

    model = joblib.load(MODEL_SAVE_PATH)
    print("✅ Model loaded successfully.")

    # Step 3: Evaluate the model on the test set
    print("\nEvaluating model performance on the test set...")
    y_pred = model.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("--------------------------------\n")

if __name__ == "__main__":
    evaluate_model()