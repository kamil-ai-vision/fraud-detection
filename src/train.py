import pandas as pd
import numpy as np
import os
import glob
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE # New import

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
    Creates more advanced features based on transaction history and time.
    """
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df['TX_DATE'] = df['TX_DATETIME'].dt.date
    
    # Sort data by time and set as index for rolling windows
    df.sort_values(by=['TX_DATETIME'], inplace=True)
    df.set_index('TX_DATETIME', inplace=True)
    
    # Feature 1: Customer's average transaction amount over last 7 days
    df['CUST_AVG_AMT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling('7D').mean().shift(1)
    )

    # Feature 2: Terminal's average transaction amount over last 7 days
    df['TERM_AVG_AMT_7D'] = df.groupby('TERMINAL_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling('7D').mean().shift(1)
    )
    
    # Feature 3: Number of transactions for a customer in the last 7 days
    df['CUST_TX_COUNT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling('7D').count().shift(1)
    )
    
    # New Features to capture fraud patterns more effectively
    df['CUST_MAX_AMT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling('7D').max().shift(1)
    )
    df['CUST_MIN_AMT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling('7D').min().shift(1)
    )
    df['TERM_FRAUD_COUNT_28D'] = df.groupby('TERMINAL_ID')['TX_FRAUD'].transform(
        lambda x: x.rolling('28D').sum().shift(1)
    )

    # Re-add TX_DATETIME and TX_DATE as columns after setting index, for other features
    df.reset_index(inplace=True)
    
    # Time-based features
    df['TX_DAY_OF_WEEK'] = df['TX_DATETIME'].dt.dayofweek
    df['TX_HOUR_OF_DAY'] = df['TX_DATETIME'].dt.hour
    
    # Fill NaN values that resulted from rolling windows
    df.fillna(0, inplace=True)
    
    # Encode categorical features
    le_customer = LabelEncoder()
    le_terminal = LabelEncoder()
    df['CUSTOMER_ID_ENC'] = le_customer.fit_transform(df['CUSTOMER_ID'])
    df['TERMINAL_ID_ENC'] = le_terminal.fit_transform(df['TERMINAL_ID'])

    return df

def train_and_save_model():
    """
    Main function to orchestrate the data loading, feature engineering,
    training, and saving process.
    """
    print("--- Starting Fraud Detection Model Training (v2) ---")

    # Step 1: Load and combine data
    print(f"Loading data from {DATA_DIR}...")
    try:
        df = load_data(DATA_DIR)
        print(f"✅ Data loaded successfully. Total transactions: {len(df)}")
    except FileNotFoundError:
        print(f"❌ Error: Data directory '{DATA_DIR}' not found.")
        return

    # Step 2: Feature Engineering
    print("Performing feature engineering...")
    df_features = feature_engineer(df.copy())
    print("✅ Feature engineering complete.")

    # Step 3: Define features and target
    # Updated feature list with new features
    features = [
        'TX_AMOUNT', 'CUST_AVG_AMT_7D', 'TERM_AVG_AMT_7D', 
        'CUST_TX_COUNT_7D', 'CUST_MAX_AMT_7D', 'CUST_MIN_AMT_7D',
        'TERM_FRAUD_COUNT_28D', 'TX_DAY_OF_WEEK', 'TX_HOUR_OF_DAY',
        'CUSTOMER_ID_ENC', 'TERMINAL_ID_ENC'
    ]
    target = 'TX_FRAUD'

    X = df_features[features]
    y = df_features[target]

    # Step 4: Split data for training and testing
    # Note: Stratify ensures the same proportion of fraud cases in train/test sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✅ Data split. Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Original fraud cases in training set: {y_train.sum()}")

    # Step 5: Handle class imbalance with SMOTE
    print("\nApplying SMOTE oversampling to the training set...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"✅ Training set after SMOTE. New size: {len(X_resampled)}")
    print(f"New fraud cases in training set: {y_resampled.sum()}")

    # Step 6: Train the LightGBM model on the resampled data
    print("\nTraining LightGBM model on resampled data...")
    model = lgb.LGBMClassifier(random_state=RANDOM_STATE)
    model.fit(X_resampled, y_resampled)
    print("✅ Model training complete.")

    # Step 7: Evaluate the model on the ORIGINAL test set
    # This is crucial for a fair evaluation!
    print("Evaluating model performance on the ORIGINAL test set...")
    y_pred = model.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation Metrics (After SMOTE) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("--------------------------------\n")
    
    # Step 8: Save the trained model
    print("Saving the final model...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

    print("--- Script finished successfully ---")

if __name__ == "__main__":
    train_and_save_model()