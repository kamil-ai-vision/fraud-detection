import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

# --- Configuration ---
MODEL_SAVE_PATH = 'D:/VS Code/fraud-detection/models/fraud_detection_model_v2.joblib'
DATA_DIR = 'D:/VS Code/fraud-detection/data/' # We need this to get a sample of existing customer/terminal IDs

def predict_transaction_fraud(model, transaction_data):
    """
    Loads the trained model and predicts if a new transaction is fraudulent.
    
    Args:
        model: The trained machine learning model.
        transaction_data (dict): A dictionary containing the features of the transaction.
    
    Returns:
        float: The predicted probability of fraud (0 to 1).
    """
    # 1. Convert the input data to a DataFrame
    input_df = pd.DataFrame([transaction_data])

    # 2. Add placeholder columns that would be generated during training
    # These must be present for the model to work
    # We will need to re-engineer these in a more robust solution
    input_df['CUST_AVG_AMT_7D'] = 0.0
    input_df['TERM_AVG_AMT_7D'] = 0.0
    input_df['CUST_TX_COUNT_7D'] = 0.0
    input_df['CUST_MAX_AMT_7D'] = 0.0
    input_df['CUST_MIN_AMT_7D'] = 0.0
    input_df['TERM_FRAUD_COUNT_28D'] = 0.0
    
    # 3. Re-create time-based and encoded features for the new transaction
    input_df['TX_DATETIME'] = pd.to_datetime(input_df['TX_DATETIME'])
    input_df['TX_DAY_OF_WEEK'] = input_df['TX_DATETIME'].dt.dayofweek
    input_df['TX_HOUR_OF_DAY'] = input_df['TX_DATETIME'].dt.hour

    # Note: In a real-world system, you'd load the saved LabelEncoders
    # For this example, we'll cast to a numerical type that LightGBM accepts
    input_df['CUSTOMER_ID_ENC'] = pd.factorize(input_df['CUSTOMER_ID'])[0]
    input_df['TERMINAL_ID_ENC'] = pd.factorize(input_df['TERMINAL_ID'])[0]

    # 4. Define the feature order for the prediction
    features = [
        'TX_AMOUNT', 'CUST_AVG_AMT_7D', 'TERM_AVG_AMT_7D', 
        'CUST_TX_COUNT_7D', 'CUST_MAX_AMT_7D', 'CUST_MIN_AMT_7D',
        'TERM_FRAUD_COUNT_28D', 'TX_DAY_OF_WEEK', 'TX_HOUR_OF_DAY',
        'CUSTOMER_ID_ENC', 'TERMINAL_ID_ENC'
    ]

    # 5. Make the prediction
    # model.predict_proba returns the probabilities for both classes (0 and 1)
    # We are interested in the probability of class 1 (fraud)
    probabilities = model.predict_proba(input_df[features])
    fraud_probability = probabilities[:, 1][0]
    
    return fraud_probability

# --- Main execution block ---
if __name__ == '__main__':
    print("--- Starting Fraud Prediction Script ---")

    # Load the trained model
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ Error: Model file not found at {MODEL_SAVE_PATH}.")
        print("Please run train.py first to save the model.")
        exit()
    
    model = joblib.load(MODEL_SAVE_PATH)
    print("✅ Model loaded successfully.")
    
    # --- Example: A new transaction to predict ---
    # This example assumes you have an existing customer and terminal ID
    sample_transaction = {
        'TX_DATETIME': datetime.now(),
        'TX_AMOUNT': 250.00,  # A transaction amount > 220, matching fraud scenario #1
        'CUSTOMER_ID': 'C_1', # A sample customer ID
        'TERMINAL_ID': 'T_100', # A sample terminal ID
    }

    print("\nPredicting a new transaction...")
    fraud_prob = predict_transaction_fraud(model, sample_transaction)

    print("\n--- Fraud Prediction Result ---")
    print(f"Transaction: Amount ${sample_transaction['TX_AMOUNT']:.2f} by Customer {sample_transaction['CUSTOMER_ID']}")
    print(f"Predicted probability of fraud: {fraud_prob:.4f}")
    
    # Determine if it's fraud based on a threshold (e.g., 50%)
    if fraud_prob > 0.5:
        print("⚠️ This transaction is likely fraudulent.")
    else:
        print("✅ This transaction is likely legitimate.")
    print("-----------------------------------")