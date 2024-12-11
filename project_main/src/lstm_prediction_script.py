import os
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_scalers(features_scaler_path, targets_scaler_path):
    scaler_features = joblib.load(features_scaler_path)
    scaler_targets = joblib.load(targets_scaler_path)
    return scaler_features, scaler_targets

def load_test_data(test_sensor_json_path, sequence_length, feature_cols):
    with open(test_sensor_json_path, 'r') as f:
        test_sensor_data = json.load(f)
    
    test_df = pd.DataFrame(test_sensor_data)
    features_df = test_df['features'].apply(pd.Series)
    features_df.columns = feature_cols
    test_df = pd.concat([test_df['timestamp'], features_df], axis=1)
    test_df = test_df.sort_values('timestamp').reset_index(drop=True)
    timestamps_test = test_df['timestamp'].values
    test_df = test_df.drop(columns=['timestamp'])
    test_features = test_df.values
    return test_features, timestamps_test

def create_sequences(X, sequence_length):
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        seq = X[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

def main():
    # Define paths
    MODEL_PATH = 'models/trained_lstm_model.h5'
    SCALER_FEATURES_PATH = 'models/scaler_features.pkl'
    SCALER_TARGETS_PATH = 'models/scaler_targets.pkl'
    
    # Define test data path
    TEST_SENSOR_JSON_PATH = 'data/Filtered Transformer Input/subject1/lstm_sensor_input.json'  # Update as needed
    
    # Define output paths
    OUTPUT_CSV_PATH = 'predictions/subject1_predictions.csv'  # Update as needed
    PLOT_PATH = 'predictions/predicted_centroids_plot.png'  # Update as needed
    
    # Define feature columns (ensure this matches your preprocessing)
    feature_cols = [
        'feature_1', 'feature_2', 'feature_3',
        'feature_4', 'feature_5', 'feature_6',
        'feature_7', 'feature_8', 'feature_9',
        'feature_10', 'feature_11', 'feature_12',
        'x_prev', 'y_prev'
        # Add additional engineered features here if applicable
    ]
    
    # Define sequence length (must match the one used during training)
    SEQUENCE_LENGTH = 10  # Example value; update as per your training
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    # Step 1: Load scalers
    scaler_features, scaler_targets = load_scalers(SCALER_FEATURES_PATH, SCALER_TARGETS_PATH)
    print("Scalers loaded successfully.")
    
    # Step 2: Load test data
    test_features, timestamps_test = load_test_data(TEST_SENSOR_JSON_PATH, SEQUENCE_LENGTH, feature_cols)
    print(f"Test data loaded. Number of samples: {len(test_features)}")
    
    # Step 3: Scale test features
    num_features = test_features.shape[2]
    test_features_reshaped = test_features.reshape(-1, num_features)
    test_features_scaled = scaler_features.transform(test_features_reshaped)
    test_features_scaled = test_features_scaled.reshape(test_features.shape)
    print("Test features scaled successfully.")
    
    # Step 4: Sequences are already created
    X_test = test_features_scaled  # Shape: (samples, sequence_length, features)
    print(f"Test sequences shape: {X_test.shape}")
    
    # Step 5: Load the trained LSTM model
    model = load_model(MODEL_PATH)
    print("Trained LSTM model loaded successfully.")
    
    # Step 6: Make predictions
    predictions_scaled = model.predict(X_test)
    print("Predictions made successfully.")
    
    # Step 7: Inverse transform predictions to original scale
    predictions = scaler_targets.inverse_transform(predictions_scaled)
    print("Predictions inverse transformed successfully.")
    
    # Step 8: Align predictions with corresponding timestamps
    adjusted_timestamps = timestamps_test[SEQUENCE_LENGTH -1 :]
    
    # Create DataFrame
    predictions_df = pd.DataFrame({
        'timestamp': adjusted_timestamps,
        'predicted_x': predictions[:, 0],
        'predicted_y': predictions[:, 1]
    })
    
    # Step 9: Save predictions to CSV
    predictions_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_CSV_PATH}")
    
    # Step 10: (Optional) Plotting Predicted Centroids
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions_df['predicted_x'], predictions_df['predicted_y'], label='Predicted Centroids', alpha=0.6, c='blue')
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted Centroids')
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    plt.show()
    print(f"Centroid predictions plotted and saved to {PLOT_PATH}")
    
    # Optional Step 11: If Actual Centroids Are Available
    # Load actual centroids corresponding to the test timestamps
    # Assuming you have a DataFrame 'actual_df' with 'timestamp', 'actual_x', 'actual_y'
    # Replace the following lines with actual loading of centroids
    
    # Example:
    # actual_df = pd.read_csv('data/Filtered Transformer Input/subject1/actual_centroids.csv')
    # merged = pd.merge(predictions_df, actual_df, on='timestamp', how='left')
    
    # Compute Evaluation Metrics
    # mae_x = mean_absolute_error(merged['actual_x'], merged['predicted_x'])
    # mae_y = mean_absolute_error(merged['actual_y'], merged['predicted_y'])
    # print(f"MAE X: {mae_x}")
    # print(f"MAE Y: {mae_y}")
    
    # # Plotting Actual vs Predicted
    # plt.figure(figsize=(10, 6))
    # plt.scatter(merged['actual_x'], merged['actual_y'], label='Actual Centroids', alpha=0.5, c='green')
    # plt.scatter(merged['predicted_x'], merged['predicted_y'], label='Predicted Centroids', alpha=0.5, c='blue')
    # plt.legend()
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Actual vs. Predicted Centroids')
    # plt.grid(True)
    # plt.savefig('predictions/predicted_vs_actual_centroids.png')
    # plt.show()
    # print("Actual vs Predicted centroids plotted successfully.")

if __name__ == "__main__":
    main()
