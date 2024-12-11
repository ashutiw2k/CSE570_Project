import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import joblib

# Define paths
data_path = 'project_main/data/Transformer Input/'
sensor_file = 'lstm_sensor_input.json'
centroid_file = 'centroids.json'

def unpack_centroid(dictval):
    """
    Unpacks the centroid tuple from the centroid data.
    """
    values = list(dictval.values())
    return tuple(values[0])

def load_and_merge_data(subject):
    """
    Loads sensor and centroid data for a given subject and merges them.
    """
    # Load sensor data
    with open(os.path.join(data_path, subject, sensor_file), 'r') as f:
        sensor_data = json.load(f)

    sensor_df = pd.DataFrame(sensor_data)

    # Define feature columns
    feature_cols = [
        'Accelo_x', 'Accelo_y', 'Accelo_z',
        'Gyro_x', 'Gyro_y', 'Gyro_z',
        'Magneto_x', 'Magneto_y', 'Magneto_z',
        'Wifi_FTM_li_range', 'Wifi_FTM_li_std', 'WiFi_rssi'
    ]

    # Expand 'features' list into separate columns
    sensor_df[feature_cols] = pd.DataFrame(sensor_df['features'].tolist(), index=sensor_df.index)

    # Load centroid data
    with open(os.path.join(data_path, subject, centroid_file), 'r') as f:
        centroid_data = json.load(f)

    centroid_df = pd.DataFrame.from_dict(centroid_data, orient='index').reset_index()
    centroid_df = centroid_df.rename(columns={'index': 'image_key', 0: 'centroid'})
    centroid_df['centroid'] = centroid_df['centroid'].apply(unpack_centroid)

    # Extract timestamp from image_key
    centroid_df['timestamp'] = centroid_df['image_key'].apply(lambda x: '_'.join(x.split('_')[:3]))

    # Merge sensor and centroid data on timestamp
    merged_df = pd.merge(sensor_df, centroid_df, on='timestamp', how='left')

    # Extract 'x' and 'y' from centroid
    merged_df[['x', 'y']] = pd.DataFrame(merged_df['centroid'].tolist(), index=merged_df.index)

    # Create lag features (previous centroids)
    merged_df['x_prev'] = merged_df['x'].shift(1)
    merged_df['y_prev'] = merged_df['y'].shift(1)

    # Drop the first row with NaN lag
    merged_df = merged_df.dropna().reset_index(drop=True)

    return merged_df

def create_sequences(data, seq_length_input, seq_length_output, feature_cols, target_cols):
    """
    Creates input and output sequences for multi-step prediction.

    Args:
        data (pd.DataFrame): Merged DataFrame with features and targets.
        seq_length_input (int): Number of past frames for input.
        seq_length_output (int): Number of future frames to predict.
        feature_cols (list): List of feature column names.
        target_cols (list): List of target column names.

    Returns:
        tuple: Arrays of input sequences and corresponding output sequences.
    """
    X = []
    y = []
    total_length = seq_length_input + seq_length_output
    for i in range(len(data) - total_length + 1):
        input_seq = data[feature_cols].iloc[i:i + seq_length_input].values
        output_seq = data[target_cols].iloc[i + seq_length_input:i + total_length].values.flatten()
        X.append(input_seq)
        y.append(output_seq)
    return np.array(X), np.array(y)

def main():
    """
    Main function to execute the LSTM workflow.
    """
    # Specify the subject
    subject = 'subject1'  # Update as needed

    # Load and merge data
    merged_df = load_and_merge_data(subject)
    print(f"Merged data for {subject} contains {len(merged_df)} records.")

    # Define feature and target columns
    feature_cols = [
        'Accelo_x', 'Accelo_y', 'Accelo_z',
        'Gyro_x', 'Gyro_y', 'Gyro_z',
        'Magneto_x', 'Magneto_y', 'Magneto_z',
        'Wifi_FTM_li_range', 'Wifi_FTM_li_std', 'WiFi_rssi',
        'x_prev', 'y_prev'
    ]
    target_cols = ['x', 'y']

    # Create sequences
    seq_length_input = 5  # Past 5 frames
    seq_length_output = 2  # Next 2 frames
    X, y = create_sequences(merged_df, seq_length_input, seq_length_output, feature_cols, target_cols)

    print(f"Total sequences created: {len(X)}")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Shuffle=False to maintain temporal order
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # # Reshape X for scaling
    # # Flatten the 3D input to 2D for scaling
    # num_features = X_train.shape[2]
    # X_train_reshaped = X_train.reshape(-1, num_features)
    # X_test_reshaped = X_test.reshape(-1, num_features)

    # # Initialize scaler
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    # X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    # # Save the scaler for future use
    # os.makedirs('models/', exist_ok=True)
    # joblib.dump(scaler, 'models/standard_scaler.pkl')
    # print("Scaler saved to 'models/standard_scaler.pkl'.")

    # Define model parameters
    input_shape = (seq_length_input, len(feature_cols))  # (5, 14)
    output_dim = seq_length_output * 2  # 2 frames: [x1, y1, x2, y2]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_dim))  # Output layer for [x1, y1, x2, y2]

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Define Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model on the test set
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {loss}")
    print(f"Test MAE: {mae}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Reshape predictions to [samples, 2 frames, 2 coordinates]
    y_pred_reshaped = y_pred.reshape(-1, seq_length_output, 2)
    y_test_reshaped = y_test.reshape(-1, seq_length_output, 2)

    # Align timestamps
    # Calculate starting index for predictions
    # Each prediction corresponds to frames seq_length_input to seq_length_input + seq_length_output
    pred_start_idx = seq_length_input + len(X_train)  # Adjust based on train-test split

    # Ensure we have corresponding timestamps
    predicted_timestamps = merged_df['timestamp'].iloc[seq_length_input + len(X_train):].values[:len(y_pred)]

    # Prepare DataFrame
    predictions_df = pd.DataFrame({
        'timestamp': predicted_timestamps,
        'predicted_x1': y_pred_reshaped[:, 0, 0],
        'predicted_y1': y_pred_reshaped[:, 0, 1],
        'predicted_x2': y_pred_reshaped[:, 1, 0],
        'predicted_y2': y_pred_reshaped[:, 1, 1],
        'actual_x1': y_test_reshaped[:, 0, 0],
        'actual_y1': y_test_reshaped[:, 0, 1],
        'actual_x2': y_test_reshaped[:, 1, 0],
        'actual_y2': y_test_reshaped[:, 1, 1]
    })

    # Save predictions to CSV
    os.makedirs('predictions/', exist_ok=True)
    predictions_df.to_csv('predictions/lstm_modified_predictions.csv', index=False)
    print("Predictions saved to 'predictions/lstm_modified_predictions.csv'.")

    # Save the trained LSTM model
    os.makedirs('models/', exist_ok=True)
    model.save('models/LeftSideMaskedLSTM.h5')
    print("LSTM model saved to 'models/LeftSideMaskedLSTM.h5'.")
