import os
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


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

def create_sequences(data, seq_length, feature_cols, target_cols):
    X = []
    y = []
    t = []
    for i in range(seq_length, len(data)):
        X.append(data[feature_cols].iloc[i-seq_length:i].values)
        y.append(data[target_cols].iloc[i].values)
        t.append(data['timestamp'].iloc[i])
    return np.array(X), np.array(y), t


def unpack_centroid(dictval):
    values = list(dictval.values())
    return tuple(values[0])


def main():
    # Define paths
    MODEL_PATH = 'LeftSideMaskedLSTM.keras'
    test_data_path = 'project_main/data/Testing/Transformer Input/'
    sensor_file = 'lstm_sensor_input.json'
    PREDICTIONS_PATH = 'project_main/predictions/'
    OUTPUT_CSV_PATH = 'predictions.csv'  # Update as needed
    PLOT_PATH = 'predicted_centroids_plot.png'  # Update as needed
    # centroid_path = 'project_main/data/Transformer Input/'
    centroid_file = 'centroids.json'

    scaler = StandardScaler()
    dataframe_list = []
    train_centroid_df_list = []
    for subject in os.listdir(test_data_path):
        with open(test_data_path + subject + '/' + sensor_file, 'r') as f:
            sensor_data = json.load(f)

        with open(test_data_path + subject + '/' + centroid_file, 'r') as f:
            centroid_data = json.load(f)

        # Convert to DataFrame
        sensor_df = pd.DataFrame(sensor_data)
        # print(centroid_data)
        feature_cols = ['Accelo_x', 'Accelo_y', 'Accelo_z',
                    'Gyro_x', 'Gyro_y', 'Gyro_z',
                    'Magneto_x', 'Magneto_y', 'Magneto_z',
                    'Wifi_FTM_li_range', 'Wifi_FTM_li_std', 'WiFi_rssi']
                    # 'x_prev', 'y_prev']
        
        sensor_df[feature_cols] = pd.DataFrame(sensor_df['features'].to_list(), index=sensor_df.index)

        centroid_df = pd.DataFrame.from_dict(centroid_data, orient='index').reset_index()
        centroid_df = centroid_df.rename(columns={'index': 'image_key', 0: 'centroid'})
        # centroid_df = centroid_df.apply(transform_centroid, axis=1)
        centroid_df['centroid'] = centroid_df['centroid'].apply(unpack_centroid)

        # print(centroid_df)

        # exit(1)

        # Extract timestamp from image_key
        centroid_df['timestamp'] = centroid_df['image_key'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1] + '_' + x.split('_')[2])

        # Merge sensor and centroid data
        merged_df = pd.merge(sensor_df, centroid_df, on='timestamp', how='left')

        # Handle null centroids
        # merged_df['centroid'] = merged_df['centroid'].apply(lambda x: x[0]['centroid'] if pd.notnull(x) else [None, None])
        # print(merged_df['centroid'].tolist())

        # exit(1)

        merged_df[['x', 'y']] = pd.DataFrame(merged_df['centroid'].tolist(), index=merged_df.index)

        # Impute missing centroids (example: forward fill)
        # merged_df[['x', 'y']] = merged_df[['x', 'y']].fillna(method='ffill')

        # Create lag features (previous centroid)
        merged_df['x_prev'] = merged_df['x'].shift(1)
        merged_df['y_prev'] = merged_df['y'].shift(1)

        # Drop the first row with NaN lag
        merged_df = merged_df.dropna()

        # Feature Columns
        feature_cols = ['Accelo_x', 'Accelo_y', 'Accelo_z',
                    'Gyro_x', 'Gyro_y', 'Gyro_z',
                    'Magneto_x', 'Magneto_y', 'Magneto_z',
                    'Wifi_FTM_li_range', 'Wifi_FTM_li_std', 'WiFi_rssi',
                    'x_prev', 'y_prev']

        # Target Columns
        target_cols = ['x', 'y']

        # Scaling
        # merged_df[feature_cols] = scaler.fit_transform(merged_df[feature_cols])

        dataframe_list.append(merged_df)


    # Sequence Parameters
    sequence_length = 10  # Number of past frames to consider

    all_X = []
    all_y = []
    all_timestamps = []
    for df in dataframe_list:
        X, y, t = create_sequences(df, sequence_length, feature_cols, target_cols)
        # Train-Test Split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        all_X.extend(X)
        # all_X_test.extend(X_test)
        all_y.extend(y)
        # all_y_test.extend(y_test)
        all_timestamps.extend(t)

    all_X = np.array(all_X)
    all_y = np.array(all_y)
                     

    # Define sequence length (must match the one used during training)
    SEQUENCE_LENGTH = 10  # Example value; update as per your training

    print(f"Test sequences shape: {all_X.shape}")

    # train_centriod_concat = pd.concat(train_centroid_df_list)
    # # train_x = [data['x'] for data in train_centriod_df.]
    # # train_x = [train_centriod_concat]
    # print(train_centriod_concat.index)
    # exit(0)
    
    # Step 5: Load the trained LSTM model
    model = load_model(MODEL_PATH)
    print("Trained LSTM model loaded successfully.")
    
    # Step 6: Make predictions
    predictions = model.predict(all_X)
    print("Predictions made successfully.")
    
    # Step 7: Inverse transform predictions to original scale
    # scaler = StandardScaler()
    # predictions = scaler.inverse_transform(predictions_scaled)
    print("Predictions inverse transformed successfully.")
    
    # Step 8: Align predictions with corresponding timestamps
    # Create DataFrame

    predictions_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'predicted_x': predictions[:, 0],
        'true_x': all_y[:, 0],
        'predicted_y': predictions[:, 1],
        'true_y' : all_y[:, 1]
    })
    
    # Step 9: Save predictions to CSV
    predictions_df.to_csv(PREDICTIONS_PATH + OUTPUT_CSV_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH + OUTPUT_CSV_PATH}")
    
    # Step 10: (Optional) Plotting Predicted Centroids
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions_df['predicted_x'], predictions_df['predicted_y'], label='Predicted Centroids', alpha=0.6, c='blue')
    
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted Centroids')
    plt.grid(True)
    plt.savefig(PREDICTIONS_PATH + PLOT_PATH)
    plt.show()
    print(f"Centroid predictions plotted and saved to {PREDICTIONS_PATH + PLOT_PATH}")

    pred_df_per_subject = []

    ctr = 0
    for df in dataframe_list:
        pred_sub_df = predictions_df[predictions_df['timestamp'].isin(df['timestamp'])]
        pred_sub_df.to_csv(PREDICTIONS_PATH+ f'Subject{ctr}/' + OUTPUT_CSV_PATH, index=False)

        plt.figure(figsize=(10, 6))
        plt.scatter(pred_sub_df['predicted_x'], pred_sub_df['predicted_y'], label='Predicted Centroids', alpha=0.6, c='blue')
        
        plt.legend()
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Predicted Centroids')
        plt.grid(True)
        plt.savefig(PREDICTIONS_PATH + f'Subject{ctr}/' + PLOT_PATH)
        plt.show()
        print(f"Centroid predictions plotted and saved to {PREDICTIONS_PATH + f'Subject{ctr}/' + PLOT_PATH}")
        
        plt.close()

        ctr += 1


    
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
