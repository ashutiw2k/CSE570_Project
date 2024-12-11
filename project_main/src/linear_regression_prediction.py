import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_sensor_data(sensor_json_path):
    """
    Load sensor data from a JSON file and convert it to a DataFrame.

    Args:
        sensor_json_path (str): Path to the sensor JSON file.

    Returns:
        pd.DataFrame: DataFrame containing sensor features and timestamps.
    """
    with open(sensor_json_path, 'r') as f:
        sensor_data = json.load(f)
    
    # Convert list of sensor entries to DataFrame
    sensor_df = pd.DataFrame(sensor_data)
    
    # Expand the 'features' list into separate columns
    features_df = sensor_df['features'].apply(pd.Series)
    features_df.columns = [f'feature_{i+1}' for i in range(features_df.shape[1])]
    
    # Combine timestamp with features
    sensor_df = pd.concat([sensor_df['timestamp'], features_df], axis=1)
    
    return sensor_df

def load_centroid_data(centroids_json_path):
    """
    Load centroid data from a JSON file and convert it to a DataFrame.

    Args:
        centroids_json_path (str): Path to the centroids JSON file.

    Returns:
        pd.DataFrame: DataFrame containing timestamps and centroid coordinates.
    """
    with open(centroids_json_path, 'r') as f:
        centroids_data = json.load(f)
    
    # Extract timestamp from image filenames and associate with centroids
    records = []
    for image_key, centroid_list in centroids_data.items():
        # Example image_key: "2020-12-23 14_09_58.686064_left.png"
        # Extract timestamp by splitting and joining first three parts
        parts = image_key.split('_')
        if len(parts) < 4:
            continue  # Skip malformed keys
        timestamp = '_'.join(parts[:3])  # "2020-12-23 14_09_58.686064"
        # Handle multiple centroids per timestamp if necessary
        # Here, we assume one centroid per timestamp; modify as needed
        centroid = centroid_list[0].get('centroid', [np.nan, np.nan])
        records.append({'timestamp': timestamp, 'centroid_x': centroid[0], 'centroid_y': centroid[1]})
    
    centroid_df = pd.DataFrame(records)
    return centroid_df

def merge_data(sensor_df, centroid_df):
    """
    Merge sensor data with centroid data based on timestamps.

    Args:
        sensor_df (pd.DataFrame): DataFrame containing sensor features and timestamps.
        centroid_df (pd.DataFrame): DataFrame containing centroid coordinates and timestamps.

    Returns:
        pd.DataFrame: Merged DataFrame containing sensor features and centroid coordinates.
    """
    merged_df = pd.merge(sensor_df, centroid_df, on='timestamp', how='left')
    
    # Drop entries with missing centroids
    merged_df.dropna(subset=['centroid_x', 'centroid_y'], inplace=True)
    
    # Reset index after dropping
    merged_df.reset_index(drop=True, inplace=True)
    
    return merged_df

def prepare_features_targets(merged_df):
    """
    Prepare feature matrix X and target matrix y from the merged DataFrame.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame with sensor features and centroids.

    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target matrix.
    """
    # Define feature columns and target columns
    feature_cols = [f'feature_{i+1}' for i in range(12)] + ['centroid_x_prev', 'centroid_y_prev']
    target_cols = ['centroid_x', 'centroid_y']
    
    # Create lag features (previous centroid)
    merged_df['centroid_x_prev'] = merged_df['centroid_x'].shift(1)
    merged_df['centroid_y_prev'] = merged_df['centroid_y'].shift(1)
    
    # Drop the first row with NaN lag
    merged_df.dropna(inplace=True)
    
    # Define X and y
    X = merged_df[feature_cols]
    y = merged_df[target_cols]
    
    return X, y



def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target matrix.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.

    Args:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training targets.

    Returns:
        LinearRegression: Trained Linear Regression model.
    """
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    return lin_reg

def train_ridge_regression(X_train, y_train, alpha=1.0):
    """
    Train a Ridge Regression model.

    Args:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training targets.
        alpha (float): Regularization strength.

    Returns:
        Ridge: Trained Ridge Regression model.
    """
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X_train, y_train)
    return ridge_reg

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate the regression model using MSE, RMSE, MAE, and R².

    Args:
        model: Trained regression model.
        X_test (np.ndarray): Scaled testing features.
        y_test (np.ndarray): Testing targets.
        model_name (str): Name of the model for display purposes.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"=== {model_name} Evaluation ===")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("\n")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}


def main():
    """
    Main function to execute the prediction workflow.
    """
    # Define file paths
    DATAFILE_PATH = 'project_main/data/Testing/Transformer Input/'
    SENSOR_JSON_PATH = 'lstm_sensor_input.json'  # Update as needed
    CENTROIDS_JSON_PATH = 'centroids.json'  # Update as needed
    
    # Define output directories
    # PLOTS_DIR = 'plots/'
    MODELS_DIR = 'project_main/models/'
    # SCALERS_DIR = 'scalers/'
    # os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    merge_df_list = []
    # SCALERS_DIR = 'scalers/'
    linear_model_path = os.path.join(MODELS_DIR, 'linear_regression_model.joblib')
    ridge_model_path = os.path.join(MODELS_DIR, 'ridge_regression_model.joblib')
    # scaler_path = os.path.join(SCALERS_DIR, 'standard_scaler.pkl')
    
    # Define output directories
    PREDICTIONS_DIR = 'project_main/predictions/'
    # PLOTS_DIR = 'plots/'
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    # os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Step 1: Load sensor data
    for sub in os.listdir(DATAFILE_PATH):
        # Step 1: Load sensor data
        sensor_df = load_sensor_data(DATAFILE_PATH + sub + '/' + SENSOR_JSON_PATH)
        print("Sensor data loaded successfully.")
        
        # Step 2: Load centroid data
        centroid_df = load_centroid_data(DATAFILE_PATH + sub + '/' + CENTROIDS_JSON_PATH)
        print("Centroid data loaded successfully.")
        
        # Step 3: Merge data
        merged_df = merge_data(sensor_df, centroid_df)
        print(f"Merged data contains {len(merged_df)} records.")

        merge_df_list.append(merged_df)

    merged_df = pd.concat(merge_df_list)
    
    # Step 4: Prepare features and targets
    X, y = prepare_features_targets(merged_df)
    print("Features and targets prepared.")
    
    # # Step 5: Load the scaler
    # scaler = joblib.load(scaler_path)
    # print("Scaler loaded successfully.")
    
    # # Step 6: Scale features
    # X_scaled = scaler.transform(X)
    # print("Features scaled successfully.")
    
    # Step 7: Load the trained models
    lin_reg = joblib.load(linear_model_path)
    ridge_reg = joblib.load(ridge_model_path)
    print("Trained Linear and Ridge Regression models loaded successfully.")
    
    # Step 8: Make predictions with Linear Regression
    y_pred_lin = lin_reg.predict(X)
    print("Predictions made with Linear Regression.")
    
    # Step 9: Make predictions with Ridge Regression
    y_pred_ridge = ridge_reg.predict(X)
    print("Predictions made with Ridge Regression.")
    
    # Step 10: Evaluate Linear Regression
    lin_mse = mean_squared_error(y, y_pred_lin)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(y, y_pred_lin)
    lin_r2 = r2_score(y, y_pred_lin)
    
    print("=== Linear Regression Evaluation ===")
    print(f"Mean Squared Error (MSE): {lin_mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {lin_rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {lin_mae:.4f}")
    print(f"R² Score: {lin_r2:.4f}")
    print("\n")
    
    # Step 11: Evaluate Ridge Regression
    ridge_mse = mean_squared_error(y, y_pred_ridge)
    ridge_rmse = np.sqrt(ridge_mse)
    ridge_mae = mean_absolute_error(y, y_pred_ridge)
    ridge_r2 = r2_score(y, y_pred_ridge)
    
    print("=== Ridge Regression Evaluation ===")
    print(f"Mean Squared Error (MSE): {ridge_mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {ridge_rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {ridge_mae:.4f}")
    print(f"R² Score: {ridge_r2:.4f}")
    print("\n")
    
    # Step 12: Save Predictions to CSV
    predictions_df = pd.DataFrame({
        'timestamp': merged_df['timestamp'],
        'centroid_x': y['centroid_x'],
        'centroid_y': y['centroid_y'],
        'predicted_centroid_x_linear': y_pred_lin[:, 0],
        'predicted_centroid_y_linear': y_pred_lin[:, 1],
        'predicted_centroid_x_ridge': y_pred_ridge[:, 0],
        'predicted_centroid_y_ridge': y_pred_ridge[:, 1]
    })
    
    output_csv_path = os.path.join(PREDICTIONS_DIR, 'predictions_regression.csv')
    predictions_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

    ctr = 0
    for df in merge_df_list:
        pred_sub_df = pd.merge(predictions_df, df, 
                               on=['timestamp', 'centroid_x', 'centroid_y'], how='inner')
        
        pred_sub_df = pred_sub_df.drop([f'feature_{i+1}' for i in range(12)], axis=1)
        
        pred_sub_df.to_csv(PREDICTIONS_DIR + f'Subject{ctr}/' + 'predictions_regression.csv', index=False)

        ctr += 1

        
    
    # # Step 13: Plot Predictions for Linear Regression
    # plot_predictions(y.values, y_pred_lin, model_name='Linear Regression', subject='subject1')
    
    # # Step 14: Plot Predictions for Ridge Regression
    # plot_predictions(y.values, y_pred_ridge, model_name='Ridge Regression', subject='subject1')
    
    # # Optional: Save evaluation metrics to a text file or JSON
    # metrics = {
    #     'Linear Regression': {
    #         'MSE': lin_mse,
    #         'RMSE': lin_rmse,
    #         'MAE': lin_mae,
    #         'R2': lin_r2
    #     },
    #     'Ridge Regression': {
    #         'MSE': ridge_mse,
    #         'RMSE': ridge_rmse,
    #         'MAE': ridge_mae,
    #         'R2': ridge_r2
    #     }
    # }
    
    # metrics_path = os.path.join(PREDICTIONS_DIR, 'regression_evaluation_metrics.json')
    # with open(metrics_path, 'w') as f:
    #     json.dump(metrics, f, indent=4)
    # print(f"Evaluation metrics saved to {metrics_path}")
    
    # Optional: Save the predictions plot
    # (Already handled in plot_predictions function)
    
if __name__ == "__main__":
    main()
