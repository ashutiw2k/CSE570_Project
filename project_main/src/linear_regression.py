import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

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
        # Assume only one centroid per timestamp; modify if multiple centroids exist
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

def plot_predictions(y_test, y_pred, model_name='Model', subject='subject1'):
    """
    Plot actual vs. predicted centroid coordinates.

    Args:
        y_test (np.ndarray): Actual centroid coordinates.
        y_pred (np.ndarray): Predicted centroid coordinates.
        model_name (str): Name of the model for labeling.
        subject (str): Subject identifier for plot title.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    
    # Plot for X Coordinate
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5, color='blue')
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
    plt.xlabel('Actual X')
    plt.ylabel('Predicted X')
    plt.title(f'{model_name}: Actual vs Predicted X (Subject: {subject})')
    plt.grid(True)
    
    # Plot for Y Coordinate
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5, color='green')
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'r--')
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.title(f'{model_name}: Actual vs Predicted Y (Subject: {subject})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_Actual_vs_Predicted_{subject}.png')  # Save the plot
    plt.show()

def main():
    """
    Main function to execute the regression workflow.
    """
    # Define file paths
    SENSOR_JSON_PATH = 'data/Filtered Transformer Input/subject1/lstm_sensor_input.json'  # Update as needed
    CENTROIDS_JSON_PATH = 'data/Centroids/subject1/centroids.json'  # Update as needed
    
    # Define output directories
    PLOTS_DIR = 'plots/'
    MODELS_DIR = 'models/'
    SCALERS_DIR = 'scalers/'
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SCALERS_DIR, exist_ok=True)
    
    # Step 1: Load sensor data
    sensor_df = load_sensor_data(SENSOR_JSON_PATH)
    print("Sensor data loaded successfully.")
    
    # Step 2: Load centroid data
    centroid_df = load_centroid_data(CENTROIDS_JSON_PATH)
    print("Centroid data loaded successfully.")
    
    # Step 3: Merge data
    merged_df = merge_data(sensor_df, centroid_df)
    print(f"Merged data contains {len(merged_df)} records.")
    
    # Step 4: Prepare features and targets
    X, y = prepare_features_targets(merged_df)
    print("Features and targets prepared.")
    
    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")
    
    # # Step 6: Feature scaling
    # X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    # print("Feature scaling completed.")
    
    # # Step 7: Save the scaler for future use
    # joblib.dump(scaler, os.path.join(SCALERS_DIR, 'standard_scaler.pkl'))
    # print("Scaler saved to 'scalers/standard_scaler.pkl'.")
    
    # Step 8: Train Linear Regression
    lin_reg = train_linear_regression(X_train, y_train)
    print("Linear Regression model trained.")
    
    # Step 9: Evaluate Linear Regression
    lin_eval = evaluate_model(lin_reg, X_test, y_test, model_name='Linear Regression')
    
    # Step 10: Plot Linear Regression Predictions
    plot_predictions(y_test.values, lin_eval['y_pred'], model_name='Linear Regression', subject='subject1')
    
    # Step 11: Train Ridge Regression
    ridge_reg = train_ridge_regression(X_train, y_train, alpha=1.0)  # Adjust alpha as needed
    print("Ridge Regression model trained.")
    
    # Step 12: Evaluate Ridge Regression
    ridge_eval = evaluate_model(ridge_reg, X_test, y_test, model_name='Ridge Regression')
    
    # Step 13: Plot Ridge Regression Predictions
    plot_predictions(y_test.values, ridge_eval['y_pred'], model_name='Ridge Regression', subject='subject1')
    
    # Step 14: Save the models
    joblib.dump(lin_reg, os.path.join(MODELS_DIR, 'linear_regression_model.joblib'))
    joblib.dump(ridge_reg, os.path.join(MODELS_DIR, 'ridge_regression_model.joblib'))
    print("Models saved successfully.")

if __name__ == "__main__":
    main()
