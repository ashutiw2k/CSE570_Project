import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load sensor and centroid data
sensor_file = 'project_main/data/Transformer Input/transformer_sensor_input.json'
centroid_file = 'project_main/data/Transformer Input/centroids.json'

with open(sensor_file, 'r') as f:
    sensor_data = json.load(f)

with open(centroid_file, 'r') as f:
    centroid_data = json.load(f)

# Convert to DataFrame
sensor_df = pd.DataFrame(sensor_data)
centroid_df = pd.DataFrame.from_dict(centroid_data, orient='index').reset_index()
centroid_df = centroid_df.rename(columns={'index': 'image_key', 0: 'centroid'})

# Extract timestamp from image_key
centroid_df['timestamp'] = centroid_df['image_key'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1] + '_' + x.split('_')[2])

# Merge sensor and centroid data
merged_df = pd.merge(sensor_df, centroid_df, on='timestamp', how='left')

# Handle null centroids
merged_df['centroid'] = merged_df['centroid'].apply(lambda x: x[0]['centroid'] if pd.notnull(x) else [None, None])
merged_df[['x', 'y']] = pd.DataFrame(merged_df['centroid'].tolist(), index=merged_df.index)

# Impute missing centroids (example: forward fill)
merged_df[['x', 'y']] = merged_df[['x', 'y']].fillna(method='ffill')

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
scaler = StandardScaler()
merged_df[feature_cols] = scaler.fit_transform(merged_df[feature_cols])

# # Save the preprocessed data
# merged_df.to_csv('preprocessed_data.csv', index=False)


# Feature Columns
feature_cols = ['Accelo_x', 'Accelo_y', 'Accelo_z',
               'Gyro_x', 'Gyro_y', 'Gyro_z',
               'Magneto_x', 'Magneto_y', 'Magneto_z',
               'Wifi_FTM_li_range', 'Wifi_FTM_li_std', 'WiFi_rssi',
               'x_prev', 'y_prev']

# Target Columns
target_cols = ['x', 'y']

# Scaling
scaler = StandardScaler()
merged_df[feature_cols] = scaler.fit_transform(merged_df[feature_cols])

# Sequence Parameters
sequence_length = 10  # Number of past frames to consider

# Prepare sequences
def create_sequences(data, seq_length, feature_cols, target_cols):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[feature_cols].iloc[i-seq_length:i].values)
        y.append(data[target_cols].iloc[i].values)
    return np.array(X), np.array(y)

X, y = create_sequences(merged_df, sequence_length, feature_cols, target_cols)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model Definition
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(sequence_length, len(feature_cols)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(2))  # Output layer for [x, y] coordinates

# Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate Model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae}")
