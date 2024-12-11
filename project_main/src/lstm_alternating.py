import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load sensor and centroid data
data_path = 'project_main/data/Transformer Input/'
sensor_file = 'lstm_sensor_input.json'
# centroid_path = 'project_main/data/Transformer Input/'
centroid_file = 'centroids.json'

def unpack_centroid(dictval):
    values = list(dictval.values())
    return tuple(values[0])


dataframe_list = []
for subject in os.listdir(data_path):
    with open(data_path + subject + '/' + sensor_file, 'r') as f:
        sensor_data = json.load(f)

    with open(data_path + subject + '/' + centroid_file, 'r') as f:
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
    # scaler = StandardScaler()
    # merged_df[feature_cols] = scaler.fit_transform(merged_df[feature_cols])

    dataframe_list.append(merged_df)


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


# Model Definition
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(sequence_length, len(feature_cols)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(2))  # Output layer for [x, y] coordinates

early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=8,          # Number of epochs to wait before stopping
    restore_best_weights=True  # Restore the weights from the best epoch
)

# Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

all_X_train = []
all_X_test = []
all_y_train = []
all_y_test = []

for df in dataframe_list:
    X, y = create_sequences(df, sequence_length, feature_cols, target_cols)
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    all_X_train.extend(X_train)
    all_X_test.extend(X_test)
    all_y_train.extend(y_train)
    all_y_test.extend(y_test)

all_X_train = np.array(all_X_train)
all_X_test = np.array(all_X_test)
all_y_train = np.array(all_y_train)
all_y_test = np.array(all_y_test)

# Train Model
    
history = model.fit(all_X_train, all_y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate Model
loss, mae = model.evaluate(all_X_test, all_y_test)
print(f"Test MAE: {mae}")
model.save('LeftSideMaskedLSTM.keras')
