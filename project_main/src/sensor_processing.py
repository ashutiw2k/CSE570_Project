import json
import numpy as np
import os

# Paths to data files
IMU_ALL_SUBJECTS= 'project_main/data/IMU Json/'
WIFI_ALL_SUBJECTS= 'project_main/data/Wifi Json/'

for subject in os.listdir(IMU_ALL_SUBJECTS):

    imu_file = IMU_ALL_SUBJECTS + subject + '/' + "imuagm9.json"  # IMU data file
    wifi_ftm_file = WIFI_ALL_SUBJECTS + subject + '/' + "wifi_ftm.json"  # WiFi FTM data file
    wifi_rssi_file = WIFI_ALL_SUBJECTS + subject + '/' + "wifi_rssi.json"  # WiFi RSSI data file
    output_file = f"project_main/data/Transformer Input/{subject}/transformer_sensor_input.json"  # Output file for transformer input

    # print(imu_file)
    # print(wifi_ftm_file)
    # print(wifi_rssi_file)
    # print(output_file)
    
    # continue

    # Load JSON files
    with open(imu_file, 'r') as f:
        imu_data = json.load(f)

    with open(wifi_ftm_file, 'r') as f:
        wifi_ftm_data = json.load(f)

    with open(wifi_rssi_file, 'r') as f:
        wifi_rssi_data = json.load(f)

    # Align timestamps
    timestamps = sorted(set(imu_data.keys()) & set(wifi_ftm_data.keys()) & set(wifi_rssi_data.keys()))

    # Feature extraction and formatting
    data = []
    for ts in timestamps:
        # Extract IMU features
        imu_features = imu_data[ts][:9]  # Accel_x, Accel_y, Accel_z, Gyro_x, Gyro_y, Gyro_z, Mag_x, Mag_y, Mag_z

        # Extract WiFi features
        ftm_features = wifi_ftm_data[ts]['ftm_li']  # Interpolated distances
        rssi_features = wifi_rssi_data[ts]['rssi']  # RSSI values

        # Combine features into a single vector
        combined_features = imu_features + ftm_features + rssi_features

        # Create an entry for the current timestamp
        data.append({
            "timestamp": ts,
            "features": combined_features  # Flattened feature vector
        })

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data successfully transformed and saved to {output_file}")







# The output json file should look something like this:


# [
#     {
#         "timestamp": "2020-12-23 14_09_58.352997",
#         "features": [
#             -0.50426155, 4.4542623, 8.362674, 0.012710561, 0.18636855, -0.23180082,
#             -16.131023, -31.06852, -23.677158, 1984.0, 1973.0, -50, -55
#         ]
#     },
#     {
#         "timestamp": "2020-12-23 14_09_58.686064",
#         "features": [
#             -1.859571, 4.1200247, 10.726638, 0.24280798, 0.053479128, -0.016352868,
#             -18.549023, -29.85508, -25.242561, 2063.991, 1770.450, -51, -57
#         ]
#     },
#     ...
# ]
