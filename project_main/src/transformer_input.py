import json
import os

# Paths to data files
CENTROID_LABELS_ALL_SUBJECTS= 'project_main/data/Transformer Input/'
SENSOR_ALL_SUBJECTS= 'project_main/data/Transformer Input/'
TRANSFORMER_ALL_SUBJECTS= 'project_main/data/Transformer Input/'

# # Load sensor inputs and bounding box labels
# for subject in os.listdir(SENSOR_ALL_SUBJECTS):

#     sensor_file = SENSOR_ALL_SUBJECTS + subject + '/' + "transformer_sensor_input.json"  # Path to sensor inputs
#     bbox_file = BB_LABELS_ALL_SUBJECTS + subject + '/' + "bounding_boxes.json"  # Path to bounding box labels
#     output_file = TRANSFORMER_ALL_SUBJECTS + subject + '/' + "transformer_input.json"  # Output file

#     # Load data
#     with open(sensor_file, 'r') as f:
#         sensor_data = json.load(f)

#     with open(bbox_file, 'r') as f:
#         bbox_data = json.load(f)

#     # Combine data
#     transformer_data = []

#     for sensor_entry in sensor_data:
#         timestamp = sensor_entry["timestamp"]
#         features = sensor_entry["features"]

#         # Create entries for `_left` and `_right`
#         for suffix in ["_left", "_right"]:
#             image_key = f"{timestamp}{suffix}.png"
#             bbox_labels = bbox_data.get(image_key, [{"bbox": [0, 0, 0, 0]}])  # Default to [0, 0, 0, 0] if missing

#             # Add combined entry
#             transformer_data.append({
#                 "timestamp": timestamp,
#                 "side": suffix.strip('_'),
#                 "features": features,
#                 "bounding_boxes": [entry["bbox"] for entry in bbox_labels]  # List of bounding boxes
#             })

#         # suffix='full'
#         # image_key = f"{timestamp}.png"
#         # bbox_labels = bbox_data.get(image_key, [{"bbox": [0, 0, 0, 0]}])  # Default to [0, 0, 0, 0] if missing

#         # # Add combined entry
#         # transformer_data.append({
#         #     "timestamp": timestamp,
#         #     "side": 'full',
#         #     "features": features,
#         #     "bounding_boxes": [entry["bbox"] for entry in bbox_labels]  # List of bounding boxes
#         # })

#     # Save to output file
#     with open(output_file, 'w') as f:
#         json.dump(transformer_data, f, indent=4)

#     print(f"Combined data saved to {output_file}")


# Iterate over each subject
for subject in os.listdir(SENSOR_ALL_SUBJECTS):
    # Define file paths
    sensor_file = os.path.join(SENSOR_ALL_SUBJECTS, subject, "transformer_sensor_input.json")  # Path to sensor inputs
    centroid_file = os.path.join(CENTROID_LABELS_ALL_SUBJECTS, subject, "centroids.json")      # Path to centroid labels
    output_file = os.path.join(TRANSFORMER_ALL_SUBJECTS, subject, "transformer_input.json")   # Output file

    # Check if sensor_file exists
    if not os.path.isfile(sensor_file):
        print(f"Sensor file not found for subject: {subject}. Skipping.")
        continue

    # Check if centroid_file exists
    if not os.path.isfile(centroid_file):
        print(f"Centroid file not found for subject: {subject}. Skipping.")
        continue

    # Load sensor data
    try:
        with open(sensor_file, 'r') as f:
            sensor_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Invalid JSON in sensor file for subject: {subject}. Skipping.")
        continue

    # Load centroid data
    try:
        with open(centroid_file, 'r') as f:
            centroid_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Invalid JSON in centroid file for subject: {subject}. Skipping.")
        continue

    # Initialize the transformer_data dictionary
    transformer_data = {}

    # Iterate through each sensor entry
    for sensor_entry in sensor_data:
        timestamp = sensor_entry.get("timestamp")
        features = sensor_entry.get("features")

        if not timestamp:
            # Skip entries without a timestamp
            continue

        # Process both _left and _right suffixes
        for suffix in ["_left", "_right"]:
            image_key = f"{timestamp}{suffix}.png"
            centroid_entry = centroid_data.get(image_key, [{"centroid": None}])  # Default to None if missing

            # Extract centroid
            centroid = centroid_entry[0].get("centroid")

            # Assign the centroid to the image_key
            transformer_data[image_key] = [{
                "centroid": centroid
            }]

    # Ensure the subject's output directory exists
    subject_output_dir = os.path.dirname(output_file)
    os.makedirs(subject_output_dir, exist_ok=True)

    # Save the combined data to the output file
    try:
        with open(output_file, 'w') as f:
            json.dump(transformer_data, f, indent=4)
        print(f"Combined data saved to {output_file}")
    except IOError as e:
        print(f"Failed to write to {output_file}: {e}")
        continue

print("All subjects processed successfully.")

# import pandas as pd
# import json
# import os
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # Load sensor and centroid data
# sensor_file = 'path_to_transformer_sensor_input.json'
# centroid_file = 'path_to_transformer_input.json'

# with open(sensor_file, 'r') as f:
#     sensor_data = json.load(f)

# with open(centroid_file, 'r') as f:
#     centroid_data = json.load(f)

# # Convert to DataFrame
# sensor_df = pd.DataFrame(sensor_data)
# centroid_df = pd.DataFrame.from_dict(centroid_data, orient='index').reset_index()
# centroid_df = centroid_df.rename(columns={'index': 'image_key', 0: 'centroid'})

# # Extract timestamp from image_key
# centroid_df['timestamp'] = centroid_df['image_key'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1] + '_' + x.split('_')[2])

# # Merge sensor and centroid data
# merged_df = pd.merge(sensor_df, centroid_df, on='timestamp', how='left')

# # Handle null centroids
# merged_df['centroid'] = merged_df['centroid'].apply(lambda x: x[0]['centroid'] if pd.notnull(x) else [None, None])
# merged_df[['x', 'y']] = pd.DataFrame(merged_df['centroid'].tolist(), index=merged_df.index)

# # Impute missing centroids (example: forward fill)
# merged_df[['x', 'y']] = merged_df[['x', 'y']].fillna(method='ffill')

# # Create lag features (previous centroid)
# merged_df['x_prev'] = merged_df['x'].shift(1)
# merged_df['y_prev'] = merged_df['y'].shift(1)

# # Drop the first row with NaN lag
# merged_df = merged_df.dropna()

# # Feature Columns
# feature_cols = ['Accelo_x', 'Accelo_y', 'Accelo_z',
#                'Gyro_x', 'Gyro_y', 'Gyro_z',
#                'Magneto_x', 'Magneto_y', 'Magneto_z',
#                'Wifi_FTM_li_range', 'Wifi_FTM_li_std', 'WiFi_rssi',
#                'x_prev', 'y_prev']

# # Target Columns
# target_cols = ['x', 'y']

# # Scaling
# scaler = StandardScaler()
# merged_df[feature_cols] = scaler.fit_transform(merged_df[feature_cols])

# # Save the preprocessed data
# merged_df.to_csv('preprocessed_data.csv', index=False)
