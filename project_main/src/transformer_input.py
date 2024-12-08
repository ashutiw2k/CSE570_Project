import json

# Paths to data files
BB_LABELS_ALL_SUBJECTS= 'project_main/data/Masked BB Labels/Subject0'
SENSOR_ALL_SUBJECTS= 'project_main/data/Transformer Input/Subject0'
TRANSFORMER_ALL_SUBJECTS= 'project_main/data/Transformer Input/Subject0'

# Load sensor inputs and bounding box labels
sensor_file = SENSOR_ALL_SUBJECTS + '/' + "transformer_sensor_input.json"  # Path to sensor inputs
bbox_file = BB_LABELS_ALL_SUBJECTS + '/' + "bounding_boxes.json"  # Path to bounding box labels
output_file = TRANSFORMER_ALL_SUBJECTS + '/' + "transformer_input.json"  # Output file

# Load data
with open(sensor_file, 'r') as f:
    sensor_data = json.load(f)

with open(bbox_file, 'r') as f:
    bbox_data = json.load(f)

# Combine data
transformer_data = []

for sensor_entry in sensor_data:
    timestamp = sensor_entry["timestamp"]
    features = sensor_entry["features"]

    # Create entries for `_left` and `_right`
    for suffix in ["_left", "_right"]:
        image_key = f"{timestamp}{suffix}.png"
        bbox_labels = bbox_data.get(image_key, [{"bbox": [0, 0, 0, 0]}])  # Default to [0, 0, 0, 0] if missing

        # Add combined entry
        transformer_data.append({
            "timestamp": timestamp,
            "side": suffix.strip('_'),
            "features": features,
            "bounding_boxes": [entry["bbox"] for entry in bbox_labels]  # List of bounding boxes
        })

# Save to output file
with open(output_file, 'w') as f:
    json.dump(transformer_data, f, indent=4)

print(f"Combined data saved to {output_file}")
