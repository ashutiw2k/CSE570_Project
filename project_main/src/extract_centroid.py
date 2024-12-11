import os
import json
import matplotlib.pyplot as plt

# Define your directory paths
BB_LABELS_ALL_SUBJECTS = "project_main/data/Masked BB Labels/"
TRANSFORMER_ALL_SUBJECTS = "project_main/data/Transformer Input/"

# Function to calculate centroid from bounding box
def calculate_centroid(bbox):
    x_min, y_min, x_max, y_max = bbox
    centroid_x = (x_min + x_max) / 2
    centroid_y = (y_min + y_max) / 2
    return [centroid_x, centroid_y]

# Iterate over each subject
for subject in os.listdir(BB_LABELS_ALL_SUBJECTS):
    bbox_file = os.path.join(BB_LABELS_ALL_SUBJECTS, subject, "bounding_boxes.json")
    output_file = os.path.join(TRANSFORMER_ALL_SUBJECTS, subject, "centroids.json")
    sensor_file = os.path.join(TRANSFORMER_ALL_SUBJECTS, subject, "transformer_sensor_input.json")

    # Load bounding box data
    try:
        with open(bbox_file, 'r') as f:
            bbox_data = json.load(f)
    except FileNotFoundError:
        print(f"Bounding box file not found for subject: {subject}. Skipping.")
        continue
    except json.JSONDecodeError:
        print(f"Invalid JSON in bounding box file for subject: {subject}. Skipping.")
        continue
    
    try:
        with open(sensor_file, 'r') as f:
            sensor_data = json.load(f)
    except FileNotFoundError:
        print(f"Sensor Data file not found for subject: {subject}. Skipping.")
        continue
    except json.JSONDecodeError:
        print(f"Invalid JSON in sensor data file for subject: {subject}. Skipping.")
        continue

    # Initialize the centriod_data dictionary
    centriod_data = {}

    # Lists to store centroids for plotting (optional)
    centroids_left = []
    centroids_right = []

    for sensor_entry in sensor_data:
        timestamp = sensor_entry.get("timestamp")

        if not timestamp:
            # Skip entries without a timestamp
            continue

        # Process both _left and _right suffixes
        for suffix in ["_left", "_right"]:
            image_key = f"{timestamp}{suffix}.png"
            bbox_labels = bbox_data.get(image_key)

            centroid = None
            if bbox_labels and bbox_labels != [{"bbox": [0, 0, 0, 0]}]:
                # Assuming there's only one bounding box per image
                bbox = bbox_labels[0].get("bbox")
                if bbox and isinstance(bbox, list) and len(bbox) == 4:
                    centroid = calculate_centroid(bbox)
                else:
                    centroid = (-1,-1) # Centriod is None.
            # else:
            #     # No valid bounding box present
            #     centroid = (-1,-1)

            # Assign the centroid to the image_key
                centriod_data[image_key] = [{"centroid": centroid}]

            # Collect centroids for plotting
            if centroid:
                if suffix == "_left":
                    centroids_left.append(centroid)
                else:
                    centroids_right.append(centroid)

    # Save the combined data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        with open(output_file, 'w') as f:
            json.dump(centriod_data, f, indent=4)
    except IOError as e:
        print(f"Failed to write to {output_file}: {e}")
        continue

    # # Plotting centroids (optional)
    # plt.figure(figsize=(10, 6))
    # if centroids_left:
    #     x_left, y_left = zip(*centroids_left)
    #     plt.scatter(x_left, y_left, c='blue', label='Left Centroids')
    # if centroids_right:
    #     x_right, y_right = zip(*centroids_right)
    #     plt.scatter(x_right, y_right, c='red', label='Right Centroids')

    # plt.title(f'Centroids for Subject: {subject}')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.legend()
    # plt.grid(True)
    # plot_file = os.path.join(TRANSFORMER_ALL_SUBJECTS, subject, "centroids_plot.png")
    # try:
    #     plt.savefig(plot_file)
    # except IOError as e:
    #     print(f"Failed to save plot for subject {subject}: {e}")
    # plt.close()

    print(f"Processed subject: {subject}")

print("All subjects processed successfully.")
